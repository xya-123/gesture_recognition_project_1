# scripts/1_data_processing.py
import cv2
import mediapipe as mp
import os
import pandas as pd
import numpy as np
from datetime import datetime
import json

# 改进的路径配置：自动获取项目根目录
# 假设脚本结构：gesture_recognition_project/scripts/1_data_processing.py
# 原先硬编码  PROJECT_ROOT = r"d:\desktop\py\gesture_recognition_project"  如果把项目复制到U盘，路径变成 e:/my_project，代码就报错了
# __file__  永远代表当前正在运行的Python文件的完整路径     此处绝对路径——>父文件夹 scripts
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # 项目根目录路径
RAW_VIDEOS_PATH = os.path.join(PROJECT_ROOT, "data", "raw_videos")  # 构建原始视频路径


def convert_to_relative(raw_landmarks):
    """
    将绝对坐标转换为相对坐标（以手腕为基准），保留所有21个点
    参数:
        raw_landmarks: 63个绝对坐标 [x0,y0,z0, x1,y1,z1, ...]
    返回:
        relative_landmarks: 63个相对坐标 [x0-x0,y0-y0,z0-z0, x1-x0,y1-y0,z1-z0, ...]
    """
    relative_landmarks = []

    if len(raw_landmarks) < 63:
        raise ValueError(f"Expected 63 coordinates, got {len(raw_landmarks)}")

    # 手腕坐标（基准点）- 点0
    wrist_x = raw_landmarks[0]  # x0
    wrist_y = raw_landmarks[1]  # y0
    wrist_z = raw_landmarks[2]  # z0

    # 计算所有21个点相对于手腕的坐标（包括手腕自己）
    for i in range(0, len(raw_landmarks), 3):
        rel_x = raw_landmarks[i] - wrist_x
        rel_y = raw_landmarks[i+1] - wrist_y
        rel_z = raw_landmarks[i+2] - wrist_z

        # extend！  不用append [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3], ...] 这是一个"列表的列表" 不是扁平结构
        relative_landmarks.extend([rel_x, rel_y, rel_z])

    return relative_landmarks


def extract_features_from_video(video_path):
    """从视频提取相对坐标特征"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return [], [], {"status": "ERROR", "detection_rate": 0}

    original_landmarks = []   # 原始相对坐标
    mirrored_landmarks = []   # 镜像相对坐标（通过图像镜像） 更接近mediapipe的识别原理 不区分左右手
    total_frames = 0
    detected_frames = 0
    frame_interval = 5  # 每5帧采样1帧，避免过于相似的数据

    try:
        # 使用 with 语句管理 MediaPipe 资源    with 创建一个对象 as 变量名:       在这里使用这个变量 离开这个区域时，自动清理资源
        with mp.solutions.hands.Hands(
            # True：图片模式 - 每张图片都重新检测（精度高，速度慢）   False：视频模式 - 第一帧检测，后续帧跟踪（速度快）
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,  # 当MediaPipe有50%以上把握认为"这是手"时，才返回结果
            min_tracking_confidence=0.5  # 在视频模式下，如果跟踪置信度低于这个值，就重新检测而不是继续跟踪 防止手部移动太快时跟丢
        ) as hands:

            while True:  # cap：之前创建的 VideoCapture 对象      .read()：读取下一帧视频
                ret, frame = cap.read()  # ret：布尔值 - 是否成功读取到帧   frame：图像数据 - 当前帧的像素数据
                if not ret:
                    break

                total_frames += 1
                if total_frames % frame_interval != 0:
                    continue

                # 处理原始图像
                # OpenCV 和 MediaPipe 使用不同的颜色格式
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # OpenCV 默认使用 BGR 格式（蓝-绿-红）MediaPipe 要 RGB 格式（红-绿-蓝）
                results_original = hands.process(rgb_frame)

                frame_detected = False

                # MediaPipe 的返回对象结构  当调用 hands.process(image) 时，返回的 results 对象包含几个属性：
                if results_original.multi_hand_landmarks:
                    # results.multi_hand_landmarks  手部关键点数据   results.multi_handedness 左手/右手信息    results.multi_hand_world_landmarks 3D世界坐标
                    # 1. 提取原始绝对坐标               # results.multi_hand_landmarks 是一个列表   每只手对应一个 hand_landmarks 对象
                    frame_detected = True
                    raw_landmarks = []
                    # 第一只手
                    for landmark in results_original.multi_hand_landmarks[0].landmark:
                        raw_landmarks.extend(
                            [landmark.x, landmark.y, landmark.z])
                        # 原始图像 → raw_landmarks → original_landmarks
                    # 2. 转换为相对坐标                                                        镜像图像 → raw_mirrored → mirrored_landmarks
                    relative_landmarks = convert_to_relative(raw_landmarks)
                    original_landmarks.append(relative_landmarks)

                # 3. 处理镜像图像（数据增强）
                frame_mirrored = cv2.flip(frame, 1)  # 1代表水平翻转
                rgb_mirrored = cv2.cvtColor(frame_mirrored, cv2.COLOR_BGR2RGB)
                results_mirrored = hands.process(rgb_mirrored)

                if results_mirrored.multi_hand_landmarks:
                    # 提取镜像图像的绝对坐标
                    frame_detected = True
                    raw_mirrored = []
                    for landmark in results_mirrored.multi_hand_landmarks[0].landmark:
                        raw_mirrored.extend(
                            [landmark.x, landmark.y, landmark.z])

                    # 转换为相对坐标
                    relative_mirrored = convert_to_relative(raw_mirrored)
                    mirrored_landmarks.append(relative_mirrored)

                if frame_detected:
                    detected_frames += 1

    except Exception as e:
        print(f"处理视频时出错 {video_path}: {e}")
        return [], [], {"status": "ERROR", "detection_rate": 0}
    finally:
        cap.release()

    # 计算检测率
    sampled_frames = total_frames // frame_interval
    detection_rate = detected_frames / sampled_frames if sampled_frames > 0 else 0

    quality_info = {
        "detection_rate": round(detection_rate, 3),  # 四舍五入保留3位小数
        "status": "PASS" if detection_rate > 0.8 else "FAIL",
        "original_samples": len(original_landmarks),
        "mirrored_samples": len(mirrored_landmarks),
        "total_samples": len(original_landmarks) + len(mirrored_landmarks)
    }

    return original_landmarks, mirrored_landmarks, quality_info


def process_all_videos():
    """处理所有视频"""
    print("开始处理视频（改进的相对坐标版本）...")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"视频数据路径: {RAW_VIDEOS_PATH}")

    if not os.path.exists(RAW_VIDEOS_PATH):
        print(f"错误：视频路径不存在: {RAW_VIDEOS_PATH}")
        return [], []

    all_features = []
    quality_reports = []

    processed_count = 0
    skipped_count = 0

    for gesture in os.listdir(RAW_VIDEOS_PATH):
        gesture_path = os.path.join(RAW_VIDEOS_PATH, gesture)
        if not os.path.isdir(gesture_path):
            continue

        print(f"\n 处理手势类别: {gesture}")

        for person in os.listdir(gesture_path):
            person_path = os.path.join(gesture_path, person)
            if not os.path.isdir(person_path):
                continue

            for i in range(1, 5):  # 每个手势有4个视频
                video_file = f"{person}_{gesture}_{i}.mp4"
                video_path = os.path.join(person_path, video_file)

                if not os.path.exists(video_path):
                    print(f"  视频文件不存在: {video_file}")
                    skipped_count += 1
                    continue

                print(f"  处理: {video_file}")
                original_features, mirrored_features, quality = extract_features_from_video(
                    video_path)

                # 空值检查：如果没有任何有效样本，跳过
                if len(original_features) == 0 and len(mirrored_features) == 0:
                    print(f"  无有效样本，跳过: {video_file}")
                    skipped_count += 1
                    continue

                quality.update({
                    "gesture": gesture,
                    "person": person,
                    "video_file": video_file
                })
                quality_reports.append(quality)

                # 保存特征数据
                for landmarks in original_features:
                    all_features.append(landmarks + [gesture])  # 添加标签
                for landmarks in mirrored_features:
                    all_features.append(landmarks + [gesture])  # 添加标签

                processed_count += 1
                status_icon = "✅" if quality["status"] == "PASS" else "⚠️"
                print(
                    f"   {status_icon} {video_file} - 样本: {quality['original_samples']}+{quality['mirrored_samples']} = {quality['total_samples']} (检测率: {quality['detection_rate']:.1%})")

    print(f"\n 处理总结:")
    print(f"   成功处理: {processed_count} 个视频")
    print(f"   跳过: {skipped_count} 个视频")
    print(f"   总样本数: {len(all_features)}")

    return all_features, quality_reports


def save_datasets(features_data, quality_data):
    """保存数据集"""
    if len(features_data) == 0:
        print(" 错误：没有有效数据可保存！")
        return None

    print(" 保存数据集...")

    # 创建版本文件夹
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_folder = os.path.join(
        PROJECT_ROOT, "data", "processed", f"version_{version}")
    os.makedirs(version_folder, exist_ok=True)

    #  注意：现在是63个特征（21个点×3坐标）
    columns = []
    for i in range(0, 21):  # 从点0到点20
        columns.extend([f'x{i}', f'y{i}', f'z{i}'])
    columns.append('label')

    full_df = pd.DataFrame(features_data, columns=columns)

    # 划分数据集
    from sklearn.model_selection import train_test_split
    X = full_df.drop('label', axis=1)
    y = full_df['label']

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp
    )  # 0.125 * 0.8 = 0.1，所以是 70% train, 10% val, 20% test

    # 保存文件
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(os.path.join(
        version_folder, "train_dataset.csv"), index=False)
    val_df.to_csv(os.path.join(version_folder, "val_dataset.csv"), index=False)
    test_df.to_csv(os.path.join(
        version_folder, "test_dataset.csv"), index=False)
    full_df.to_csv(os.path.join(
        version_folder, "full_dataset.csv"), index=False)

    # 保存质量报告
    pd.DataFrame(quality_data).to_csv(
        os.path.join(version_folder, "quality_report.csv"),
        index=False
    )

    # 版本信息
    version_info = {
        "version": version,
        "feature_type": "relative_coordinates_21points",  # 标明特征类型
        "feature_dimension": 63,                          # 63维特征
        "total_samples": len(full_df),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "classes": list(y.unique()),
        "class_distribution": y.value_counts().to_dict(),
        "processing_date": datetime.now().isoformat()
    }

    with open(os.path.join(version_folder, "version_info.json"), 'w', encoding='utf-8') as f:
        json.dump(version_info, f, indent=2, ensure_ascii=False)

    print(f"\n 数据集统计（改进的相对坐标）:")
    print(f"   特征维度: 63维 (21个关键点×3坐标)")
    print(f"   训练集: {len(X_train)} 样本 ({len(X_train)/len(full_df):.1%})")
    print(f"   验证集: {len(X_val)} 样本 ({len(X_val)/len(full_df):.1%})")
    print(f"   测试集: {len(X_test)} 样本 ({len(X_test)/len(full_df):.1%})")
    print(f"   总样本: {len(full_df)} 样本")
    print(f"   手势类别: {list(y.unique())}")
    print(f"   类别分布: {y.value_counts().to_dict()}")

    return version_folder


if __name__ == "__main__":
    print(" 手势识别项目 - 改进的相对坐标版本")
    print("=" * 50)
    print("特征: 63维相对坐标 (保留所有21个关键点)")
    print("增强: 图像镜像 + 相对坐标转换")
    print("=" * 50)

    features, quality_info = process_all_videos()

    if len(features) > 0:
        version_path = save_datasets(features, quality_info)
        print(f"\n 处理完成!")
        print(f"   数据版本: {version_path}")
        print(f"   下一步: 运行 2_model_training.py 来训练模型")
    else:
        print("\n 处理失败：没有生成任何有效数据！")
        print("   请检查：")
        print("   1. RAW_VIDEOS_PATH 路径是否正确")
        print("   2. 视频文件是否存在且格式正确")
        print("   3. 摄像头或MediaPipe是否能正常检测手部")
