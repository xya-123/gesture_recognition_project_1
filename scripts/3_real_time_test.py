# scripts/3_real_time_test.py
import cv2
import mediapipe as mp
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier

# 自动获取项目路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def find_latest_model():
    """自动找到最新的训练模型"""
    models_dir = os.path.join(PROJECT_ROOT, "models")
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"模型目录不存在: {models_dir}")

    # 找到所有模型版本文件夹
    model_versions = [d for d in os.listdir(
        models_dir) if d.startswith("gesture_model_")]
    if not model_versions:
        raise FileNotFoundError("没有找到任何训练好的模型")

    # 返回最新的模型版本
    latest_version = sorted(model_versions)[-1]
    model_dir = os.path.join(models_dir, latest_version)
    model_path = os.path.join(model_dir, "model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    return model_path, model_dir


def load_model(model_path):
    """加载训练好的模型"""
    print("加载训练好的模型...")
    model = joblib.load(model_path)
    print(f"模型加载成功! 特征维度: {model.n_features_in_}")
    print(f"支持的手势类别: {list(model.classes_)}")
    return model


def convert_to_relative(raw_landmarks):
    """
    将绝对坐标转换为相对坐标（与训练时一致）
    参数:
        raw_landmarks: 63个绝对坐标 [x0,y0,z0, x1,y1,z1, ...]
    返回:
        relative_landmarks: 63个相对坐标
    """
    relative_landmarks = []

    if len(raw_landmarks) < 63:
        return None

    # 手腕坐标（基准点）- 点0
    wrist_x = raw_landmarks[0]
    wrist_y = raw_landmarks[1]
    wrist_z = raw_landmarks[2]

    # 计算所有21个点相对于手腕的坐标
    for i in range(0, len(raw_landmarks), 3):
        rel_x = raw_landmarks[i] - wrist_x
        rel_y = raw_landmarks[i+1] - wrist_y
        rel_z = raw_landmarks[i+2] - wrist_z
        relative_landmarks.extend([rel_x, rel_y, rel_z])

    return relative_landmarks


def process_frame(frame, hands, model):
    """
    处理单帧图像：检测手部、提取特征、预测手势
    返回: 处理后的帧, 预测结果, 置信度
    """
    # 转换颜色空间
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe手部检测
    results = hands.process(rgb_frame)

    prediction = "无手势"
    confidence = 0.0
    hand_landmarks = None

    if results.multi_hand_landmarks:
        # 只处理第一只检测到的手
        hand_landmarks = results.multi_hand_landmarks[0]

        # 提取绝对坐标
        raw_landmarks = []
        for landmark in hand_landmarks.landmark:
            raw_landmarks.extend([landmark.x, landmark.y, landmark.z])

        # 转换为相对坐标
        relative_landmarks = convert_to_relative(raw_landmarks)

        if relative_landmarks:
            # 模型预测
            probabilities = model.predict_proba([relative_landmarks])[0]
            predicted_class_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_class_idx]

            # 只在高置信度时显示结果
            if confidence > 0.6:  # 置信度阈值
                prediction = model.classes_[predicted_class_idx]
            else:
                prediction = "不确定"
                confidence = 0.0

    return frame, prediction, confidence, hand_landmarks


def draw_landmarks(frame, hand_landmarks, prediction, confidence):
    """在帧上绘制手部关键点和识别结果"""
    if hand_landmarks:
        # 绘制手部关键点和连接线
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style()
        )

    # 绘制识别结果
    result_text = f"{prediction}"
    if confidence > 0:
        result_text += f" - {confidence:.1%}"

    # 根据置信度选择颜色
    if confidence > 0.8:
        color = (0, 255, 0)  # 绿色 - 高置信度
    elif confidence > 0.6:
        color = (0, 255, 255)  # 黄色 - 中置信度
    else:
        color = (0, 0, 255)  # 红色 - 低置信度或无手势

    # 在画面顶部显示结果
    cv2.putText(frame, result_text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    # 显示操作提示
    cv2.putText(frame, "按 'Q' 退出", (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame


def real_time_test():
    """实时手势识别测试"""
    print("启动实时手势识别测试")
    print("=" * 50)

    # 在try外面定义变量，确保finally能访问到
    cap = None

    try:
        # 1. 加载模型
        model_path, model_dir = find_latest_model()
        model = load_model(model_path)

        # 2. 初始化MediaPipe手部检测
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:

            # 3. 初始化摄像头
            print("初始化摄像头...")
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                raise RuntimeError("无法打开摄像头")

            # 设置摄像头分辨率
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            print("摄像头初始化成功!")
            print("提示: 面对摄像头做出手势，按 'Q' 键退出")
            print("-" * 50)

            # 4. 主循环
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ 无法读取摄像头帧")
                    break

                # 水平翻转帧（镜像效果，更符合直觉）
                frame = cv2.flip(frame, 1)

                # 处理当前帧
                processed_frame, prediction, confidence, hand_landmarks = process_frame(
                    frame, hands, model
                )

                # 绘制结果
                final_frame = draw_landmarks(
                    processed_frame, hand_landmarks, prediction, confidence
                )

                # 显示帧
                cv2.imshow('手势识别 - 实时测试', final_frame)

                # 检查退出键
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n用户退出")
                    break

    except FileNotFoundError as e:
        print(f"\n❌ 文件错误: {e}")
        print("请检查:")
        print("1. 是否已经运行过模型训练? (运行 2_model_training.py)")
        print("2. models/ 目录下是否有模型文件?")
        print("3. 模型文件路径是否正确?")

    except ImportError as e:
        print(f"\n❌ 导入错误: {e}")
        print("请检查依赖库:")
        print("1. MediaPipe: pip install mediapipe")
        print("2. scikit-learn: pip install scikit-learn")
        print("3. OpenCV: pip install opencv-python")

    except RuntimeError as e:
        if "摄像头" in str(e) or "camera" in str(e).lower():
            print(f"\n❌ 摄像头错误: {e}")
            print("请检查:")
            print("1. 摄像头是否被其他程序占用")
            print("2. 摄像头驱动是否正常")
            print("3. 尝试重启电脑或更换摄像头")
        else:
            print(f"\n❌ 运行时错误: {e}")

    except Exception as e:
        print(f"\n❌ 未知错误: {e}")
        print("错误类型:", type(e).__name__)
        print("请检查:")
        print("1. 所有依赖库版本")
        print("2. 系统环境配置")

    finally:
        # 清理资源
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("✅ 资源已释放")


def main():
    """主函数"""
    print("手势识别实时测试")
    print("功能: 实时摄像头手势识别演示")
    print("=" * 50)

    real_time_test()

    print("\n实时测试结束!")
    print("感谢使用手势识别系统!")


if __name__ == "__main__":
    main()
