# scripts/2_model_training.py
import pandas as pd
import numpy as np
import os  # 管理文件和路径
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  # 保存和加载训练好的模型

# 自动获取项目路径
# __file__  当前执行脚本的完整路径  绝对路径  os.path.dirname(路径)获取该路径的父目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # 再次获取父目录，得到项目根目录
# D:/desktop/py/gesture_recognition_project/  ← 项目根目录(PROJECT_ROOT)       scripts/  ← 脚本目录(SCRIPT_DIR)      2_model_training.py     ← 当前文件(__file__)


def find_latest_data_version():
    """自动找到最新的数据版本文件夹"""
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    if not os.path.exists(processed_dir):
        raise FileNotFoundError(f"处理后的数据目录不存在: {processed_dir}")

    # 找到所有版本文件夹
    versions = [d for d in os.listdir(
        processed_dir) if d.startswith("version_")]
    if not versions:
        raise FileNotFoundError("没有找到任何数据版本文件夹")

    # 返回最新的版本
    latest_version = sorted(versions)[-1]
    return os.path.join(processed_dir, latest_version)


def load_datasets(version_folder):
    """加载训练集、验证集、测试集"""
    print("加载数据集...")

    train_path = os.path.join(version_folder, "train_dataset.csv")
    val_path = os.path.join(version_folder, "val_dataset.csv")
    test_path = os.path.join(version_folder, "test_dataset.csv")

    # 检查文件是否存在
    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"数据文件不存在: {path}")

    # 读取数据
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print(f"✅ 训练集: {len(train_df)} 样本")
    print(f"✅ 验证集: {len(val_df)} 样本")
    print(f"✅ 测试集: {len(test_df)} 样本")

    # 分离特征和标签
    X_train = train_df.drop('label', axis=1)
    y_train = train_df['label']
    X_val = val_df.drop('label', axis=1)
    y_val = val_df['label']
    X_test = test_df.drop('label', axis=1)
    y_test = test_df['label']

    return X_train, y_train, X_val, y_val, X_test, y_test


def parameter_search(X_train, y_train, X_val, y_val):
    """参数搜索 - 找到最佳参数组合"""
    print("\n开始参数搜索...")

    # 定义参数网格
    param_grid = {
        'n_estimators': [50, 100, 150],    # 树的数量
        'max_depth': [10, 15, None],       # 树的最大深度
    }

    best_score = 0
    best_params = None
    best_model = None
    results = []

    # 遍历所有参数组合
    for n_est in param_grid['n_estimators']:
        for max_d in param_grid['max_depth']:
            print(f"尝试参数: n_estimators={n_est}, max_depth={max_d}", end="")

            # 训练模型
            model = RandomForestClassifier(
                n_estimators=n_est,
                max_depth=max_d,
                random_state=42  # 固定随机种子确保结果可重现
            )
            model.fit(X_train, y_train)

            # 在验证集上评估
            y_val_pred = model.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)

            print(f" → 验证集准确率: {val_accuracy:.3f}")

            results.append({
                'n_estimators': n_est,
                'max_depth': max_d,
                'val_accuracy': val_accuracy
            })

            # 更新最佳模型
            if val_accuracy > best_score:
                best_score = val_accuracy
                best_params = {'n_estimators': n_est, 'max_depth': max_d}
                best_model = model

    # 输出参数搜索结果
    print(f"\n最佳参数: {best_params}")
    print(f"最佳验证集准确率: {best_score:.3f}")

    return best_model, best_params, best_score, results


def evaluate_final_model(model, X_test, y_test):
    """在测试集上评估最终模型"""
    print("\n在测试集上评估最终模型...")

    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"  测试集准确率: {test_accuracy:.3f}")

    # 详细分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_test_pred))

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_test_pred)
    print("混淆矩阵:")
    print(cm)

    return test_accuracy


def save_model_and_results(model, best_params, val_accuracy, test_accuracy, search_results):
    """保存模型和训练结果"""
    # 创建模型保存目录
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)

    # 创建带时间戳的模型版本
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_version = f"gesture_model_{timestamp}"
    model_dir = os.path.join(models_dir, model_version)
    os.makedirs(model_dir, exist_ok=True)

    # 保存模型
    model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(model, model_path)

    # 保存训练结果
    results_info = {
        'model_version': model_version,
        'training_date': datetime.now().isoformat(),
        'best_parameters': best_params,
        'validation_accuracy': val_accuracy,
        'test_accuracy': test_accuracy,
        'parameter_search_results': search_results,
        'feature_dimension': model.n_features_in_,
        'classes': list(model.classes_)
    }

    results_path = os.path.join(model_dir, "training_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_info, f, indent=2, ensure_ascii=False)

    print(f"\n模型已保存: {model_path}")
    print(f"训练结果已保存: {results_path}")

    return model_dir


def main():
    """主函数"""
    print("手势识别 - 模型训练开始")
    print("=" * 50)

    try:
        # 1. 自动找到最新数据版本
        data_folder = find_latest_data_version()
        print(f"使用数据版本: {os.path.basename(data_folder)}")

        # 2. 加载数据
        X_train, y_train, X_val, y_val, X_test, y_test = load_datasets(
            data_folder)

        # 3. 参数搜索
        best_model, best_params, val_accuracy, search_results = parameter_search(
            X_train, y_train, X_val, y_val
        )

        # 4. 最终评估
        test_accuracy = evaluate_final_model(best_model, X_test, y_test)

        # 5. 保存结果
        model_dir = save_model_and_results(
            best_model, best_params, val_accuracy, test_accuracy, search_results
        )

        print(f"\n模型训练完成!")
        print(f"    模型位置: {model_dir}")
        print(f"    最终测试准确率: {test_accuracy:.3f}")

    except Exception as e:
        print(f"\n❌ 训练过程中出错: {e}")
        print("请检查数据文件是否存在且格式正确")


if __name__ == "__main__":
    main()
