# 手势识别系统

一个基于传统机器学习的手势识别项目，使用MediaPipe提取手部关键点，随机森林进行分类，训练好模型后可以实现实时手势识别。

## 识别手势

- ✌️ peace
- ✊ fist
- ✋ palm
- 🤟 love_you
- 👌 ok
- 👉 finger_gun

## 📁 项目结构

``` text
gesture_recognition_project/
├── data/                       # 数据目录
│   ├── raw_videos/             # 原始视频数据
│   │   ├── peace/ 
│   │   ├── fist/
│   │   ├── palm/ 
│   │   ├── love_you/
│   │   ├── ok/  
│   │   └── finger_gun/
│   └── processed/              # 处理后的特征数据
├── scripts/                    # 核心代码
│   ├── 1_data_processing.py
│   ├── 2_model_training.py  
│   └── 3_real_time_test.py
├── models/                     # 训练好的模型
├── presentations/              # 我个人的汇报ppt记录一下捏
├── requirements.txt            # 依赖包列表
├── PROJECT_JOURNEY.md          # 个人实践过程中遇到的问题、解决方法和心得总结
└── README.md                   # 项目说明
```

## 快速开始

### 环境要求

- Python 3.8+ (推荐3.8-3.11)
- 摄像头设备

### 安装依赖

```bash
pip install -r requirements.txt
```

### 使用步骤

```bash
cd scripts

# 选项1: 直接运行实时识别（使用预训练模型）
python 3_real_time_test.py

# 选项2: 完整流程（从零开始训练 可以根据自己的需求更改视频原素材raw_videos）
python 1_data_processing.py  # 数据预处理
python 2_model_training.py   # 模型训练  
python 3_real_time_test.py   # 实时测试
```

## 总体表现

本项目作为我的第一个学习项目收获颇丰，也发现了很多问题，到时候写进**PROJECT_JOURNEY.md**中，也会继续采用深度学习的方法来进一步研究。等我有时间了再更新吧。

总体来说，本框架用较为简单的方法实现了对简单手势的识别，非常适合新手学习。之所以是简单手势，是因为我发现像原定动作**call_me**已经很难实现准确的、合理的识别，更换的**ok**也与理想差别甚远。有些动作在人眼中看也许差别不大，但对机器来说却完全不同，尤其涉及到旋转、角度等问题时。后续会在**PROJECT_JOURNEY.md**中详细说明。

动作受**角度**、**正反**、**远近**、**环境亮度**等影响大，不如静态测试那么完美。两手结果较为相似，关于镜像图片的选择是正确的。

简单手势，比如项目里面的**fist**、**palm**、**love_you**、**finger_gun**，基本能准确识别，得分很高，90-100%。**peace**也能过关，只是受各因素影响较大，80—90%。**ok**已经经过反复拍摄了，依然受角度影响大，需要选择很严格的角度，得分差异大。

所以建议多多尝试，选择合适的环境。
