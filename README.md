# EfficientNet-UNet
针对东北黑土区侵蚀沟遥感影像存在类别极端不平衡、细小沟谷边缘分割模糊、传统模型参数量大、推理速度慢等问题，本发明构建轻量化改进语义分割模型。 该方法以 EfficientNet-B4 作为编码器骨干，依托 MBConv 模块与 SE 注意力机制高效提取多尺度特征，结合 UNet 解码器结构与跳跃连接保留浅层细节信息。解码器采用渐进式混合激活函数，深层使用 ReLU、中层 LeakyReLU、浅层 SiLU，提升侵蚀沟边缘细节表征能力。同时构建Dice-Focal 复合损失函数，兼顾难易样本与正负样本不平衡问题，搭配 AdamW 优化器、学习率调度与早停策略，提升模型训练稳定性与泛化能力。



------

## 📁 项目结构

```
code/
├── .idea/                     
├── datasets/                   # 数据集目录
│   ├── test/
│   │   ├── images/             # 测试集影像
│   │   └── labels/             # 测试集标签
│   ├── train/
│   │   ├── images/             # 训练集影像
│   │   └── labels/             # 训练集标签
│   ├── val/
│   │   ├── images/             # 验证集影像
│   │   └── labels/             # 验证集标签
│   └── class_dict.csv          # 类别映射表
├── model/                      # 模型实现
│   ├── efficientunet.py        # 核心模型：EfficientNet-UNet
│   ├── linkent.py              # 对比模型：LinkNet
│   ├── manet.py                # 对比模型：MANet
│   ├── segnet.py               # 对比模型：SegNet
│   └── unet++.py               # 对比模型：UNet++
├── output/                     # 输出目录（权重、日志、预测结果）
├── utils/                     
│   ├── augmentation.py         # 数据增强
│   ├── Dataset.py              # 自定义数据集类
│   ├── datasets_split.py       # 数据集划分脚本
│   ├── DiceFocalLoss.py        # Dice-Focal混合复失函数
│   ├── LabelProcessor.py       # 标签预处理工具
│   ├── Metrics.py              # 评价指标计算
│   ├── config.py               # 全局配置（路径、超参数）
│   ├── draw.py                 # 绘图
│   ├── predict_image.py        # 预测
│   ├── test.py                 # 测试
│   └── train.py                # 训练（入口）
└── README.md                   # 项目说明文档
```

------

## 快速开始

### 1. 环境依赖

| 描述         | 工具          |
| ------------ | ------------- |
| 深度学习框架 | Pytorch       |
| 操作系统     | Ubuntu 24.04  |
| CUDA         | CUDA 12.8     |
| Arcgis Pro   | 10.8          |
| Python       | Python 3.11   |
| Pytorch      | Pytorch 2.0.0 |

### 2. 数据集准备

- 按 `datasets/train/val/test` 目录结构存放影像与标签
- 运行 `utils/datasets_split.py` 可自动完成数据集划分

### 3. 模型训练

```
python train.py
```

### 4. 模型测试

```
python test.py
```

### 5. 单张影像预测

```
python predict_image.py
```

------

##  核心特性

-  **轻量化主干**：EfficientNet-B4 作为编码器，兼顾精度与效率
- **类别不平衡优化**：Dice-Focal 复合损失函数，适配侵蚀沟小目标场景
-  **完整对比实验**：支持 UNet++、SegNet、LinkNet、MANet 等模型对比
- **模块化设计**：数据增强、损失函数、评价指标可灵活替换

------

## 实验结果

|         模型          |  mIoU(%)  | Recall(%) | Precision(%) | F1-Score  |
| :-------------------: | :-------: | :-------: | :----------: | :-------: |
|        UNet++         |   73.54   |   83.01   |    79.63     |   80.63   |
|        SegNet         |   69.75   |   72.86   |    71.82     |   72.31   |
|        LinkNet        |   76.39   |   84.32   |    82.95     |   83.61   |
|         MANet         |   78.86   |   86.54   |    84.88     |   85.69   |
| **EfficientNet-UNet** | **78.95** | **86.76** |  **86.68**   | **86.72** |
