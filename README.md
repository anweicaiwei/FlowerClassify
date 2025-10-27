# 花卉分类项目

## 项目概述
这是一个基于深度学习的花卉图像分类项目，使用ResNet系列模型对花卉图像进行自动分类。项目支持多种模型架构、数据增强、正则化技术以及丰富的优化策略，可以高效地训练和部署花卉分类模型。

## 项目结构
```angular2html
├── .git/ # Git版本控制文件 
├── .idea/ # IDE配置文件 
├── code/ # 源代码目录 
│ ├── pycache/ # Python编译缓存 
│ ├── models.py # 模型定义 
│ ├── predict.py # 预测脚本 
│ ├── requirements.txt # 依赖包列表 
│ ├── train.py # 训练脚本 
│ └── utils.py # 工具函数 
├── model/ # 模型存储目录 
│ ├── best-model.pt # 最佳模型权重 
│ └── config.toml # 配置文件 
├── results/ # 结果存储目录 
│ └── submission.csv # 预测结果 
└── test/ # 测试数据目录 
├── img_000056.jpg # 测试图像 
├── img_000060.jpg # 测试图像 
└── ... # 更多测试图像
```


## 核心功能

### 1. 模型架构
- 基于ResNet系列模型（resnet18、resnet34、resnet50）
- 自定义FlowerNet类，包含卷积层、池化层、Dropout层
- 支持LayerNorm层和全连接分类层

### 2. 数据处理
- 支持训练集和验证集自动划分
- 提供数据增强变换
- 支持带标签和无标签两种测试模式

### 3. 训练流程
- 实现早停机制，防止过拟合
- 支持梯度裁剪，稳定训练过程
- 自动保存最佳模型和最新模型
- 支持多种损失函数和优化器

### 4. 预测功能
- 批量处理测试图像
- 计算置信度并生成预测结果
- 输出CSV格式的预测结果文件

## 技术栈
- Python 3.9+
- PyTorch
- torchvision
- tqdm (4.67.1)
- numpy
- pandas

## 安装指南

1. 克隆项目代码

2. 安装依赖包
```bash
pip install -r code/requirements.txt
```

## 使用指南

### 配置项目
编辑`model/config.toml`文件，设置相关参数：
- 计算设备：`device = "cuda"`（或"cpu"）
- 数据集路径：`data-root`和`data-label`
- 模型架构：`model-name = "resnet34"`
- 训练参数：`batch-size`, `num-epochs`等
- 优化策略：`loss-function`, `optimizer-type`, `learning-rate`等

### 训练模型
```bash
python code/train.py
```

### 生成预测结果
```bash
python code/predict.py test/test results/submission.csv
```

## 高级功能

### 数据增强
- 支持图像Resize、Normalize等变换
- 可自定义数据增强策略

### 正则化技术
- 支持Dropout (0.5和0.3)
- 支持LayerNorm
- 支持L1正则化

### 优化策略
- 支持多种优化器：adam、sgd、rmsprop
- 支持多种学习率调度器：step、multi_step、exponential、cosine、cosine_warm_restarts
- 支持早停机制

## 性能评估
模型训练过程中会自动计算验证集准确率，并保存准确率最高的模型。训练结束后，会显示最佳准确率和最终准确率。

## 注意事项
- 确保训练数据和测试数据格式正确
- 训练过程中会自动使用GPU（如果可用）
- 如需调整模型复杂度，可以在config.toml中修改model-name参数

## License
未指定
