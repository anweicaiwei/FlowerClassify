# FlowerClassify 花卉分类识别系统
<i><u>v2.0.0 新变化 

进行数据增强，包括随机水平翻转、随机垂直翻转、随机旋转、随机缩放、随机裁剪、颜色抖动等

模型结构优化，增加LayerNorm层、全局平均池化和扁平化、双 Dropout 层、两层全连接分类层

增加多种损失函数、优化器、学习率调度器等，以提高模型的训练效率和泛化能力，增加梯度裁剪 </u></i>

## 项目简介

本项目为一个基于 ResNet 的花卉分类识别系统，能有效区分多种不同类别的花卉，采用 ResNet 系列模型作为主干网络，包含模型的训练、测试。

- 基于 [PyTorch](https://pytorch.org/) 框架进行模型的训练及测试。
- 支持单GPU训练和分布式训练两种模式。

## 系统架构

### 模型架构

项目使用了自定义的 FlowerNet 模型，基于 ResNet 架构实现，支持以下 ResNet 变体：
- ResNet18
- ResNet34
- ResNet50

模型主要特点：
- 保留 ResNet 的卷积层和池化层作为特征提取器
- 添加双 Dropout 层防止过拟合
- 可选的 LayerNorm 层增强模型稳定性
- 两层全连接分类器增强分类能力
- 支持预训练权重加载

### 数据处理

项目使用 FlowerDataset 类进行数据加载和预处理：
- 支持从 CSV 文件读取数据集信息
- 包含图像读取和异常处理机制
- 支持图像变换和数据增强
- 数据集按类别分割为 train/valid/test 三部分

## 使用说明

### 安装环境依赖

首先使用 pip 安装本项目相关的依赖包：

```shell-session
pip install -r requirements.txt
```

### 数据准备

项目已提供 `data_preparation.py` 脚本用于数据集预处理：
- 支持将数据集分割为 train/valid/test 三部分
- 生成对应的 CSV 文件
- 自动创建类别映射

### 模型训练

#### 单GPU训练

若要使用单GPU训练模型，运行以下命令：

```shell-session
python train.py
```

训练配置文件为 `configs/config_OneGPU.toml`，主要配置项包括：

|           字段名           |                     字段描述                     |
|:-----------------------:|:--------------------------------------------:|
|         device          |          设备名称，与 PyTorch 的设备名称保持一致。           |
|        data-root        |                   原始数据集路径                    |
|       data-label        |                    标签文件路径                    |
|    valid-split-ratio    |                   验证集划分比例                    |
|    test-split-ratio     |                   测试集划分比例                    |
|       num-epochs        |                   训练迭代次数。                    |
|       num-workers       |                训练及评估数据加载进程数。                 |
|       batch-size        |                   训练数据批大小。                   |
|      learning-rate      |                   模型训练学习率。                   |
|      weight-decay       |                  模型训练权重衰减。                   |
|       num-classes       |                   模型输出类别数。                   |
|      log-interval       |                   日志输出频率。                    |
|     load-pretrained     |              是否使用预训练参数初始化模型权重。               |
|       model-name        | 使用的 ResNet 模型名称（resnet18/resnet34/resnet50）。 |
|     use-layer-norm      |              是否使用 LayerNorm 层。               |
|      loss-function      |  损失函数类型（默认为 l1_regularized_cross_entropy）。   |
|     optimizer-type      |               优化器类型（默认为 adam）。               |
|    lr-scheduler-type    |            学习率调度器类型（默认为 cosine）。             |
| lr-scheduler-step-size  |                  学习率调度器步长。                   |
|   lr-scheduler-gamma    |                  学习率调度器衰减率。                  |
| early-stopping-patience |                   早停机制耐心值。                   |
|        l1-lambda        |                   L1正则化系数。                   |
|      use-grad-clip      |                  是否使用梯度裁剪。                   |
|     grad-clip-value     |                   梯度裁剪阈值。                    |
|     load-checkpoint     |            是否加载 checkpoint 继续训练。             |
|  load-checkpoint-path   |                 训练初始模型的加载路径。                 |
|  checkpoint-timestamp   |                  检查点时间戳目录。                   |
|     checkpoint-type     |          检查点类型（可选值为"best"或"last"）。           |
| custom-checkpoint-path  |                  自定义检查点路径。                   |
|  best-checkpoint-path   |              训练中当前验证集最优模型保存路径。               |
|  last-checkpoint-path   |               训练中最后一次训练模型保存路径。               |

#### 分布式训练

若要使用多GPU分布式训练，运行以下命令：

```shell-session
sh train_distributed.sh
```

Windows系统下可使用批处理替代方案。分布式训练配置文件为 `configs/config_Distributed.toml`。

### 模型评估

模型训练完成后，运行以下命令评估模型性能：

```shell-session
python eval.py
```

评估脚本会加载测试集并计算模型在测试集上的准确率。

## 支持的花卉种类

根据数据集 `test_split.csv` 显示，系统支持识别多种花卉，包括但不限于：
- 紫叶竹节秋海棠（紫竹梅）
- 龙牙草（仙鹤草）
- 络石（风车茉莉）
- 湖北荚蒾
- 旱金莲
- 白花雪果
- 吊兰
- ...等多种花卉

### 数据集格式
```csv
filename,category_id,chinese_name,english_name
img_000051.jpg,164,紫叶竹节秋海棠（紫竹梅）,Tradescantia pallida
img_000052.jpg,164,紫叶竹节秋海棠（紫竹梅）,Tradescantia pallida
img_000053.jpg,164,紫叶竹节秋海棠（紫竹梅）,Tradescantia pallida
img_000054.jpg,164,紫叶竹节秋海棠（紫竹梅）,Tradescantia pallida
img_000055.jpg,164,紫叶竹节秋海棠（紫竹梅）,Tradescantia pallida
img_000056.jpg,164,紫叶竹节秋海棠（紫竹梅）,Tradescantia pallida
...
```


## 项目结构

```
FlowerClassify/ 
├── checkpoints/ # 模型权重保存目录 
│ ├── OneGPU/ # 单GPU训练模型 
│ └── Distributed/ # 分布式训练模型 
├── configs/ # 配置文件目录 
│ ├── config_OneGPU.toml # 单GPU训练配置 
│ └── config_Distributed.toml # 分布式训练配置 
├── datasets/ # 数据集目录 
│ ├── train/ # 训练图像数据 
│ ├── valid/ # 验证图像数据 
│ ├── test/ # 测试图像数据 
│ ├── train_split.csv # 训练集分割文件 
│ ├── valid_split.csv # 验证集分割文件 
│ └── test_split.csv # 测试集分割文件 
├── training_logs/ # 训练日志目录 
├── utils/ # 工具脚本目录 
│ ├── logging_utils.py # 日志工具脚本 
│ ├── optim_utils.py # 优化器选择工具脚本 
│ └── plot_utils.py # 绘图工具脚本 
├── data_preparation.py # 数据预处理脚本 
├── models.py # 模型定义 
├── train.py # 单GPU训练脚本 
├── train_distributed.py # 分布式训练脚本 
├── train_distributed.sh # 分布式训练启动脚本 
├── eval.py # 模型评估脚本 
├── requirements.txt # 项目依赖 
└── README.md # 项目说明文档
```
