# FlowerClassify 花卉分类识别系统

*<u>v1.4.0 新变化：调整数据增强重新训练模型。</u>*

## 项目简介

本项目为一个基于 ResNet 的花卉分类识别系统，能有效区分多种不同类别的花卉（从数据集来看支持至少232种花卉），采用 ResNet 系列模型作为主干网络，包含模型的训练、测试以及线上部署（提供容器化部署）。

- 基于 [PyTorch](https://pytorch.org/) 框架进行模型的训练及测试。
- 模型采用 [ONNX](https://onnx.org.cn/onnx/index.html) 格式部署，采用 [ONNX Runtime](https://onnxruntime.ai/) 进行推理。
- 基于 [Flask](https://flask.palletsprojects.com/en/stable/) 框架实现 Web 接口。
- 使用 [Docker](https://www.docker.com/) 进行容器化部署。
- 支持单GPU训练和分布式训练两种模式。

训练数据集来自 [Kaggle](https://www.kaggle.com/)，融合了多个数据集并进行了数据清洗，基于预训练模型进行训练。

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

### Web 服务

基于 Flask 实现的 Web 服务提供花卉识别接口：
- 使用 ONNX Runtime 进行模型推理
- 支持图像上传和预处理
- 提供 JSON 格式的识别结果（类别索引和置信度）

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

| 字段名 | 字段描述 |
| :--------------------: | :---------------------------------------------------------------------------------------: |
| device | 设备名称，与 PyTorch 的设备名称保持一致。 |
| num-epochs | 训练迭代次数。 |
| num-workers | 训练及评估数据加载进程数。 |
| batch-size | 训练数据批大小。 |
| learning-rate | 模型训练学习率。 |
| weight-decay | 模型训练权重衰减。 |
| num-classes | 模型输出类别数。 |
| log-interval | 日志输出频率。 |
| load-pretrained | 是否使用预训练参数初始化模型权重。 |
| model-name | 使用的 ResNet 模型名称（resnet18/resnet34/resnet50）。 |
| use-layer-norm | 是否使用 LayerNorm 层。 |
| loss-function | 损失函数类型（默认为 cross_entropy）。 |
| optimizer-type | 优化器类型（默认为 adam）。 |
| lr-scheduler-type | 学习率调度器类型（默认为 step）。 |
| load-checkpoint | 是否加载 checkpoint 继续训练。 |
| load-checkpoint-path | 训练初始模型的加载路径。 |
| best-checkpoint-path | 训练中当前验证集最优模型保存路径。 |
| last-checkpoint-path | 训练中最后一次训练模型保存路径。 |

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

### 启动 Web 服务

本项目 Web 服务的配置文件为 `servers/configs/config.toml`，其中各个字段描述如下：

| 字段名        | 字段描述                                      |
|:----------:|:-----------------------------------------:|
| precision  | 模型推理精度，取值为 "fp32" (单精度) 和 "fp16" (半精度) 。  |
| providers  | 模型推理 ONNX Runtime Execution Providers 列表。 |
| model-path | 模型加载路径。                                   |

从本项目 Release 中下载 [ONNX](https://onnx.org.cn/onnx/index.html) 格式的模型权重文件放入 `servers/models` 目录后，执行以下命令启动 Web 服务：

```shell-session
flask --app servers.server run --host="0.0.0.0" --port=9500
```

### API 使用说明

Web 服务提供 `/flowerclassify` POST 接口用于花卉识别：

- 请求方式：POST
- 请求参数：表单数据，包含名为 'image' 的图像文件
- 返回格式：JSON，包含识别结果（类别索引和置信度）

示例响应：
```json
{
  "index": 164,
  "score": 0.987
}
```

### 构建镜像

模型部署前需要转换为 [ONNX](https://onnx.org.cn/onnx/index.html) 格式放入 `servers/models` 目录中。构建镜像使用的 Dockerfile 位于 `docker` 目录中，请参考 [Docker 官方文档](https://docs.docker.com/) 进行镜像的构建和容器的运行。

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
