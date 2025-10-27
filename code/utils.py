import os  # 导入os用于路径处理

import numpy as np
import pandas as pd  # 导入pandas用于读取CSV
import toml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, Subset
from torchvision.io import read_image  # 导入读取图像的函数

# 加载配置
try:
    configs = toml.load('model/config.toml')
except Exception as e:
    print(f"警告：无法加载配置文件，某些功能可能受限: {e}")
    configs = {}

# ===== 数据准备相关功能 =====

def create_category_mapping(df):
    """创建类别ID到索引的映射，确保类别索引从0开始连续"""
    unique_categories = sorted(df['category_id'].unique())
    return {category: idx for idx, category in enumerate(unique_categories)}


class FlowerDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """初始化数据集
        Args:
            csv_file (string): 标签CSV文件的路径
            img_dir (string): 图像文件夹的路径
            transform (callable, optional): 应用于图像的变换函数
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        # 创建类别映射
        self.category_to_idx = create_category_mapping(self.data_frame)
        self.num_classes = len(self.category_to_idx)

    # 返回数据集大小
    def __len__(self):
        return len(self.data_frame)

    # 获取数据集中的第idx个样本
    def __getitem__(self, idx):
        # 获取图像文件名和类别ID
        img_name = self.data_frame.iloc[idx, 0]
        category_id = self.data_frame.iloc[idx, 1]

        # 构建完整的图像路径
        img_path = os.path.join(self.img_dir, img_name)

        # 读取图像，添加异常处理
        try:
            image = read_image(img_path)

            # 应用变换
            if self.transform:
                image = self.transform(image)

            # 将类别ID映射为索引
            label = self.category_to_idx[category_id]

            return image, label
        except Exception as e:
            # 处理图像读取错误，记录错误信息并跳过该图像
            print(f"Error reading image {img_path}: {e}")
            # 返回前一个有效的图像，防止训练中断
            if idx > 0:
                return self.__getitem__(idx - 1)
            else:
                # 如果是第一个图像，创建一个空白图像和随机标签
                dummy_image = torch.zeros(3, 224, 224, dtype=torch.float32)
                dummy_label = 0
                return dummy_image, dummy_label


# ===== 仅用于无标签推理的数据集类 =====
class InferenceDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        """初始化无标签推理数据集
        Args:
            img_dir (string): 图像文件夹的路径
            transform (callable, optional): 应用于图像的变换函数
        """
        self.img_dir = img_dir
        self.transform = transform
        # 获取图像文件夹中的所有图像文件
        self.image_files = []
        # 支持的图像文件扩展名
        supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        
        # 遍历文件夹获取图像文件
        try:
            for file in os.listdir(img_dir):
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in supported_extensions:
                    self.image_files.append(file)
        except Exception as e:
            print(f"读取图像文件夹时出错: {e}")
        
        # 类别数量从配置中获取
        self.num_classes = configs.get('num-classes', 102)  # 假设默认有102个花卉类别
        
        # 创建默认的类别映射（索引到类别ID）
        self.category_to_idx = {i: i for i in range(self.num_classes)}

    # 返回数据集大小
    def __len__(self):
        return len(self.image_files)

    # 获取数据集中的第idx个样本
    def __getitem__(self, idx):
        # 获取图像文件名
        img_name = self.image_files[idx]
        
        # 构建完整的图像路径
        img_path = os.path.join(self.img_dir, img_name)

        # 读取图像，添加异常处理
        try:
            image = read_image(img_path)

            # 应用变换
            if self.transform:
                image = self.transform(image)

            # 返回图像、占位符标签和文件名
            return image, -1, img_name
        except Exception as e:
            # 处理图像读取错误
            print(f"读取图像 {img_path} 时出错: {e}")
            # 返回空白图像和占位符信息
            dummy_image = torch.zeros(3, 224, 224, dtype=torch.float32)
            return dummy_image, -1, f"error_{idx}.jpg"


# ========== 用于predict.py的测试集相关代码 ==========

def get_test_dataset(test_img_dir, transform=None):
    """创建无标签测试集数据集（仅在predict.py中使用）
    Args:
        test_img_dir (string): 测试集图像文件夹的路径
        transform (callable, optional): 应用于图像的变换函数
    Returns:
        Dataset: 无标签测试集数据集对象
    """
    # 检查图像目录是否存在
    if not test_img_dir or not os.path.exists(test_img_dir):
        print(f"错误: 测试图像目录不存在或未指定: {test_img_dir}")
        return None
    
    # 创建无标签推理数据集
    try:
        test_dataset = InferenceDataset(test_img_dir, transform)
        print(f"测试集大小: {len(test_dataset)}")
        print(f"使用的类别数量: {test_dataset.num_classes}")
        return test_dataset
    except Exception as e:
        print(f"创建无标签推理数据集时出错: {e}")
        return None


# 处理训练集和验证集的划分

def prepare_train_valid_datasets(csv_file, img_dir, valid_ratio=0.2):
    """准备训练集和验证集
    Args:
        csv_file (string): 原始标签CSV文件的路径
        img_dir (string): 原始图像文件夹的路径
        valid_ratio (float): 验证集占总数据集的比例
    Returns:
        tuple: (train_indices, valid_indices, category_to_idx, num_classes)
               训练集和验证集的索引列表，类别映射字典，类别数量
    """
    # 读取原始CSV文件
    df = pd.read_csv(csv_file)

    # 创建类别映射
    category_to_idx = create_category_mapping(df)
    num_classes = len(category_to_idx)

    # 按类别分割数据集，确保每个类别的数据都能被合理分割
    train_indices = []
    valid_indices = []

    # 按类别ID分组
    grouped = df.groupby('category_id')

    # 为每个类别分配原始索引
    all_indices = {}
    for category_id, group in grouped:
        all_indices[category_id] = group.index.tolist()

    # 分割数据
    for category_id, indices in all_indices.items():
        # 对每个类别的索引进行打乱
        np.random.seed(42)
        np.random.shuffle(indices)

        # 计算验证集大小
        valid_size = int(len(indices) * valid_ratio)

        # 分割数据
        valid_indices.extend(indices[:valid_size])
        train_indices.extend(indices[valid_size:])

    # 输出统计信息
    print(f"训练集大小: {len(train_indices)}")
    print(f"验证集大小: {len(valid_indices)}")
    print(f"类别数量: {num_classes}")

    return train_indices, valid_indices, category_to_idx, num_classes


# 根据索引创建Subset数据集

def create_subset_datasets(dataset, train_indices, valid_indices):
    """根据索引创建训练集和验证集的Subset
    Args:
        dataset (Dataset): 完整的数据集对象
        train_indices (list): 训练集索引列表
        valid_indices (list): 验证集索引列表
    Returns:
        tuple: (train_dataset, valid_dataset)
    """
    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)
    return train_dataset, valid_dataset


# 直接创建完整数据集并划分训练集和验证集

def get_train_valid_datasets(csv_file, img_dir, valid_ratio=0.2, transform=None):
    """直接创建完整数据集并划分训练集和验证集
    Args:
        csv_file (string): 标签CSV文件的路径
        img_dir (string): 图像文件夹的路径
        valid_ratio (float): 验证集占总数据集的比例
        transform (callable, optional): 应用于图像的变换函数
    Returns:
        tuple: (train_dataset, valid_dataset, category_to_idx, num_classes)
    """
    # 创建完整数据集
    full_dataset = FlowerDataset(csv_file, img_dir, transform)

    # 获取划分的索引
    train_indices, valid_indices, category_to_idx, num_classes = \
        prepare_train_valid_datasets(csv_file, img_dir, valid_ratio)

    # 创建Subset数据集
    train_dataset, valid_dataset = create_subset_datasets(full_dataset, train_indices, valid_indices)

    return train_dataset, valid_dataset, category_to_idx, num_classes


# ========== 以下是仅在predict.py中使用的测试集相关代码 ==========

def get_test_dataset(test_csv_file=None, test_img_dir=None, transform=None):
    """创建测试集数据集（仅在predict.py中使用）
    Args:
        test_csv_file (string, optional): 测试集标签CSV文件的路径，可选
        test_img_dir (string): 测试集图像文件夹的路径
        transform (callable, optional): 应用于图像的变换函数
    Returns:
        Dataset: 测试集数据集对象
    """
    # 检查图像目录是否存在
    if not test_img_dir or not os.path.exists(test_img_dir):
        print(f"错误: 测试图像目录不存在或未指定: {test_img_dir}")
        return None
        
    # 如果提供了CSV文件，则使用传统的FlowerDataset
    if test_csv_file and os.path.exists(test_csv_file):
        try:
            # 创建测试集数据集
            test_dataset = FlowerDataset(test_csv_file, test_img_dir, transform)
            print(f"测试集大小: {len(test_dataset)}")
            print(f"测试集类别数量: {test_dataset.num_classes}")
            return test_dataset
        except Exception as e:
            print(f"使用CSV文件创建测试集时出错: {e}")
            print("尝试使用无标签推理模式...")
    else:
        # 没有提供CSV文件或文件不存在，使用无标签推理模式
        if test_csv_file:
            print(f"警告: 测试集标签文件 {test_csv_file} 不存在，使用无标签推理模式")
        else:
            print("未提供测试集标签文件，使用无标签推理模式")
    
    # 使用无标签推理数据集
    try:
        test_dataset = InferenceDataset(test_img_dir, transform)
        print(f"测试集大小: {len(test_dataset)}")
        print(f"使用的类别数量: {test_dataset.num_classes}")
        return test_dataset
    except Exception as e:
        print(f"创建无标签推理数据集时出错: {e}")
        return None


def load_custom_test_data(test_csv_file, test_img_dir):
    """加载自定义测试数据（仅在predict.py中使用）
    Args:
        test_csv_file (string): 测试集标签CSV文件的路径
        test_img_dir (string): 测试集图像文件夹的路径
    Returns:
        tuple: (test_df, category_to_idx) 如果文件存在
        None: 如果文件不存在
    """
    if not os.path.exists(test_csv_file):
        print(f"错误: 测试集标签文件 {test_csv_file} 不存在")
        return None

    # 读取测试集CSV文件
    test_df = pd.read_csv(test_csv_file)

    # 创建类别映射
    category_to_idx = create_category_mapping(test_df)

    return test_df, category_to_idx


# ===== 优化器工具相关功能 =====

def get_loss_function(loss_function_type):
    """
    根据指定的损失函数类型创建并返回相应的损失函数
    
    参数:
        loss_function_type: 字符串，表示损失函数类型
    
    返回:
        创建的损失函数实例
    """
    # CrossEntropyLoss损失函数：交叉熵损失函数，适用于多分类任务
    if loss_function_type == 'cross_entropy':
        return nn.CrossEntropyLoss()
    # NLLLoss损失函数：负对数似然损失函数，适用于单标签分类任务
    elif loss_function_type == 'nll_loss':
        return nn.NLLLoss()
    # BCEWithLogitsLoss损失函数：二分类任务的损失函数，适用于输出为logits的情况
    elif loss_function_type == 'bce_with_logits':
        return nn.BCEWithLogitsLoss()
    # L1正则化损失函数
    elif loss_function_type == 'l1_regularized_cross_entropy':
        return L1RegularizedLoss()
    else:
        # 默认使用CrossEntropyLoss
        print(f"警告：未知的损失函数类型 '{loss_function_type}'，默认使用CrossEntropyLoss")
        return nn.CrossEntropyLoss()


class L1RegularizedLoss(nn.Module):
    """
    结合L1正则化的交叉熵损失函数
    用于同时优化模型分类性能和参数稀疏性
    """
    def __init__(self, l1_lambda=0.001, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super().__init__()
        # 支持标准CrossEntropyLoss的所有参数
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=weight, size_average=size_average,
                                                     ignore_index=ignore_index, reduce=reduce,
                                                     reduction=reduction)
        self.l1_lambda = l1_lambda
        # 存储需要计算L1正则化的模型
        self.model = None
        
    def set_model(self, model):
        """设置需要计算L1正则化的模型"""
        self.model = model
        return self  # 支持链式调用
        
    def set_l1_lambda(self, l1_lambda):
        """动态调整L1正则化系数"""
        self.l1_lambda = l1_lambda
        return self  # 支持链式调用
        
    def forward(self, outputs, targets):
        """标准PyTorch损失函数接口，只接受outputs和targets"""
        if self.model is None:
            raise ValueError("模型未设置，请先调用set_model方法")
            
        # 计算交叉熵损失
        ce_loss = self.cross_entropy_loss(outputs, targets)
        
        # 计算L1正则化项
        l1_reg = 0.0
        with torch.no_grad():
            for param in self.model.parameters():
                if param.requires_grad:
                    l1_reg += torch.sum(torch.abs(param))  # 使用torch.abs和torch.sum更高效
        
        # 组合损失
        total_loss = ce_loss + self.l1_lambda * l1_reg
        return total_loss


def get_optimizer(model_parameters, optimizer_type, learning_rate, weight_decay, l1_lambda=None):
    """
    根据指定的优化器类型创建并返回相应的优化器
    
    参数:
        model_parameters: 模型的参数
        optimizer_type: 字符串，表示优化器类型
        learning_rate: 浮点数，表示学习率
        weight_decay: 浮点数，表示权重衰减
        l1_lambda: 浮点数，表示L1正则化系数，如果为None则不使用参数分组
    
    返回:
        创建的优化器实例
    """
    # 如果指定了L1正则化系数，则进行参数分组
    if l1_lambda is not None and l1_lambda > 0:
        # 将参数分为两组：带L1正则化的权重参数和其他参数
        decay_params = []  # 带权重衰减的参数
        no_decay_params = []  # 不带权重衰减的参数（偏置、批归一化参数等）
        
        # 检查model_parameters是否是named_parameters
        if hasattr(model_parameters, 'named_parameters'):
            # 如果model_parameters是模型实例，获取named_parameters
            for name, param in model_parameters.named_parameters():
                if param.requires_grad:
                    if 'bias' in name or 'bn' in name or 'norm' in name.lower():
                        no_decay_params.append(param)
                    else:
                        decay_params.append(param)
        else:
            # 假设model_parameters是parameters()的结果，无法获取名称信息
            # 这里简化处理，所有参数都应用相同的权重衰减
            decay_params = list(model_parameters)
            no_decay_params = []
        
        param_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
    else:
        param_groups = model_parameters
    
    #  Adam优化器：自适应学习率优化器，适用于大多数情况
    if optimizer_type == 'adam':
        return optim.Adam(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay
        )
    # SGD优化器：随机梯度下降优化器，适用于小批量数据
    elif optimizer_type == 'sgd':
        return optim.SGD(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9  # SGD的动量参数
        )
    # RMSprop优化器：均方根传播优化器，适用于处理稀疏梯度
    elif optimizer_type == 'rmsprop':
        return optim.RMSprop(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9
        )
    else:
        # 默认使用Adam优化器
        print(f"警告：未知的优化器类型 '{optimizer_type}'，默认使用Adam优化器")
        return optim.Adam(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay
        )


def get_lr_scheduler(optimizer, scheduler_type='step', **kwargs):
    """
    创建并返回学习率调度器
    
    参数:
        optimizer: 优化器实例
        scheduler_type: 学习率调度器类型
        **kwargs: 其他调度器参数
    
    返回:
        创建的学习率调度器实例
    """
    #  StepLR学习率调度器：每隔固定步数衰减学习率
    if scheduler_type == 'step':
        step_size = kwargs.get('step_size', 10)
        gamma = kwargs.get('gamma', 0.5)
        return optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=step_size, 
            gamma=gamma
        )
    # MultiStepLR学习率调度器：在指定的epoch数后衰减学习率
    elif scheduler_type == 'multi_step':
        milestones = kwargs.get('milestones', [10, 20, 30])
        gamma = kwargs.get('gamma', 0.5)
        return optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=milestones, 
            gamma=gamma
        )
    # ExponentialLR学习率调度器：指数衰减学习率
    elif scheduler_type == 'exponential':
        gamma = kwargs.get('gamma', 0.95)
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    # CosineAnnealingLR学习率调度器：余弦退火学习率调度器
    # 按照余弦函数方式逐渐降低学习率，适用于大规模数据集
    elif scheduler_type == 'cosine':
        # T_max是余弦周期的一半，通常设为总训练轮数
        T_max = kwargs.get('T_max', 50)
        # eta_min是最小学习率，默认为0
        eta_min = kwargs.get('eta_min', 0)
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=T_max, 
            eta_min=eta_min
        )
    # CosineAnnealingWarmRestarts学习率调度器：带热重启的余弦退火学习率调度器
    # 在余弦退火的基础上，定期重新开始学习率调度，有助于跳出局部最优
    elif scheduler_type == 'cosine_warm_restarts':
        # T_0是第一个重启周期的大小
        T_0 = kwargs.get('T_0', 10)
        # T_mult控制后续重启周期的增长因子
        T_mult = kwargs.get('T_mult', 1)
        # eta_min是最小学习率，默认为0
        eta_min = kwargs.get('eta_min', 0)
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=T_0, 
            T_mult=T_mult, 
            eta_min=eta_min
        )
    else:
        # 默认使用StepLR
        print(f"警告：未知的学习率调度器类型 '{scheduler_type}'，默认使用StepLR")
        return optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=10, 
            gamma=0.5
        )