import math
import os
import random
import shutil  # 导入shutil用于文件复制

import numpy as np
import pandas as pd  # 导入pandas用于读取CSV
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms


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

        try:
            # 读取图像
            image = Image.open(img_path).convert('RGB')

            # 应用变换
            if self.transform:
                image = self.transform(image)

            # 将类别ID转换为索引
            label = self.category_to_idx[category_id]
        except Exception as e:
            print(f"读取图像 {img_name} 时出错: {e}")
            # 返回默认值
            image = torch.zeros(3, 224, 224)  # 假设图像大小为224x224
            label = 0  # 返回第一个类别的索引作为默认值

        return image, label


class InferenceDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        """初始化推理数据集
        Args:
            img_dir (string): 图像文件夹的路径
            transform (callable, optional): 应用于图像的变换函数
        """
        self.img_dir = img_dir
        self.transform = transform
        # 获取目录中所有图像文件
        self.img_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]

    # 返回数据集大小
    def __len__(self):
        return len(self.img_files)

    # 获取数据集中的第idx个样本
    def __getitem__(self, idx):
        # 获取图像文件名
        img_name = self.img_files[idx]
        # 构建完整的图像路径
        img_path = os.path.join(self.img_dir, img_name)

        try:
            # 读取图像
            image = Image.open(img_path).convert('RGB')

            # 应用变换
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"读取图像 {img_name} 时出错: {e}")
            # 返回默认值 - 修改为400x400以匹配变换设置
            image = torch.zeros(3, 400, 400)  # 与RandomResizedCrop(400)保持一致

        return image, img_name


# ===== 测试集相关代码 =====

def get_test_dataset(test_dir, transform=None):
    """获取测试数据集
    Args:
        test_dir (string): 测试集图像文件夹的路径
        transform (callable, optional): 应用于图像的变换函数
    Returns:
        InferenceDataset: 测试数据集
    """
    if transform is None:
        # 默认变换
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return InferenceDataset(test_dir, transform)


def load_custom_test_data(test_dir):
    """加载自定义测试数据
    Args:
        test_dir (string): 测试数据目录
    Returns:
        list: 图像文件列表和对应的变换后张量列表
    """
    # 定义默认变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_files = []
    image_tensors = []

    # 遍历目录中的所有文件
    for file_name in os.listdir(test_dir):
        file_path = os.path.join(test_dir, file_name)
        if os.path.isfile(file_path) and file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                # 读取并预处理图像
                image = Image.open(file_path).convert('RGB')
                image_tensor = transform(image)

                image_files.append(file_name)
                image_tensors.append(image_tensor)
            except Exception as e:
                print(f"处理图像 {file_name} 时出错: {e}")
                continue

    return image_files, image_tensors


# ===== 数据集划分功能 =====

def prepare_train_valid_datasets(df, img_dir, valid_ratio=0.2, seed=42):
    """准备训练集和验证集
    Args:
        df (DataFrame): 包含图像路径和标签的DataFrame
        img_dir (string): 图像文件夹的路径
        valid_ratio (float, optional): 验证集占总数据的比例
        seed (int, optional): 随机种子，确保结果可复现
    Returns:
        tuple: (训练数据集, 验证数据集, 类别到索引的映射)
    """
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)

    # 创建类别映射
    category_to_idx = create_category_mapping(df)
    num_classes = len(category_to_idx)

    # 按类别分割数据集
    train_indices = []
    valid_indices = []

    # 对每个类别进行分割
    for category in category_to_idx:
        # 获取当前类别的所有样本索引
        category_indices = df[df['category_id'] == category].index.tolist()
        # 打乱索引顺序
        random.shuffle(category_indices)
        # 计算验证集大小
        valid_size = int(len(category_indices) * valid_ratio)
        # 分割索引
        valid_indices.extend(category_indices[:valid_size])
        train_indices.extend(category_indices[valid_size:])

    # 默认变换
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建完整数据集
    full_dataset = FlowerDataset(df, img_dir)

    # 创建训练集和验证集
    train_dataset = Subset(full_dataset, train_indices)
    valid_dataset = Subset(full_dataset, valid_indices)

    # 设置变换
    train_dataset.dataset.transform = train_transform
    valid_dataset.dataset.transform = valid_transform

    return train_dataset, valid_dataset, category_to_idx


# ===== 数据增强相关功能 =====

def get_augmentations():
    """获取数据增强方法
    Returns:
        dict: 包含各种数据增强方法的字典
    """
    augmentations = {
        'original': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'horizontal_flip': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'vertical_flip': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'rotation': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'brightness': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'contrast': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(contrast=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    return augmentations


def apply_all_augmentations(img_path, output_dir):
    """应用所有增强方法到图像
    Args:
        img_path (string): 输入图像路径
        output_dir (string): 输出目录
    Returns:
        list: 生成的增强图像路径列表
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有增强方法
    augmentations = get_augmentations()

    # 读取原始图像
    try:
        image = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"读取图像 {img_path} 时出错: {e}")
        return []

    # 获取图像文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    # 应用每种增强方法
    augmented_paths = []
    for aug_name, transform in augmentations.items():
        # 应用变换
        augmented_img = transform(image)
        # 构建输出路径
        output_path = os.path.join(output_dir, f"{base_name}_{aug_name}.jpg")
        # 保存增强后的图像
        try:
            # 将张量转换回PIL图像
            img_pil = transforms.ToPILImage()(augmented_img)
            img_pil.save(output_path)
            augmented_paths.append(output_path)
        except Exception as e:
            print(f"保存增强图像 {output_path} 时出错: {e}")
            continue

    return augmented_paths


# ===== 数据集处理功能 =====

def prepare_datasets(df, img_dir, train_dir, valid_dir, valid_ratio=0.2, seed=42):
    """准备数据集，包括创建目录、分割数据、复制图像
    Args:
        df (DataFrame): 包含图像和标签信息的DataFrame
        img_dir (string): 原始图像目录
        train_dir (string): 训练集输出目录
        valid_dir (string): 验证集输出目录
        valid_ratio (float): 验证集比例
        seed (int): 随机种子
    Returns:
        tuple: (训练集DataFrame, 验证集DataFrame)
    """
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)

    # 清空并创建目录
    def clear_and_create_dir(dir_path):
        if os.path.exists(dir_path):
            # 获取目录中的所有文件
            files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
            if files:
                print(f"清空目录: {dir_path}")
                # 普通循环，移除tqdm进度条
                for file_name in files:
                    file_path = os.path.join(dir_path, file_name)
                    os.remove(file_path)
        # 确保目录存在
        os.makedirs(dir_path, exist_ok=True)

    # 清空并创建训练集和验证集目录
    clear_and_create_dir(train_dir)
    clear_and_create_dir(valid_dir)

    # 按类别分割数据集，确保每个类别的数据都能被合理分割
    train_data = []
    valid_data = []

    # 按类别ID分组
    grouped = df.groupby('category_id')

    # 分割数据
    for category_id, group in grouped:
        # 对每个类别的数据进行打乱
        group = group.sample(frac=1, random_state=seed).reset_index(drop=True)

        # 计算验证集大小
        valid_size = int(len(group) * valid_ratio)

        # 分割数据
        valid_group = group.iloc[:valid_size].copy()
        train_group = group.iloc[valid_size:].copy()

        # 添加到训练集和验证集列表
        train_data.append(train_group)
        valid_data.append(valid_group)

    # 合并所有类别的数据
    train_df = pd.concat(train_data, ignore_index=True)
    valid_df = pd.concat(valid_data, ignore_index=True)

    # 统计需要处理的总文件数
    total_files = len(train_df) + len(valid_df)
    print(f"开始处理数据集，总计 {total_files} 个文件...")

    # 只复制原始训练集图像，不进行增强
    print("\n复制训练集图像...")
    # 普通循环，移除tqdm进度条
    for _, row in train_df.iterrows():
        src_path = os.path.join(img_dir, row['filename'])
        dst_path = os.path.join(train_dir, row['filename'])

        # 直接复制原始文件
        shutil.copy2(src_path, dst_path)

    # 复制验证集图像
    print("\n复制验证集图像...")
    # 普通循环，移除tqdm进度条
    for _, row in valid_df.iterrows():
        src_path = os.path.join(img_dir, row['filename'])
        dst_path = os.path.join(valid_dir, row['filename'])
        # 直接复制文件，覆盖已存在的文件
        shutil.copy2(src_path, dst_path)

    print("数据集准备完成！")
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(valid_df)}")

    return train_df, valid_df


# ===== 优化器工具功能 =====

# 定义常用的损失函数
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """初始化Focal Loss
        Args:
            alpha (float): 平衡因子
            gamma (float): 聚焦参数
            reduction (string): 损失聚合方式
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        """前向传播
        Args:
            inputs (tensor): 模型输出
            targets (tensor): 真实标签
        Returns:
            tensor: 损失值
        """
        ce_loss = self.cross_entropy(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=3, smoothing=0.1, dim=-1):
        """初始化标签平滑损失
        Args:
            classes (int): 类别数量
            smoothing (float): 平滑因子
            dim (int): 维度
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        """前向传播
        Args:
            pred (tensor): 模型输出
            target (tensor): 真实标签
        Returns:
            tensor: 损失值
        """
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# 定义学习率调度器
class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, warmup_type='linear', target_lr=None):
        """初始化预热学习率调度器
        Args:
            optimizer (Optimizer): 优化器
            warmup_epochs (int): 预热轮数
            warmup_type (string): 预热类型（linear或cosine）
            target_lr (float, optional): 目标学习率
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_type = warmup_type
        self.current_epoch = 0
        # 获取基础学习率
        self.base_lr = [param_group['lr'] for param_group in optimizer.param_groups]
        # 设置目标学习率
        self.target_lr = target_lr if target_lr is not None else self.base_lr[0]

    def step(self):
        """更新学习率"""
        if self.current_epoch < self.warmup_epochs:
            # 预热阶段
            lr_values = self.get_warmup_lr()
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = lr_values[i]
        else:
            # 预热结束后，保持目标学习率
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.target_lr

        # 更新当前轮数
        self.current_epoch += 1

    def get_warmup_lr(self):
        """根据当前轮数计算预热学习率"""
        progress = self.current_epoch / self.warmup_epochs

        if self.warmup_type == 'linear':
            # 线性增长学习率
            lr_factor = progress
        elif self.warmup_type == 'cosine':
            # 余弦增长学习率
            lr_factor = 0.5 * (1 - math.cos(math.pi * progress))
        else:
            lr_factor = 1.0

        return [self.target_lr * lr_factor for _ in self.base_lr]


class MultiStepLR:
    def __init__(self, optimizer, milestones, gamma=0.1):
        """初始化多步学习率调度器
        Args:
            optimizer (Optimizer): 优化器
            milestones (list): 学习率衰减的里程碑（轮数）
            gamma (float): 学习率衰减因子
        """
        self.optimizer = optimizer
        self.milestones = milestones
        self.gamma = gamma
        self.current_epoch = 0
        # 获取基础学习率
        self.base_lr = [param_group['lr'] for param_group in optimizer.param_groups]

    def step(self):
        """更新学习率"""
        # 检查是否达到里程碑
        if self.current_epoch in self.milestones:
            # 衰减学习率
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.gamma

        # 更新当前轮数
        self.current_epoch += 1


class ExponentialLR:
    def __init__(self, optimizer, gamma=0.95):
        """初始化指数学习率调度器
        Args:
            optimizer (Optimizer): 优化器
            gamma (float): 学习率衰减因子
        """
        self.optimizer = optimizer
        self.gamma = gamma
        self.current_epoch = 0

    def step(self):
        """更新学习率"""
        # 衰减学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.gamma

        # 更新当前轮数
        self.current_epoch += 1


class CosineAnnealingLR:
    def __init__(self, optimizer, T_max, eta_min=0):
        """初始化余弦退火学习率调度器
        Args:
            optimizer (Optimizer): 优化器
            T_max (int): 余弦周期
            eta_min (float): 最小学习率
        """
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.current_epoch = 0
        # 获取基础学习率
        self.base_lr = [param_group['lr'] for param_group in optimizer.param_groups]

    def step(self):
        """更新学习率"""
        # 计算当前学习率
        for i, param_group in enumerate(self.optimizer.param_groups):
            lr = self.eta_min + (self.base_lr[i] - self.eta_min) * \
                 (1 + math.cos(math.pi * self.current_epoch / self.T_max)) / 2
            param_group['lr'] = lr

        # 更新当前轮数
        self.current_epoch += 1


class CosineAnnealingWarmRestarts:
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0):
        """初始化带热重启的余弦退火学习率调度器
        Args:
            optimizer (Optimizer): 优化器
            T_0 (int): 第一个周期的长度
            T_mult (int): 周期乘法因子
            eta_min (float): 最小学习率
        """
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.current_epoch = 0
        self.T_cur = 0
        self.T_i = T_0
        # 获取基础学习率
        self.base_lr = [param_group['lr'] for param_group in optimizer.param_groups]

    def step(self):
        """更新学习率"""
        # 计算当前学习率
        for i, param_group in enumerate(self.optimizer.param_groups):
            lr = self.eta_min + (self.base_lr[i] - self.eta_min) * \
                 (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            param_group['lr'] = lr

        # 更新当前轮数
        self.current_epoch += 1
        self.T_cur += 1

        # 检查是否需要重启
        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.T_i *= self.T_mult


# 优化器工厂函数

def get_optimizer(model, config):
    """获取优化器
    Args:
        model (nn.Module): 模型
        config (dict): 优化器配置
    Returns:
        Optimizer: 优化器
    """
    optimizer_type = config.get('type', 'adam').lower()
    lr = config.get('lr', 0.001)
    weight_decay = config.get('weight_decay', 0)

    params = model.parameters()

    if optimizer_type == 'sgd':
        momentum = config.get('momentum', 0.9)
        nesterov = config.get('nesterov', False)
        optimizer = optim.SGD(params, lr=lr, momentum=momentum,
                              weight_decay=weight_decay, nesterov=nesterov)
    elif optimizer_type == 'adam':
        betas = config.get('betas', (0.9, 0.999))
        eps = config.get('eps', 1e-8)
        optimizer = optim.Adam(params, lr=lr, betas=betas,
                               eps=eps, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        betas = config.get('betas', (0.9, 0.999))
        eps = config.get('eps', 1e-8)
        optimizer = optim.AdamW(params, lr=lr, betas=betas,
                                eps=eps, weight_decay=weight_decay)
    elif optimizer_type == 'rmsprop':
        alpha = config.get('alpha', 0.99)
        eps = config.get('eps', 1e-8)
        momentum = config.get('momentum', 0)
        optimizer = optim.RMSprop(params, lr=lr, alpha=alpha,
                                  eps=eps, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")

    return optimizer


# 学习率调度器工厂函数

def get_lr_scheduler(optimizer, config):
    """获取学习率调度器
    Args:
        optimizer (Optimizer): 优化器
        config (dict): 学习率调度器配置
    Returns:
        object: 学习率调度器
    """
    scheduler_type = config.get('type', 'constant').lower()

    if scheduler_type == 'constant':
        # 恒定学习率，不进行调整
        scheduler = None
    elif scheduler_type == 'warmup':
        # 预热学习率调度器
        warmup_epochs = config.get('warmup_epochs', 5)
        warmup_type = config.get('warmup_type', 'linear')
        target_lr = config.get('target_lr')
        scheduler = WarmupScheduler(optimizer, warmup_epochs, warmup_type, target_lr)
    elif scheduler_type == 'multistep':
        # 多步学习率调度器
        milestones = config.get('milestones', [30, 60, 90])
        gamma = config.get('gamma', 0.1)
        scheduler = MultiStepLR(optimizer, milestones, gamma)
    elif scheduler_type == 'exponential':
        # 指数学习率调度器
        gamma = config.get('gamma', 0.95)
        scheduler = ExponentialLR(optimizer, gamma)
    elif scheduler_type == 'cosine':
        # 余弦退火学习率调度器
        T_max = config.get('T_max', 100)
        eta_min = config.get('eta_min', 0)
        scheduler = CosineAnnealingLR(optimizer, T_max, eta_min)
    elif scheduler_type == 'cosine_warm_restarts':
        # 带热重启的余弦退火学习率调度器
        T_0 = config.get('T_0', 50)
        T_mult = config.get('T_mult', 1)
        eta_min = config.get('eta_min', 0)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult, eta_min)
    else:
        raise ValueError(f"不支持的学习率调度器类型: {scheduler_type}")

    return scheduler


# 损失函数工厂函数

def get_loss_function(config):
    """获取损失函数
    Args:
        config (dict): 损失函数配置
    Returns:
        nn.Module: 损失函数
    """
    loss_type = config.get('type', 'cross_entropy').lower()

    if loss_type == 'cross_entropy':
        loss_fn = nn.CrossEntropyLoss()
    elif loss_type == 'focal':
        alpha = config.get('alpha', 1)
        gamma = config.get('gamma', 2)
        loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
    elif loss_type == 'label_smoothing':
        classes = config.get('classes', 3)
        smoothing = config.get('smoothing', 0.1)
        loss_fn = LabelSmoothingLoss(classes=classes, smoothing=smoothing)
    else:
        raise ValueError(f"不支持的损失函数类型: {loss_type}")

    return loss_fn

# ===== 主函数（仅在直接运行此脚本时执行） =====

def main():
    """主函数，当脚本直接运行时执行数据预处理"""
    try:
        import toml
        configs = toml.load('configs/config_OneGPU.toml')

        # 调用数据预处理函数，分割为train、valid两个数据集
        prepare_datasets(
            csv_file=configs['data-label'],
            img_dir=configs['data-root'],
            valid_ratio=configs['valid-split-ratio']
        )

        print("\n数据预处理完成！已生成train、valid两个数据集及其CSV文件。")
    except Exception as e:
        print(f"运行主函数时出错: {e}")


if __name__ == '__main__':
    main()