import os
import random
import shutil
import json  # 使用json模块读取配置
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

# 从指定的config.json读取配置
with open('../model/config.json', 'r', encoding='utf-8') as f:
    configs = json.load(f)

# 将数据增强功能移到模块级别，确保可以在任何地方访问

def get_augmentations():
    """创建并返回一个综合性的数据增强方法
    Returns:
        callable: 包含多种增强操作的Compose对象
    """
    # 移除内部的resize，因为后续训练代码中会统一调整大小
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.RandomVerticalFlip(p=0.3),    # 随机垂直翻转
        transforms.RandomRotation(degrees=(-10, 10)),  # 随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 随机颜色抖动
        transforms.RandomGrayscale(p=0.2),  # 随机灰度转换
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # 随机高斯模糊
    ])

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
        self.transforms = transform
        # 创建类别映射
        self.category_to_idx = create_category_mapping(self.data_frame)
        self.num_classes = len(self.category_to_idx)
        # 从配置中获取图像大小，默认为224
        self.img_size = configs.get('image-size', 224)

    # 返回数据集大小
    def __len__(self):
        return len(self.data_frame)
    # 获取数据集中的第idx个样本
    # 修改FlowerDataset类的__getitem__方法
    def __getitem__(self, idx):
        # 获取图像文件名和类别ID
        img_name = self.data_frame.iloc[idx, 0]
        category_id = self.data_frame.iloc[idx, 1]
        
        # 构建完整的图像路径
        img_path = os.path.join(self.img_dir, img_name)
        
        # 读取图像，添加更稳健的异常处理
        try:
            # 使用PIL直接读取图像
            # 添加参数支持处理截断的图像文件
            image = Image.open(img_path).convert('RGB')
            
            # 验证图像是否完整，通过尝试加载全部像素
            image.verify()  # 验证图像文件的完整性
            image = Image.open(img_path).convert('RGB')  # 重新打开图像，因为verify()后文件指针在末尾
            
            # 应用变换
            if self.transforms:
                if isinstance(self.transforms, list):
                    selected_transform = random.choice(self.transforms)
                    image = selected_transform(image)
                else:
                    image = self.transforms(image)
            
            # 将类别ID映射为索引
            label = self.category_to_idx[category_id]
            
            return image, label
        except Exception as e:
            # 更详细的错误信息
            print(f"Error reading image {img_path}: {e}")
            # 使用配置中的图像大小创建dummy图像
            dummy_image = torch.rand(3, self.img_size, self.img_size, dtype=torch.float32)
            dummy_label = random.randint(0, self.num_classes - 1)
            return dummy_image, dummy_label


# 修改后的函数，只将数据集分为train、valid两部分

def prepare_datasets(csv_file, img_dir, valid_ratio=0.15):
    """准备训练集和验证集
    Args:
        csv_file (string): 原始标签CSV文件的路径
        img_dir (string): 原始图像文件夹的路径
        valid_ratio (float): 验证集占总数据集的比例
    """
    # 读取原始CSV文件
    df = pd.read_csv(csv_file)
    
    # 创建类别映射
    category_to_idx = create_category_mapping(df)
    num_classes = len(category_to_idx)
    
    # 创建保存训练集和验证集的目录
    train_img_dir = os.path.join('datasets', 'train')
    valid_img_dir = os.path.join('datasets', 'valid')
    
    # 清空已有的目录（如果存在）
    for dir_path in [train_img_dir, valid_img_dir]:
        if os.path.exists(dir_path):
            # 获取目录中的所有文件
            files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
            if files:
                print(f"清空目录: {dir_path}")
                # 移除tqdm进度条，直接遍历文件
                for file_name in files:
                    file_path = os.path.join(dir_path, file_name)
                    os.remove(file_path)
        # 确保目录存在
        os.makedirs(dir_path, exist_ok=True)
    
    # 按类别分割数据集，确保每个类别的数据都能被合理分割
    train_data = []
    valid_data = []
    
    # 按类别ID分组
    grouped = df.groupby('category_id')
    
    # 分割数据
    for category_id, group in grouped:
        # 对每个类别的数据进行打乱
        group = group.sample(frac=1, random_state=42).reset_index(drop=True)
        
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
    # 移除tqdm进度条，直接遍历
    for _, row in train_df.iterrows():
        src_path = os.path.join(img_dir, row['filename'])
        dst_path = os.path.join(train_img_dir, row['filename'])
        
        # 直接复制原始文件
        shutil.copy2(src_path, dst_path)
    
    # 复制验证集图像
    print("\n复制验证集图像...")
    # 移除tqdm进度条，直接遍历
    for _, row in valid_df.iterrows():
        src_path = os.path.join(img_dir, row['filename'])
        dst_path = os.path.join(valid_img_dir, row['filename'])
        # 直接复制文件，覆盖已存在的文件
        shutil.copy2(src_path, dst_path)
    
    # 保存新的CSV文件
    print("\n正在保存CSV文件...")
    train_csv_path = os.path.join('datasets', 'train_split.csv')
    valid_csv_path = os.path.join('datasets', 'valid_split.csv')
    
    train_df.to_csv(train_csv_path, index=False)  # 保存原始训练集，不进行增强
    valid_df.to_csv(valid_csv_path, index=False)
    
    print(f"\n训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(valid_df)}")
    print(f"类别数量: {num_classes}")
    print(f"训练集CSV已保存至: {train_csv_path}")
    print(f"验证集CSV已保存至: {valid_csv_path}")
    print(f"训练集图像已复制至: {train_img_dir}")
    print(f"验证集图像已复制至: {valid_img_dir}")
    print("注意：数据增强将在训练过程中随机应用，不再提前生成增强图像")


# 优化器工具模块相关函数

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
        """扩展forward方法以支持Mixup的多参数输入"""
        if self.model is None:
            raise ValueError("模型未设置，请先调用set_model方法")
        
        # 检查是否是Mixup数据格式 (targets是包含多个元素的元组)
        if isinstance(targets, tuple) and len(targets) == 3:
            # Mixup格式: (label1, label2, lam)
            label1, label2, lam = targets
            # 计算混合损失
            ce_loss = lam * self.cross_entropy_loss(outputs, label1) + (1 - lam) * self.cross_entropy_loss(outputs, label2)
        else:
            # 标准分类格式
            ce_loss = self.cross_entropy_loss(outputs, targets)
        
        # 计算L1正则化项
        l1_reg = 0.0
        with torch.no_grad():
            # 只对分类头参数计算L1正则化，避免在解冻所有层时损失爆炸
            if hasattr(self.model, 'fc1'):  # 确保模型有fc1属性
                for param in self.model.fc1.parameters():
                    if param.requires_grad:
                        l1_reg += torch.sum(torch.abs(param))
            if hasattr(self.model, 'fc2'):
                for param in self.model.fc2.parameters():
                    if param.requires_grad:
                        l1_reg += torch.sum(torch.abs(param))
            if hasattr(self.model, 'fc3'):
                for param in self.model.fc3.parameters():
                    if param.requires_grad:
                        l1_reg += torch.sum(torch.abs(param))
        
        # 组合损失
        total_loss = ce_loss + self.l1_lambda * l1_reg
        # 确保返回的是标量
        if total_loss.dim() > 0:
            total_loss = total_loss.mean()
        return total_loss


class MixupDataset(Dataset):
    def __init__(self, dataset, alpha=0.4):
        self.dataset = dataset
        self.alpha = alpha
        self.num_classes = dataset.num_classes
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        # 获取原始数据
        img1, label1 = self.dataset[index]
        
        # 随机选择另一个样本索引
        index2 = random.randint(0, len(self.dataset) - 1)
        img2, label2 = self.dataset[index2]
        
        # 生成mixup系数
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        # 确保lam是与batch_size匹配的张量格式
        # 注意：在__getitem__中我们只处理单个样本，所以这里保持简单
        
        # 混合图像
        mixed_img = lam * img1 + (1 - lam) * img2
        
        return mixed_img, label1, label2, lam


class WarmupScheduler:
    """
    学习率预热调度器
    用于在训练初期逐步增加学习率到目标值，然后再应用常规学习率调度策略
    """
    def __init__(self, optimizer, warmup_epochs, target_lr, scheduler_type='cosine', warmup_type='linear', **kwargs):
        """
        初始化学习率预热调度器
        
        参数:
            optimizer: 优化器实例
            warmup_epochs: 预热轮数
            target_lr: 预热结束后的目标学习率
            scheduler_type: 预热结束后使用的学习率调度器类型
            warmup_type: 预热类型，'linear'表示线性增长，'cosine'表示余弦增长
            **kwargs: 传递给后续学习率调度器的参数
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr
        self.warmup_type = warmup_type
        self.current_epoch = 0
        
        # 保存初始学习率（用于预热）
        self.base_lr = [group['lr'] for group in optimizer.param_groups]
        
        # 初始化预热后的学习率调度器
        self.scheduler = get_lr_scheduler(optimizer, scheduler_type, **kwargs)
    
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
    
    def step(self):
        """更新学习率"""
        if self.current_epoch < self.warmup_epochs:
            # 预热阶段
            warmup_lrs = self.get_warmup_lr()
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lrs):
                param_group['lr'] = lr
        else:
            # 预热结束后，使用正常学习率调度器
            self.scheduler.step()
        
        self.current_epoch += 1
    
    def get_last_lr(self):
        """获取当前学习率"""
        if hasattr(self.scheduler, 'get_last_lr'):
            return self.scheduler.get_last_lr()
        return [group['lr'] for group in self.optimizer.param_groups]


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
    
    # Adam优化器：自适应学习率优化器，适用于大多数情况
    if optimizer_type == 'adam':
        return optim.Adam(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'adamw':
        # AdamW优化器：对权重衰减实现更合理的Adam变种
        # 使用带有权重衰减修正的AdamW
        return optim.AdamW(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.99),  # 默认参数，但明确写出以便调整
            eps=1e-6
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
        **kwargs: 其他调度器参数，支持warmup_epochs参数用于启用学习率预热
    
    返回:
        创建的学习率调度器实例
    """
    # 检查是否需要学习率预热
    warmup_epochs = kwargs.pop('warmup_epochs', 0)
    if warmup_epochs > 0:
        # 获取目标学习率，默认为当前优化器的学习率
        target_lr = kwargs.pop('target_lr', optimizer.param_groups[0]['lr'])
        warmup_type = kwargs.pop('warmup_type', 'linear')
        
        # 创建带预热的学习率调度器
        return WarmupScheduler(
            optimizer, 
            warmup_epochs, 
            target_lr, 
            scheduler_type, 
            warmup_type, 
            **kwargs
        )
    
    # StepLR学习率调度器：每隔固定步数衰减学习率
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
        T_0 = kwargs.get('T_0', 10)  # 第一次重启的迭代次数
        T_mult = kwargs.get('T_mult', 2)  # 重启周期的乘数因子
        eta_min = kwargs.get('eta_min', 0)  # 最小学习率
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=T_0, 
            T_mult=T_mult, 
            eta_min=eta_min
        )
    else:
        # 默认使用StepLR学习率调度器
        print(f"警告：未知的学习率调度器类型 '{scheduler_type}'，默认使用StepLR")
        step_size = kwargs.get('step_size', 10)
        gamma = kwargs.get('gamma', 0.5)
        return optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=step_size, 
            gamma=gamma
        )


# 主函数，用于数据预处理和数据集分割
if __name__ == '__main__':
    # 配置参数
    csv_file = '../../FlowerClassify/datasets/train_labels.csv'  # 原始标签文件路径
    img_dir = '../../FlowerClassify/datasets/train'  # 原始图像文件夹路径
    valid_ratio = 0.15  # 验证集比例
    
    # 调用函数准备数据集
    prepare_datasets(csv_file, img_dir, valid_ratio)