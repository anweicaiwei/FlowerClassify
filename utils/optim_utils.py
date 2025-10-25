"""
    优化器工具模块

    该模块提供了创建损失函数、优化器和学习率调度器的函数。
"""

import torch
import torch.nn as nn
import torch.optim as optim


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