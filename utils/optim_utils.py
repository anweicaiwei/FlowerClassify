"""
    优化器工具模块

    该模块提供了创建损失函数、优化器和学习率调度器的函数。
"""

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
    else:
        # 默认使用CrossEntropyLoss
        print(f"警告：未知的损失函数类型 '{loss_function_type}'，默认使用CrossEntropyLoss")
        return nn.CrossEntropyLoss()


def get_optimizer(model_parameters, optimizer_type, learning_rate, weight_decay):
    """
    根据指定的优化器类型创建并返回相应的优化器
    
    参数:
        model_parameters: 模型的参数
        optimizer_type: 字符串，表示优化器类型
        learning_rate: 浮点数，表示学习率
        weight_decay: 浮点数，表示权重衰减
    
    返回:
        创建的优化器实例
    """
    #  Adam优化器：自适应学习率优化器，适用于大多数情况
    if optimizer_type == 'adam':
        return optim.Adam(
            model_parameters,
            lr=learning_rate,
            weight_decay=weight_decay
        )
    # SGD优化器：随机梯度下降优化器，适用于小批量数据
    elif optimizer_type == 'sgd':
        return optim.SGD(
            model_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9  # SGD的动量参数
        )
    # RMSprop优化器：均方根传播优化器，适用于处理稀疏梯度
    elif optimizer_type == 'rmsprop':
        return optim.RMSprop(
            model_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9
        )
    else:
        # 默认使用Adam优化器
        print(f"警告：未知的优化器类型 '{optimizer_type}'，默认使用Adam优化器")
        return optim.Adam(
            model_parameters,
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