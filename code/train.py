import os
from datetime import datetime

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import FlowerNet
from utils import get_loss_function, get_optimizer, get_lr_scheduler, get_train_valid_datasets


def main():
    """主函数，执行模型训练"""
    configs = {
        'device': 'cuda',
        'data-root': 'D:/ProjectDevelop/PyCharm/FlowerClassify/datasets/data/train',
        'data-label': 'D:/ProjectDevelop/PyCharm/FlowerClassify/datasets/data_labels.csv',
        'train-ratio': 0.8,
        'valid-split-ratio': 0.2,
        'random-seed': 42,
        'custom-model-params-path': 'model/best-model.pt',
        'custom-output-path': 'results/submission.csv',
        'model-name': 'resnet34',
        'batch-size': 64,
        'num-epochs': 100,
        'num-workers': 4,
        'num-classes': 100,
        'log-interval': 10,
        'load-checkpoint': False,
        'load-pretrained': False,
        'load-checkpoint-path': 'checkpoints/best-ckpt.pt',
        'loss-function': 'l1_regularized_cross_entropy',
        'learning-rate': 0.0001,
        'weight-decay': 0.0005,
        'optimizer-type': 'adam',
        'lr-scheduler-type': 'cosine',
        'lr-scheduler-step-size': 10,
        'lr-scheduler-gamma': 0.5,
        'early-stopping-patience': 10,
        'l1-lambda': 0.000001,
        'use-layer-norm': True,
        'use-grad-clip': True,
        'grad-clip-value': 1.0
    }
    
    # 创建训练集和验证集的变换
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        # transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomErasing(),
    ])
    
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # 使用utils.py中的函数直接创建并划分训练集和验证集
    train_dataset, valid_dataset, category_to_idx, num_classes = get_train_valid_datasets(
        csv_file=configs['data-label'],
        img_dir=configs['data-root'],
        valid_ratio=configs['valid-split-ratio'],
        transform=train_transform  # 为了简单起见，对训练集和验证集使用相同的变换
    )
    
    # 获取数据集大小
    configs['num-classes'] = num_classes
    train_dataset_size = len(train_dataset)
    valid_dataset_size = len(valid_dataset)
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=configs['batch-size'],
        num_workers=configs['num-workers'],
        shuffle=True
    )
    
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=configs['batch-size'],
        num_workers=configs['num-workers'],
        shuffle=False
    )
    
    log_interval = configs['log-interval']
    best_accuracy = 0.0
    last_accuracy = 0.0
    
    # 设置设备
    device = torch.device(configs['device'])
    
    # 初始化模型
    use_layer_norm = configs['use-layer-norm']
    model = FlowerNet(
        num_classes=num_classes, 
        pretrained=configs['load-pretrained'], 
        model_name=configs['model-name'],
        use_layer_norm=use_layer_norm
    )
    model = model.to(device)
    
    # 获取损失函数
    loss_function_type = configs['loss-function']
    criterion = get_loss_function(loss_function_type)
    l1_lambda = 0
    
    # 如果是L1正则化损失，设置模型
    if loss_function_type == 'l1_regularized_cross_entropy':
        criterion = criterion.set_model(model)
        l1_lambda = configs['l1-lambda']
        criterion = criterion.set_l1_lambda(l1_lambda)
    
    # 获取优化器
    optimizer_type = configs['optimizer-type']
    optimizer = get_optimizer(
        model,
        optimizer_type,
        learning_rate=configs['learning-rate'],
        weight_decay=configs['weight-decay'],
        l1_lambda=l1_lambda if l1_lambda > 0 else None
    )
    
    # 添加学习率调度器
    scheduler_type = configs['lr-scheduler-type']
    scheduler = get_lr_scheduler(
        optimizer,
        scheduler_type,
        step_size=configs['lr-scheduler-step-size'],
        gamma=configs['lr-scheduler-gamma']
    )
    
    # 早停机制相关参数
    early_stopping_patience = configs['early-stopping-patience']
    early_stopping_counter = 0
    
    # 梯度裁剪参数
    use_grad_clip = configs['use-grad-clip']
    grad_clip_value = configs['grad-clip-value']
    
    # 创建基于时间戳的checkpoint目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoints_dir = os.path.join('checkpoints', 'OneGPU', timestamp)
    os.makedirs(checkpoints_dir, exist_ok=True)
    print(f"检查点将保存至目录: {checkpoints_dir}")
    
    # 设置检查点路径
    load_checkpoint_path = configs['load-checkpoint-path']
    best_checkpoint_path = os.path.join(checkpoints_dir, 'best-model.pt')
    last_checkpoint_path = os.path.join(checkpoints_dir, 'last-ckpt.pt')
    
    # 确保目录存在
    os.makedirs(os.path.dirname(best_checkpoint_path), exist_ok=True)
    
    if configs['load-checkpoint']:
        try:
            model.load_state_dict(
                torch.load(load_checkpoint_path, map_location=device, weights_only=True),
                strict=False
            )
            print(f"成功加载检查点（非严格模式）: {load_checkpoint_path}")
        except Exception as e:
            print(f"加载检查点失败: {e}")
            print("将从头开始训练模型")
    
    print(f'\n---------- training start at: {device} ----------\n')
    print(f"损失函数: {loss_function_type}")
    print(f"优化器: {optimizer_type}")
    print(f"学习率调度器: {scheduler_type}")
    print(f"早停机制配置: 连续{early_stopping_patience}个epoch无提升则停止训练")
    if l1_lambda > 0:
        print(f"使用L1正则化，系数: {l1_lambda}")
    if use_grad_clip:
        print(f"使用梯度裁剪，阈值: {grad_clip_value}")
    if use_layer_norm:
        print(f"使用LayerNorm层")
    
    # 训练循环
    for epoch in range(configs['num-epochs']):
        model.train()
        epoch_loss = 0.0
        
        for batch, (images, labels) in enumerate(train_dataloader, start=1):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 可选的梯度裁剪
            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
            
            optimizer.step()
            epoch_loss += loss.item()
            
            # 记录训练批次信息
            if batch % log_interval == 0:
                print(f'[train] [{epoch:03d}/{configs["num-epochs"]:03d}] [{batch:04d}/{len(train_dataloader):04d}] loss: {loss.item():.5f}')
        
        # 计算当前epoch的平均损失
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        
        # 验证循环
        model.eval()
        with torch.no_grad():
            accuracy = 0.0
            for batch, (images, labels) in enumerate(valid_dataloader, start=1):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                accuracy += (torch.argmax(outputs, dim=1) == labels).sum().item()
                
                # 记录验证批次信息
                if batch % log_interval == 0:
                    print(f'[valid] [{epoch:03d}/{configs["num-epochs"]:03d}] [{batch:04d}/{len(valid_dataloader):04d}]')
            
            accuracy /= valid_dataset_size
            
            # 保存最佳模型
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), best_checkpoint_path)
                print(f"新的最佳模型已保存: {best_checkpoint_path}")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                print(f"早停计数器: {early_stopping_counter}/{early_stopping_patience}")
            
            # 保存最新模型
            last_accuracy = accuracy
            torch.save(model.state_dict(), last_checkpoint_path)
        
        print(f'[valid] [{epoch:03d}/{configs["num-epochs"]:03d}] accuracy: {accuracy:.4f}')
        print(f'当前学习率: {optimizer.param_groups[0]["lr"]:.8f}')
        
        # 更新学习率
        scheduler.step()
        
        # 检查早停条件
        if early_stopping_counter >= early_stopping_patience:
            print(f"早停机制触发：连续{early_stopping_patience}个epoch无性能提升")
            break
    
    print(f'best accuracy: {best_accuracy:.3f}')
    print(f'last accuracy: {last_accuracy:.3f}')
    
    print('\n---------- training finished ----------\n')


if __name__ == '__main__':
    main()