import os
from datetime import datetime  # 添加datetime导入

import toml
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from data_preparation import FlowerDataset  # 只导入必要的类和函数
from models import FlowerNet
# 导入优化器和损失函数工具
from utils.optim_utils import get_loss_function, get_optimizer, get_lr_scheduler
# 导入绘图工具
from utils.plot_utils import PlotManager


def main():
    """主函数，执行模型训练"""
    # 加载配置
    configs = toml.load('configs/config_OneGPU.toml')
    
    # 不再调用prepare_datasets函数，而是直接使用已分割好的数据集
    # 假设数据集已经通过data_preparation.py生成并保存在相应位置
    
    # 创建训练集和验证集的变换
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ConvertImageDtype(torch.float32),  # 将uint8转换为float32
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomErasing(),
    ])
    
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ConvertImageDtype(torch.float32),  # 将uint8转换为float32
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # 直接加载已分割好的数据集
    train_dataset = FlowerDataset(
        csv_file='datasets/train_split.csv',
        img_dir='datasets/train',  # 修改为从新创建的训练集目录加载图像
        transform=train_transform
    )
    
    valid_dataset = FlowerDataset(
        csv_file='datasets/valid_split.csv',
        img_dir='datasets/valid',
        transform=valid_transform
    )
    
    # 获取类别数量
    num_classes = train_dataset.num_classes
    configs['num-classes'] = num_classes  # 更新配置中的类别数量
    
    # 获取数据集大小
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
    
    train_dataloader_size = len(train_dataloader)
    valid_dataloader_size = len(valid_dataloader)
    
    log_interval = configs['log-interval']
    
    best_accuracy = 0.0
    last_accuracy = 0.0
    
    # 设置设备
    device = torch.device(configs['device'])
    
    # 初始化模型，添加LayerNorm选项
    use_layer_norm = configs.get('use-layer-norm', False)
    model = FlowerNet(
        num_classes=num_classes, 
        pretrained=configs['load-pretrained'], 
        model_name=configs['model-name'],
        use_layer_norm=use_layer_norm
    )
    model = model.to(device)
    
    # 初始化绘图管理器
    try:
        plot_manager = PlotManager()
        # 获取绘图管理器生成的时间戳，确保两者使用相同的时间戳
        timestamp = plot_manager.timestamp
    except Exception as e:
        print(f"无法初始化绘图管理器: {e}")
        plot_manager = None
        # 如果无法初始化绘图管理器，则生成新的时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 从配置文件读取损失函数类型并获取损失函数
    loss_function_type = configs.get('loss-function', 'cross_entropy')
    criterion = get_loss_function(loss_function_type)

    # 如果是L1正则化损失，设置模型
    if loss_function_type == 'l1_regularized_cross_entropy':
        criterion = criterion.set_model(model)
        # 可选：从配置中设置l1_lambda
        l1_lambda = configs.get('l1-lambda', 0.001)
        criterion = criterion.set_l1_lambda(l1_lambda)
        # 从配置文件读取L1正则化相关参数
        use_l1_regularization = True


    
    # 从配置文件读取优化器类型并获取优化器
    optimizer_type = configs.get('optimizer-type', 'adam')
    optimizer = get_optimizer(
        model,  # 传递整个模型实例而不仅仅是参数
        optimizer_type,
        learning_rate=configs['learning-rate'],
        weight_decay=configs['weight-decay'],
        l1_lambda=l1_lambda if l1_lambda > 0 else None
    )
    
    # 添加学习率调度器
    scheduler_type = configs.get('lr-scheduler-type', 'step')
    scheduler = get_lr_scheduler(
        optimizer,
        scheduler_type,
        step_size=configs.get('lr-scheduler-step-size', 10),
        gamma=configs.get('lr-scheduler-gamma', 0.5)
    )
    
    # 早停机制相关参数
    early_stopping_patience = configs.get('early-stopping-patience', 15)
    early_stopping_counter = 0
    
    # 梯度裁剪参数
    use_grad_clip = configs.get('use-grad-clip', False)
    grad_clip_value = configs.get('grad-clip-value', 1.0)
    
    # 创建基于时间戳的checkpoint目录
    checkpoints_dir = os.path.join('checkpoints', 'OneGPU',timestamp)
    os.makedirs(checkpoints_dir, exist_ok=True)
    print(f"检查点将保存至目录: {checkpoints_dir}")
    
    # 使用配置中的文件名，但保存路径改为基于时间戳的目录
    load_checkpoint_path = configs['load-checkpoint-path']
    best_checkpoint_path = os.path.join(checkpoints_dir, 'best-ckpt.pt')
    last_checkpoint_path = os.path.join(checkpoints_dir, 'last-ckpt.pt')
    
    # 创建检查点目录（确保父目录存在）
    os.makedirs(os.path.dirname(best_checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(last_checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(load_checkpoint_path), exist_ok=True)
    
    if configs['load-checkpoint']:
        try:
            # 使用strict=False来允许部分加载，忽略不匹配的键
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
    if use_l1_regularization and l1_lambda > 0:
        print(f"使用L1正则化，系数: {l1_lambda}")
    if use_grad_clip:
        print(f"使用梯度裁剪，阈值: {grad_clip_value}")
    if use_layer_norm:
        print(f"使用LayerNorm层")
    
    # 训练循环
    for epoch in range(configs['num-epochs']):
        model.train()
        epoch_loss = 0.0  # 用于记录当前epoch的总损失
        
        for batch, (images, labels) in enumerate(train_dataloader, start=1):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # 根据是否使用L1正则化选择不同的损失计算方式
            if use_l1_regularization:
                loss = criterion(outputs, labels)
            else:
                loss = criterion(outputs, labels)
            
            loss.backward()
            
            # 可选的梯度裁剪
            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
            
            optimizer.step()
            
            # 累加损失值
            epoch_loss += loss.item()
            
            if batch % log_interval == 0:
                print(f'[train] [{epoch:03d}/{configs["num-epochs"]:03d}] [{batch:04d}/{train_dataloader_size:04d}] loss: {loss.item():.5f}')
        
        # 计算当前epoch的平均损失
        avg_epoch_loss = epoch_loss / train_dataloader_size
        
        # 验证循环
        model.eval()

        # 将模型切换到评估模式
        with torch.no_grad():
            accuracy = 0.0
            # 遍历验证集进行验证
            for batch, (images, labels) in enumerate(valid_dataloader, start=1):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                accuracy += (torch.argmax(outputs, dim=1) == labels).sum().item()
                
                if batch % log_interval == 0:
                    print(f'[valid] [{epoch:03d}/{configs["num-epochs"]:03d}] [{batch:04d}/{valid_dataloader_size:04d}]')
            
            accuracy /= valid_dataset_size
            
            # 保存最佳模型
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), best_checkpoint_path)
                print(f"新的最佳模型已保存: {best_checkpoint_path}")
                # 重置早停计数器
                early_stopping_counter = 0
            else:
                # 增加早停计数器
                early_stopping_counter += 1
                print(f"早停计数器: {early_stopping_counter}/{early_stopping_patience}")
            
            # 保存最新模型
            last_accuracy = accuracy
            torch.save(model.state_dict(), last_checkpoint_path)
        
        print(f'[valid] [{epoch:03d}/{configs["num-epochs"]:03d}] accuracy: {accuracy:.4f}')
        print(f'当前学习率: {optimizer.param_groups[0]["lr"]:.8f}')
        
        # 更新绘图
        if plot_manager:
            plot_manager.update(epoch + 1, avg_epoch_loss, accuracy, configs['num-epochs'])
        
        # 更新学习率
        scheduler.step()
        
        # 检查早停条件
        if early_stopping_counter >= early_stopping_patience:
            print(f"早停机制触发：连续{early_stopping_patience}个epoch无性能提升")
            break
    
    print(f'best accuracy: {best_accuracy:.3f}')
    print(f'last accuracy: {last_accuracy:.3f}')
    
    # 保存绘图结果
    if plot_manager:
        plot_manager.save_plot()
        plot_manager.close()
    
    print('\n---------- training finished ----------\n')


if __name__ == '__main__':
    main()