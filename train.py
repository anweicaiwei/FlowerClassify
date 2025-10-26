import os
from datetime import datetime

import toml
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from data_preparation import FlowerDataset
from models import FlowerNet
from utils.optim_utils import get_loss_function, get_optimizer, get_lr_scheduler
from utils.plot_utils import PlotManager
from utils.logging_utils import TrainingLogger, TrainingProcessLogger  # 导入日志记录器


def main():
    """主函数，执行模型训练"""
    # 加载配置
    config_path = 'configs/config_OneGPU.toml'
    configs = toml.load(config_path)
    
    # 初始化日志记录器
    logger = TrainingLogger()
    timestamp = logger.get_timestamp()  # 获取时间戳用于checkpoint和日志
    
    # 初始化训练过程记录器，使用与参数记录器相同的时间戳
    process_logger = TrainingProcessLogger(timestamp=timestamp)
    
    # 确保 training_logs 目录存在（虽然 TrainingLogger 初始化时已经会创建，但这里再次确认以增加健壮性）
    os.makedirs('training_logs', exist_ok=True)
    
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
    
    # 直接加载已分割好的数据集
    train_dataset = FlowerDataset(
        csv_file='datasets/train_split.csv',
        img_dir='datasets/train',
        transform=train_transform
    )
    
    valid_dataset = FlowerDataset(
        csv_file='datasets/valid_split.csv',
        img_dir='datasets/valid',
        transform=valid_transform
    )
    
    # 获取类别数量和数据集大小
    num_classes = train_dataset.num_classes
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
    except Exception as e:
        print(f"无法初始化绘图管理器: {e}")
        plot_manager = None
    
    # 从配置文件读取损失函数类型并获取损失函数
    loss_function_type = configs.get('loss-function', 'cross_entropy')
    criterion = get_loss_function(loss_function_type)
    l1_lambda = 0
    
    # 如果是L1正则化损失，设置模型
    if loss_function_type == 'l1_regularized_cross_entropy':
        criterion = criterion.set_model(model)
        l1_lambda = configs.get('l1-lambda', 0.001)
        criterion = criterion.set_l1_lambda(l1_lambda)
    
    # 从配置文件读取优化器类型并获取优化器
    optimizer_type = configs.get('optimizer-type', 'adam')
    optimizer = get_optimizer(
        model,
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
    checkpoints_dir = os.path.join('checkpoints', 'OneGPU', timestamp)
    os.makedirs(checkpoints_dir, exist_ok=True)
    print(f"检查点将保存至目录: {checkpoints_dir}")
    
    # 使用配置中的文件名，但保存路径改为基于时间戳的目录
    load_checkpoint_path = configs['load-checkpoint-path']
    best_checkpoint_path = os.path.join(checkpoints_dir, 'best-ckpt.pt')
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
            process_logger.log_training_event("checkpoint_loaded", f"成功加载检查点: {load_checkpoint_path}")
        except Exception as e:
            print(f"加载检查点失败: {e}")
            print("将从头开始训练模型")
            process_logger.log_training_event("checkpoint_load_failed", f"加载检查点失败: {str(e)}")
    
    # 记录训练信息到日志
    logger.load_config(config_path)
    logger.add_training_info({
        "dataset_size": {
            "train": train_dataset_size,
            "valid": valid_dataset_size
        },
        "model_name": configs['model-name'],
        "checkpoint_dir": checkpoints_dir,
        "device": str(device),
        "use_layer_norm": use_layer_norm,
        "use_l1_regularization": loss_function_type == 'l1_regularized_cross_entropy',
        "l1_lambda": l1_lambda,
        "use_grad_clip": use_grad_clip,
        "grad_clip_value": grad_clip_value,
        "early_stopping_patience": early_stopping_patience
    })
    
    # 保存初始日志
    logger.save_log("initial")
    
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
    
    process_logger.log_training_event("training_start", f"训练开始于设备: {device}")
    
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
                process_logger.log_training_batch(epoch, batch, len(train_dataloader), loss.item())
        
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
                    process_logger.log_validation_batch(epoch, batch, len(valid_dataloader))
            
            accuracy /= valid_dataset_size
            
            # 保存最佳模型
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), best_checkpoint_path)
                print(f"新的最佳模型已保存: {best_checkpoint_path}")
                process_logger.log_model_saving(epoch, best_checkpoint_path, is_best=True)
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                print(f"早停计数器: {early_stopping_counter}/{early_stopping_patience}")
                process_logger.log_training_event("early_stopping", f"早停计数器: {early_stopping_counter}/{early_stopping_patience}")
            
            # 保存最新模型
            last_accuracy = accuracy
            torch.save(model.state_dict(), last_checkpoint_path)
            process_logger.log_model_saving(epoch, last_checkpoint_path, is_best=False)
        
        print(f'[valid] [{epoch:03d}/{configs["num-epochs"]:03d}] accuracy: {accuracy:.4f}')
        print(f'当前学习率: {optimizer.param_groups[0]["lr"]:.8f}')
        
        # 记录epoch结束信息
        process_logger.log_epoch_end(epoch, avg_epoch_loss, accuracy, optimizer.param_groups[0]["lr"], best_accuracy)
        
        # 更新绘图
        if plot_manager:
            plot_manager.update(epoch + 1, avg_epoch_loss, accuracy, configs['num-epochs'])
        
        # 更新学习率
        scheduler.step()
        
        # 检查早停条件
        if early_stopping_counter >= early_stopping_patience:
            print(f"早停机制触发：连续{early_stopping_patience}个epoch无性能提升")
            process_logger.log_training_event("early_stopping_triggered", f"早停机制触发：连续{early_stopping_patience}个epoch无性能提升")
            break
    
    print(f'best accuracy: {best_accuracy:.3f}')
    print(f'last accuracy: {last_accuracy:.3f}')
    
    # 更新日志并保存最终版本
    logger.add_training_info({
        "best_accuracy": best_accuracy,
        "last_accuracy": last_accuracy,
        "epochs_completed": epoch + 1,
        "total_epochs": configs['num-epochs']
    })
    final_log_path = logger.save_log("final")

    # 保存绘图结果
    if plot_manager:
        plot_manager.save_plot()
        plot_manager.close()
    
    print('\n---------- training finished ----------\n')
    process_logger.log_training_event("training_finished", "训练完成")


if __name__ == '__main__':
    main()