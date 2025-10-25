import os
import sys
import argparse
from datetime import datetime

import toml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from data_preparation import FlowerDataset
from models import FlowerNet
from utils.plot_utils import PlotManager
from utils.optim_utils import get_loss_function, get_optimizer, get_lr_scheduler


def setup(rank, world_size, master_addr, master_port):
    """
    初始化分布式环境
    
    参数:
        rank: 当前进程的编号
        world_size: 进程总数
        master_addr: 主节点地址
        master_port: 主节点端口
    """
    # 设置环境变量
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # 初始化进程组，使用nccl后端
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # 设置当前设备
    torch.cuda.set_device(rank)


def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()


def train(rank, world_size, configs):
    """
    分布式训练函数
    
    参数:
        rank: 当前进程的编号
        world_size: 进程总数
        configs: 配置参数
    """
    # 初始化分布式环境
    setup(rank, world_size, configs['master-addr'], configs['master-port'])
    
    # 只在主进程中初始化绘图管理器
    plot_manager = None
    timestamp = None
    if rank == 0:
        try:
            plot_manager = PlotManager()
            timestamp = plot_manager.timestamp
        except Exception as e:
            print(f"无法初始化绘图管理器: {e}")
            plot_manager = None
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    else:
        # 其他进程使用与主进程相同的时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 创建训练集和验证集的变换
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.RandomRotation(10),
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
    
    # 加载数据集
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
    
    # 获取类别数量
    num_classes = train_dataset.num_classes
    configs['num-classes'] = num_classes
    
    # 获取数据集大小
    train_dataset_size = len(train_dataset)
    valid_dataset_size = len(valid_dataset)
    
    # 创建分布式采样器
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    # 验证集不需要随机打乱
    valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=configs['batch-size'] // world_size,  # 每个进程处理的批次大小
        sampler=train_sampler,
        num_workers=configs['num-workers'],
        pin_memory=True
    )
    
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=configs['batch-size'] // world_size,
        sampler=valid_sampler,
        num_workers=configs['num-workers'],
        pin_memory=True
    )
    
    train_dataloader_size = len(train_dataloader)
    valid_dataloader_size = len(valid_dataloader)
    
    log_interval = configs['log-interval']
    
    # 初始化模型
    model = FlowerNet(num_classes=configs['num-classes'], pretrained=configs['load-pretrained'], model_name=configs['model-name'])
    model = model.to(rank)  # 将模型放到当前GPU上
    
    # 使用DDP包装模型
    model = DDP(model, device_ids=[rank])
    
    # 获取损失函数
    loss_function_type = configs.get('loss-function', 'cross_entropy')
    criterion = get_loss_function(loss_function_type)
    
    # 获取优化器
    optimizer_type = configs.get('optimizer-type', 'adam')
    optimizer = get_optimizer(
        model.parameters(),
        optimizer_type,
        configs['learning-rate'],
        configs['weight-decay']
    )
    
    # 获取学习率调度器
    lr_scheduler_step_size = configs.get('lr-scheduler-step-size', 10)
    lr_scheduler_gamma = configs.get('lr-scheduler-gamma', 0.5)
    scheduler_type = configs.get('lr-scheduler-type', 'step')
    scheduler = get_lr_scheduler(
        optimizer,
        scheduler_type,
        step_size=lr_scheduler_step_size,
        gamma=lr_scheduler_gamma
    )
    
    # 早停机制相关参数
    early_stopping_patience = configs.get('early-stopping-patience', 15)
    early_stopping_counter = 0
    best_accuracy = 0.0
    last_accuracy = 0.0
    
    # 创建基于时间戳的checkpoint目录
    checkpoints_dir = os.path.join('checkpoints', 'Distributed', timestamp)
    if rank == 0:
        os.makedirs(checkpoints_dir, exist_ok=True)
        print(f"检查点将保存至目录: {checkpoints_dir}")
    
    # 使用配置中的文件名，但保存路径改为基于时间戳的目录
    load_checkpoint_path = configs['load-checkpoint-path']
    best_checkpoint_path = os.path.join(checkpoints_dir, 'best-ckpt.pt')
    last_checkpoint_path = os.path.join(checkpoints_dir, 'last-ckpt.pt')
    
    # 创建检查点目录（确保父目录存在）
    if rank == 0:
        os.makedirs(os.path.dirname(best_checkpoint_path), exist_ok=True)
        os.makedirs(os.path.dirname(last_checkpoint_path), exist_ok=True)
        os.makedirs(os.path.dirname(load_checkpoint_path), exist_ok=True)
    
    # 等待主进程创建目录
    dist.barrier()
    
    # 加载检查点（只在主进程加载，然后广播给其他进程）
    if configs['load-checkpoint']:
        if rank == 0:
            # 主进程加载模型权重
            checkpoint = torch.load(load_checkpoint_path, map_location={'cuda:0': f'cuda:{rank}'}, weights_only=True)
            model.load_state_dict(checkpoint)
        # 广播模型权重给所有进程
        dist.broadcast_object_list([model.state_dict()], src=0)
        if rank != 0:
            # 其他进程加载广播的权重
            model.load_state_dict(torch.load(load_checkpoint_path, map_location={'cuda:0': f'cuda:{rank}'}, weights_only=True))
    
    if rank == 0:
        print(f'\n---------- training start at: {configs["device"]} with {world_size} GPUs ----------\n')
        print(f"损失函数: {loss_function_type}")
        print(f"优化器: {optimizer_type}")
        print(f"学习率调度器: {scheduler_type}，每{lr_scheduler_step_size}个epoch降低为原来的{lr_scheduler_gamma}倍")
        print(f"早停机制配置: 连续{early_stopping_patience}个epoch无提升则停止训练")
    
    # 训练循环
    for epoch in range(configs['num-epochs']):
        # 设置采样器的epoch，确保每个epoch的打乱方式不同
        train_sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0.0
        
        for batch, (images, labels) in enumerate(train_dataloader, start=1):
            images = images.to(rank, non_blocking=True)
            labels = labels.to(rank, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # 累加损失值
            epoch_loss += loss.item()
            
            # 只在主进程和指定的log_interval打印日志
            if rank == 0 and batch % log_interval == 0:
                print(f'[train] [{epoch:03d}/{configs["num-epochs"]:03d}] [{batch:04d}/{train_dataloader_size:04d}] loss: {loss.item():.5f}')
        
        # 计算当前epoch的平均损失（所有进程的平均值）
        avg_epoch_loss = epoch_loss / train_dataloader_size
        
        # 收集所有进程的平均损失
        avg_epoch_loss_tensor = torch.tensor(avg_epoch_loss).to(rank)
        dist.all_reduce(avg_epoch_loss_tensor, op=dist.ReduceOp.AVG)
        avg_epoch_loss = avg_epoch_loss_tensor.item()
        
        # 验证循环
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            
            for batch, (images, labels) in enumerate(valid_dataloader, start=1):
                images = images.to(rank, non_blocking=True)
                labels = labels.to(rank, non_blocking=True)
                
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 只在主进程和指定的log_interval打印日志
                if rank == 0 and batch % log_interval == 0:
                    print(f'[valid] [{epoch:03d}/{configs["num-epochs"]:03d}] [{batch:04d}/{valid_dataloader_size:04d}]')
            
            # 计算准确率
            accuracy = correct / total
            
            # 收集所有进程的正确数和总数
            correct_tensor = torch.tensor(correct).to(rank)
            total_tensor = torch.tensor(total).to(rank)
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
            
            # 计算全局准确率
            global_accuracy = correct_tensor.item() / total_tensor.item()
            
            # 只在主进程中保存模型和更新早停计数器
            if rank == 0:
                last_accuracy = global_accuracy
                
                # 保存最佳模型
                if global_accuracy > best_accuracy:
                    best_accuracy = global_accuracy
                    torch.save(model.module.state_dict(), best_checkpoint_path)  # 注意使用model.module
                    print(f"新的最佳模型已保存: {best_checkpoint_path}")
                    # 重置早停计数器
                    early_stopping_counter = 0
                else:
                    # 增加早停计数器
                    early_stopping_counter += 1
                    print(f"早停计数器: {early_stopping_counter}/{early_stopping_patience}")
                
                # 保存最新模型
                torch.save(model.module.state_dict(), last_checkpoint_path)
                
                print(f'[valid] [{epoch:03d}/{configs["num-epochs"]:03d}] accuracy: {global_accuracy:.4f}')
                print(f'当前学习率: {optimizer.param_groups[0]["lr"]:.8f}')
                
                # 更新绘图
                if plot_manager:
                    plot_manager.update(epoch + 1, avg_epoch_loss, global_accuracy, configs['num-epochs'])
        
        # 更新学习率
        scheduler.step()
        
        # 检查早停条件（通过广播机制同步早停状态）
        if rank == 0:
            early_stopping_triggered = early_stopping_counter >= early_stopping_patience
        else:
            early_stopping_triggered = False
        
        # 广播早停状态
        early_stopping_triggered_tensor = torch.tensor(early_stopping_triggered).to(rank)
        dist.broadcast(early_stopping_triggered_tensor, src=0)
        early_stopping_triggered = early_stopping_triggered_tensor.item()
        
        if early_stopping_triggered:
            if rank == 0:
                print(f"早停机制触发：连续{early_stopping_patience}个epoch无性能提升")
            break
    
    # 训练结束
    if rank == 0:
        print(f'best accuracy: {best_accuracy:.3f}')
        print(f'last accuracy: {last_accuracy:.3f}')
        
        # 保存绘图结果
        if plot_manager:
            plot_manager.save_plot()
            plot_manager.close()
        
        print('\n---------- training finished ----------\n')
    
    # 清理分布式环境
    cleanup()


def main():
    """主函数，启动分布式训练"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='分布式花卉分类训练')
    parser.add_argument('--config', type=str, default='configs/config_distribute.toml', help='配置文件路径')
    parser.add_argument('--world-size', type=int, default=torch.cuda.device_count(), help='进程总数')
    args = parser.parse_args()
    
    # 加载配置
    configs = toml.load(args.config)
    
    # 启动多进程分布式训练
    mp.spawn(
        train,
        args=(args.world_size, configs),
        nprocs=args.world_size,
        join=True
    )


if __name__ == '__main__':
    main()