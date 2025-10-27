import os
import sys

import toml
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from data_preparation import FlowerDataset  # 使用我们自定义的数据集类
from models import FlowerNet


def main():
    """主函数，执行模型评估"""
    # 加载配置
    configs = toml.load('configs/config_OneGPU.toml')
    
    # 创建测试集的变换
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ConvertImageDtype(torch.float32),  # 将uint8转换为float32
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # 使用FlowerDataset加载测试集，与训练时保持一致
    test_dataset = FlowerDataset(
        csv_file='datasets/test_split.csv',
        img_dir='datasets/test',
        transform=test_transform
    )
    
    # 获取数据集大小
    dataset_size = len(test_dataset)
    
    # 获取类别数量并更新配置
    num_classes = test_dataset.num_classes
    configs['num-classes'] = num_classes
    
    # 创建数据加载器
    dataloader = DataLoader(
        test_dataset,
        batch_size=configs['batch-size'],
        num_workers=configs['num-workers'],
        shuffle=False
    )
    
    dataloader_size = len(dataloader)
    
    # 设置设备
    device = torch.device(configs['device'])
    
    # 初始化模型
    model = FlowerNet(num_classes=configs['num-classes'], pretrained=False)
    model = model.to(device)
    
    log_interval = configs['log-interval']
    
    print(f'\n---------- evaluation start at: {device} ----------\n')
    
    # 从配置中获取时间戳目录和checkpoint类型
    timestamp_dir = configs.get('checkpoint-timestamp', '')
    checkpoint_type = configs.get('checkpoint-type', 'best')
    # 新增：获取自定义检查点路径
    custom_checkpoint_path = configs.get('custom-checkpoint-path', '')

    # 根据时间戳目录和checkpoint类型构建完整的checkpoint路径
    if custom_checkpoint_path:
        # 优先使用自定义检查点路径
        checkpoint_path = custom_checkpoint_path
        
        # 确保指定的checkpoint文件存在
        if not os.path.exists(checkpoint_path):
            print(f"错误：指定的自定义checkpoint文件不存在: {checkpoint_path}")
            sys.exit(1)
    elif timestamp_dir:
        # 如果指定了时间戳目录，则从该目录加载checkpoint
        checkpoint_filename = 'best-ckpt.pt' if checkpoint_type == 'best' else 'last-ckpt.pt'
        checkpoint_path = os.path.join('checkpoints', timestamp_dir, checkpoint_filename)
        
        # 确保指定的checkpoint目录存在
        if not os.path.exists(os.path.dirname(checkpoint_path)):
            print(f"错误：指定的checkpoint目录不存在: {os.path.dirname(checkpoint_path)}")
            print("请在config.toml中提供有效的checkpoint-timestamp值")
            sys.exit(1)
            
        # 确保指定的checkpoint文件存在
        if not os.path.exists(checkpoint_path):
            print(f"错误：指定的checkpoint文件不存在: {checkpoint_path}")
            # 尝试加载另一种类型的checkpoint作为备选
            alt_checkpoint_filename = 'last-ckpt.pt' if checkpoint_type == 'best' else 'best-ckpt.pt'
            alt_checkpoint_path = os.path.join('checkpoints', timestamp_dir, alt_checkpoint_filename)
            if os.path.exists(alt_checkpoint_path):
                print(f"尝试加载备选checkpoint: {alt_checkpoint_path}")
                checkpoint_path = alt_checkpoint_path
            else:
                sys.exit(1)
    else:
        # 如果未指定时间戳目录，则报错退出
        print("错误：未指定checkpoint时间戳目录")
        print("请在config.toml中设置checkpoint-timestamp配置项或custom-checkpoint-path配置项")
        sys.exit(1)
    
        print(f"正在加载模型检查点: {checkpoint_path}")
    
        with torch.no_grad():
            top1_accuracy = 0.0
            top2_accuracy = 0.0
            top3_accuracy = 0.0
            
            # 加载模型检查点
            try:
                model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
                model.eval()
                print(f"模型加载成功！开始评估...")
            except Exception as e:
                print(f"加载模型时出错: {e}")
                sys.exit(1)
            
            for batch, (images, labels) in enumerate(dataloader, start=1):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                
                # 计算top-k准确率
                _, top1_indices = torch.topk(outputs, 1, dim=1)
                _, top2_indices = torch.topk(outputs, 2, dim=1)
                _, top3_indices = torch.topk(outputs, 3, dim=1)
                
                labels = labels.view(-1, 1)
                
                top1_accuracy += (top1_indices == labels).sum().item()
                top2_accuracy += (top2_indices == labels).sum().item()
                top3_accuracy += (top3_indices == labels).sum().item()
                
                if batch % log_interval == 0:
                    print(f'[valid] [{batch:04d}/{dataloader_size:04d}]')
            
            # 计算最终准确率
            top1_accuracy /= dataset_size
            top2_accuracy /= dataset_size
            top3_accuracy /= dataset_size
        
        # 打印评估结果
        print('\n--------------------------------------')
        print(f'top1 accuracy: {top1_accuracy:.3f}')
        print(f'top2 accuracy: {top2_accuracy:.3f}')
        print(f'top3 accuracy: {top3_accuracy:.3f}')
        print(f'评估的模型路径: {checkpoint_path}')
        print('--------------------------------------\n')
        
        print('---------- evaluation finished ----------\n')
    

if __name__ == '__main__':
    main()