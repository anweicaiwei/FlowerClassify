import os
import sys

import pandas as pd
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
    
    # 从配置文件中读取测试集CSV文件路径和图像目录
    try:
        test_csv_file = configs['test-csv-file']
        test_img_dir = configs['test-img-dir']
    except KeyError as e:
        print(f"错误：配置文件中缺少必要的测试集配置项: {e}")
        print("请在config.toml中添加'test-csv-file'和'test-img-dir'配置项")
        sys.exit(1)
    
    # 确保指定的文件和目录存在
    if not os.path.exists(test_csv_file):
        print(f"错误：指定的测试集CSV文件不存在: {test_csv_file}")
        sys.exit(1)
    
    if not os.path.exists(test_img_dir):
        print(f"错误：指定的测试集图像目录不存在: {test_img_dir}")
        sys.exit(1)
    
    # 使用FlowerDataset加载测试集
    test_dataset = FlowerDataset(
        csv_file=test_csv_file,
        img_dir=test_img_dir,
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
    
    # 初始化模型 - 根据配置文件传递所有必要参数
    model = FlowerNet(
        num_classes=configs['num-classes'],
        pretrained=configs.get('load-pretrained', False),
        model_name=configs.get('model-name', 'resnet18'),
        use_layer_norm=configs.get('use-layer-norm', False)  # 关键修复：添加这个参数以匹配训练时的配置
    )
    model = model.to(device)
    
    log_interval = configs['log-interval']
    
    print(f'\n---------- evaluation start at: {device} ----------\n')
    print(f'使用测试集: {test_csv_file}')
    print(f'图像目录: {test_img_dir}')
    
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
        checkpoint_path = os.path.join('checkpoints', 'OneGPU', timestamp_dir, checkpoint_filename)
        
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
    
    # 创建一个反向映射：从索引映射回原始类别ID
    idx_to_category = {v: k for k, v in test_dataset.category_to_idx.items()}
    
    # 用于存储预测结果
    predictions = []
    
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
            
            # 收集预测结果
            # 获取批次中每个图像的文件名
            batch_start = (batch - 1) * configs['batch-size']
            batch_end = min(batch_start + configs['batch-size'], dataset_size)
            
            for i in range(batch_end - batch_start):
                # 获取原始图像文件名
                img_idx = batch_start + i
                img_name = test_dataset.data_frame.iloc[img_idx, 0]
                
                # 获取预测类别和置信度
                predicted_idx = top1_indices[i].item()
                predicted_category = idx_to_category[predicted_idx]
                
                # 计算置信度（使用softmax）
                probabilities = torch.nn.functional.softmax(outputs[i], dim=0)
                confidence = probabilities[predicted_idx].item()
                
                # 添加到预测列表
                predictions.append([img_name, predicted_category, confidence])
            
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
    
    # 保存预测结果到CSV文件
    # 确定输出文件路径（保存在与模型相同的目录）
    output_dir = os.path.dirname(checkpoint_path)
    output_file = os.path.join(output_dir, 'predictions.csv')
    
    # 创建DataFrame并保存
    df = pd.DataFrame(predictions, columns=['filename', 'category_id', 'confidence'])
    # 保存时不包含索引列，但包含列名
    df.to_csv(output_file, index=False)
    print(f'预测结果已保存到: {output_file}')
    print('---------- evaluation finished ----------\n')
    

if __name__ == '__main__':
    main()