import os
import sys
import json
import argparse
import glob
from PIL import Image

import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from utils import FlowerDataset  # 使用我们自定义的数据集类
from model import FlowerNet

# 定义一个简单的数据集类，只用于加载图像文件
class SimpleImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        # 获取目录下所有图像文件
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_files.extend(glob.glob(os.path.join(img_dir, ext)))
        
        # 确保有图像文件
        if len(self.image_files) == 0:
            raise FileNotFoundError(f"在目录 {img_dir} 中未找到任何图像文件")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        img_name = os.path.basename(img_path)
        
        if self.transform:
            image = self.transform(image)
        
        # 返回图像和文件名（不返回标签）
        return image, img_name

def main():
    """主函数，执行模型预测"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='花卉分类预测程序')
    parser.add_argument('test_dir', help='测试文件夹路径')
    parser.add_argument('output_path', help='结果输出CSV文件路径')
    
    # 检查是否提供了命令行参数
    if len(sys.argv) < 3:
        print("错误：缺少必要的命令行参数")
        print("使用格式：python predict.py 测试文件夹 'result/submission.csv'")
        sys.exit(1)
    
    args = parser.parse_args()
    test_dir = args.test_dir
    output_path = args.output_path
    
    # 加载配置 - 从JSON文件读取
    config_path = "../model/config.json"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            configs = json.load(f)
        print(f"成功加载配置文件: {config_path}")
    except Exception as e:
        print(f"错误：无法加载配置文件 {config_path}: {e}")
        sys.exit(1)

    # 创建测试集的变换
    test_transform = transforms.Compose([
        transforms.RandomResizedCrop(400, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # 使用自定义的简单数据集类，只加载图像文件
    try:
        test_dataset = SimpleImageDataset(
            img_dir=test_dir,
            transform=test_transform
        )
    except Exception as e:
        print(f"错误：加载测试数据集时出错: {e}")
        sys.exit(1)

    # 获取数据集大小
    dataset_size = len(test_dataset)
    print(f"找到 {dataset_size} 张图像进行预测")

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
    # 获取激活函数配置，默认为gelu
    activation_fn = configs.get('activation-fn', 'gelu')
    
    # 从配置中获取类别数量，如果没有则默认为100
    num_classes = configs.get('num-classes', 100)
    
    model = FlowerNet(
        num_classes=num_classes,
        model_name=configs.get('model-name', 'resnet18'),
        use_layer_norm=configs.get('use-layer-norm', False),
        activation_fn=activation_fn
    )
    model = model.to(device)

    log_interval = configs['log-interval']

    print(f'\n---------- prediction start at: {device} ----------\n')
    print(f'图像目录: {test_dir}')
    # 添加激活函数信息打印
    print(f'使用激活函数: {activation_fn}')

    # 从配置中获取自定义模型参数路径
    try:
        model_params_path = configs['custom-model-params-path']
    except KeyError:
        print("错误：配置文件中缺少必要的模型参数路径配置项")
        print("请在config.json中添加'custom-model-params-path'配置项")
        sys.exit(1)

    # 确保指定的模型参数文件存在
    if not os.path.exists(model_params_path):
        print(f"错误：指定的模型参数文件不存在: {model_params_path}")
        sys.exit(1)

    print(f"正在加载模型参数: {model_params_path}")

    # 尝试从与模型参数相同的目录加载类别映射文件
    category_map_path = os.path.join(os.path.dirname(model_params_path), 'category_mapping.json')
    if os.path.exists(category_map_path):
        with open(category_map_path, 'r') as f:
            category_to_idx = json.load(f)
        print(f"成功加载类别映射文件: {category_map_path}")
        idx_to_category = {int(v): k for k, v in category_to_idx.items()}
    else:
        # 如果没有找到类别映射文件，创建默认映射
        print(f"警告：未找到类别映射文件，使用默认索引映射: {category_map_path}")
        idx_to_category = {i: i for i in range(num_classes)}

    # 用于存储预测结果
    predictions = []

    with torch.no_grad():
        # 加载模型检查点
        try:
            model.load_state_dict(torch.load(model_params_path, map_location=device, weights_only=True))
            model.eval()
            print(f"模型加载成功！开始预测...")
        except Exception as e:
            print(f"加载模型时出错: {e}")
            sys.exit(1)

        for batch, (images, img_names) in enumerate(dataloader, start=1):
            images = images.to(device)
            outputs = model(images)

            # 计算top-1准确率的预测结果
            _, top1_indices = torch.topk(outputs, 1, dim=1)

            # 收集预测结果
            for i in range(len(images)):
                # 获取图像文件名
                img_name = img_names[i]
                
                # 获取预测类别和置信度
                predicted_idx = top1_indices[i].item()
                predicted_category = idx_to_category[predicted_idx]
                
                # 计算置信度（使用softmax）
                probabilities = torch.nn.functional.softmax(outputs[i], dim=0)
                confidence = probabilities[predicted_idx].item()

                # 添加到预测列表（只包含文件名、预测类别和置信度）
                predictions.append([img_name, predicted_category, confidence])

            if batch % log_interval == 0:
                print(f'[predict] [{batch:04d}/{dataloader_size:04d}]')

    # 使用命令行参数中的输出路径
    output_file = output_path
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        print(f"警告：输出文件目录不存在，将创建目录: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    # 创建DataFrame并保存（只包含文件名、预测类别和置信度）
    df = pd.DataFrame(predictions, columns=['filename', 'predicted_category_id', 'confidence'])
    # 保存时不包含索引列，但包含列名
    df.to_csv(output_file, index=False)
    print(f'预测结果已保存到: {output_file}')
    print(f'共预测 {len(predictions)} 张图像\n')
    print('---------- prediction finished ----------\n')


if __name__ == '__main__':
    main()