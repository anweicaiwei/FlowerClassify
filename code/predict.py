import argparse
import glob
import json
import os
import sys

import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

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
        
        # 不再过滤掉无法打开的图像文件，而是全部保留
        print(f"找到 {len(self.image_files)} 个图像文件")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img_name = os.path.basename(img_path)
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            # 返回图像、文件名和有效标志
            return image, img_name, True
        except Exception as e:
            print(f"警告：无法打开或处理图像文件 {img_path}: {e}")
            # 创建一个黑色占位图像作为替代
            placeholder = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                placeholder = self.transform(placeholder)
            # 返回占位图像、文件名和无效标志
            return placeholder, img_name, False


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

    img_size = configs.get('image-size', 224)
    # 创建测试集的变换
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
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
    print(f"共有 {dataset_size} 个文件待处理")

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
    # 从配置中获取类别数量，如果没有则默认为100
    num_classes = configs.get('num-classes', 100)
    use_layer_norm = configs.get('use-layer-norm', False)
    
    # 在预测时完全不加载预训练模型
    model = FlowerNet(
        num_classes=num_classes,
        use_layer_norm=use_layer_norm,
        model_name=configs.get('model-name', 'dinov2_vitb14'),
        load_pretrained=False
    )
    model = model.to(device)

    log_interval = configs['log-interval']

    print(f'\n---------- prediction start at: {device} ----------\n')
    print(f'图像目录: {test_dir}')
    if use_layer_norm:
        print(f'使用LayerNorm层')

    # # 直接使用model文件夹下的best_model.pt文件
    # model_params_path = "../model/best_model.pt"
    #
    # # 确保指定的模型参数文件存在
    # if not os.path.exists(model_params_path):
    #     print(f"错误：指定的模型参数文件不存在: {model_params_path}")
    #     sys.exit(1)
    #
    # print(f"正在加载模型参数: {model_params_path}")

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

    # 花卉类别ID到模型索引的映射
    category_to_idx = {
        "164": 0, "165": 1, "166": 2, "167": 3, "169": 4, "171": 5, "172": 6, "173": 7, 
        "174": 8, "176": 9, "177": 10, "178": 11, "179": 12, "180": 13, "183": 14, "184": 15,
        "185": 16, "186": 17, "188": 18, "189": 19, "190": 20, "192": 21, "193": 22, "194": 23,
        "195": 24, "197": 25, "198": 26, "199": 27, "200": 28, "201": 29, "202": 30, "203": 31,
        "204": 32, "205": 33, "206": 34, "207": 35, "208": 36, "209": 37, "210": 38, "211": 39,
        "212": 40, "213": 41, "214": 42, "215": 43, "216": 44, "217": 45, "218": 46, "220": 47,
        "221": 48, "222": 49, "223": 50, "224": 51, "225": 52, "226": 53, "227": 54, "228": 55,
        "229": 56, "230": 57, "231": 58, "232": 59, "233": 60, "234": 61, "235": 62, "236": 63,
        "237": 64, "238": 65, "239": 66, "240": 67, "241": 68, "242": 69, "243": 70, "244": 71,
        "245": 72, "1734": 73, "1743": 74, "1747": 75, "1749": 76, "1750": 77, "1751": 78,
        "1759": 79, "1765": 80, "1770": 81, "1772": 82, "1774": 83, "1776": 84, "1777": 85,
        "1780": 86, "1784": 87, "1785": 88, "1786": 89, "1789": 90, "1796": 91, "1797": 92,
        "1801": 93, "1805": 94, "1806": 95, "1808": 96, "1818": 97, "1827": 98, "1833": 99
    }
    
    # 创建反向映射：从模型索引到原始类别ID
    idx_to_category = {int(v): int(k) for k, v in category_to_idx.items()}
    print(f"已使用硬编码的类别映射，包含{len(category_to_idx)}个类别")
    
    # 用于存储预测结果
    predictions = []

    with torch.no_grad():
        # 加载模型检查点
        try:
            model.load_state_dict(torch.load(model_params_path, map_location=device, weights_only=True), strict=False)
            model.eval()
            print(f"模型加载成功！开始预测...")
        except Exception as e:
            print(f"加载模型时出错: {e}")
            sys.exit(1)

        for batch, (images, img_names, is_valid) in enumerate(dataloader, start=1):
            images = images.to(device)
            outputs = model(images)

            # 计算top-1准确率的预测结果
            _, top1_indices = torch.topk(outputs, 1, dim=1)

            # 收集预测结果
            for i in range(len(images)):
                # 获取图像文件名
                img_name = img_names[i]
                
                if is_valid[i]:
                    # 正常图像 - 获取预测类别和置信度
                    predicted_idx = top1_indices[i].item()
                    predicted_category = idx_to_category[predicted_idx]
                    
                    # 计算置信度（使用softmax）
                    probabilities = torch.nn.functional.softmax(outputs[i], dim=0)
                    confidence = probabilities[predicted_idx].item()
                else:
                    # 无效图像 - 设置默认类别和置信度为0
                    # 可以选择任意一个默认类别ID，这里使用第一个类别ID
                    predicted_category = int(list(category_to_idx.keys())[0])
                    confidence = 0.0

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
    # 按照filename列升序排序
    df = df.sort_values('filename', ascending=True)
    # 保存时不包含索引列，但包含列名
    df.to_csv(output_file, index=False)
    print(f'预测结果已保存到: {output_file}')
    print(f'共处理 {len(predictions)} 个文件\n')
    print('---------- prediction finished ----------\n')


if __name__ == '__main__':
    main()