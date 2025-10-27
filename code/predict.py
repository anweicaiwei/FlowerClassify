import os
import sys
import argparse  # 添加命令行参数解析模块

import pandas as pd
import toml
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models import FlowerNet
from utils import get_test_dataset  # 使用我们自定义的数据集类


def main():
    """主函数，执行模型预测（仅支持无标签推理）"""
    # 解析命令行参数 - 修改为必需参数
    parser = argparse.ArgumentParser(description='模型预测脚本（仅支持无标签推理）')
    parser.add_argument('test_folder', help='测试文件夹路径')
    parser.add_argument('output_file', help='输出文件路径')
    args = parser.parse_args()

    # 加载配置 - 仍然需要配置文件中的其他参数
    configs = toml.load('model/config.toml')

    # 直接使用命令行参数，不再从配置文件读取或覆盖
    test_img_dir = args.test_folder
    output_file = args.output_file

    # 创建测试集的变换
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ConvertImageDtype(torch.float32),  # 将uint8转换为float32
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 使用utils.py中的函数创建无标签测试集
    # 修改函数调用参数顺序，确保正确传递参数
    test_dataset = get_test_dataset(None, test_img_dir, test_transform)
    
    if test_dataset is None:
        print("无法创建测试集，程序退出")
        sys.exit(1)

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
        model_name=configs.get('model-name', 'resnet34'),
        use_layer_norm=configs.get('use-layer-norm', False)
    )
    model = model.to(device)

    log_interval = configs['log-interval']

    print(f'\n---------- prediction start at: {device} ----------\n')
    print(f'图像目录: {test_img_dir}')
    print('使用无标签推理模式')

    # 从配置中获取自定义模型参数路径
    try:
        model_params_path = configs['custom-model-params-path']
    except KeyError:
        print("错误：配置文件中缺少必要的模型参数路径配置项")
        print("请在config.toml中添加'custom-model-params-path'配置项")
        sys.exit(1)

    # 确保指定的模型参数文件存在
    if not os.path.exists(model_params_path):
        print(f"错误：指定的模型参数文件不存在: {model_params_path}")
        sys.exit(1)

    print(f"正在加载模型参数: {model_params_path}")

    # 创建一个反向映射：从索引映射回原始类别ID
    idx_to_category = {v: k for k, v in test_dataset.category_to_idx.items()}

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

        # 处理无标签数据
        for batch, (images, _, filenames) in enumerate(dataloader, start=1):
            images = images.to(device)
            outputs = model(images)

            # 只计算top1预测结果
            _, top1_indices = torch.topk(outputs, 1, dim=1)

            # 收集预测结果
            batch_size = images.size(0)

            for i in range(batch_size):
                # 获取预测类别和置信度
                predicted_idx = top1_indices[i].item()
                predicted_category = idx_to_category[predicted_idx]

                # 计算置信度（使用softmax）
                probabilities = torch.nn.functional.softmax(outputs[i], dim=0)
                confidence = probabilities[predicted_idx].item()

                # 获取图像文件名
                img_name = filenames[i]

                # 添加到预测列表
                predictions.append([img_name, predicted_category, confidence])

            if batch % log_interval == 0:
                print(f'[predict] [{batch:04d}/{dataloader_size:04d}]')

    # 打印预测信息
    print('\n--------------------------------------')
    print('无标签数据，已完成预测')
    print(f'预测图像总数: {dataset_size}')
    print(f'使用的模型参数路径: {model_params_path}')
    print('--------------------------------------\n')

    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        print(f"警告：输出文件目录不存在，将创建目录: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    # 创建DataFrame并保存
    df = pd.DataFrame(predictions, columns=['filename', 'category_id', 'confidence'])
    # 保存时不包含索引列，但包含列名
    df.to_csv(output_file, index=False)
    print(f'预测结果已保存到: {output_file}')
    print('---------- prediction finished ----------\n')


if __name__ == '__main__':
    main()