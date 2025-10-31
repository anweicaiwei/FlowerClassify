import os
import sys
import json

import pandas as pd
import toml
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from data_preparation import FlowerDataset  # 使用我们自定义的数据集类
from models import FlowerNet


def main():
    """主函数，执行模型预测"""
    # 加载配置
    configs = toml.load('configs/config_OneGPU.toml')

    # 创建测试集的变换
    test_transform = transforms.Compose([
        transforms.RandomResizedCrop(400, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
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
    # 获取激活函数配置，默认为gelu
    activation_fn = configs.get('activation-fn', 'gelu')
    
    model = FlowerNet(
        num_classes=configs['num-classes'],
        pretrained=configs.get('load-pretrained', False),
        model_name=configs.get('model-name', 'resnet18'),
        use_layer_norm=configs.get('use-layer-norm', False),
        activation_fn=activation_fn
    )
    model = model.to(device)

    log_interval = configs['log-interval']

    print(f'\n---------- prediction start at: {device} ----------\n')
    print(f'使用测试集: {test_csv_file}')
    print(f'图像目录: {test_img_dir}')
    # 添加激活函数信息打印
    print(f'使用激活函数: {activation_fn}')

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

    # 尝试从与模型参数相同的目录加载类别映射文件
    category_map_path = os.path.join(os.path.dirname(model_params_path), 'category_mapping.json')
    if os.path.exists(category_map_path):
        with open(category_map_path, 'r') as f:
            category_to_idx = json.load(f)
        print(f"成功加载类别映射文件: {category_map_path}")
        idx_to_category = {int(v): k for k, v in category_to_idx.items()}
    else:
        # 如果没有找到类别映射文件，则使用数据集的映射
        print(f"警告：未找到类别映射文件，将使用数据集的映射: {category_map_path}")
        # 创建一个反向映射：从索引映射回原始类别ID
        idx_to_category = {v: k for k, v in test_dataset.category_to_idx.items()}

    # 用于存储预测结果
    predictions = []
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        top1_accuracy = 0.0
        top2_accuracy = 0.0
        top3_accuracy = 0.0

        # 加载模型检查点
        try:
            model.load_state_dict(torch.load(model_params_path, map_location=device, weights_only=True))
            model.eval()
            print(f"模型加载成功！开始预测...")
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
                
                # 获取真实标签
                true_category_id = test_dataset.data_frame.iloc[img_idx, 1]
                true_labels.append(true_category_id)

                # 获取预测类别和置信度
                predicted_idx = top1_indices[i].item()
                predicted_category = idx_to_category[predicted_idx]
                predicted_labels.append(predicted_category)

                # 计算置信度（使用softmax）
                probabilities = torch.nn.functional.softmax(outputs[i], dim=0)
                confidence = probabilities[predicted_idx].item()

                # 添加到预测列表
                predictions.append([img_name, predicted_category, confidence, true_category_id])

            if batch % log_interval == 0:
                print(f'[predict] [{batch:04d}/{dataloader_size:04d}]')

        # 计算最终准确率
        top1_accuracy /= dataset_size
        top2_accuracy /= dataset_size
        top3_accuracy /= dataset_size

    # 打印评估结果
    print('\n--------------------------------------')
    print(f'top1 accuracy: {top1_accuracy:.3f}')
    print(f'top2 accuracy: {top2_accuracy:.3f}')
    print(f'top3 accuracy: {top3_accuracy:.3f}')
    print(f'使用的模型参数路径: {model_params_path}')
    print('--------------------------------------\n')

    # 保存预测结果到CSV文件
    # 获取用户自定义的输出文件路径（如果有）
    custom_output_path = configs.get('custom-output-path', '')
    
    if custom_output_path:
        # 使用用户自定义的输出文件路径
        output_file = custom_output_path
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            print(f"警告：输出文件目录不存在，将创建目录: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
    else:
        # 默认保存在与模型相同的目录下，文件名为predictions.csv
        output_dir = os.path.dirname(model_params_path)
        output_file = os.path.join(output_dir, 'predictions.csv')

    # 创建DataFrame并保存
    df = pd.DataFrame(predictions, columns=['filename', 'predicted_category_id', 'confidence', 'true_category_id'])
    # 保存时不包含索引列，但包含列名
    df.to_csv(output_file, index=False)
    print(f'预测结果已保存到: {output_file}')
    
    # 额外进行预测结果与测试标签的比对，计算准确率
    print('\n--- 预测结果与测试标签比对分析 ---')
    
    # 1. 从保存的预测结果和原始测试标签CSV重新计算准确率
    # 读取保存的预测结果
    pred_df = pd.read_csv(output_file)
    
    # 读取原始测试标签
    test_df = pd.read_csv(test_csv_file)
    
    # 合并预测结果和测试标签（确保顺序一致）
    merged_df = pd.merge(test_df, pred_df, on='filename', how='inner')
    
    # 计算准确率
    correct_predictions = (merged_df['category_id'] == merged_df['predicted_category_id']).sum()
    total_predictions = len(merged_df)
    comparison_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    print(f"比对准确率: {comparison_accuracy:.3f} ({correct_predictions}/{total_predictions})")
    print('--------------------------------------\n')
    print('---------- prediction finished ----------\n')


if __name__ == '__main__':
    main()