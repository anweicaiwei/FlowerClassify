import os
import sys
import argparse
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 直接导入模型类
from model import FlowerNet
# 导入测试数据集工具函数
from utils import get_test_dataset

# 设置固定的配置参数，替代从配置文件读取
def main():
    """主函数，执行模型预测（无标签推理）"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='花卉分类模型预测脚本')
    parser.add_argument('test_folder', help='测试文件夹路径')
    parser.add_argument('output_file', help='输出文件路径')
    args = parser.parse_args()

    test_img_dir = args.test_folder
    output_file = args.output_file

    # 创建测试集的变换
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 创建无标签测试集
    test_dataset = get_test_dataset(test_img_dir=test_img_dir, transform=test_transform)
    
    if test_dataset is None:
        sys.exit(1)

    # 创建数据加载器
    dataloader = DataLoader(
        test_dataset,
        batch_size=64,  # 固定批量大小
        num_workers=4,   # 固定工作进程数
        shuffle=False
    )

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型 - 使用默认参数
    model = FlowerNet(
        num_classes=100,  # 类别数
        pretrained=False,   # 不使用预训练权重
        model_name='resnet34',  # 模型架构
        use_layer_norm=True  # 使用LayerNorm
    )
    model = model.to(device)

    # 固定模型参数路径
    model_params_path = '../model/best-model.pt'

    # 确保指定的模型参数文件存在
    if not os.path.exists(model_params_path):
        print(f"错误：指定的模型参数文件不存在: {model_params_path}")
        sys.exit(1)

    # 类别映射
    category_to_idx = {
        "164": 0, "165": 1, "166": 2, "167": 3, "169": 4, "171": 5, "172": 6, "173": 7, "174": 8, "176": 9,
        "177": 10, "178": 11, "179": 12, "180": 13, "183": 14, "184": 15, "185": 16, "186": 17, "188": 18,
        "189": 19, "190": 20, "192": 21, "193": 22, "194": 23, "195": 24, "197": 25, "198": 26, "199": 27,
        "200": 28, "201": 29, "202": 30, "203": 31, "204": 32, "205": 33, "206": 34, "207": 35, "208": 36,
        "209": 37, "210": 38, "211": 39, "212": 40, "213": 41, "214": 42, "215": 43, "216": 44, "217": 45,
        "218": 46, "220": 47, "221": 48, "222": 49, "223": 50, "224": 51, "225": 52, "226": 53, "227": 54,
        "228": 55, "229": 56, "230": 57, "231": 58, "232": 59, "233": 60, "234": 61, "235": 62, "236": 63,
        "237": 64, "238": 65, "239": 66, "240": 67, "241": 68, "242": 69, "243": 70, "244": 71, "245": 72,
        "1734": 73, "1743": 74, "1747": 75, "1749": 76, "1750": 77, "1751": 78, "1759": 79, "1765": 80,
        "1770": 81, "1772": 82, "1774": 83, "1776": 84, "1777": 85, "1780": 86, "1784": 87, "1785": 88,
        "1786": 89, "1789": 90, "1796": 91, "1797": 92, "1801": 93, "1805": 94, "1806": 95, "1808": 96,
        "1818": 97, "1827": 98, "1833": 99
    }
    
    # 创建反向映射：从索引映射回原始类别ID
    idx_to_category = {int(v): k for k, v in category_to_idx.items()}

    # 用于存储预测结果
    predictions = []

    with torch.no_grad():
        # 加载模型检查点
        try:
            model.load_state_dict(torch.load(model_params_path, map_location=device, weights_only=True))
            model.eval()
        except Exception as e:
            print(f"加载模型时出错: {e}")
            sys.exit(1)

        # 处理无标签数据
        for images, _, filenames in dataloader:
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

    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 创建DataFrame并保存
    df = pd.DataFrame(predictions, columns=['filename', 'category_id', 'confidence'])
    df.to_csv(output_file, index=False)
    print(f"预测结果已保存到: {output_file}")

if __name__ == '__main__':
    main()