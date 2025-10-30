import os  # 导入os用于路径处理
import random
import shutil  # 导入shutil用于文件复制

import pandas as pd  # 导入pandas用于读取CSV
import toml
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image  # 导入读取图像的函数
from tqdm import tqdm  # 导入tqdm用于显示进度条

configs = toml.load('configs/config_OneGPU.toml')

# 将数据增强功能移到模块级别，确保可以在任何地方访问

def get_augmentations():
    """创建并返回所有数据增强方法的列表
    Returns:
        list: 包含所有增强方法的列表
    """
    # 增强方式1：随机水平翻转+颜色抖动
    aug1 = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Resize((600, 600)),  # 输出尺寸为600x600
    ])
    # 增强方式2：随机旋转+放大
    aug2 = transforms.Compose([
        transforms.RandomRotation(degrees=(-10, 10)),
        transforms.Resize((600, 600)),
        # # 直接放大到目标尺寸的1.3倍
        # transforms.Resize((int(600*1.3), int(600*1.3))),
        transforms.CenterCrop((600, 600)),  # 裁剪中心600x600，去除黑边
    ])
    # 增强方式3：亮度调整+高斯模糊
    aug3 = transforms.Compose([
        transforms.ColorJitter(brightness=(0.7, 1.3)),
        transforms.GaussianBlur(kernel_size=3),
        transforms.Resize((600, 600)),
    ])
    # 增强方式4：垂直翻转+饱和度调整
    aug4 = transforms.Compose([
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ColorJitter(saturation=(0.7, 1.3)),
        transforms.Resize((600, 600)),
    ])
    # 增强方式5：随机灰度+对比度调整
    aug5 = transforms.Compose([
        transforms.RandomGrayscale(p=1.0),
        transforms.ColorJitter(contrast=(0.7, 1.3)),
        transforms.Resize((600, 600)),
    ])
    
    # 返回所有增强方法的列表
    return [aug1, aug2, aug3, aug4, aug5]


def apply_all_augmentations(image_path, save_path, augmentations):
    """应用所有数据增强方法并保存图像
    Args:
        image_path (str): 原始图像路径
        save_path (str): 保存增强图像的路径
        augmentations (list): 所有增强方法的列表
    Returns:
        list: 包含所有生成的增强图像文件名的列表
    """
    generated_files = []
    try:
        # 使用PIL打开图像
        image = Image.open(image_path).convert('RGB')
        
        # 保存原始图像
        base_name, ext = os.path.splitext(save_path)
        image.save(save_path)
        generated_files.append(os.path.basename(save_path))
        
        # 应用每一种数据增强方法并保存
        for i, augmentation in enumerate(augmentations):
            augmented_image = augmentation(image)
            # 保存增强图像，添加增强标记
            augmented_save_path = os.path.join(os.path.dirname(save_path), f"{os.path.basename(base_name)}_aug{i}{ext}")
            augmented_image.save(augmented_save_path)
            generated_files.append(os.path.basename(augmented_save_path))
        
        return generated_files
    except Exception as e:
        print(f"处理图像时出错 {image_path}: {e}")
        return []


def create_category_mapping(df):
    """创建类别ID到索引的映射，确保类别索引从0开始连续"""
    unique_categories = sorted(df['category_id'].unique())
    return {category: idx for idx, category in enumerate(unique_categories)}


class FlowerDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """初始化数据集
        Args:
            csv_file (string): 标签CSV文件的路径
            img_dir (string): 图像文件夹的路径
            transform (callable, optional): 应用于图像的变换函数
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transforms = transform
        # 创建类别映射
        self.category_to_idx = create_category_mapping(self.data_frame)
        self.num_classes = len(self.category_to_idx)

    # 返回数据集大小
    def __len__(self):
        return len(self.data_frame)
    # 获取数据集中的第idx个样本
    def __getitem__(self, idx):
        # 获取图像文件名和类别ID
        img_name = self.data_frame.iloc[idx, 0]
        category_id = self.data_frame.iloc[idx, 1]
        
        # 构建完整的图像路径
        img_path = os.path.join(self.img_dir, img_name)
        
        # 读取图像，添加异常处理
        try:
            image = read_image(img_path)
            
            # 将PIL图像转换为Tensor并应用变换
            if self.transforms:
                # 检查transforms是否为列表
                if isinstance(self.transforms, list):
                    # 如果是列表，则随机选择一个变换
                    selected_transform = random.choice(self.transforms)
                    image = selected_transform(image)
                else:
                    # 如果不是列表，则直接应用变换
                    image = self.transforms(image)
                
            # 将类别ID映射为索引
            label = self.category_to_idx[category_id]
            
            return image, label
        except Exception as e:
            # 处理图像读取错误，记录错误信息并跳过该图像
            print(f"Error reading image {img_path}: {e}")
            # 返回前一个有效的图像，防止训练中断
            # 注意：这种方式是临时解决方案，建议后续清理损坏的图像
            if idx > 0:
                return self.__getitem__(idx - 1)
            else:
                # 如果是第一个图像，创建一个空白图像和随机标签
                import torch
                dummy_image = torch.zeros(3, 224, 224, dtype=torch.float32)
                dummy_label = 0
                return dummy_image, dummy_label


# 修改后的函数，支持将数据集分为train、valid、test三部分

def prepare_datasets(csv_file, img_dir, valid_ratio=0.15, test_ratio=0.15):
    """准备训练集、验证集和测试集
    Args:
        csv_file (string): 原始标签CSV文件的路径
        img_dir (string): 原始图像文件夹的路径
        valid_ratio (float): 验证集占总数据集的比例
        test_ratio (float): 测试集占总数据集的比例
    """
    # 获取所有增强方法
    augmentations = get_augmentations()
    
    # 读取原始CSV文件
    df = pd.read_csv(csv_file)
    
    # 创建类别映射
    category_to_idx = create_category_mapping(df)
    num_classes = len(category_to_idx)
    
    # 创建保存训练集、验证集和测试集的目录
    train_img_dir = os.path.join('datasets', 'train')
    valid_img_dir = os.path.join('datasets', 'valid')
    test_img_dir = os.path.join('datasets', 'test')
    
    # 清空已有的目录（如果存在）并显示进度条
    for dir_path in [train_img_dir, valid_img_dir, test_img_dir]:
        if os.path.exists(dir_path):
            # 获取目录中的所有文件
            files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
            if files:
                print(f"清空目录: {dir_path}")
                # 使用tqdm显示清空进度
                for file_name in tqdm(files, desc=f"清空{os.path.basename(dir_path)}", unit="文件"):
                    file_path = os.path.join(dir_path, file_name)
                    os.remove(file_path)
        # 确保目录存在
        os.makedirs(dir_path, exist_ok=True)
    
    # 按类别分割数据集，确保每个类别的数据都能被合理分割
    train_data = []
    valid_data = []
    test_data = []
    
    # 按类别ID分组
    grouped = df.groupby('category_id')
    
    # 分割数据，但不立即复制文件
    for category_id, group in grouped:
        # 对每个类别的数据进行打乱
        group = group.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # 计算验证集和测试集大小
        valid_size = int(len(group) * valid_ratio)
        test_size = int(len(group) * test_ratio)
        
        # 分割数据
        valid_group = group.iloc[:valid_size].copy()
        test_group = group.iloc[valid_size:valid_size+test_size].copy()
        train_group = group.iloc[valid_size+test_size:].copy()
        
        # 添加到训练集、验证集和测试集列表
        train_data.append(train_group)
        valid_data.append(valid_group)
        test_data.append(test_group)
    
    # 合并所有类别的数据
    train_df = pd.concat(train_data, ignore_index=True)
    valid_df = pd.concat(valid_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)
    
    # 统计需要处理的总文件数
    total_files = len(train_df) + len(valid_df) + len(test_df)
    print(f"开始处理数据集，总计 {total_files} 个文件...")
    
    # 复制训练集图像并应用增强，显示总体进度
    print("\n复制训练集图像并应用所有数据增强方法...")
    train_data_augmented = []
    
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="训练集", unit="文件"):
        src_path = os.path.join(img_dir, row['filename'])
        dst_path = os.path.join(train_img_dir, row['filename'])
        
        # 应用所有增强方法并保存图像
        generated_files = apply_all_augmentations(src_path, dst_path, augmentations)
        
        if generated_files:
            # 添加所有生成的图像到增强数据集
            for file_name in generated_files:
                augmented_row = row.copy()
                augmented_row['filename'] = file_name
                train_data_augmented.append(augmented_row)
    
    # 创建增强后的训练数据集DataFrame
    train_augmented_df = pd.DataFrame(train_data_augmented)
    
    # 复制验证集图像，显示总体进度
    print("\n复制验证集图像...")
    for _, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="验证集", unit="文件"):
        src_path = os.path.join(img_dir, row['filename'])
        dst_path = os.path.join(valid_img_dir, row['filename'])
        # 直接复制文件，覆盖已存在的文件
        shutil.copy2(src_path, dst_path)
    
    # 复制测试集图像，显示总体进度
    print("\n复制测试集图像...")
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="测试集", unit="文件"):
        src_path = os.path.join(img_dir, row['filename'])
        dst_path = os.path.join(test_img_dir, row['filename'])
        # 直接复制文件，覆盖已存在的文件
        shutil.copy2(src_path, dst_path)
    
    # 保存新的CSV文件
    print("\n正在保存CSV文件...")
    train_csv_path = os.path.join('datasets', 'train_split.csv')
    valid_csv_path = os.path.join('datasets', 'valid_split.csv')
    test_csv_path = os.path.join('datasets', 'test_split.csv')
    
    train_augmented_df.to_csv(train_csv_path, index=False)  # 保存增强后的训练集
    valid_df.to_csv(valid_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    
    print(f"\n训练集大小(包含增强图像): {len(train_augmented_df)}")
    print(f"验证集大小: {len(valid_df)}")
    print(f"测试集大小: {len(test_df)}")
    print(f"类别数量: {num_classes}")
    print(f"训练集CSV已保存至: {train_csv_path}")
    print(f"验证集CSV已保存至: {valid_csv_path}")
    print(f"测试集CSV已保存至: {test_csv_path}")
    print(f"训练集图像已复制至: {train_img_dir}")
    print(f"验证集图像已复制至: {valid_img_dir}")
    print(f"测试集图像已复制至: {test_img_dir}")
    print(f"对每个训练图像应用了 {len(augmentations)} 种不同的数据增强方法")
    
    # 根据用户需求，只生成一次数据，不需要返回数据集对象供训练使用
    # 因此不创建数据集实例，只生成文件即可


def main():
    """主函数，当脚本直接运行时执行数据预处理"""
    # 调用数据预处理函数，分割为train、valid、test三个数据集
    prepare_datasets(
        csv_file=configs['data-label'],
        img_dir=configs['data-root'],
        valid_ratio=configs['valid-split-ratio'],
        test_ratio=configs['test-split-ratio']
    )
    
    print("\n数据预处理完成！已生成train、valid、test三个数据集及其CSV文件。")


if __name__ == '__main__':
    main()