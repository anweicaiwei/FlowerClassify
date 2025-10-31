import os  # 添加os模块导入
import random
import shutil  # 导入shutil用于文件复制

import pandas as pd  # 导入pandas用于读取CSV
import toml
import torch
import torchvision.transforms as transforms
from PIL import Image  # 添加PIL Image导入
from torch.utils.data import Dataset
from tqdm import tqdm  # 导入tqdm用于显示进度条

configs = toml.load('configs/config_OneGPU.toml')

# 将数据增强功能移到模块级别，确保可以在任何地方访问

# 修改get_augmentations函数，移除不必要的resize操作
def get_augmentations():
    """创建并返回一个综合性的数据增强方法
    Returns:
        callable: 包含多种增强操作的Compose对象
    """
    # 移除内部的resize，因为后续训练代码中会统一调整大小
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.RandomVerticalFlip(p=0.3),    # 随机垂直翻转
        transforms.RandomRotation(degrees=(-10, 10)),  # 随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 随机颜色抖动
        transforms.RandomGrayscale(p=0.2),  # 随机灰度转换
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # 随机高斯模糊
    ])

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
            # 方法1: 使用PIL直接读取图像（推荐）
            image = Image.open(img_path).convert('RGB')
            
            # # 方法2: 如果坚持使用read_image，可以添加转换步骤
            # image_tensor = read_image(img_path)
            # # 将Tensor转换为PIL Image
            # image = transforms.ToPILImage()(image_tensor)
            
            # 应用变换
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
                dummy_image = torch.zeros(3, 224, 224, dtype=torch.float32)
                dummy_label = 0
                return dummy_image, dummy_label


# 修改后的函数，只将数据集分为train、valid两部分

def prepare_datasets(csv_file, img_dir, valid_ratio=0.15):
    """准备训练集和验证集
    Args:
        csv_file (string): 原始标签CSV文件的路径
        img_dir (string): 原始图像文件夹的路径
        valid_ratio (float): 验证集占总数据集的比例
    """
    # 读取原始CSV文件
    df = pd.read_csv(csv_file)
    
    # 创建类别映射
    category_to_idx = create_category_mapping(df)
    num_classes = len(category_to_idx)
    
    # 创建保存训练集和验证集的目录
    train_img_dir = os.path.join('datasets', 'train')
    valid_img_dir = os.path.join('datasets', 'valid')
    
    # 清空已有的目录（如果存在）并显示进度条
    for dir_path in [train_img_dir, valid_img_dir]:
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
    
    # 按类别ID分组
    grouped = df.groupby('category_id')
    
    # 分割数据
    for category_id, group in grouped:
        # 对每个类别的数据进行打乱
        group = group.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # 计算验证集大小
        valid_size = int(len(group) * valid_ratio)
        
        # 分割数据
        valid_group = group.iloc[:valid_size].copy()
        train_group = group.iloc[valid_size:].copy()
        
        # 添加到训练集和验证集列表
        train_data.append(train_group)
        valid_data.append(valid_group)
    
    # 合并所有类别的数据
    train_df = pd.concat(train_data, ignore_index=True)
    valid_df = pd.concat(valid_data, ignore_index=True)
    
    # 统计需要处理的总文件数
    total_files = len(train_df) + len(valid_df)
    print(f"开始处理数据集，总计 {total_files} 个文件...")
    
    # 只复制原始训练集图像，不进行增强
    print("\n复制训练集图像...")
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="训练集", unit="文件"):
        src_path = os.path.join(img_dir, row['filename'])
        dst_path = os.path.join(train_img_dir, row['filename'])
        
        # 直接复制原始文件
        shutil.copy2(src_path, dst_path)
    
    # 复制验证集图像，显示总体进度
    print("\n复制验证集图像...")
    for _, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="验证集", unit="文件"):
        src_path = os.path.join(img_dir, row['filename'])
        dst_path = os.path.join(valid_img_dir, row['filename'])
        # 直接复制文件，覆盖已存在的文件
        shutil.copy2(src_path, dst_path)
    
    # 保存新的CSV文件
    print("\n正在保存CSV文件...")
    train_csv_path = os.path.join('datasets', 'train_split.csv')
    valid_csv_path = os.path.join('datasets', 'valid_split.csv')
    
    train_df.to_csv(train_csv_path, index=False)  # 保存原始训练集，不进行增强
    valid_df.to_csv(valid_csv_path, index=False)
    
    print(f"\n训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(valid_df)}")
    print(f"类别数量: {num_classes}")
    print(f"训练集CSV已保存至: {train_csv_path}")
    print(f"验证集CSV已保存至: {valid_csv_path}")
    print(f"训练集图像已复制至: {train_img_dir}")
    print(f"验证集图像已复制至: {valid_img_dir}")
    print("注意：数据增强将在训练过程中随机应用，不再提前生成增强图像")
    
    # 根据用户需求，只生成一次数据，不需要返回数据集对象供训练使用
    # 因此不创建数据集实例，只生成文件即可


def main():
    """主函数，当脚本直接运行时执行数据预处理"""
    # 调用数据预处理函数，分割为train、valid两个数据集
    prepare_datasets(
        csv_file=configs['data-label'],
        img_dir=configs['data-root'],
        valid_ratio=configs['valid-split-ratio']
    )
    
    print("\n数据预处理完成！已生成train、valid两个数据集及其CSV文件。")


if __name__ == '__main__':
    main()