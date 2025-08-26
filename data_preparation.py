import pandas as pd  # 导入pandas用于读取CSV
import os  # 导入os用于路径处理
import shutil  # 导入shutil用于文件复制
import random  # 导入random用于随机操作

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.io import read_image  # 导入读取图像的函数


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
        self.transform = transform
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
            if self.transform:
                image = self.transform(image)
            
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
    Returns:
        tuple: (train_dataset, valid_dataset, test_dataset, num_classes) 如果需要返回数据集对象
        或者只打印信息不返回数据集对象
    """
    # 读取原始CSV文件
    df = pd.read_csv(csv_file)
    
    # 创建类别映射
    category_to_idx = create_category_mapping(df)
    num_classes = len(category_to_idx)
    
    # 创建保存训练集、验证集和测试集的目录
    train_img_dir = os.path.join('datasets', 'train')
    valid_img_dir = os.path.join('datasets', 'valid')
    test_img_dir = os.path.join('datasets', 'test')
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(valid_img_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)
    
    # 按类别分割数据集，确保每个类别的数据都能被合理分割
    train_data = []
    valid_data = []
    test_data = []
    
    # 按类别ID分组
    grouped = df.groupby('category_id')
    
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
        
        # 将训练集图像复制到新目录
        for _, row in train_group.iterrows():
            src_path = os.path.join(img_dir, row['filename'])
            dst_path = os.path.join(train_img_dir, row['filename'])
            # 如果文件不存在才复制，避免覆盖
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
        
        # 将验证集图像复制到新目录
        for _, row in valid_group.iterrows():
            src_path = os.path.join(img_dir, row['filename'])
            dst_path = os.path.join(valid_img_dir, row['filename'])
            # 如果文件不存在才复制，避免覆盖
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
        
        # 将测试集图像复制到新目录
        for _, row in test_group.iterrows():
            src_path = os.path.join(img_dir, row['filename'])
            dst_path = os.path.join(test_img_dir, row['filename'])
            # 如果文件不存在才复制，避免覆盖
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
    
        # 添加到训练集、验证集和测试集列表
        train_data.append(train_group)
        valid_data.append(valid_group)
        test_data.append(test_group)
    
    # 合并所有类别的数据
    train_df = pd.concat(train_data, ignore_index=True)
    valid_df = pd.concat(valid_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)
    
    # 保存新的CSV文件
    train_csv_path = os.path.join('datasets', 'train_split.csv')
    valid_csv_path = os.path.join('datasets', 'valid_split.csv')
    test_csv_path = os.path.join('datasets', 'test_split.csv')
    
    train_df.to_csv(train_csv_path, index=False)
    valid_df.to_csv(valid_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(valid_df)}")
    print(f"测试集大小: {len(test_df)}")
    print(f"类别数量: {num_classes}")
    print(f"训练集CSV已保存至: {train_csv_path}")
    print(f"验证集CSV已保存至: {valid_csv_path}")
    print(f"测试集CSV已保存至: {test_csv_path}")
    print(f"训练集图像已复制至: {train_img_dir}")
    print(f"验证集图像已复制至: {valid_img_dir}")
    print(f"测试集图像已复制至: {test_img_dir}")
    
    # 根据用户需求，只生成一次数据，不需要返回数据集对象供训练使用
    # 因此不创建数据集实例，只生成文件即可


def main():
    """主函数，当脚本直接运行时执行数据预处理"""
    # 调用数据预处理函数，分割为train、valid、test三个数据集
    prepare_datasets(
        csv_file='datasets/data_labels.csv',
        img_dir='datasets/data/train',
        valid_ratio=0.15,  # 15%的数据作为验证集
        test_ratio=0.15    # 15%的数据作为测试集
    )
    
    print("数据预处理完成！已生成train、valid、test三个数据集及其CSV文件。")


if __name__ == '__main__':
    main()