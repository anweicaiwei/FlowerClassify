import os
import pandas as pd
import re
from tqdm import tqdm  # 导入tqdm库用于显示进度条


def generate_augmented_labels(original_csv_path, augmented_img_dir, output_csv_path):
    """
    为增强后的数据生成与原始CSV相同结构的标签文件
    
    参数:
        original_csv_path (str): 原始标签CSV文件的路径
        augmented_img_dir (str): 增强后图像所在的目录
        output_csv_path (str): 输出CSV文件的路径
    """
    # 读取原始CSV文件
    print(f"正在读取原始标签文件: {original_csv_path}")
    df_original = pd.read_csv(original_csv_path)
    
    # 创建增强数据的标签列表
    augmented_data = []
    
    # 获取目录中的所有图像文件
    image_files = []
    for filename in os.listdir(augmented_img_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_files.append(filename)
    
    print(f"找到 {len(image_files)} 个图像文件，开始处理...")
    
    # 遍历增强后图像目录中的所有文件，并显示进度条
    for filename in tqdm(image_files, desc="处理进度", unit="个文件"):
        # 尝试处理增强后的文件名格式: img_000051_aug1.jpg
        match = re.match(r'(img_\d+)(_aug\d+)(\.\w+)?', filename)
        if match:
            original_base_name = match.group(1)
            extension = match.group(3) or '.jpg'  # 如果没有扩展名，默认为.jpg
            
            # 构建原始图像文件名
            original_filename = f"{original_base_name}{extension}"
        else:
            # 直接使用原始文件名
            original_filename = filename
        
        # 在原始数据中查找对应的标签
        original_row = df_original[df_original['filename'] == original_filename]
        
        if not original_row.empty:
            # 复制原始行数据，但替换文件名为当前文件名（可能是增强后的）
            new_row = original_row.iloc[0].copy()
            new_row['filename'] = filename
            augmented_data.append(new_row)
        else:
            # 仅在有问题时打印警告，避免进度条混乱
            if len(image_files) < 100 or tqdm.get_lock().acquire(False):
                try:
                    print(f"警告: 未找到 {original_filename} 对应的原始标签")
                finally:
                    tqdm.get_lock().release()
    
    # 创建增强数据的DataFrame
    if augmented_data:
        df_augmented = pd.DataFrame(augmented_data)
        
        # 保存为CSV文件，保持与原始文件相同的结构
        df_augmented.to_csv(output_csv_path, index=False)
        print(f"\n已成功生成增强数据的标签文件: {output_csv_path}")
        print(f"增强数据样本数量: {len(df_augmented)}")
    else:
        print("\n警告: 未找到有效的增强数据，未生成标签文件")


if __name__ == "__main__":
    # 示例用法
    original_csv = "datasets/data_labels.csv"  # 原始标签文件路径
    augmented_dir = "D:/ProjectDevelop/Data/Data/train_augmented"  # 增强后图像目录
    output_csv = "D:/ProjectDevelop/Data/Data/train_augmented_labels.csv"  # 输出标签文件路径

    generate_augmented_labels(original_csv, augmented_dir, output_csv)