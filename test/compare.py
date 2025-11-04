import csv

# 文件路径
test_labels_path = r'test_labels.csv'
submission_path = r'../results/submission.csv'

# 读取测试标签文件，建立文件名到正确category_id的映射
true_labels = {}
with open(test_labels_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        filename = row['filename']
        category_id = int(row['category_id'])
        true_labels[filename] = category_id

# 读取提交结果文件，进行比较
total_count = 0
correct_count = 0

with open(submission_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        filename = row['filename']
        predicted_category_id = int(row['predicted_category_id'])
        
        total_count += 1
        
        # 检查文件是否在测试标签中
        if filename in true_labels:
            true_category_id = true_labels[filename]
            if predicted_category_id == true_category_id:
                correct_count += 1
        else:
            print(f"警告: 文件 {filename} 在测试标签中未找到")

# 计算准确率
if total_count > 0:
    accuracy = correct_count / total_count
    print(f"总样本数: {total_count}")
    print(f"正确预测数: {correct_count}")
    print(f"准确率: {accuracy:.6f} ({accuracy*100:.2f}%)")
else:
    print("没有找到任何预测结果")
