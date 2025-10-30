import os
from datetime import datetime

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import FlowerNet
from utils import get_loss_function, get_optimizer, get_lr_scheduler, get_train_valid_datasets

# 添加Mixup数据增强
class MixupDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, alpha=0.8):
        self.dataset = dataset
        self.alpha = alpha
        self.num_classes = dataset.num_classes
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img1, label1 = self.dataset[index]
        # 随机选择第二个样本
        index2 = torch.randint(0, len(self.dataset), (1,)).item()
        img2, label2 = self.dataset[index2]
        
        # 生成mixup系数
        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        else:
            lam = 1.0
        
        # mixup图像和标签
        mixed_img = lam * img1 + (1 - lam) * img2
        
        # 返回混合后的图像和原始标签
        return mixed_img, label1, label2, lam

# 渐进式训练类
class ProgressiveTraining:
    def __init__(self, model, configs):
        self.model = model
        self.configs = configs
        self.current_stage = 0
        self.stage_epochs = configs.get('stage_epochs', [10, 15, 20])  # 分阶段训练的轮数
        self.freeze_layers = configs.get('freeze_layers', True)  # 是否冻结低层
    
    def update_stage(self, epoch):
        # 计算当前阶段
        total_epochs = 0
        for i, stage_epoch in enumerate(self.stage_epochs):
            total_epochs += stage_epoch
            if epoch < total_epochs:
                new_stage = i
                break
        else:
            new_stage = len(self.stage_epochs) - 1
        
        # 如果进入新阶段，更新模型训练策略
        if new_stage != self.current_stage:
            self.current_stage = new_stage
            self._update_model_training_status()
            print(f"进入训练阶段 {new_stage+1}/{len(self.stage_epochs)}")
    
    def _update_model_training_status(self):
        # 根据不同阶段解冻不同数量的层
        if not self.freeze_layers:
            return
        
        # 渐进式解冻模型层
        if self.current_stage == 0:
            # 第一阶段：只训练分类头
            for param in self.model.conv1.parameters():
                param.requires_grad = False
            for param in self.model.layer1.parameters():
                param.requires_grad = False
            for param in self.model.layer2.parameters():
                param.requires_grad = False
            for param in self.model.layer3.parameters():
                param.requires_grad = False
            # 只训练layer4和分类头
        elif self.current_stage == 1:
            # 第二阶段：训练layer3、layer4和分类头
            for param in self.model.conv1.parameters():
                param.requires_grad = False
            for param in self.model.layer1.parameters():
                param.requires_grad = False
            for param in self.model.layer2.parameters():
                param.requires_grad = False
        else:
            # 第三阶段：训练所有层
            for param in self.model.parameters():
                param.requires_grad = True

def main():
    """主函数，执行模型训练"""
    configs = {
        # "device": "cuda",
        "data-root": "../data/flowerclassify/train/train",
        "data-label": "../data/flowerclassify/train_labels.csv",
        "valid-split-ratio": 0.15,
        "test-split-ratio": 0.15,
        "test-csv-file": "datasets/test_split.csv",
        "test-img-dir": "datasets/test",
        "custom-model-params-path": "../model/best-model.pt",
        "custom-output-path": "checkpoints/OneGPU/20251025_071825/predictions.csv",
        "model-name": "resnet50",
        "batch-size": 64,
        "num-epochs": 100,
        "num-workers": 20,
        "num-classes": 100,
        "log-interval": 10,
        "load-checkpoint": False,
        "load-pretrained": True,
        "load-checkpoint-path": "checkpoints/best-ckpt.pt",
        "loss-function": "label_smoothing_cross_entropy",
        "learning-rate": 0.00005,
        "weight-decay": 0.0005,
        "optimizer-type": "adamw",
        "lr-scheduler-type": "cosine_warm_restarts",
        "lr-scheduler-step-size": 10,
        "lr-scheduler-gamma": 0.5,
        "warmup_epochs": 5,
        "warmup_type": "cosine",
        "early-stopping-patience": 10,
        "l1-lambda": 0.0000005,
        "use-layer-norm": True,
        "use-grad-clip": True,
        "grad-clip-value": 1.0,
        "activation-fn": "gelu",
        "use-mixup": True,
        "mixup-alpha": 0.8,
        "stage_epochs": [10, 15, 20],
        "freeze_layers": True
    }
    
    # 创建训练集和验证集的变换
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        # transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomErasing(),
    ])
    
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # 使用utils.py中的函数直接创建并划分训练集和验证集
    train_dataset, valid_dataset, category_to_idx, num_classes = get_train_valid_datasets(
        csv_file=configs['data-label'],
        img_dir=configs['data-root'],
        valid_ratio=configs['valid-split-ratio'],
        transform=train_transform  # 为了简单起见，对训练集和验证集使用相同的变换
    )
    
    # 如果启用Mixup数据增强
    if configs.get('use-mixup', False):
        train_dataset = MixupDataset(train_dataset, alpha=configs.get('mixup-alpha', 0.8))
        print(f"使用Mixup数据增强，alpha值: {configs.get('mixup-alpha', 0.8)}")
    
    # 获取数据集大小
    configs['num-classes'] = num_classes
    train_dataset_size = len(train_dataset)
    valid_dataset_size = len(valid_dataset)
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=configs['batch-size'],
        num_workers=configs['num-workers'],
        shuffle=True
    )
    
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=configs['batch-size'],
        num_workers=configs['num-workers'],
        shuffle=False
    )
    
    log_interval = configs['log-interval']
    best_accuracy = 0.0
    last_accuracy = 0.0
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化模型
    use_layer_norm = configs['use-layer-norm']
    activation_fn = configs.get('activation-fn', 'gelu')
    
    model = FlowerNet(
        num_classes=num_classes, 
        pretrained=configs['load-pretrained'], 
        model_name=configs['model-name'],
        use_layer_norm=use_layer_norm,
        activation_fn=activation_fn
    )
    model = model.to(device)
    
    # 获取损失函数
    loss_function_type = configs['loss-function']
    criterion = get_loss_function(loss_function_type)
    l1_lambda = 0
    
    # 如果是L1正则化损失，设置模型
    if loss_function_type == 'l1_regularized_cross_entropy':
        criterion = criterion.set_model(model)
        l1_lambda = configs['l1-lambda']
        criterion = criterion.set_l1_lambda(l1_lambda)
    
    # 获取优化器
    optimizer_type = configs['optimizer-type']
    optimizer = get_optimizer(
        model,
        optimizer_type,
        learning_rate=configs['learning-rate'],
        weight_decay=configs['weight-decay'],
        l1_lambda=l1_lambda if l1_lambda > 0 else None
    )
    
    # 获取预热相关参数
    warmup_epochs = configs.get('warmup_epochs', 0)
    warmup_type = configs.get('warmup_type', 'linear')
    
    # 添加学习率调度器
    scheduler_type = configs['lr-scheduler-type']
    scheduler = get_lr_scheduler(
        optimizer,
        scheduler_type,
        step_size=configs['lr-scheduler-step-size'],
        gamma=configs['lr-scheduler-gamma'],
        T_max=configs.get('num-epochs', 100),  # 余弦退火调度器需要的T_max参数
        eta_min=configs['learning-rate'] * 0.01,
        warmup_epochs=warmup_epochs,
        warmup_type=warmup_type
    )
    
    # 早停机制相关参数
    early_stopping_patience = configs['early-stopping-patience']
    early_stopping_counter = 0
    
    # 梯度裁剪参数
    use_grad_clip = configs['use-grad-clip']
    grad_clip_value = configs['grad-clip-value']
    
    # 创建基于时间戳的checkpoint目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoints_dir = os.path.join('checkpoints', 'OneGPU', timestamp)
    os.makedirs(checkpoints_dir, exist_ok=True)
    print(f"检查点将保存至目录: {checkpoints_dir}")
    
    # 设置检查点路径
    load_checkpoint_path = configs['load-checkpoint-path']
    best_checkpoint_path = os.path.join(checkpoints_dir, 'best-model.pt')
    last_checkpoint_path = os.path.join(checkpoints_dir, 'last-ckpt.pt')
    
    # 确保目录存在
    os.makedirs(os.path.dirname(best_checkpoint_path), exist_ok=True)
    
    if configs['load-checkpoint']:
        try:
            model.load_state_dict(
                torch.load(load_checkpoint_path, map_location=device, weights_only=True),
                strict=False
            )
            print(f"成功加载检查点（非严格模式）: {load_checkpoint_path}")
        except Exception as e:
            print(f"加载检查点失败: {e}")
            print("将从头开始训练模型")
    
    print(f'\n---------- training start at: {device} ----------\n')
    print(f"损失函数: {loss_function_type}")
    print(f"优化器: {optimizer_type}")
    print(f"学习率调度器: {scheduler_type}")
    if warmup_epochs > 0:
        print(f"使用学习率预热: {warmup_type}方式，预热轮数: {warmup_epochs}")
    print(f"早停机制配置: 连续{early_stopping_patience}个epoch无提升则停止训练")
    if l1_lambda > 0:
        print(f"使用L1正则化，系数: {l1_lambda}")
    if use_grad_clip:
        print(f"使用梯度裁剪，阈值: {grad_clip_value}")
    if use_layer_norm:
        print(f"使用LayerNorm层")
    print(f"使用激活函数: {activation_fn}")
    if configs.get('freeze_layers', False):
        print(f"使用渐进式训练，阶段划分: {configs.get('stage_epochs', [10, 15, 20])}个epoch")
    
    # 初始化渐进式训练
    progressive_trainer = ProgressiveTraining(model, configs)
    
    # 训练循环
    for epoch in range(configs['num-epochs']):
        # 更新训练阶段
        progressive_trainer.update_stage(epoch)
        
        model.train()
        epoch_loss = 0.0
        
        for batch, data in enumerate(train_dataloader, start=1):
            # 处理普通批次和Mixup批次
            if len(data) == 4:  # Mixup批次有4个元素
                images, labels1, labels2, lam = data
                images = images.to(device)
                labels1 = labels1.to(device)
                labels2 = labels2.to(device)
                lam = lam.to(device)
            else:  # 普通批次有2个元素
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # 计算损失
            if len(data) == 4:  # Mixup损失计算
                loss = lam * criterion(outputs, labels1) + (1 - lam) * criterion(outputs, labels2)
            else:  # 普通损失计算
                loss = criterion(outputs, labels)
            
            loss.backward()
            
            # 可选的梯度裁剪
            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
            
            optimizer.step()
            epoch_loss += loss.item()
            
            # 记录训练批次信息
            if batch % log_interval == 0:
                print(f'[train] [{epoch:03d}/{configs["num-epochs"]:03d}] [{batch:04d}/{len(train_dataloader):04d}] loss: {loss.item():.5f}')
        
        # 计算当前epoch的平均损失
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        
        # 更新学习率
        scheduler.step()
        
        # 验证循环
        model.eval()
        with torch.no_grad():
            accuracy = 0.0
            for batch, (images, labels) in enumerate(valid_dataloader, start=1):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                accuracy += (torch.argmax(outputs, dim=1) == labels).sum().item()
                
                # 记录验证批次信息
                if batch % log_interval == 0:
                    print(f'[valid] [{epoch:03d}/{configs["num-epochs"]:03d}] [{batch:04d}/{len(valid_dataloader):04d}]')
            
            accuracy /= valid_dataset_size
            
            # 保存最佳模型
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), best_checkpoint_path)
                print(f"新的最佳模型已保存: {best_checkpoint_path}")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                print(f"早停计数器: {early_stopping_counter}/{early_stopping_patience}")
            
            # 保存最新模型
            last_accuracy = accuracy
            torch.save(model.state_dict(), last_checkpoint_path)

        # 保存类别映射文件（在第一个epoch结束后保存一次即可）
        if epoch == 0:
            category_map_path = os.path.join(checkpoints_dir, 'category_mapping.json')
            import json
            category_mapping = {int(k): v for k, v in train_dataset.category_to_idx.items()}
            with open(category_map_path, 'w') as f:
                json.dump(category_mapping, f)
            print(f"类别映射已保存至: {category_map_path}")

        print(f'[valid] [{epoch:03d}/{configs["num-epochs"]:03d}] accuracy: {accuracy:.4f}')
        print(f'当前学习率: {optimizer.param_groups[0]["lr"]:.8f}')
        
        # 检查早停条件
        if early_stopping_counter >= early_stopping_patience:
            print(f"早停机制触发：连续{early_stopping_patience}个epoch无性能提升")
            break
    
    print(f'best accuracy: {best_accuracy:.3f}')
    print(f'last accuracy: {last_accuracy:.3f}')
    
    print('\n---------- training finished ----------\n')


if __name__ == '__main__':
    main()