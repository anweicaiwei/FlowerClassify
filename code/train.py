import json
import os
from datetime import datetime

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import FlowerNet
from utils import FlowerDataset, MixupDataset, L1RegularizedLoss  # 导入数据增强函数
from utils import get_loss_function, get_optimizer, get_lr_scheduler


def main():
    """主函数，执行模型训练"""
    # 加载配置
    config_path = '../model/config.json'
    with open(config_path, 'r') as f:
        configs = json.load(f)

    # 生成时间戳用于checkpoint
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 获取数据增强方法，用于训练集
    # augmentation = get_augmentations()

    # 创建训练集的变换（包含基本处理）
    # 改进数据增强策略
    # 修改训练集数据增强
    img_size = configs.get('image-size', 224)

    # 创建共享的基础变换
    base_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # 定义训练集变换（添加数据增强）
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        base_transform
    ])

    # 定义验证集变换（不使用数据增强）
    valid_transform = base_transform

    # 直接加载已分割好的数据集
    train_dataset = FlowerDataset(
        csv_file='../../FlowerClassify/datasets/train_split.csv',
        img_dir='../../FlowerClassify/datasets/train',
        transform=train_transform  # 直接传递单一变换
    )

    valid_dataset = FlowerDataset(
        csv_file='../../FlowerClassify/datasets/valid_split.csv',
        img_dir='../../FlowerClassify/datasets/valid',
        transform=valid_transform  # 验证集不使用增强
    )

    # 获取类别数量和数据集大小
    num_classes = train_dataset.num_classes
    configs['num-classes'] = num_classes
    train_dataset_size = len(train_dataset)
    valid_dataset_size = len(valid_dataset)

    # 在初始化数据集后添加
    if configs.get('use-mixup', False):
        train_dataset = MixupDataset(train_dataset, alpha=configs.get('mixup-alpha', 0.4))
        print(f"已启用Mixup数据增强，alpha={configs.get('mixup-alpha', 0.4)}")

    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=configs['batch-size'],
        num_workers=configs['num-workers'],
        shuffle=True  # 每次epoch都会打乱数据顺序 增强模型泛化能力 防止过拟合
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

    # 添加阶段性解冻相关配置
    use_layer_wise_unfreeze = configs.get('use-layer-wise-unfreeze', False)
    unfreeze_epochs = configs.get('unfreeze-epochs', [5, 10, 15])
    early_fc_only = configs.get('early-fc-only', True)

    # 设置设备
    device = torch.device(configs['device'])

    # 初始化模型
    use_layer_norm = configs.get('use-layer-norm', False)

    model = FlowerNet(
        num_classes=num_classes,
        use_layer_norm=use_layer_norm,
        model_name=configs.get('model-name', 'dinov2_vitb14'),
    )
    model = model.to(device)

    # 从配置文件读取损失函数类型并获取损失函数
    loss_function_type = configs.get('loss-function', 'cross_entropy')
    criterion = get_loss_function(loss_function_type)
    l1_lambda = 0

    # 如果是L1正则化损失，设置模型
    if loss_function_type == 'l1_regularized_cross_entropy':
        criterion = criterion.set_model(model)
        l1_lambda = configs.get('l1-lambda', 0.001)
        criterion = criterion.set_l1_lambda(l1_lambda)
    # 如果使用Mixup且不是L1正则化损失，自动切换到L1RegularizedLoss
    elif configs.get('use-mixup', False):
        print("注意：检测到使用Mixup数据增强，自动切换到L1RegularizedLoss以支持Mixup标签格式")
        criterion = L1RegularizedLoss().set_model(model)

    # 从配置文件读取优化器类型并获取优化器
    optimizer_type = configs.get('optimizer-type', 'adam')
    optimizer = get_optimizer(
        model.parameters(),  # 修改这里，传递参数迭代器而不是整个模型
        optimizer_type,
        learning_rate=configs['learning-rate'],
        weight_decay=configs['weight-decay'],
        l1_lambda=l1_lambda if l1_lambda > 0 else None
    )

    # 添加学习率调度器
    scheduler_type = configs.get('lr-scheduler-type', 'step')
    # 获取预热相关参数
    warmup_epochs = configs.get('warmup_epochs', 0)
    warmup_type = configs.get('warmup_type', 'linear')

    # 创建学习率调度器时传递预热参数
    scheduler = get_lr_scheduler(
        optimizer,
        scheduler_type,
        step_size=configs.get('lr-scheduler-step-size', 10),
        gamma=configs.get('lr-scheduler-gamma', 0.5),
        T_max=configs.get('num-epochs', 100),  # 余弦退火调度器需要的T_max参数
        eta_min=configs['learning-rate'] * 0.01,
        warmup_epochs=warmup_epochs,
        warmup_type=warmup_type
    )

    # 早停机制相关参数
    # 修改早停参数
    # 针对CosineAnnealingWarmRestarts的推荐值
    if scheduler_type == 'cosine_warm_restarts':
        # 设置更大的早停轮数，至少为T_0*2
        T_0 = 10
        early_stopping_patience = max(30, T_0 * 3)  # 至少30轮，或T_0的3倍
    else:
        early_stopping_patience = configs.get('early-stopping-patience', 15)
    early_stopping_counter = 0

    # 梯度裁剪参数
    use_grad_clip = configs.get('use-grad-clip', False)
    grad_clip_value = configs.get('grad-clip-value', 1.0)

    # 创建基于时间戳的checkpoint目录
    checkpoints_dir = os.path.join('checkpoints', 'OneGPU', timestamp)
    os.makedirs(checkpoints_dir, exist_ok=True)
    print(f"检查点将保存至目录: {checkpoints_dir}")

    # 使用配置中的文件名，但保存路径改为基于时间戳的目录
    load_checkpoint_path = configs['load-checkpoint-path']
    best_checkpoint_path = os.path.join(checkpoints_dir, 'best_model.pt')
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
    # 添加预热参数信息
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
    print("使用动态数据增强: 每次训练时随机应用增强策略")  # 提示使用动态数据增强

    # 打印阶段性解冻配置信息
    if use_layer_wise_unfreeze:
        print(f"使用阶段性解冻策略")
        print(f"解冻轮次配置: {unfreeze_epochs}")
        print(f"初始阶段仅训练全连接层: {early_fc_only}")

    # 改进的阶段性解冻函数
    def freeze_layers(model, unfreeze_level):
        """根据解冻级别冻结或解冻模型的不同层"""
        # 首先冻结所有层
        for param in model.parameters():
            param.requires_grad = False

        if unfreeze_level == 0:
            # 仅解冻全连接层
            for param in model.fc1.parameters():
                param.requires_grad = True
            for param in model.fc2.parameters():
                param.requires_grad = True
            if hasattr(model, 'fc3'):
                for param in model.fc3.parameters():
                    param.requires_grad = True
            if hasattr(model, 'fc_bn1'):
                for param in model.fc_bn1.parameters():
                    param.requires_grad = True
            if hasattr(model, 'fc_bn2'):
                for param in model.fc_bn2.parameters():
                    param.requires_grad = True
        elif unfreeze_level == 1:
            # 解冻DINOv2的最后几层和全连接层
            # 对于不同大小的DINOv2模型，解冻不同数量的层
            num_layers_to_unfreeze = 1
            if hasattr(model, 'model_size') and model.model_size in ['large', 'giant']:
                num_layers_to_unfreeze = 2

            # 解冻全连接层
            for param in model.fc1.parameters():
                param.requires_grad = True
            for param in model.fc2.parameters():
                param.requires_grad = True
            if hasattr(model, 'fc3'):
                for param in model.fc3.parameters():
                    param.requires_grad = True
            if hasattr(model, 'fc_bn1'):
                for param in model.fc_bn1.parameters():
                    param.requires_grad = True
            if hasattr(model, 'fc_bn2'):
                for param in model.fc_bn2.parameters():
                    param.requires_grad = True

            # 解冻DINOv2的最后几个transformer层
            if hasattr(model.backbone, 'encoder') and hasattr(model.backbone.encoder, 'layer'):
                for param in model.backbone.encoder.layer[-num_layers_to_unfreeze:].parameters():
                    param.requires_grad = True
        elif unfreeze_level == 2:
            # 改为渐进式解冻：解冻约一半的transformer层，而不是全部
            # 解冻全连接层
            for param in model.fc1.parameters():
                param.requires_grad = True
            for param in model.fc2.parameters():
                param.requires_grad = True
            if hasattr(model, 'fc3'):
                for param in model.fc3.parameters():
                    param.requires_grad = True
            if hasattr(model, 'fc_bn1'):
                for param in model.fc_bn1.parameters():
                    param.requires_grad = True
            if hasattr(model, 'fc_bn2'):
                for param in model.fc_bn2.parameters():
                    param.requires_grad = True

        #     # 对于large模型，解冻约一半的transformer层
        #     if hasattr(model.backbone, 'encoder') and hasattr(model.backbone.encoder, 'layer'):
        #         total_layers = len(model.backbone.encoder.layer)
        #         # 解冻最后一半的层
        #         num_layers_to_unfreeze = total_layers // 2
        #         for param in model.backbone.encoder.layer[-num_layers_to_unfreeze:].parameters():
        #             param.requires_grad = True
        elif unfreeze_level >= 3:
            # 最终阶段才解冻所有层
            for param in model.parameters():
                param.requires_grad = True

    # 初始化阶段：如果启用了early_fc_only，则只训练全连接层
    current_unfreeze_level = 0
    if use_layer_wise_unfreeze and early_fc_only:
        freeze_layers(model, current_unfreeze_level)
        print(f"初始阶段: 仅训练全连接层")

    # 训练循环
    for epoch in range(configs['num-epochs']):
        # 检查是否需要解冻更多层
        if use_layer_wise_unfreeze and epoch in unfreeze_epochs:
            # 获取基于索引的解冻级别
            level_index = unfreeze_epochs.index(epoch) + 1
            # 添加限制：确保解冻级别不超过2
            current_unfreeze_level = min(level_index, 2)
            
            freeze_layers(model, current_unfreeze_level)
            print(f"\n第{epoch}轮: 解冻更多层，当前解冻级别: {current_unfreeze_level}")
            
            # 解冻层后需要重新初始化优化器以包含新解冻的参数
            # 根据解冻级别调整学习率
            learning_rate = configs['learning-rate']
            if current_unfreeze_level == 2:
                # 当解冻更多层时降低学习率
                learning_rate = configs['learning-rate'] * 0.5
                print(f"解冻到级别{current_unfreeze_level}，学习率调整为: {learning_rate}")
            elif current_unfreeze_level >= 3:
                # 最终阶段进一步降低学习率
                learning_rate = configs['learning-rate'] * 0.3
                print(f"解冻到级别{current_unfreeze_level}，学习率调整为: {learning_rate}")
            
            optimizer = get_optimizer(
                filter(lambda p: p.requires_grad, model.parameters()),
                optimizer_type,
                learning_rate=learning_rate,
                weight_decay=configs['weight-decay'],
                l1_lambda=l1_lambda if l1_lambda > 0 else None
            )
            
            # 同时重新创建学习率调度器
            scheduler = get_lr_scheduler(
                optimizer,
                scheduler_type,
                step_size=configs.get('lr-scheduler-step-size', 10),
                gamma=configs.get('lr-scheduler-gamma', 0.5),
                T_max=configs.get('num-epochs', 100),
                eta_min=configs['learning-rate'] * 0.01,
                warmup_epochs=0,  # 解冻时不再需要预热
                warmup_type=warmup_type
            )

        model.train()
        epoch_loss = 0.0
        train_correct = 0
        train_total = 0

        # 获取梯度累积步数配置
        gradient_accumulation_steps = configs.get('gradient-accumulation-steps', 1)

        # 在训练循环中修改
        for batch, data in enumerate(train_dataloader, start=1):
            # 检查data的长度，以确定是否使用了Mixup
            if len(data) == 2:
                # 标准数据格式
                images, labels = data
            else:
                # Mixup数据格式: (mixed_img, label1, label2, lam)
                images, label1, label2, lam = data
                # 将Mixup数据打包为元组，便于传递给损失函数
                labels = (label1, label2, lam)
            
            images = images.to(device)
            # 根据数据格式决定如何处理labels
            if isinstance(labels, tuple) and len(labels) == 3:
                # Mixup格式，分别将label1和label2移至设备
                label1, label2, lam = labels
                label1 = label1.to(device)
                label2 = label2.to(device)
                # 确保lam是正确形状的标量张量
                if isinstance(lam, torch.Tensor):
                    lam = lam.to(device).view(-1)  # 重塑为1D张量
                else:
                    # 创建一个与batch_size匹配的lam张量
                    lam = torch.full((label1.size(0),), lam, device=device)
                labels = (label1, label2, lam)
            else:
                labels = labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 梯度累积
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            # 只有当批次是梯度累积步数的倍数时才更新参数
            if batch % gradient_accumulation_steps == 0 or batch == len(train_dataloader):
                # 可选的梯度裁剪
                if use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
            
                optimizer.step()
                optimizer.zero_grad()
            
            # 记录实际的损失值（乘以梯度累积步数）
            epoch_loss += loss.item() * gradient_accumulation_steps
            
            # 计算训练准确率 - 修复Mixup数据的准确率计算
            if isinstance(labels, tuple) and len(labels) == 3:
                label1, label2, lam = labels
                # 使用softmax获取概率
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                train_total += label1.size(0)
                
                # 计算基于混合标签的准确率
                # 对于每个样本，如果预测概率最高的类别是label1或label2中的任何一个，则计为正确
                _, predicted = torch.max(probabilities, 1)
                correct = ((predicted == label1) | (predicted == label2)).sum().item()
                train_correct += correct
            else:
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # 打印训练批次信息
            if batch % log_interval == 0:
                print(
                    f'[train] [{epoch:03d}/{configs["num-epochs"]:03d}] [{batch:04d}/{len(train_dataloader):04d}] loss: {loss.item():.5f}')

        # 计算当前epoch的平均损失和训练准确率
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        train_accuracy = train_correct / train_total

        # 更新学习率（在验证前更新，符合PyTorch推荐实践）
        scheduler.step()

        # 验证循环
        model.eval()
        with torch.no_grad():
            valid_correct = 0
            valid_total = 0
            valid_loss = 0.0
            for batch, (images, labels) in enumerate(valid_dataloader, start=1):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

                # 计算验证准确率
                _, predicted = torch.max(outputs.data, 1)
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()

                # 打印验证批次信息
                if batch % log_interval == 0:
                    print(
                        f'[valid] [{epoch:03d}/{configs["num-epochs"]:03d}] [{batch:04d}/{len(valid_dataloader):04d}]')

            valid_accuracy = valid_correct / valid_total
            avg_valid_loss = valid_loss / len(valid_dataloader)

            # 保存最佳模型
            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                torch.save(model.state_dict(), best_checkpoint_path)
                print(f"新的最佳模型已保存: {best_checkpoint_path}")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                print(f"早停计数器: {early_stopping_counter}/{early_stopping_patience}")

            # 保存最新模型
            last_accuracy = valid_accuracy
            torch.save(model.state_dict(), last_checkpoint_path)
            # 保存类别映射文件（在第一个epoch结束后保存一次即可）
            if epoch == 0:
                category_map_path = os.path.join(checkpoints_dir, 'category_mapping.json')
                # 修改这里，使用正确的属性名dataset访问category_to_idx
                category_mapping = {int(k): v for k, v in train_dataset.dataset.category_to_idx.items()}
                with open(category_map_path, 'w') as f:
                    json.dump(category_mapping, f)
                print(f"类别映射已保存至: {category_map_path}")
            # 打印当前epoch的结果
            print(f"Epoch: {epoch + 1}/{configs['num-epochs']}")
            print(f"Train Loss: {avg_epoch_loss:.5f} | Train Accuracy: {train_accuracy:.4f}")
            print(f"Valid Loss: {avg_valid_loss:.5f} | Valid Accuracy: {valid_accuracy:.4f}")
            print(f"Best Accuracy: {best_accuracy:.4f}")

            # 早停检查
            if early_stopping_counter >= early_stopping_patience:
                print(f"早停机制触发，已连续{early_stopping_patience}个epoch无性能提升")
                break

    print(f"\n训练完成! 最佳验证准确率: {best_accuracy:.4f}")
    print(f"最佳模型已保存至: {best_checkpoint_path}")
    print(f"最新模型已保存至: {last_checkpoint_path}")


if __name__ == '__main__':
    main()