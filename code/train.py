import os
import json
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datetime import datetime

from utils import FlowerDataset, get_augmentations  # 导入数据增强函数
from model import FlowerNet
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
    augmentation = get_augmentations()

    # 创建训练集的变换（包含基本处理）
    train_transform = transforms.Compose([
        # 使用RandomResizedCrop代替直接的Resize，更符合预训练模型的训练方式
        transforms.RandomResizedCrop(400, scale=(0.8, 1.0)),
        augmentation,  # 应用单一的增强变换
        transforms.ToTensor(),  # 转换为张量
        # 修改为ResNet预训练模型使用的归一化参数
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # 创建验证集的变换（不包含增强，仅基本处理）
    valid_transform = transforms.Compose([
        transforms.RandomResizedCrop(400, scale=(0.8, 1.0)),
        transforms.ToTensor(),  # 转换为张量
        # 验证集也使用相同的归一化参数
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # 直接加载已分割好的数据集
    train_dataset = FlowerDataset(
        csv_file='datasets/train_split.csv',
        img_dir='datasets/train',
        transform=train_transform  # 直接传递单一变换
    )

    valid_dataset = FlowerDataset(
        csv_file='datasets/valid_split.csv',
        img_dir='datasets/valid',
        transform=valid_transform  # 验证集不使用增强
    )

    # 获取类别数量和数据集大小
    num_classes = train_dataset.num_classes
    configs['num-classes'] = num_classes
    train_dataset_size = len(train_dataset)
    valid_dataset_size = len(valid_dataset)

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
    # 获取激活函数配置，默认为gelu
    activation_fn = configs.get('activation-fn', 'gelu')

    model = FlowerNet(
        num_classes=num_classes,
        pretrained=configs['load-pretrained'],
        model_name=configs['model-name'],
        use_layer_norm=use_layer_norm,
        activation_fn=activation_fn
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
    best_checkpoint_path = os.path.join(checkpoints_dir, 'best-ckpt.pt')
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
    # 添加激活函数信息打印
    print(f"使用激活函数: {activation_fn}")
    print("使用动态数据增强: 每次训练时随机应用增强策略")  # 提示使用动态数据增强

    # 打印阶段性解冻配置信息
    if use_layer_wise_unfreeze:
        print(f"使用阶段性解冻策略")
        print(f"解冻轮次配置: {unfreeze_epochs}")
        print(f"初始阶段仅训练全连接层: {early_fc_only}")

    # 阶段性解冻函数定义
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
            if hasattr(model, 'fc_bn'):
                for param in model.fc_bn.parameters():
                    param.requires_grad = True
        elif unfreeze_level == 1:
            # 解冻layer4和全连接层
            for param in model.layer4.parameters():
                param.requires_grad = True
            for param in model.fc1.parameters():
                param.requires_grad = True
            for param in model.fc2.parameters():
                param.requires_grad = True
            if hasattr(model, 'fc_bn'):
                for param in model.fc_bn.parameters():
                    param.requires_grad = True
        elif unfreeze_level == 2:
            # 解冻layer3-4和全连接层
            for param in model.layer3.parameters():
                param.requires_grad = True
            for param in model.layer4.parameters():
                param.requires_grad = True
            for param in model.fc1.parameters():
                param.requires_grad = True
            for param in model.fc2.parameters():
                param.requires_grad = True
            if hasattr(model, 'fc_bn'):
                for param in model.fc_bn.parameters():
                    param.requires_grad = True
        elif unfreeze_level >= 3:
            # 解冻所有层
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
            current_unfreeze_level = unfreeze_epochs.index(epoch) + 1
            freeze_layers(model, current_unfreeze_level)
            print(f"\n第{epoch}轮: 解冻更多层，当前解冻级别: {current_unfreeze_level}")

            # 解冻层后需要重新初始化优化器以包含新解冻的参数
            optimizer = get_optimizer(
                filter(lambda p: p.requires_grad, model.parameters()),
                optimizer_type,
                learning_rate=configs['learning-rate'],
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

        for batch, (images, labels) in enumerate(train_dataloader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # 可选的梯度裁剪
            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

            optimizer.step()
            epoch_loss += loss.item()

            # 计算训练准确率
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
                category_mapping = {int(k): v for k, v in train_dataset.category_to_idx.items()}
                with open(category_map_path, 'w') as f:
                    json.dump(category_mapping, f)
                print(f"类别映射已保存至: {category_map_path}")

        print(f'[valid] [{epoch:03d}/{configs["num-epochs"]:03d}] accuracy: {valid_accuracy:.4f}')
        print(f'当前学习率: {optimizer.param_groups[0]["lr"]:.8f}')
        print(f'训练准确率: {train_accuracy:.4f}, 训练损失: {avg_epoch_loss:.5f}')
        print(f'验证准确率: {valid_accuracy:.4f}, 验证损失: {avg_valid_loss:.5f}')

        # 检查早停条件
        if early_stopping_counter >= early_stopping_patience:
            print(f"早停机制触发：连续{early_stopping_patience}个epoch无性能提升")
            break

    print(f'best accuracy: {best_accuracy:.3f}')
    print(f'last accuracy: {last_accuracy:.3f}')

    print('\n---------- training finished ----------\n')


if __name__ == '__main__':
    main()