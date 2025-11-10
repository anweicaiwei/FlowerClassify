import os

import torch
import torch.nn as nn


class FlowerNet(nn.Module):
    def __init__(self, num_classes=100, use_layer_norm=False, model_name='dinov2_vitb14', load_pretrained=True):
        super().__init__()

        # 创建模型架构
        if model_name == 'dinov2_vits14':
            model_size = 'small'
            self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=False)
        elif model_name == 'dinov2_vitb14':
            model_size = 'base'
            self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=False)
        elif model_name == 'dinov2_vitl14':
            model_size = 'large'
            self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=False)
        elif model_name == 'dinov2_vitg14':
            model_size = 'giant'
            self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14', pretrained=False)
        else:
            print(f"警告：不支持的模型名称 '{model_name}'，默认使用 'dinov2_vitb14'")
            model_size = 'base'
            self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=False)

        # 只有在需要加载预训练模型时才执行以下代码
        if load_pretrained:
            try:
                # 用户指定的本地缓存目录
                local_cache_dir = "../model/cache"
                
                # 确保缓存目录存在
                os.makedirs(local_cache_dir, exist_ok=True)
                print(f"使用本地缓存目录: {local_cache_dir}")
                
                # 定义模型在缓存中的路径模式
                model_cache_patterns = {
                    'dinov2_vits14': os.path.join(local_cache_dir, 'dinov2_vits14.pt'),
                    'dinov2_vitb14': os.path.join(local_cache_dir, 'dinov2_vitb14.pt'),
                    'dinov2_vitl14': os.path.join(local_cache_dir, 'dinov2_vitl14.pt'),
                    'dinov2_vitg14': os.path.join(local_cache_dir, 'dinov2_vitg14.pt')
                }
                
                # 仅当本地不存在模型文件时才下载
                if model_name in model_cache_patterns and os.path.exists(model_cache_patterns[model_name]):
                    print(f"检测到本地缓存目录已存在{model_name}预训练模型，直接加载...")
                    
                    # 从本地文件加载模型权重
                    state_dict = torch.load(model_cache_patterns[model_name], map_location=torch.device('cpu'), weights_only=False)
                    # 加载预训练权重
                    self.backbone.load_state_dict(state_dict)
                else:
                    print(f"本地缓存目录未找到{model_name}预训练模型，正在下载...")
                    
                    # 从官方仓库下载模型
                    if model_name == 'dinov2_vits14':
                        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
                    elif model_name == 'dinov2_vitb14':
                        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
                    elif model_name == 'dinov2_vitl14':
                        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
                    elif model_name == 'dinov2_vitg14':
                        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
                    
                    # 下载完成后保存模型到本地缓存目录
                    if model_name in model_cache_patterns:
                        print(f"将{model_name}预训练模型保存到本地缓存目录...")
                        torch.save(self.backbone.state_dict(), model_cache_patterns[model_name])
                        print(f"模型已成功保存到: {model_cache_patterns[model_name]}")
            except Exception as e:
                print(f"加载模型失败: {e}")
                print("请确保您已经安装了所需的依赖，或者手动下载模型文件到 ../model/cache 目录")
                raise
        else:
            print(f"不加载预训练模型，仅使用模型架构")

        # 获取特征维度（不同模型大小的特征维度不同）
        feature_dims = {
            'small': 384,
            'base': 768,
            'large': 1024,
            'giant': 1536
        }
        feature_dim = feature_dims[model_size]

        # 可选的LayerNorm层
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(feature_dim)

        # 分类头
        # 保持类似的分类头结构，但简化为两层
        # 优化分类头，添加更多的自适应能力
        self.fc1 = nn.Linear(in_features=feature_dim, out_features=1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc_relu1 = nn.GELU()

        self.fc2 = nn.Linear(in_features=1024, out_features=768)  # 增加中间层神经元数量
        self.fc_bn2 = nn.BatchNorm1d(768)
        self.fc_relu2 = nn.GELU()

        self.fc3 = nn.Linear(in_features=768, out_features=512)
        self.fc_bn3 = nn.BatchNorm1d(512)  # 添加额外的BatchNorm层
        self.fc_relu3 = nn.GELU()

        self.fc4 = nn.Linear(in_features=512, out_features=num_classes)

        # 调整Dropout策略
        self.dropout1 = nn.Dropout(p=0.4)  # 降低Dropout率以保留更多特征
        self.dropout2 = nn.Dropout(p=0.3)
        self.dropout3 = nn.Dropout(p=0.2)

    def forward(self, inputs):
        outputs = self.backbone(inputs)
        if self.use_layer_norm:
            outputs = self.layer_norm(outputs)

        outputs = self.dropout1(outputs)
        outputs = self.fc1(outputs)
        outputs = self.fc_bn1(outputs)
        outputs = self.fc_relu1(outputs)

        outputs = self.dropout2(outputs)
        outputs = self.fc2(outputs)
        outputs = self.fc_bn2(outputs)
        outputs = self.fc_relu2(outputs)

        outputs = self.dropout3(outputs)
        outputs = self.fc3(outputs)
        outputs = self.fc_bn3(outputs)
        outputs = self.fc_relu3(outputs)

        outputs = self.fc4(outputs)

        return outputs