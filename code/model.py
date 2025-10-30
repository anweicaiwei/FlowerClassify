import torch.nn as nn
import torchvision.models as models


class FlowerNet(nn.Module):
    def __init__(self, num_classes, pretrained, model_name, use_layer_norm=False, activation_fn='gelu',
                 use_attention=False):
        super().__init__()

        # 根据指定的模型名称选择适当的ResNet变体
        if model_name == 'resnet18':
            if pretrained:
                resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                resnet = models.resnet18()
            feature_dim = 512
        elif model_name == 'resnet34':
            if pretrained:
                resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            else:
                resnet = models.resnet34()
            feature_dim = 512
        elif model_name == 'resnet50':
            if pretrained:
                resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            else:
                resnet = models.resnet50()
            feature_dim = 2048
        else:
            raise ValueError(f"不支持的模型名称: {model_name}")

        # 保留ResNet的卷积层和池化层
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1

        # 替换激活函数
        self.activation_fn = activation_fn
        if activation_fn == 'gelu':
            self.relu = nn.GELU()
        elif activation_fn == 'swish':
            self.relu = nn.SiLU()
        elif activation_fn == 'mish':
            self.relu = nn.Mish()
        elif activation_fn == 'leaky_relu':
            self.relu = nn.LeakyReLU(negative_slope=0.1)
        else:
            self.relu = resnet.relu
            print(f"警告: 不支持的激活函数 '{activation_fn}'，已使用默认的ReLU")

        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # 全局平均池化和扁平化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        # 可选的注意力机制
        self.use_attention = use_attention
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 4),
                nn.Tanh(),
                nn.Linear(feature_dim // 4, feature_dim),
                nn.Sigmoid()
            )

        # 改进的分类头 - 两层MLP
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.5)

        # 可选的LayerNorm层
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(feature_dim)

        # 添加中间隐藏层增强模型表达能力
        self.fc1 = nn.Linear(in_features=feature_dim, out_features=feature_dim // 2)
        self.fc2 = nn.Linear(in_features=feature_dim // 2, out_features=num_classes)

    def forward(self, inputs):
        # 主干特征提取
        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = self.relu(outputs)
        outputs = self.maxpool(outputs)

        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)

        # 池化和扁平化
        outputs = self.avgpool(outputs)
        outputs = self.flatten(outputs)

        # 可选的LayerNorm
        if self.use_layer_norm:
            outputs = self.layer_norm(outputs)

        # 可选的注意力机制
        if self.use_attention:
            attention_weights = self.attention(outputs)
            outputs = outputs * attention_weights

        # 分类层，带Dropout防止过拟合
        outputs = self.dropout1(outputs)
        outputs = self.fc1(outputs)
        outputs = self.relu(outputs)  # 在隐藏层后添加激活函数
        outputs = self.dropout2(outputs)
        outputs = self.fc2(outputs)

        return outputs