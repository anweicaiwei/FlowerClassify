import torch.nn as nn
import torchvision.models as models


class FlowerNet(nn.Module):
    def __init__(self, num_classes,  model_name, pretrained=False, use_layer_norm=False, activation_fn='gelu'):
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
            # GELU激活函数: x * Φ(x)，Φ为标准正态分布的累积分布函数
            self.relu = nn.GELU()
        elif activation_fn == 'swish':
            # Swish激活函数: x * sigmoid(x)
            self.relu = nn.SiLU()  # PyTorch 1.7+中的SiLU就是Swish
        elif activation_fn == 'mish':
            # Mish激活函数: x * tanh(softplus(x))
            self.relu = nn.Mish()
        elif activation_fn == 'leaky_relu':
            # LeakyReLU激活函数: max(0, x) + negative_slope * min(0, x)
            self.relu = nn.LeakyReLU(negative_slope=0.1)
        elif activation_fn == 'relu':
            # 标准ReLU激活函数
            self.relu = nn.ReLU()
        else:
            # 默认为ReLU
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

        # 添加Dropout层来防止过拟合，特别适合大数据集
        self.dropout = nn.Dropout(p=0.5)
        # 增加第二个Dropout层以增强正则化效果
        self.dropout2 = nn.Dropout(p=0.3)
        
        # 可选的LayerNorm层
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(feature_dim)

        # 增强分类头，增加一个全连接层
        self.fc1 = nn.Linear(in_features=feature_dim, out_features=1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc_relu1 = nn.ReLU()

        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.fc_bn2 = nn.BatchNorm1d(512)
        self.fc_relu2 = nn.ReLU()

        self.fc3 = nn.Linear(in_features=512, out_features=num_classes)

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

        # 分类层，带Dropout防止过拟合
        outputs = self.dropout(outputs)
        outputs = self.fc1(outputs)
        outputs = self.fc_bn1(outputs)
        outputs = self.fc_relu1(outputs)
        outputs = self.dropout2(outputs)

        outputs = self.fc2(outputs)
        outputs = self.fc_bn2(outputs)
        outputs = self.fc_relu2(outputs)
        outputs = self.dropout2(outputs)

        outputs = self.fc3(outputs)

        return outputs