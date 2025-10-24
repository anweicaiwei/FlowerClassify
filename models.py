import torch.nn as nn
import torchvision.models as models


class FlowerNet(nn.Module):
    def __init__(self, num_classes, pretrained, model_name):
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
        self.relu = resnet.relu
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

        # 全连接分类层，使用两层结构增强分类能力
        self.fc1 = nn.Linear(in_features=feature_dim, out_features=512)
        self.fc_bn = nn.BatchNorm1d(512)
        self.fc_relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)

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

        # 分类层，带Dropout防止过拟合
        outputs = self.dropout(outputs)
        outputs = self.fc1(outputs)
        outputs = self.fc_bn(outputs)
        outputs = self.fc_relu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.fc2(outputs)

        return outputs