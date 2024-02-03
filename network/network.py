import torch
from torch import Tensor
from torch.nn import Module, Linear
from torchvision.models import resnet18, ResNet18_Weights, ResNet


class ResNet18WithClassifier(Module):
    def __init__(self, num_classes: int):
        super(ResNet18WithClassifier, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = Linear(num_features, num_classes)
        torch.nn.init.xavier_uniform(self.resnet.fc.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.resnet(x)
        return x

    def get_embedding(self, x: Tensor) -> Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x


from layers.arcface import ArcFace


class ResNet18WithArcFace(Module):
    def __init__(self, device, num_classes: int):
        super(ResNet18WithArcFace, self).__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = ArcFace(in_features=num_features, out_features=num_classes, device=device)

    def forward(self, x: Tensor, labels) -> Tensor:
        x = self.backbone(x, labels=labels)
        return x

    def get_embedding(self, x: Tensor) -> Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x
