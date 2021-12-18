import torch.nn as nn
import torch


class DConvNetV2(nn.Module):
    def __init__(self, num_classes):
        super(DConvNetV2, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d((2, 2))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.DeformConv2d(32, 64, 3, padding=1, modulation=True)
        self.bn2 = nn.BatchNorm2d(64)

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))

        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x
