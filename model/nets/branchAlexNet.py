import torch
from torch import nn

class BranchedAlexNet(nn.Module):
    def __init__(self, num_classes=2, dropout = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64*3, kernel_size=11, stride=4, padding=2, groups=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64*3, 192*3, kernel_size=5, padding=2, groups=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192*3, 384*3, kernel_size=3, padding=1, groups=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(384*3, 256*3, kernel_size=3, padding=1, groups=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(256*3, 256*3, kernel_size=3, padding=1, groups=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear((256*3) * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x