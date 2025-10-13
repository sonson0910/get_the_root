import torch
import torch.nn as nn


# LeNet
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


lenet = LeNet()
lenet_params = sum(p.numel() for p in lenet.parameters())
print(f"LeNet params: {lenet_params:,}")

from torchvision.models import alexnet
alexnet_model = alexnet()
alexnet_params = sum(p.numel() for p in alexnet_model.parameters())
print(f"AlexNet params: {alexnet_params:,}")
