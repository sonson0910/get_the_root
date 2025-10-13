import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniAlexNet(nn.Module):
    def __init__(self):
        super(MiniAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),  # Conv1
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # Pool1

            nn.Conv2d(64, 128, 3, padding=1), # Conv2
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # Pool2

            nn.Conv2d(128, 256, 3, padding=1),# Conv3
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 8 * 8, 512),     # Adjust input size based on image dimensions
            nn.ReLU(),
            nn.Linear(512, 10),               # Assuming 10 classes
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
