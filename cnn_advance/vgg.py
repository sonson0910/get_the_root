import torch
import torch.nn as nn

class VGG9(nn.Module):
    def __init__(self):
        super(VGG9, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),  # Conv1
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), # Conv2
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # Pool1

            nn.Conv2d(64, 128, 3, padding=1), # Conv3
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),# Conv4
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # Pool2

            nn.Conv2d(128, 256, 3, padding=1),# Conv5
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),# Conv6
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # Pool3
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),     # Adjust input size based on image dimensions
            nn.ReLU(),
            nn.Linear(256, 10),               # Assuming 10 classes
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model = VGG9()
model_params = sum(p.numel() for p in model.parameters())
print(f"VGG9 params: {model_params:,}")
