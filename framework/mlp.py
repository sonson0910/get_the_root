import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.model(x)

model = MLP()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters in the model: {total_params}")
print(model)
