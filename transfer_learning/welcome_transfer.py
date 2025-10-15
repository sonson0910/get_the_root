import torch
import torch.nn as nn
from torchvision.models import resnet18

model = resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

print(model)

for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
print("Number of trainable parameters:", len(trainable_params))

 
for name, module in model.named_children():
    print(f'{name}: {module.__class__.__name__}')
