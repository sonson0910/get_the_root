import torch
import torch.nn as nn

x = torch.randn(1, 1, 5, 5)

conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)
y1 = conv1(x)
print("Output shape after convolution:", y1.shape)

conv2 = nn.Conv2d(1, 1, 3, stride=1, padding=1)
y2 = conv2(x)
print("Output shape after convolution with padding:", y2.shape)

conv3 = nn.Conv2d(1, 1, 3, stride=2, padding=0)
y3 = conv3(x)
print("Output shape after convolution with stride 2:", y3.shape)

pool = nn.MaxPool2d(kernel_size=2, stride=2)
y4 = pool(x)
print("Output shape after max pooling:", y4.shape)
