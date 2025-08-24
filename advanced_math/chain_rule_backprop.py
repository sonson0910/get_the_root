import torch

x = torch.tensor(3.0, requires_grad=True)

y = 2 * x
z = y ** 2

z.backward()
print(x.grad)

