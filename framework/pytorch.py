import torch
x = torch.tensor([1.0, 2.0, 3.0])
print(x)

a = torch.rand(2, 3)
b = torch.randn(2, 3)
z = torch.zeros(2, 3)
print(a)
print(b)
print(z)

x = torch.tensor([[1., 2.], [3., 4.]])
y = torch.tensor([[5., 6.], [7., 8.]])

print(x + y)
print(x @ y)
print(x * y)
print(x.T)

import numpy as np

arr = np.array([[1, 2, 3]])
t = torch.from_numpy(arr)
arr2 = t.numpy()

x = torch.randn(1000, 1000)
x = x.to('cuda')
y = torch.randn(1000, 1000).to('cuda')
z = x @ y
print(z)

x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 3 * x + 1
y.backward()
print(x.grad)

import numpy as np, torch, time

size = 10000
x_np = np.random.rand(size, size)
y_np = np.random.rand(size, size)

start = time.time()
np.dot(x_np, y_np)
print("Numpy time:", time.time() - start)

x_t = torch.rand(size, size)
y_t = torch.rand(size, size)

start = time.time()
torch.mm(x_t, y_t)
print("PyTorch (CPU) time:", time.time() - start)

x_t = x_t.to('cuda')
y_t = y_t.to('cuda')
torch.cuda.synchronize()
start = time.time()
torch.mm(x_t, y_t)
torch.cuda.synchronize()
print("PyTorch (GPU) time:", time.time() - start)
