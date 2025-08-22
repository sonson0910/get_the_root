import numpy as np
import math

'''
example 1: 
'''

def f(x):
    return math.sin(x)

def f_prime(x):
    return math.cos(x)

def numerical_derivative(f, x, h=1e-5):
    return (f(x+h) - f(x)) / h

x0 = math.pi / 4
print("f(x0): ", f(x0))
print("f_prime(x0): ", f_prime(x0))
print("numerical_derivative(f, x0): ", numerical_derivative(f, x0))

'''
example 2: 
'''

def g(v):
    x, y = v
    return x**2 * y + 3 * y

def grad_g(v, h=1e-5):
    x, y = v
    df_dx = (g([x+h, y]) - g([x, y])) / h
    df_dy = (g([x, y+h]) - g([x, y])) / h
    return np.array([df_dx, df_dy])

v0 = np.array([1, 2])
print("g(v0): ", g(v0))
print("grad_g(v0): ", grad_g(v0))

'''
example 3: 
'''

import torch

x = torch.tensor(3.0, requires_grad=True)
y = torch.tensor(4.0, requires_grad=True)

f = x**2 + y**2

f.backward()

print("df/dx: ", x.grad.item())
print("df/dy: ", y.grad.item())



