import numpy as np

def f(x):
    return x**2

def f_prime(x):
    return 2*x

def numerical_derivative(f, x, h=1e-5):
    return (f(x+h) - f(x)) / h

x0 = 2.0
print("f(x0): ", f(x0))
print("f_prime(x0): ", f_prime(x0))
print("numerical_derivative(f, x0): ", numerical_derivative(f, x0))

def g(v):
    x, y = v
    return x**2 + y**2

def grad_g(v, h=1e-5):
    x, y = v
    dx = (g([x+h, y]) - g([x, y])) / h
    dy = (g([x, y+h]) - g([x, y])) / h
    return np.array([dx, dy])
    
print("g([1, 2]): ", g([1, 2]))
print("grad_g([1, 2]): ", grad_g([1, 2]))





