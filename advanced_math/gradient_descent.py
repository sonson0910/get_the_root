import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2

def grad(x):
    return 2*x

def gradient_descent(start_x, lr=0.1, n_iter=20):
    x = start_x
    history = [x]
    for i in range(n_iter):
        x = x - lr*grad(x)
        history.append(x)
    return history

x0 = 0
hist_small = gradient_descent(x0, lr=0.01, n_iter=50)
hist_good = gradient_descent(x0, lr=0.1, n_iter=50)

