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

x0 = 5
hist_small = gradient_descent(x0, lr=0.01, n_iter=50)
hist_good = gradient_descent(x0, lr=0.1, n_iter=50)
hist_big = gradient_descent(x0, lr=1.0, n_iter=50)

X = np.linspace(-6, 6, 100)
Y = f(X)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(X, Y)
plt.scatter(hist_small, [f(x) for x in hist_small], color='red')
plt.title('learning rate=0.01(small)')

plt.subplot(1, 3, 2)
plt.plot(X, Y)
plt.scatter(hist_good, [f(x) for x in hist_good], color='green')
plt.title('learning rate=0.1(good)')

plt.subplot(1, 3, 3)
plt.plot(X, Y)
plt.scatter(hist_big, [f(x) for x in hist_big], color='blue')
plt.title('learning rate=1.0(big)')


plt.show()