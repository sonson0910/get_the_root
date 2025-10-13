import numpy as np
import matplotlib.pyplot as plt

X = np.random.rand(100, 2)
y = np.where(X[:, 0] + X[:, 1] > 0, 1, 0)

w = np.zeros(2)
b = 0
lr = 0.1

def predict(X):
    z = np.dot(X, w) + b
    return np.where(z > 0, 1, 0)

for epoch in range(20):
    for i in range(len(X)):
        y_pred = predict(X[i])
        error = y[i] - y_pred
        w += lr * error * X[i]
        b += lr * error

x1 = np.linspace(-3, 3, 100)
x2 = -(w[0] * x1 + b)
plt.scatter(X[:, 0], X[:, 1], c = y, cmap = 'bwr')
plt.plot(x1, x2, 'k--')
plt.show()
