import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3  * X + np.random.randn(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

m, b = 0.0, 0.0
lr = 0.1
n_iter = 1000
N = len(X)

for i in range(n_iter):
    y_pred = m * X + b
    error = y_pred - y
    dm = (2/N) * np.sum(error * X)
    db = (2/N) * np.sum(error)
    m -= lr * dm
    b -= lr * db

print("Code Liner Regression youself: ")
print("m (slope): ", m)
print("b (intercept): ", b)

lin_reg = LinearRegression()
lin_reg.fit(X, y)

print("Code Liner Regression sklearn: ")
print("m (slope): ", lin_reg.coef_[0][0])
print("b (intercept): ", lin_reg.intercept_[0])

plt.scatter(X, y, color='blue', alpha=0.5, label='Data')
plt.plot(X, m * X + b, color='red', label='Linear Regression')
plt.plot(X, lin_reg.predict(X), color='green', linestyle='--', label='Linear Regression sklearn')
plt.legend()
plt.show()