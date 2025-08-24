from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np

'''Liner Regression dataset sklearn'''

data = fetch_california_housing()

X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))

print("R-squared: ", model.score(X_test, y_test))

print("Coefficients: ", model.coef_)

print("Intercept: ", model.intercept_)

print("Predicted Values: ", y_pred)

print("Actual Values: ", y_test)

'''Logistic Regression dataset sklearn'''

data = load_iris()

X, y = data.data, data.target

X = X[y != 2]
y = y[y != 2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ctf = LogisticRegression()
ctf.fit(X_train, y_train)

y_pred = ctf.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))

print("Predicted Values: ", y_pred)

print("Actual Values: ", y_test)

'''Liner Regression dataset from scratch'''

housing = fetch_california_housing()

X, y = housing.data, housing.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = np.c_[np.ones((X.shape[0], 1)), X]

def linear_regression(X, y, lr=0.1, epochs=1000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)

    loss_history = []
    for i in range(epochs):
        y_pred = X.dot(w)
        loss = np.mean((y_pred - y) ** 2)
        loss_history.append(loss)

        gradient = 2 * X.T @ (y_pred - y) / n_samples
        w = w - lr * gradient

        if i % 200 == 0:
            print(f"Epoch {i}, Loss: {loss}")

    return w, loss_history

w, loss_history = linear_regression(X, y, lr=0.1, epochs=2000)
lin_reg = LinearRegression()
lin_reg.fit(X, y)

print("\nLinear Regression from scratch")
print("weights from scratch, GD: ", w[:5], ...)
print("weights from sklearn: ", lin_reg.coef_[:5], "...")

plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss History")
plt.show()