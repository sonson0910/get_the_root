import numpy as np

# Use standalone `keras` import which maps to the same Keras bundled with TensorFlow
# This often helps language servers (Pylance) resolve the import when tensorflow.* paths
# are not detected. Runtime behavior remains the same.
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0

Y_train = np.eye(10)[y_train]
Y_test = np.eye(10)[y_test]

np.random.seed(0)
W1 = np.random.randn(784, 128) * 0.01
b1 = np.zeros((1, 128))
W2 = np.random.randn(128, 10) * 0.01
b2 = np.zeros((1, 10))
lr = 0.1

def relu(X): return np.maximum(0, X)
def relu_deriv(X): return (X > 0).astype(float)
def softmax(X):
    exp = np.exp(X - np.max(X, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

for epoch in range(20):
    z1 = np.dot(X_train, W1) + b1
    A1 = relu(z1)
    Z2 = np.dot(A1, W2) + b2
    Y_hat = softmax(Z2)

    loss = -np.mean(np.sum(Y_train * np.log(Y_hat + 1e-8), axis=1))

    dZ2 = Y_hat - Y_train
    dW2 = np.dot(A1.T, dZ2) / len(X_train)
    db2 = np.sum(dZ2, axis=0, keepdims=True) / len(X_train)
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_deriv(z1)
    dW1 = np.dot(X_train.T, dZ1) / len(X_train)
    db1 = np.sum(dZ1, axis=0, keepdims=True) / len(X_train)

    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    if epoch % 2 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

Z1 = np.dot(X_test, W1) + b1
A1 = relu(Z1)
Z2 = np.dot(A1, W2) + b2
Y_hat_test = softmax(Z2)
predictions = np.argmax(Y_hat_test, axis=1)
accuracy = np.mean(predictions == y_test)
print(f"Test Accuracy: {accuracy:.4f}")
