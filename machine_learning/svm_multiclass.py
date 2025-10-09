from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_linear = SVC(kernel="linear", random_state=42)
svm_linear.fit(X_train, y_train)
y_pred_linear = svm_linear.predict(X_test)

svm_rbf = SVC(kernel="rbf", gamma="scale", random_state=42)
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)

acc_linear = accuracy_score(y_test, y_pred_linear)
acc_rbf = accuracy_score(y_test, y_pred_rbf)

print(f"Linear Kernel Accuracy: {acc_linear:.2f}")
print(f"RBF Kernel Accuracy: {acc_rbf:.2f}")

# --- Prepare 2D versions (first two features) for plotting decision boundaries ---
X_train_2 = X_train[:, :2]
X_test_2 = X_test[:, :2]

# Train separate classifiers on the first two features so their .predict expects 2 features
svm_linear_2 = SVC(kernel="linear", random_state=42)
svm_linear_2.fit(X_train_2, y_train)
y_pred_linear_2 = svm_linear_2.predict(X_test_2)

svm_rbf_2 = SVC(kernel="rbf", gamma="scale", random_state=42)
svm_rbf_2.fit(X_train_2, y_train)
y_pred_rbf_2 = svm_rbf_2.predict(X_test_2)

acc_linear_2 = accuracy_score(y_test, y_pred_linear_2)
acc_rbf_2 = accuracy_score(y_test, y_pred_rbf_2)

print(f"Linear Kernel Accuracy (first 2 features): {acc_linear_2:.2f}")
print(f"RBF Kernel Accuracy (first 2 features): {acc_rbf_2:.2f}")


def plot_decision_boundary(model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
    plt.title(title)
    plt.xlabel("Sepal length (standardized)")
    plt.ylabel("Sepal width (standardized)")
    plt.show()


plot_decision_boundary(
    svm_linear_2,
    X_train_2,
    y_train,
    f"SVM with Linear Kernel (2D Acc: {acc_linear_2:.2f})",
)
plot_decision_boundary(
    svm_rbf_2, X_train_2, y_train, f"SVM with RBF Kernel (2D Acc: {acc_rbf_2:.2f})"
)
