import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

X = X[y != 2]
y = y[y != 2]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


def plot_decision_boundary(X, y, clf, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cmap_light = ListedColormap(["#FFAAAA", "#AAAAFF"])
    cmap_bold = ["red", "blue"]

    plt.figure(figsize=(6, 4))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
    plt.scatter(
        X[:, 0], X[:, 1], c=y, edgecolor="k", s=50, cmap=ListedColormap(cmap_bold)
    )
    plt.xlabel("sepal length (cm)")
    plt.ylabel("sepal width (cm)")
    plt.title(title)
    plt.show()


clf_linear = SVC(kernel="linear", C=1.0)
clf_linear.fit(X_train, y_train)
acc_linear = clf_linear.score(X_test, y_test)
print(f"Linear Kernel Accuracy: {acc_linear:.2f}")
plot_decision_boundary(
    X, y, clf_linear, f"SVM with Linear Kernel (Accuracy: {acc_linear:.2f})"
)

clf_rbf = SVC(kernel="rbf", gamma=0.7, C=1.0)
clf_rbf.fit(X_train, y_train)
acc_rbf = clf_rbf.score(X_test, y_test)
print(f"RBF Kernel Accuracy: {acc_rbf:.2f}")
plot_decision_boundary(X, y, clf_rbf, f"SVM with RBF Kernel (Accuracy: {acc_rbf:.2f})")
