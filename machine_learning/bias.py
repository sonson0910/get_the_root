import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

train_scores = []
test_scores = []
depths = range(1, 15)

for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    clf.fit(x_train, y_train)

    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)

    train_scores.append(clf.score(x_train, y_train))
    test_scores.append(clf.score(x_test, y_test))

plt.figure(figsize=(8, 5))
plt.plot(depths, train_scores, label="Training Accuracy", marker='o')
plt.plot(depths, test_scores, label="Testing Accuracy", marker='s')
plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree Depth vs. Accuracy")
plt.legend()
plt.grid()
plt.show()
