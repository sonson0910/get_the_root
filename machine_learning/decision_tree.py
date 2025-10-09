import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


titanic = sns.load_dataset("titanic")

features = ["pclass", "sex", "age", "fare", "alone"]
data = titanic[features + ["survived"]].dropna()

data['sex'] = data['sex'].map({'male': 0, 'female': 1})

X = data[features]
y = data["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_train, y_train)

forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_train, y_train)

tree_pred = tree.predict(X_test)
forest_pred = forest.predict(X_test)


print(f"Decision Tree Accuracy: {accuracy_score(y_test, tree_pred):.2f}")
print(f"Random Forest Accuracy: {accuracy_score(y_test, forest_pred):.2f}")

plt.figure(figsize=(16, 8))
plot_tree(tree, feature_names=features, class_names=["Not Survived", "Survived"], filled=True)
plt.title("Decision Tree")
plt.show()


importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title("Feature Importances from Random Forest")
sns.barplot(x = importances[indices], y = np.array(features)[indices], palette="viridis")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()
