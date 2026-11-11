import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X = iris.data
y = iris.target
print("Dataset loaded successfully!")
print("Total samples:", X.shape[0])
print("Total features:", X.shape[1])
print("Flower classes:", iris.target_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Data split completed!")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
print("KNN model trained successfully!")

y_pred = model.predict(X_test)
print("Predictions completed!")
print("First 5 predictions:", y_pred[:5])

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
for row in cm:
    print(row)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Step 7: Visualize the confusion matrix

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix - Iris Flower Classification")
plt.show()





