from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train k-NN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Print correct and incorrect predictions
print("\nPrediction Results:\n")
for i in range(len(y_test)):
    print(f"Predicted: {y_pred[i]}, Actual: {y_test[i]}, "
          f"{'Correct' if y_pred[i] == y_test[i] else 'Incorrect'}")