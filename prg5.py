import numpy as np
from tensorflow import keras
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -----------------------------
# 1. Load Iris Dataset
# -----------------------------
iris = load_iris()
X, y = iris.data, iris.target

# -----------------------------
# 2. One-Hot Encoding (New sklearn version)
# -----------------------------
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.reshape(-1, 1))

# -----------------------------
# 3. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Standardization
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# 5. Build ANN Model
# -----------------------------
model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

# -----------------------------
# 6. Compile Model
# -----------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# 7. Train Model
# -----------------------------
model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=10,
    validation_data=(X_test, y_test),
    verbose=1
)

# -----------------------------
# 8. Evaluate Model
# -----------------------------
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("\nModel Accuracy on Test Data: {:.2f}%".format(test_accuracy * 100))

# -----------------------------
# 9. Predictions
# -----------------------------
y_pred = model.predict(X_test)

y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

print("\nSample Predictions (True vs Predicted Labels):")
print(np.vstack((y_test_classes[:10], y_pred_classes[:10])).T)
