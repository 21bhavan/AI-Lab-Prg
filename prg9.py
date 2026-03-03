import numpy as np
import matplotlib.pyplot as plt

def kernel_weight(query_x, X_bias, tau):
    """Compute weights using Gaussian kernel."""
    diff = X_bias[:, 1:] - query_x.reshape(1, -1)  # exclude bias for distance
    distances = np.sum(diff ** 2, axis=1)
    weights = np.exp(-distances / (2 * tau ** 2))
    return np.diag(weights)

def locally_weighted_regression(X, y, tau, query_points):
    """Perform Locally Weighted Regression."""
    X_bias = np.c_[np.ones(len(X)), X]  # Add bias term
    y_pred = []

    for query_x in query_points:
        W = kernel_weight(query_x, X_bias, tau)

        # Compute theta using weighted normal equation
        theta = np.linalg.pinv(X_bias.T @ W @ X_bias) @ (X_bias.T @ W @ y)

        # Add bias to query point
        query_x_bias = np.hstack(([1], query_x))

        y_pred.append(query_x_bias @ theta)

    return np.array(y_pred)

# Generate sample data
np.random.seed(42)
X = np.linspace(-3, 3, 30).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(scale=0.2, size=X.shape[0])

# Test points
X_test = np.linspace(-3, 3, 100).reshape(-1, 1)

# Bandwidth parameter
tau = 0.5

# Predict
y_pred = locally_weighted_regression(X, y, tau, X_test)

# Plot
plt.scatter(X, y, label="Data Points")
plt.plot(X_test, y_pred, label=f"LWR Curve (tau={tau})")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Locally Weighted Regression (LWR)")
plt.legend()
plt.show()