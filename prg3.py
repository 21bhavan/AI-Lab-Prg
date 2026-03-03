import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

# Generate a sample dataset (100 past cases)
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # Normalize for better distance calculation

# Case-Based Reasoning (CBR) Function
def retrieve_similar_cases(X_cases, y_cases, new_case, n_similar=3):
 """Finds the most similar past cases using Euclidean distance."""
 nbrs = NearestNeighbors(n_neighbors=n_similar, metric='euclidean').fit(X_cases)
 distances, indices = nbrs.kneighbors(new_case) # FIXED: Removed extra brackets
 return indices[0], distances[0]

# New case to predict
new_case = np.array([[0.5]]) # Example new input
new_case_scaled = scaler.transform(new_case) # Keeps it 2D

# Retrieve most similar cases
similar_indices, similar_distances = retrieve_similar_cases(X_scaled, y, new_case_scaled)
similar_cases = X[similar_indices] # Original scale for visualization
predicted_value = np.mean(y[similar_indices]) # Average output of similar cases

# Output results
print(f"New Case Input: {new_case.flatten()[0]:.2f}")
print(f"Most Similar Cases (X values): {similar_cases.flatten()}")
print(f"Corresponding Outputs (y values): {y[similar_indices]}")
print(f"Predicted Output: {predicted_value:.2f}")

# Plot cases and prediction
plt.scatter(X, y, label="Past Cases", color="blue", alpha=0.6)
plt.scatter(similar_cases, y[similar_indices], label="Similar Cases", color="red", marker="s", s=100)
plt.scatter(new_case, predicted_value, label="Predicted Case", color="green", marker="*", s=150)
plt.xlabel("Feature Value")
plt.ylabel("Output Value")
plt.title("Case-Based Reasoning Prediction")
plt.legend()
plt.show()
