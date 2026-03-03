import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC

# Dataset
X, y = datasets.make_classification(
    n_samples=100, n_features=2,
    n_classes=2, n_redundant=0,
    random_state=42
)

# Train SVM
svm = SVC(kernel='linear')
svm.fit(X, y)

# Plot points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')

# Plot decision boundary
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svm.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, Z, levels=[0], linewidths=2)

plt.title("SVM Decision Boundary")
plt.show()