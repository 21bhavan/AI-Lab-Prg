import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier, plot_tree 
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
# Load dataset (Iris dataset for classification) 
iris = load_iris() 
X, y = iris.data, iris.target 
feature_names = iris.feature_names 
class_names = iris.target_names 
# Split into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
# Train Decision Tree Classifier using ID3 (entropy-based) 
clf = DecisionTreeClassifier(criterion='entropy', random_state=42) 
clf.fit(X_train, y_train) 
# Plot the decision tree 
plt.figure(figsize=(12, 6)) 
plot_tree(clf, feature_names=feature_names, class_names=class_names, filled=True) 
plt.title("Decision Tree using ID3 Algorithm") 
plt.show() 
# Print model accuracy 
accuracy = clf.score(X_test, y_test) 
print(f"\nModel Accuracy: {accuracy * 100:.2f}% ") 
# Print feature importance 
feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': clf.feature_importances_}) 
print("\nFeature Importance:\n", feature_importance.sort_values(by='Importance', ascending=False))