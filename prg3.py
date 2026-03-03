import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier, plot_tree 
from sklearn.model_selection import train_test_split 
from sklearn import datasets 
# Load a sample dataset 
iris = datasets.load_iris() 
X = iris.data  #Use all features for better accuracy 
y = iris.target 
# Split the dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=75) 
# Train the ID3 Decision Tree model with additional parameters to avoid over/underfitting 
id3_tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=75) 
id3_tree.fit(X_train, y_train) 
# Visualize the Decision Tree 
plt.figure(figsize=(12, 8)) 
plot_tree(id3_tree,feature_names=iris.feature_names,class_names=iris.target_names,filled=True) 
plt.title("ID3 Decision Tree Visualization") 
plt.show() 
# Model Accuracy 
accuracy = id3_tree.score(X_test, y_test) 
print(f"Model Accuracy: {accuracy * 100:.2f}%")