import os
import numpy as np
import pandas as pd
import pydotplus
import graphviz
import seaborn as sns; sns.set()
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold


# plt.figure(figsize=(6, 4))
# xx = np.linspace(0,1,50)
# plt.plot(xx, [2 * x * (1-x) for x in xx], label='gini')
# plt.plot(xx, [4 * x * (1-x) for x in xx], label='2*gini')
# plt.plot(xx, [(-x) * np.log2(x) - (1-x) * np.log2(1 - x)  for x in xx], label='entropy')
# plt.plot(xx, [1 - max(x, 1-x) for x in xx], label='missclass')
# plt.plot(xx, [2 - 2 * max(x, 1-x) for x in xx], label='2*missclass')
# plt.xlabel('p+')
# plt.ylabel('criterion')
# plt.title('Criteria of quality as a function of p+ (binary classification)')

np.random.seed(20)
train_data = np.random.normal(size=(100, 2))
train_labels = np.zeros(100)

train_data = np.r_[train_data, np.random.normal(size=(100, 2), loc=2)]
train_labels = np.r_[train_labels, np.ones(100)]

# plt.figure(figsize=(10, 8))
# plt.scatter(train_data[:, 0], train_data[:, 1], c = train_labels, s= 100, cmap="autumn", edgecolors="black", linewidths=1.0)
# plt.plot(range(-2, 5), range(4, -3,-1))

def get_grid(data):
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    return np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
clf_tree = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=17)
clf_tree.fit(train_data, train_labels)
# tree.plot_tree(clf_tree.fit(train_data, train_labels))
x, y = get_grid(train_data)
prediction = clf_tree.predict(np.c_[x.ravel(), y.ravel()]).reshape(x.shape)
plt.pcolormesh(x, y, prediction, cmap="autumn") 
plt.scatter(train_data[:, 0], train_data[:, 1], c = train_labels, s= 100, cmap="autumn", edgecolors="black", linewidths=1.0)

# def treeplot(tree, feature_names, png_file):
#     tree_str = export_graphviz(tree, feature_names=feature_names, filled=True, out_file=None)
#     graph = pydotplus.graph_from_dot_data(tree_str)
#     colors = ('turquoise', 'orange')
#     graph.write_png(png_file)
# treeplot(tree=clf_tree, feature_names=["x1", "x2"], png_file="Decision Tree.png")
# print(train_labels)

n_train = 150        
n_test = 1000       
noise = 0.1

def f(x):
    x = x.ravel()
    return np.exp(-x ** 2) + 1.5 * np.exp(-(x - 2) ** 2)

def generate(n_samples, noise):
    X = np.random.rand(n_samples) * 10 - 5
    X = np.sort(X).ravel()
    y = np.exp(-X ** 2) + 1.5 * np.exp(-(X - 2) ** 2) + \
    np.random.normal(0.0, noise, n_samples)
    X = X.reshape((n_samples, 1))
    return X, y

X_train, y_train = generate(n_samples=n_train, noise=noise)
X_test, y_test = generate(n_samples=n_test, noise=noise)

from sklearn.tree import DecisionTreeRegressor

reg_tree = DecisionTreeRegressor(max_depth=5, random_state=17)

reg_tree.fit(X_train, y_train)
reg_tree_pred = reg_tree.predict(X_test)

plt.figure(figsize=(10, 6))
plt.plot(X_test, f(X_test), "b")
plt.scatter(X_train, y_train, c="b", s=20)
plt.plot(X_test, reg_tree_pred, "g", lw=2)
plt.xlim([-5, 5])
plt.title("Decision tree regressor, MSE = %.2f" % (np.sum((y_test - reg_tree_pred) ** 2) / n_test))

plt.show()
