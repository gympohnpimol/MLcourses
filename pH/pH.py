
import pandas as pd 
import numpy as np 
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from sklearn import metrics
df = pd.read_csv("~/Desktop/Practice/pH/ph_data.csv")
df_clean = df.apply(lambda x: sum(x.isnull()))

plt.figure(figsize=(5,5))
# plt.subplot(2, 2, 1)
# sns.violinplot(x="label", y="blue", data=df)
# plt.subplot(2, 2, 2)
# sns.violinplot(x="label", y="green", data=df)
# plt.subplot(2, 2, 3)
# sns.violinplot(x="label", y="red", data=df)

# df = df.drop(["label"], axis=1)
# sns.heatmap(df.corr(), annot=True, cmap="cubehelix_r")
# plt.show()

def ph(row):
    if row["label"] < 7:
        return "acidic"
    elif row["label"] > 7:
        return "alkaline"
    else:
        return "natural"
df["pH"] = df.apply(lambda row: ph(row), axis=1)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection = "3d")
x = df["blue"]
y = df["green"]
z = df["red"]
ax.scatter(x , y, z, label=df, cmap="viridis")
ax.set_xlabel("blue")
ax.set_ylabel("green")
ax.set_zlabel("red")

train, test = train_test_split(df, test_size=0.2)
x_train = train[["blue", "green", "red"]]
y_train = train.label

x_test = test[["blue", "green", "red"]]
y_test = test.label

feature = df[["blue", "green", "red"]]
label = df.label

classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(feature, label)
# print(classifier.predict([[182,184,10]]))
# tree.plot_tree(classifier.fit(df, label))
# plt.show()

mdl = KNeighborsClassifier()
mdl.fit(x_train, y_train)
prediction = mdl.predict(x_test)
print("The accuracy of KNN: ", metrics.accuracy_score(y_test, prediction))
# print(metrics.confusion_matrix(y_test,prediction))


