
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
import seaborn as sns
import warnings 
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sns.set()

figure_formats = {"retina"}
df = pd.read_csv("/Users/gympohnpimol/Desktop/MLcourse/Dataset/ml_telecom_churn.csv")
print(df.head())
plt.style.use('ggplot')
features = ["total intl calls"]

"""Histogram plots"""
df[features].hist(figsize=(10,5)) 
print(df["total day minutes"].unique())
plt.hist(df[features].T); 

"""density plots"""
df[features].plot(kind="density", subplots=True, layout=(1,2), sharex=False, figsize=(10,4));

"""Seaborn's distplot kernel density estimate (KDE)"""
sns.distplot(df["total intl calls"])

"""Seaborn's boxplot"""
sns.boxplot(x = "total intl calls", data=df);

"""Violin plots"""
_, axes = plt.subplots(1, 2, sharey = True, figsize = (6, 4))
sns.boxplot(data = df["total intl calls"], ax=axes[0])
sns.violinplot(data=df["total intl calls"], ax=axes[1])

"""Dataframe description"""
print(df[features].describe())

"""Frequency table"""
print(df["churn"].value_counts())

"""Bar plot"""
_, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
sns.countplot(x="churn", data=df, ax=axes[0])
sns.countplot(x="customer service calls", data=df, ax=axes[1])

"""Correlation matrix"""
numerical = list(set(df.columns) - set(["state", "international plan", "voice mail plan",
            "area code", "churn", "customer service calls"])) #drop non-numerical variables

corr_matrix = df[numerical].corr()
sns.heatmap(corr_matrix)

"""Scatter plot"""
plt.scatter(df["total day minutes"], df["total night minutes"]) #x, y axis

"""Scatter plot with histogram using jointplot"""
sns.jointplot(x="total day minutes", y="total night minutes", data=df, kind="scatter")

"""Bivariate version of the Kernel Density Plot"""
sns.jointplot("total day minutes", "total night minutes", data=df, kind="kde", color ="r")

"""Scatterplot matrix"""
figure_formats = {"retina"}
sns.pairplot(df[list(set(df.columns))])

"""Quantitative plots"""
sns.lmplot("total day minutes", "total night minutes", data=df, hue = "churn", fit_reg=False)

"""Box plots to visualize the distribution statistics"""
features.append("customer service calls")
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(10,7))
for idx, feat in enumerate(features):
    ax = axes[int(idx / 4), idx % 4]
    sns.boxplot(x="churn", y=feat, data=df, ax=ax)
    ax.set_xlabel(" ")
    ax.set_ylabel(feat)
fig.tight_layout()

"""Box and violin plots grouped by the target variable"""
_, axes = plt.subplots(1, 2, sharey=True, figsize=(10,5))
sns.boxplot(x="churn", y="total day minutes", data=df, ax=axes[0])
sns.violinplot(x="churn", y="total day minutes", data=df, ax=axes[1])

"""Catplot() for observation on average"""
sns.catplot(x = "churn", y= "total day minutes", col = "customer service calls",
            data= df[df["customer service calls"]< 8], kind="box", col_wrap=4, height = 3,aspect= .8)

"""Catagory comparison"""
sns.countplot(x="customer service calls", hue= "churn", data=df)
_, axes =plt.subplots(1, 2, sharey= True, figsize=(10,5))
sns.countplot(x="international plan", hue="churn", data=df, ax=axes[0])
sns.countplot(x="voice mail plan", hue="churn", data=df, ax=axes[1])

"""Contingency table (or) cross tabulation"""
print(pd.crosstab(df["state"], df["churn"]).T)
print(df.groupby(['state'])['churn'].agg([np.mean]).sort_values(by='mean', ascending=False).T)

"""t-SNE"""
X = df.drop(['churn', 'state'], axis=1)
X['international plan'] = X['international plan'].map({'Yes': 1, 'No': 0})
X['voice mail plan'] = X['voice mail plan'].map({'Yes': 1, 'No': 0})
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
tsne = TSNE(random_state=15)
tsne_repr = tsne.fit_transform(X_scaled)
plt.scatter(tsne_repr[:, 0], tsne_repr[:, 1], alpha=.5);
plt.scatter(tsne_repr[:, 0], tsne_repr[:, 1], c=df["churn"].map({False: "blue", True: "orange"}), alpha=.5)

_, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 5))

for i, name in enumerate(['International plan', 'Voice mail plan']):
    axes[i].scatter(tsne_repr[:, 0], tsne_repr[:, 1], 
                    c=df[name].map({'Yes': 'orange', 'No': 'blue'}), alpha=.5);
    axes[i].set_title(name)

plt.show()
