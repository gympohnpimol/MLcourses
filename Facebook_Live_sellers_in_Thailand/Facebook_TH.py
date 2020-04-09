
import pandas as pd
import numpy as np 
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("~/Desktop/Practice/Facebook_Live_sellers_in_Thailand/Live.csv")
df_clean1 = df.apply(lambda x: sum(x.isnull()))
df = df.drop(["Column1","Column2", "Column3", "Column4", "status_id"], axis=1)
df["status_published"] = pd.to_datetime(df["status_published"])
df["year"] = df["status_published"].dt.year
df["month"] = df["status_published"].dt.month
df["day"] = df["status_published"].dt.dayofweek
df["hour"] = df["status_published"].dt.hour

# df.hist(edgecolor="black", linewidth = 1)
# hist = plt.gcf()
# hist.set_size_inches(10,7)

# plt.figure(figsize=(10,10))
# sns.heatmap(df.corr(), annot=True, cmap="cubehelix_r")
# plt.subplot(3, 3, 1)
# sns.violinplot(data= df,x="status_type", y="num_reactions")
# plt.subplot(3, 3, 2)
# sns.violinplot(data= df,x="status_type", y="num_comments")
# plt.subplot(3, 3, 3)
# sns.violinplot(data= df,x="status_type", y="num_shares")
# plt.subplot(3, 3, 4)
# sns.violinplot(data= df,x="status_type", y="num_likes")
# plt.subplot(3, 3, 5)
# sns.violinplot(data =df,x="status_type", y="num_loves")
# plt.subplot(3, 3, 6)
# sns.violinplot(data= df,x="status_type", y="num_wows")
# plt.subplot(3, 3, 7)
# sns.violinplot(data= df,x="status_type", y="num_hahas")
# plt.subplot(3, 3, 8)
# sns.violinplot(data= df,x="status_type", y="num_sads")
# plt.subplot(3, 3, 9)
# sns.violinplot(data= df,x="status_type", y="num_angrys")

# plt.show()

df["status_type_isvideo"] = df["status_type"].map(lambda x:1 if(x=="video") else 0)
df.drop("status_type", axis=1, inplace=True)
reaction = ['num_reactions', 'num_comments', 'num_shares', 'num_likes', 'num_loves', 'num_wows', 'num_hahas',
            'num_sads', 'num_angrys']
y2016 = df[df["year"]==2016]
EDA = y2016.groupby("status_type_isvideo")[reaction].mean()
# sns.heatmap(y2016[reaction].corr(), cmap="coolwarm", annot=True)
EDA2 = df.groupby("status_type_isvideo")[reaction].mean()

std_scalar = StandardScaler()
# y2016_2 = y2016[reaction]
# y2016_2 = std_scalar.fit_transform(y2016_2)
all_year = df[reaction]
all_year = std_scalar.fit_transform(all_year)
# print(y2016_2)

from sklearn.decomposition import PCA
pca = PCA(svd_solver="randomized", random_state=123)
pca.fit(all_year)

# fig = plt.figure(figsize=(10,5))
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel("components")
# plt.ylabel("cumulative explained variance")
# plt.show()

PC_A = pd.DataFrame({"PC1":pca.components_[0], "PC2":pca.components_[1], "Feature": reaction})

# print(PC_A)
# plt.scatter(PC_A.PC1, PC_A.PC2)
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# for i, txt in enumerate(PC_A.Feature):
#     plt.annotate(txt, (PC_A.PC1[i], PC_A.PC2[i]))
# plt.tight_layout()
# plt.show()

df.groupby(["year", "hour"]).sum()[reaction]
# df.groupby(["year", "hour"]).sum()[reaction].plot(figsize=(12,5))
# df.groupby("year").sum()[reaction]
# df.groupby("year").sum()[reaction].plot(figsize=(12,5))
# plt.show()
# print(p)
df.groupby(["year", "status_type_isvideo"]).sum()[reaction]

# plt.figure(1)
# df[df["status_type_isvideo"]==0].groupby("year").sum()[reaction].plot(figsize=(10,5), title = "Photo Content", linestyle = "--")
# plt.figure(2)
# df[df["status_type_isvideo"]==1].groupby("year").sum()[reaction].plot(figsize=(10,5), title = "Video Content", linestyle = "--")
# plt.figure(3)
# df[df["status_type_isvideo"]==0].groupby("month").sum()[reaction].plot(figsize=(10,5), title = "Photo Content", linestyle = "--")
# plt.figure(4)
# df[df["status_type_isvideo"]==1].groupby("month").sum()[reaction].plot(figsize=(10,5), title = "Video Content", linestyle = "--")
# plt.figure(5)
# df[df["status_type_isvideo"]==0].groupby("day").sum()[reaction].plot(figsize=(10,5), title = "Photo Content", linestyle = "--")
# plt.figure(6)
# df[df["status_type_isvideo"]==1].groupby("day").sum()[reaction].plot(figsize=(10,5), title = "Video Content", linestyle = "--")
# plt.figure(7)
# df[df["status_type_isvideo"]==0].groupby("hour").sum()[reaction].plot(figsize=(10,5), title = "Photo Content", linestyle = "--")
# plt.figure(8)
# df[df["status_type_isvideo"]==1].groupby("hour").sum()[reaction].plot(figsize=(10,5), title = "Video Content", linestyle = "--")

plt.show()

