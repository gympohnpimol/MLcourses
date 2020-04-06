
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
df = df[df["year"]==2016]
df["status_type_isvideo"] = df["status_type"].map(lambda x:1 if(x=="video") else 0)
df.drop("status_type", axis=1, inplace=True)
reaction = ['num_reactions', 'num_comments', 'num_shares', 'num_likes', 'num_loves', 'num_wows', 'num_hahas',
            'num_sads', 'num_angrys']

EDA = df.groupby("status_type_isvideo")[reaction].mean()
sns.heatmap(df[reaction].corr(), cmap="coolwarm", annot=True)
# plt.show()
print(df)
