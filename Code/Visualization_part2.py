
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
import seaborn as sns
import warnings 
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


plt.rcParams["figure.figsize"] = 8, 5
plt.rcParams["image.cmap"]= "viridis"
df = pd.read_csv("/Users/gympohnpimol/Desktop/MLcourse/Dataset/video_games_sales.csv").dropna()

df["User_Score"] = df["User_Score"].astype("float64")
df["Year_of Release"] = df["Year_of_Release"].astype("int64")
df["User_Count"] = df["User_Count"].astype("int64")
df["Critic_Count"] = df["Critic_Count"].astype("int64")
# print(df.info())

"""Dataframe Plot"""
# df[[ x for x in df.columns if "Sale" in x] + 
#     ["Year_of_Release"]].groupby("Year_of_Release").sum().plot(kind="bar", rot=45)

"""Pairplot"""
# sns.pairplot(df[['Global_Sales', 'Critic_Score', 'Critic_Count', 
#                  'User_Score', 'User_Count']])

"""distplot"""
# sns.distplot(df["User_Score"])

"""jointplot"""
# sns.jointplot(x="Critic_Score", y="User_Score", data=df, kind="scatter")

"""boxplot"""
# top_platforms = df["Platform"].value_counts().sort_values(ascending=False).head(5).index.values
# sns.boxplot(y="Platform", x="Critic_Score", data=df[df["Platform"].isin(top_platforms)], orient= "h")

"""heatmap"""
platform_genre_sales = df.pivot_table( 
                        index="Platform",
                        columns="Genre",
                        values="Global_Sales",
                        aggfunc= sum).fillna(0).applymap(float)
sns.heatmap(platform_genre_sales, annot=True, fmt=".1f", linewidths=.5)
plt.show()