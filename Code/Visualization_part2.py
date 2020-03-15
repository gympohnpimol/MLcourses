
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
import seaborn as sns
import warnings 
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly
import plotly.graph_objs as go
init_notebook_mode(connected = True)

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
# platform_genre_sales = df.pivot_table( 
#                         index="Platform",
#                         columns="Genre",
#                         values="Global_Sales",
#                         aggfunc= sum).fillna(0).applymap(float)
# sns.heatmap(platform_genre_sales, annot=True, fmt=".1f", linewidths=.5)

platforms_df = df.groupby('Platform')[['Global_Sales']].sum().join(
    df.groupby('Platform')[['Name']].count()
)
platforms_df.columns = ['Global_Sales', 'Number_of_Games']
platforms_df.sort_values('Global_Sales', ascending=False, inplace=True)
# Create a bar for the global sales
trace0 = go.Bar(
    x=platforms_df.index,
    y=platforms_df['Global_Sales'],
    name='Global Sales'
)

# Create a bar for the number of games released
trace1 = go.Bar(
    x=platforms_df.index,
    y=platforms_df['Number_of_Games'],
    name='Number of games released'
)

# Get together the data and style objects
data = [trace0, trace1]
layout = {'title': 'Market share by gaming platform'}

# Create a `Figure` and plot it
fig = go.Figure(data=data, layout=layout)
iplot(fig, show_link=False)

data = []

# Create a box trace for each genre in our dataset
for genre in df.Genre.unique():
    data.append(
        go.Box(y=df[df.Genre == genre].Critic_Score, name=genre)
    )

plt.show()