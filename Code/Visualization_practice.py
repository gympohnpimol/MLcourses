
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
import seaborn as sns
import warnings 
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

df = df = pd.read_csv("/Users/gympohnpimol/Desktop/MLcourse/Dataset/mlbootcamp5_train.csv")
"""Basic Observation"""
df["gender"].value_counts() #f-45530,m-24470
df[df["alco"]==1]["gender"].value_counts() #f-1161, m-2603
x = df[df["smoke"]==1]["gender"].value_counts()

female = 813 #20
male = 5356
all = female + male
female_pct = round(female/45330*100,0)
male_pct = round(male/24470*100,0)


# print(male_pct)
