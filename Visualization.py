
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings 

warnings.filterwarnings("ignore")
sns.set()


df = pd.read_csv("ml_telecom_churn.csv")
# print(df.head())
plt.style.use('ggplot')
features = ["total intl calls"]

###histogram plots###
# df[features].hist(figsize=(10,5)) 
# print(df["total day minutes"].unique())
# plt.hist(df[features].T); 

###density plots###
# df[features].plot(kind="density", subplots=True, layout=(1,2), sharex=False, figsize=(10,4));

###Seaborn's distplot kernel density estimate (KDE)###
sns.distplot(df["total intl calls"])


plt.show()