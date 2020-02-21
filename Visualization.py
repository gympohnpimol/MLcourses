
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
features = ["total day minutes"]
# print(df["total day minutes"].unique())
plt.hist(df[features].T);
plt.show()

# , "total intl calls"