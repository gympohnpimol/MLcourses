
import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings
pd.set_option("display.max.columns", 100)
warnings.filterwarnings("ignore")

df = pd.read_csv("adult.data.csv")
#Q1
Q1 = df[df["sex"]=="Female"]

#Q2
Q2 = df[df["sex"]=="Female"]['age'].mean()
# print(Q2)

#Q3
Q3 = df[df["native-country"]=="Germany"]
print(Q3.count())

Q4 = df[df["salary"]==">50K"].describe()
# print(Q4)

Q6 = df[df["salary"]==">50K"]["education"]
# print(Q6)

# for (race, sex), sub_df in df.groupby(['race', 'sex']):
#     print("Race: {0}, sex: {1}".format(race, sex))
#     print(sub_df['age'].describe())

Q7 = df.loc[(df['sex'] == 'Male') &
     (df['marital-status'].isin(['Never-married', 
                                   'Separated', 
                                   'Divorced',
                                   'Widowed'])), 'salary'].value_counts()
# print(Q7)

Q9 = df["hours-per-week"].max()
num_workaholics = df[df['hours-per-week'] == Q9].shape[0]
# print(num_workaholics)

# for (country, salary), sub_df in df.groupby(['native-country', 'salary']):
#     print(country, salary, round(sub_df['hours-per-week'].mean(), 2))