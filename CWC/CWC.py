
import pandas as pd 
import numpy as np 
import os 

world_cup = pd.read_csv('~/Desktop/Practice/CWC/World_Cup_2019_Dataset.csv')
results = pd.read_csv('~/Desktop/Practice/CWC/results.csv')
world_cup_clean1 = world_cup.apply(lambda x: sum(x.isnull()))
results_clean1 = results.apply(lambda x: sum(x.isnull()))
world_cup_teams = list(world_cup['Team'])
team1 = results[results['Team_1'].isin(world_cup_teams)]
team2 = results[results['Team_2'].isin(world_cup_teams)]
team = pd.concat((team1, team2))
team.drop_duplicates()
team.count()
team2010 = results.drop(['date', 'Margin','Ground'], axis=1)

team2010 = team2010.reset_index(drop=True)
team2010.loc[team2010.Winner == team2010.Team_1, "winning_team"] = 1
team2010.loc[team2010.Winner == team2010.Team_2, "winning_team"] = 2
final = pd.get_dummies(team2010, prefix=["Team_1", "Team_2"], columns=["Team_1", "Team_2"])

#Separate X and y sets
X = final.drop(["Winner"], axis = 1)
y = final.Winner
print(y)