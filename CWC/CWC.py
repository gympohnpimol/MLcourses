
import pandas as pd 
import numpy as np 
import os 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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
y = final["Winner"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)
model = LogisticRegression()
model.fit(X_train, y_train)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
# print(test_score)

ranking = pd.read_csv('~/Desktop/Practice/CWC/icc_rankings.csv')
fixtures = pd.read_csv('~/Desktop/Practice/CWC/fixtures.csv')
pred_set = []

fixtures.insert(1, "first_position", fixtures["Team_1"].map(ranking.set_index('Team')['Position']))
fixtures.insert(2, "second_position", fixtures["Team_2"].map(ranking.set_index('Team')['Position']))

fixtures = fixtures.iloc[:45,:]
fixtures = fixtures.tail()
for index, row in fixtures.iterrows():
    if row["first_position"] < row["second_position"]:
        pred_set.append({"Team_1": row["Team_1"], "Team_2": row["Team_2"], "winning_team": None})
    else:
        pred_set.append({"Team_1": row["Team_1"], "Team_2": row["Team_2"], "winning_team": None})

pred_set = pd.DataFrame(pred_set)
backup_pred_set = pred_set
print(pred_set.head())
# Get dummy variables and drop winning_team column
pred_set = pd.get_dummies(pred_set, prefix = ["Team_1", "Team_2"],columns = ["Team_1", "Team_2"])
missing_cols = set(final.columns) -  set(pred_set.columns)
for c in missing_cols:
    pred_set[c] = 0
pred_set = pred_set[final.columns]
# pred_set = pred_set.drop(["Winner"], axis=1)
pred_set = pred_set.drop(["winning_team"], axis =1)


predictions = model.predict(pred_set)
for i in range(fixtures.shape[0]):
    print(backup_pred_set.iloc[i, 1] + "  and  "+ backup_pred_set.iloc[i, 0])
    if predictions[i] == 1:
        print("Winner: " + backup_pred_set.iloc[i, 1])
    else:
        print("Winner: " + backup_pred_set.iloc[i, 0])
    print("")

semi = [("New Zealand", "India"),
        ("England", "Australia")]
final = [("India", "England")]
def semifinal(match, ranking, final, model):
    positions = []

    for m in match:
        positions.append(ranking.loc[ranking["Team"] == match[0], "Position"].iloc[0])
        positions.append(ranking.loc[ranking["Team"] == match[1], "Position"].iloc[0])
    
    pred_set = []
    i = 0
    j = 0
    while i < len(positions):
        dict1 = {}

        if positions[i] < positions[i+1]:
            dict1.update({"Team1": match[j][0], "Team2": match[j][1]})
        else:
            dict1.update({"Team1": match[j][1], "Team2": match[j][0]})
    
    pred_set = pd.DataFrame(pred_set)
    backup_pred_set = pred_set

    pred_set = pd.get_dummies(pred_set, prefix=["Team1", "Team2"], columns=["Team1", "Team2"])

    missing_cols2 = set(final.columns) - set(pred_set.columns)
    for c in missing_cols2:
        pred_set[c] = 0
    pred_set = pred_set[final.columns]
    pred_set = pred_set.drop(["Winner"], axis=1)

    predictions = model.predict(pred_set)
    for i in range(len(pred_set)):
        print(backup_pred_set.iloc[i,1] + "  and  " + backup_pred_set.iloc[i,0])
        if predictions[i] == 1:
            print("Winner: " + backup_pred_set.iloc[i,1])
        else:
            print("Winner: " + backup_pred_set.iloc[i,0])
        print("")

    return semifinal(semi, ranking, final, model)
    return semifinal(final, ranking, final, model)
# final = [("India", "England")]