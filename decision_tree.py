from sklearn import tree
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import metrics
from operator import itemgetter
import pandas as pd
import numpy as np
import graphviz
import matplotlib.pyplot as plt

# get data
df = pd.read_csv("shots.csv", index_col="shot_num")
# turn categorical into numerical
df["play_pattern"] = pd.Categorical(df["play_pattern"])
df["play_pattern"] = df["play_pattern"].cat.codes

df = df.sample(frac=1, random_state=42) # shuffle dataset
print(df.shape[0])
features = ["distant_to_goal","play_pattern", "duration", "angle_to_goal", "players_between_goal", "within_1", "within_5", "within_10"]
# original X and Y
X_orig = df[features].to_numpy()
Y_orig = df["goal"].to_numpy()

# oversampling
max_size = df["goal"].value_counts().max()
sampled = [df]
for class_index, group in df.groupby("goal"):
    sampled.append(group.sample(max_size-len(group), replace=True))

df_new = pd.concat(sampled)

print(df_new.shape[0])

# split into X and Y
X_train = df_new[features]
Y_train = df_new["goal"]

"""
train = df.sample(frac=0.7, random_state=56)
test = df.drop(train.index)

X_train = train[features]
Y_train = train["goal"]

X_test = test[features]
Y_test = test["goal"]
"""

# do cross validation to find the best depth
depth = []
for i in range(3, 6):
    dt = tree.DecisionTreeClassifier(max_depth=i)
    # 5-fold cross validation
    scores = cross_val_score(estimator=dt, X=X_train, y=Y_train, cv=5, n_jobs=2)
    depth.append((i, scores.mean()))

print(depth)
m_depth = max(depth, key=itemgetter(1))[0] # get max depth from cross validation
print(m_depth)
dt_shots = tree.DecisionTreeClassifier(max_depth=m_depth)
dt_shots = dt_shots.fit(X_train, Y_train)

dot_data = tree.export_graphviz(dt_shots, out_file=None, class_names=["0", "1"],feature_names=features, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("shots_dt")

y_pred = dt_shots.predict(X_orig)
print("Accuracy:", metrics.accuracy_score(Y_orig, y_pred))