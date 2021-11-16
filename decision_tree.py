from sklearn import tree
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from operator import itemgetter
import pandas as pd
import numpy as np
import graphviz
import matplotlib.pyplot as plt

# get data for LOCF and mean preprocessed
df = pd.read_csv("shots_LOCF.csv", index_col="shot_num")
# turn categorical into numerical
df["play_pattern"] = pd.Categorical(df["play_pattern"])
df["play_pattern"] = df["play_pattern"].cat.codes
df["shot_taker_type"] = pd.Categorical(df["shot_taker_type"])
df["shot_taker_type"] = df["shot_taker_type"].cat.codes
df["shot_technique"] = pd.Categorical(df["shot_technique"])
df["shot_technique"] = df["shot_technique"].cat.codes
df["shot_body_part"] = pd.Categorical(df["shot_body_part"])
df["shot_body_part"] = df["shot_body_part"].cat.codes

df_mean = pd.read_csv("shots_mean.csv", index_col="shot_num")
df_mean["play_pattern"] = pd.Categorical(df_mean["play_pattern"])
df_mean["play_pattern"] = df_mean["play_pattern"].cat.codes
df_mean["shot_taker_type"] = pd.Categorical(df_mean["shot_taker_type"])
df_mean["shot_taker_type"] = df_mean["shot_taker_type"].cat.codes
df_mean["shot_technique"] = pd.Categorical(df_mean["shot_technique"])
df_mean["shot_technique"] = df_mean["shot_technique"].cat.codes
df_mean["shot_body_part"] = pd.Categorical(df_mean["shot_body_part"])
df_mean["shot_body_part"] = df_mean["shot_body_part"].cat.codes

df = df.sample(frac=1, random_state=42) # shuffle dataset
df_mean = df_mean.sample(frac=1, random_state=42) # shuffle dataset
# print(df.shape[0])
features = df.columns.tolist()
features.remove("goal")
# original X and Y
X_orig = df[features].to_numpy()
Y_orig = df["goal"].to_numpy()

# for mean
Xm_orig = df_mean[features].to_numpy()
Ym_orig = df_mean["goal"].to_numpy()

# oversampling for LOCF
max_size = df["goal"].value_counts().max()
sampled = [df]
for class_index, group in df.groupby("goal"):
    sampled.append(group.sample(max_size-len(group), replace=True))

# LOCF
df_new = pd.concat(sampled)

m_max_size = df_mean["goal"].value_counts().max()
sampled_m = [df_mean]
for class_index, group in df_mean.groupby("goal"):
    sampled_m.append(group.sample(max_size-len(group), replace=True))

df_mnew = pd.concat(sampled_m)
# print(df_new.shape[0])
# print(df_mnew.shape[0])

# split into X and Y
X_train = df_new[features]
Y_train = df_new["goal"]

Xm_train = df_mnew[features]
Ym_train = df_mnew["goal"]

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
depth_m = []
for i in range(3, 6):
    dt = tree.DecisionTreeClassifier(max_depth=i)
    dt_m = tree.DecisionTreeClassifier(max_depth=i)
    # 5-fold cross validation
    scores = cross_val_score(estimator=dt, X=X_train, y=Y_train, cv=5, n_jobs=2)
    scores_m = cross_val_score(estimator=dt_m, X=Xm_train, y=Ym_train, cv=5, n_jobs=2)
    depth.append((i, scores.mean()))
    depth_m.append((i, scores_m.mean()))

# print(depth)
# print(depth_m)
m_depth = max(depth, key=itemgetter(1))[0] # get max depth from cross validation
mm_depth = max(depth_m, key=itemgetter(1))[0]
# print(m_depth)
# print(mm_depth)
dt_shots = tree.DecisionTreeClassifier(max_depth=m_depth)
dt_shots = dt_shots.fit(X_train, Y_train)

dtm_shots = tree.DecisionTreeClassifier(max_depth=mm_depth)
dtm_shots = dtm_shots.fit(Xm_train, Ym_train)

dot_data = tree.export_graphviz(dt_shots, out_file=None, class_names=["0", "1"],feature_names=features, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("shots_dt")

dotm_data = tree.export_graphviz(dtm_shots, out_file=None, class_names=["0", "1"], feature_names=features, filled=True, rounded=True, special_characters=True)
graph_m = graphviz.Source(dotm_data)
graph_m.render("shots_dtm")

y_pred = dt_shots.predict(X_orig)
y_pred_m = dtm_shots.predict(Xm_orig)

print("LOCF Report")
print(classification_report(Y_orig, y_pred))

print("Mean Report")
print(classification_report(Ym_orig, y_pred_m))

print()
print("LOCF Confusion Matrix")
print(confusion_matrix(Y_orig, y_pred))

print("Mean Confusion Matrix")
print(confusion_matrix(Ym_orig, y_pred_m))