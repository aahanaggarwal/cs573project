#%%

from os import replace
from sklearn import neural_network
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

# read locf data
df_locf = pd.read_csv('data/shots_LOCF.csv', index_col='shot_num')
# read mean mode data
df_mean = pd.read_csv('data/shots_mean.csv', index_col='shot_num')

categorical_features = [
    'play_pattern',
    'shot_taker_type',
    'shot_technique',
    'shot_body_part'
]

for feature in categorical_features:
    df_locf[feature] = pd.Categorical(df_locf[feature])
    df_mean[feature] = pd.Categorical(df_mean[feature])
    df_locf[feature] = df_locf[feature].cat.codes
    df_mean[feature] = df_mean[feature].cat.codes

df_locf_test = df_locf.sample(frac=0.2)
df_locf = df_locf.drop(df_locf_test.index)

df_mean_test = df_mean.sample(frac=0.2)
df_mean = df_mean.drop(df_mean_test.index)

all_features = df_locf.columns.difference(['goal'])
X_locf = df_locf[all_features]
y_locf = df_locf['goal']

X_mean = df_mean[all_features]
y_mean = df_mean['goal']

X_locf_test = df_locf_test[all_features]
y_locf_test = df_locf_test['goal']

X_mean_test = df_mean_test[all_features]
y_mean_test = df_mean_test['goal']

# oversample the minority class
max_count = y_locf.value_counts().max()
sampled = [df_locf]
for class_index, group in df_locf.groupby('goal'):
    sampled.append(group.sample(n=max_count - group.shape[0], replace=True))

df_locf_oversampled = pd.concat(sampled)
X_locf_oversampled = df_locf_oversampled[all_features]
y_locf_oversampled = df_locf_oversampled['goal']

max_count = y_mean.value_counts().max()
sampled = [df_mean]
for class_index, group in df_mean.groupby('goal'):
    sampled.append(group.sample(n=max_count - group.shape[0], replace=True))

df_mean_oversampled = pd.concat(sampled)

X_mean_oversampled = df_mean_oversampled[all_features]
y_mean_oversampled = df_mean_oversampled['goal']

# oversampled preds
nn_locf_oversampled = neural_network.MLPClassifier(hidden_layer_sizes=(128, 64, 32))
nn_locf_oversampled.fit(X_locf_oversampled, y_locf_oversampled)

nn_mean_oversampled = neural_network.MLPClassifier(hidden_layer_sizes=(128, 64, 32))
nn_mean_oversampled.fit(X_mean_oversampled, y_mean_oversampled)

y_locf_oversampled_pred = nn_locf_oversampled.predict(X_locf_oversampled)
y_mean_oversampled_pred = nn_mean_oversampled.predict(X_mean_oversampled)

y_locf_oversampled_prob = nn_locf_oversampled.predict_proba(X_locf_oversampled)
y_mean_oversampled_prob = nn_mean_oversampled.predict_proba(X_mean_oversampled)

print('Oversampled')
print('LOCF')
print(classification_report(y_locf_oversampled, y_locf_oversampled_pred))
print('Mean')
print(classification_report(y_mean_oversampled, y_mean_oversampled_pred))

print()

# original preds
nn_locf = neural_network.MLPClassifier(hidden_layer_sizes=(128, 64, 32))
nn_locf.fit(X_locf, y_locf)

nn_mean = neural_network.MLPClassifier(hidden_layer_sizes=(128, 64, 32))
nn_mean.fit(X_mean, y_mean)

y_orig_locf_pred = nn_locf.predict(X_locf_test)
y_orig_mean_pred = nn_mean.predict(X_mean_test)

y_orig_locf_prob = nn_locf.predict_proba(X_locf)
y_orig_mean_prob = nn_mean.predict_proba(X_mean)

# print classification report
print('Original')
print('LOCF')
print(classification_report(y_locf_test, y_orig_locf_pred))
print('Mean')
print(classification_report(y_mean_test, y_orig_mean_pred))

print()

#%%

df_locf_og = pd.read_csv('shots_LOCF.csv', index_col='shot_num')
# read mean mode data
df_mean_og = pd.read_csv('shots_mean.csv', index_col='shot_num')

for feature in categorical_features:
    df_locf_og[feature] = pd.Categorical(df_locf_og[feature])
    df_mean_og[feature] = pd.Categorical(df_mean_og[feature])
    df_locf_og[feature] = df_locf_og[feature].cat.codes
    df_mean_og[feature] = df_mean_og[feature].cat.codes

X_locf = df_locf_og[all_features]
y_locf = df_locf_og['goal']

X_mean = df_mean_og[all_features]
y_mean = df_mean_og['goal']


y_orig_locf_prob = nn_locf.predict_proba(X_locf)
y_orig_mean_prob = nn_mean.predict_proba(X_mean)


y_locf_oversampled_prob = nn_locf_oversampled.predict_proba(X_locf)
y_mean_oversampled_prob = nn_mean_oversampled.predict_proba(X_locf)

print("Comparing Sums")
print("Original Dataset Number of Goals")
print("LOCF")
print(np.sum(y_locf))
print("Mean")
print(np.sum(y_mean))

print("Oversampling Probability Sums")
print("LOCF")
print(np.sum(y_locf_oversampled_prob, axis=0)[0])
print("Mean")
print(np.sum(y_mean_oversampled_prob, axis=0)[0])

print("Original Probability Sums")
print("LOCF")
print(np.sum(y_orig_locf_prob, axis=0)[0])
print("Mean")
print(np.sum(y_orig_mean_prob, axis=0)[0])

#%%

ycum_orig_locf = np.add.reduceat(y_orig_locf_prob[:, 1], np.arange(0, len(y_orig_locf_prob[:, 1]), 1000))
ycum_orig_mean = np.add.reduceat(y_orig_mean_prob[:, 1], np.arange(0, len(y_orig_mean_prob[:, 1]), 1000))

ycum_oversampled_locf = np.add.reduceat(y_locf_oversampled_prob[:, 1], np.arange(0, len(y_locf_oversampled_prob[:, 1]), 1000))
ycum_oversampled_mean = np.add.reduceat(y_mean_oversampled_prob[:, 1], np.arange(0, len(y_mean_oversampled_prob[:, 1]), 1000))

ycum_og_mean = np.add.reduceat(np.array(y_mean), np.arange(0, 23618, 1000))

# print(ycum_orig_locf)
# print(ycum_orig_mean)
# print(ycum_oversampled_locf)
# print(ycum_oversampled_mean)
print(ycum_og_mean)

#%%

import matplotlib.pyplot as plt

# convert to log plot
# ycum_orig_locf = np.log(ycum_orig_locf)
# ycum_orig_mean = np.log(ycum_orig_mean)

# ycum_oversampled_locf = np.log(ycum_oversampled_locf)
# ycum_oversampled_mean = np.log(ycum_oversampled_mean)

# ycum_og_locf = np.log(ycum_og_locf)
# ycum_og_mean = np.log(ycum_og_mean)

# divide by shot num
# ycum_orig_locf = ycum_orig_locf / np.arange(5, len(ycum_orig_locf) + 5)
# ycum_orig_mean = ycum_orig_mean / np.arange(5, len(ycum_orig_mean) + 5)

# ycum_oversampled_locf = ycum_oversampled_locf / np.arange(5, len(ycum_oversampled_locf) + 5)
# ycum_oversampled_mean = ycum_oversampled_mean / np.arange(5, len(ycum_oversampled_mean) + 5)

# ycum_og_locf = ycum_og_locf / np.arange(5, len(ycum_og_locf) + 5)
# ycum_og_mean = ycum_og_mean / np.arange(5, len(ycum_og_mean) + 5)

from scipy.interpolate import BSpline, make_interp_spline

spl = make_interp_spline(np.arange(0, len(ycum_orig_locf)), ycum_orig_locf, k = 3)
ycum_orig_locf_smooth = spl(ycum_orig_locf)

spl = make_interp_spline(np.arange(0, len(ycum_orig_mean)), ycum_orig_mean, k=3)
ycum_orig_mean_smooth = spl(ycum_orig_mean)

spl = make_interp_spline(np.arange(0, len(ycum_og_mean)), ycum_og_mean, k=3)
ycum_og_mean_smooth = spl(ycum_og_mean)

plt.figure(figsize=(20, 10))
plt.plot(ycum_orig_locf_smooth, label='Original Data LOCF Prob')
plt.plot(ycum_orig_mean_smooth, label='Original Data Mean Prob')

# plt.plot(ycum_oversampled_locf, label='Oversampled Data LOCF Prob')
# plt.plot(ycum_oversampled_mean, label='Oversampled Data Mean Prob')

plt.plot(ycum_og_mean_smooth, label='Actual Goal Sums')

# plt.xlim(0, 220)
# plt.ylim(0, 20)
plt.legend()
plt.xlabel('100\'s of Games')
plt.ylabel('Smoothed Goals per 100 games')
plt.savefig('media/understat_graph_comp.png')
plt.show()