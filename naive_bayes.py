from os import replace
from sklearn import naive_bayes
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

# read locf data
df_locf = pd.read_csv('shots_LOCF.csv', index_col='shot_num')
# read mean mode data
df_mean = pd.read_csv('shots_mean.csv', index_col='shot_num')

discrete_features = [
    'play_pattern',
    'shot_taker_type',
    'shot_technique',
    'shot_body_part',
    'goal',
    'goalkeeper_in_the_way',
    'one_on_one']

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

continous_features = df_locf.columns.difference(discrete_features)


est = KBinsDiscretizer(n_bins=5, encode='ordinal')
for feature in continous_features:
    df_locf[feature] = est.fit_transform(df_locf[feature].values.reshape(-1, 1))
    df_mean[feature] = est.fit_transform(df_mean[feature].values.reshape(-1, 1))

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

# undersample the majority class
min_count = y_locf.value_counts().min()
sampled = []
for class_index, group in df_locf.groupby('goal'):
    sampled.append(group.sample(n=min_count, replace=True))

df_locf_undersampled = pd.concat(sampled)
X_locf_undersampled = df_locf_undersampled[all_features]
y_locf_undersampled = df_locf_undersampled['goal']

min_count = y_mean.value_counts().min()
sampled = []
for class_index, group in df_mean.groupby('goal'):
    sampled.append(group.sample(n=min_count, replace=True))

df_mean_undersampled = pd.concat(sampled)
X_mean_undersampled = df_mean_undersampled[all_features]
y_mean_undersampled = df_mean_undersampled['goal']

# Oversampling
nbc_locf_oversampled = naive_bayes.CategoricalNB()
nbc_locf_oversampled.fit(X_locf_oversampled, y_locf_oversampled)

nbc_mean_oversampled = naive_bayes.CategoricalNB()
nbc_mean_oversampled.fit(X_mean_oversampled, y_mean_oversampled)

y_locf_predicted_oversampled = nbc_locf_oversampled.predict(X_locf_test)
y_mean_predicted_oversampled = nbc_mean_oversampled.predict(X_mean_test)

y_locf_prob_oversampled = nbc_locf_oversampled.predict_proba(X_locf_test)
y_mean_prob_oversampled = nbc_mean_oversampled.predict_proba(X_mean_test)

# Undersampling

# nbc_locf_undersampled = naive_bayes.CategoricalNB()
# nbc_locf_undersampled.fit(X_locf_undersampled, y_locf_undersampled)

# nbc_mean_undersampled = naive_bayes.CategoricalNB()
# nbc_mean_undersampled.fit(X_mean_undersampled, y_mean_undersampled)

# y_locf_predicted_undersampled = nbc_locf_undersampled.predict(X_locf)
# y_mean_predicted_undersampled = nbc_mean_undersampled.predict(X_mean)

# y_locf_prob_undersampled = nbc_locf_undersampled.predict_proba(X_locf)
# y_mean_prob_undersampled = nbc_mean_undersampled.predict_proba(X_mean)

# original

nbc_locf = naive_bayes.CategoricalNB()
nbc_locf.fit(X_locf, y_locf)

nbc_mean = naive_bayes.CategoricalNB()
nbc_mean.fit(X_mean, y_mean)

y_orig_predicted_locf = nbc_locf.predict(X_locf_test)
y_orig_predicted_mean = nbc_mean.predict(X_mean_test)

y_orig_prob_locf = nbc_locf.predict_proba(X_locf_test)
y_orig_prob_mean = nbc_mean.predict_proba(X_mean_test)


# print accuracy, precision, recall, f1 score
print('Original')
print('LOCF')
print(classification_report(y_locf_test, y_orig_predicted_locf))
print(confusion_matrix(y_locf_test, y_orig_predicted_locf))
print('Mean')
print(classification_report(y_mean_test, y_orig_predicted_mean))
print(confusion_matrix(y_mean_test, y_orig_predicted_mean))

print()

print('Oversampled')
print('LOCF')
print(classification_report(y_locf_test, y_locf_predicted_oversampled))
print(confusion_matrix(y_locf_test, y_locf_predicted_oversampled))
print('Mean')
print(classification_report(y_mean_test, y_mean_predicted_oversampled))
print(confusion_matrix(y_mean_test, y_mean_predicted_oversampled))

# print('Undersampled')
# print('LOCF')
# print(classification_report(y_locf_undersampled, y_locf_predicted_undersampled))
# print('Mean')
# print(classification_report(y_mean_undersampled, y_mean_predicted_undersampled))

print()

print("Comparing Sums")
print("Original Dataset Number of Goals")
print("LOCF")
print(np.sum(y_locf))
print("Mean")
print(np.sum(y_mean))

print("Oversampling Probability Sums")
print("LOCF")
print(np.sum(y_locf_prob_oversampled, axis=0)[0])
print("Mean")
print(np.sum(y_mean_prob_oversampled, axis=0)[0])

print("Original Probability Sums")
print("LOCF")
print(np.sum(y_orig_prob_locf, axis=0)[0])
print("Mean")
print(np.sum(y_orig_prob_mean, axis=0)[0])