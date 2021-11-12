# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv('shots.csv')
print(df.head())

df.describe()

# %%
df.corr()

#%%
x = df['distance_to_goal']
y = df['pass_length']

colors = ('r', 'b')

plt.xlabel('distance_to_goal')
plt.ylabel('pass_length')
plt.scatter(x, y)

# %%
import seaborn as sns



corr = df.corr()
plt.figure(figsize=(15, 15))
sns.heatmap(corr, vmax=1, square=True, annot=True, cmap='viridis')
plt.title('Correlation between features')
plt.savefig('media/corr.png')

# %%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# the following contains some classification. I'll seperate it later
# we need to fix the following stuff to be corrected by under/oversampling
# Lets use https://github.com/scikit-learn-contrib/imbalanced-learn

new_df = df.drop(['play_pattern', 'shot_taker_type', 'shot_technique', 'shot_body_part'], axis=1)

X = new_df.iloc[:, 0:len(new_df.columns)-1]
y = new_df.iloc[:, len(new_df.columns)-1]

X = scaler.fit_transform(X)
# %%

pca = PCA()
principalComponents = pca.fit_transform(X)

explained_variance = pca.explained_variance_ratio_
print(explained_variance)

# %%

with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(10, 10))
    plt.bar(range(len(explained_variance)), explained_variance, label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('media/pca_explained_variance.png')
# %%

pca = PCA(n_components=7)
X_new = pca.fit_transform(X)

explained_variance = pca.explained_variance_ratio_
print(explained_variance)

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

X_train.shape

# %%
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
estimators = np.arange(10, 200, 10)
scores = []
for n in estimators:
    model.set_params(n_estimators=n)
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))
    print(scores[-1])

print(scores)

# %%

plt.plot(estimators, scores)
plt.xlabel("Number of Estimators")
plt.ylabel("Score")
plt.title("Effect of Estimators on Score")
plt.savefig("media/estimators_vs_model_score.png")

# %%
print(y)