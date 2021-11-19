#%%

from os import replace
from numpy.core.numeric import indices
from sklearn import naive_bayes
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

df = pd.read_csv('shots.csv', index_col='shot_num')
df.dropna(inplace=True)
categorical_features = [
    'play_pattern',
    'shot_taker_type',
    'shot_technique',
    'shot_body_part'
]
df.drop(categorical_features, axis=1, inplace=True)

df_subset = df.sample(frac=0.1, random_state=1)

#%%
# df_subset.plot(kind='scatter', x='distance_to_goal', y='angle_to_goal', c='goal', colormap='viridis')
# df_subset.plot(kind='scatter', x='pass_speed', y='pass_length', c='goal', colormap='viridis')

# Do pca and get 2 features
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X = df.iloc[:, 0:len(df.columns)-1]
y = df.iloc[:, len(df.columns)-1]

X = scaler.fit_transform(X)

pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_)

# get features of the first two components
df_pca = pd.DataFrame(pca.transform(X), columns=['pc1', 'pc2'])
df_pca['goal'] = y

print(df_pca.head())

# plot pca components with goal as color
df_pca.plot(kind='scatter', x='pc1', y='pc2', c='goal', colormap='viridis')
plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#%%

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=200),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

#%%

X = df_pca.iloc[:, 0:len(df_pca.columns)-1]

# get player_name and pass_name cols of df
# some_cols = ['players_within_5', 'pass_speed']
# df_some_cols = df[some_cols]
# print(df_some_cols.head())

store_df = df.copy()
store_df_pca = df_pca.copy()
store_y = y.copy()

#%%

number_of_samples = int(350 / 2)

# pick half of samples where y is 0
df_0 = store_df[store_df['goal'] == 0]
df_0_pca = store_df_pca[store_df_pca['goal'] == 0]
df_0_y = store_y[store_y == 0]

df_1 = store_df[store_df['goal'] == 1]
df_1_pca = store_df_pca[store_df_pca['goal'] == 1]
df_1_y = store_y[store_y == 1]

one_indices = np.random.choice(df_1_y.index, number_of_samples, replace=False)
print(one_indices)
zero_indices = np.random.choice(df_0_y.index, number_of_samples, replace=False)
print(zero_indices)

indices = np.concatenate((one_indices, zero_indices))

df = store_df.iloc[indices]
df_pca = store_df_pca.iloc[indices]
y = store_y.iloc[indices]

datasets = [
    (df[['players_within_5', 'shot_speed']], y),
    (df[['distance_to_goal', 'players_between_goal']], y),
    (df_pca[['pc1', 'pc2']], y),
]

for ds in datasets:
    print(ds)

#%%

print(X.shape, y.shape)

h = 0.02 
figure = plt.figure(figsize=(27, 9))
i = 1

for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdYlGn
    cm_bright = ListedColormap(["#FF0000", "#00FF00"])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")

    if i == 1:
        ax.set_ylabel("Ppl w/in 5 vs Shot Spd")

    if i == 12:
        ax.set_ylabel("Dist vs ppl b/w Goal")

    if i == 23:
        ax.set_ylabel("PCA components")

    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    # Plot the testing points
    ax.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
    )
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)

        # Plot the training points
        ax.scatter(
            X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
        )
        # Plot the testing points
        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_test,
            cmap=cm_bright,
            edgecolors="k",
            alpha=0.6,
        )

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(
            xx.max() - 0.3,
            yy.min() + 0.3,
            ("%.2f" % score).lstrip("0"),
            size=15,
            horizontalalignment="right",
        )
        i += 1

plt.savefig("media/lots_of_models.png", dpi=300)

plt.tight_layout()

#%%
