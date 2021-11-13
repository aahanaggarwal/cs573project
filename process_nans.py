import pandas as pd
import numpy as np

shots = pd.read_csv("shots.csv")
# replace -1 with NaN for LOCF
shots_LOCF = shots.replace(-1.0, np.nan)
shots_LOCF = shots_LOCF.fillna(method="ffill")

shots_LOCF.to_csv("shots_LOCF.csv")

# for mean_mode
shots_MEAN = shots.replace(-1.0, np.nan)
shots_MEAN['shot_speed'].fillna(value=shots_MEAN['shot_speed'].mean(), inplace=True)
shots_MEAN['pass_length'].fillna(value=shots_MEAN['pass_length'].mean(), inplace=True)
shots_MEAN['pass_angle'].fillna(value=shots_MEAN['pass_angle'].mean(), inplace=True)
shots_MEAN['pass_speed'].fillna(value=shots_MEAN['pass_speed'].mean(), inplace=True)

shots_MEAN.to_csv("shots_mean.csv")