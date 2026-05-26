# This script process the raw data and saves it to the output/data directory.
import pickle
import os
import pandas as pd
import numpy as np

from modules.read_data import read_category_data
from modules.data_management import manage_data
from modules.feature_extraction import convert_to_features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df_app = read_category_data('data/application_Gen 2 Sucrose PR')

len(df_app)
# 19 recordings

truncate_size = 3
max_press = 400
standardize = False
min_press_len = 10
df = manage_data(df_app, 
                  truncate_size=truncate_size,
                  max_press=max_press,
                  standardize=standardize,
                  min_press_len=min_press_len)
len(df)
# 1068 bar presses


# feature dataset 

params = {
    "distance":28,
    "height":2,
    "plateau_size":None,
    "prominence":2.6,
    "rel_height":0.8,
    "threshold":None,
    "width":12,
    "wlen":None
}

x = convert_to_features(df, params=params) 
feature_names = x.columns.tolist()

ML_feature_names = ['total_press_duration', 'max_force', 'pk_widths_max', 'pk_widths_mean', 'pk_widths_min',  'pk_sharp_mean', 'pk_sharp_min', 'skewness', 'kurtosis', 'avg_first_5', 'avg_last_5', 'sex']

assert set(ML_feature_names).issubset(set(feature_names)), "Some ML feature names are not in the extracted feature names."


# exclude sex
x_no_sex = x.drop(columns=['sex'])
std_feature_names = x_no_sex.columns.tolist()
# load_scaler
with open("output/data/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

x_scaled = scaler.transform(x_no_sex)
# convert back to dataframe
x_scaled = pd.DataFrame(x_scaled, columns=std_feature_names)


# df_app_ML_unscaled = pd.concat([df[['id', 'category', 'data']], x], axis=1)

df_app_ML = pd.concat([df[['id', 'category', 'data', 'data_index']], x_scaled, x[['sex']]], axis=1)

with open(f'output/data/df_app_ML.pkl', 'wb') as f:
    pickle.dump(df_app_ML, f)
