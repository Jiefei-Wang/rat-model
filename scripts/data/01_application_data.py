# This script process the raw data and saves it to the output/data directory.
import pickle
import os
import pandas as pd
import sys

# use project root as the import path
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from modules.read_data import read_category_data
from modules.data_management import manage_data
from modules.feature_extraction import extract_barpress_features

df_app = read_category_data('data/cohort3_application_Gen 2 Sucrose PR')
df_app['cohort'] = 3
df_app['rat_type'] = 'long-evans'

rat_meta = pd.read_excel('data/Profiling-Weights_age_sex.xlsx')
rat_meta['id'] = rat_meta['id'].astype(str)
rat_meta['cohort'] = rat_meta['cohort'].astype(int)

df_app['id'] = df_app['id'].astype(str)
df_app['cohort'] = df_app['cohort'].astype(int)
df_app = df_app.merge(rat_meta, on=['id', 'cohort'], how='left')

df_app['id'] = "C3:"+df_app['id']
df_app['id'] = df_app['id'].apply(lambda x: x[:-1] if x[-1] in ['M', 'F'] else x)


missing_meta = df_app[['sex', 'age', 'weight']].isna().any(axis=1)
if missing_meta.any():
    missing_ids = sorted(df_app.loc[missing_meta, 'id'].unique().tolist())
    raise ValueError(f'Missing cohort 3 metadata for ids: {missing_ids}')

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
# 988 bar presses

# feature dataset 

params = {
    "distance": 20,
    "height": 20,
    "prominence": 0.2,
    "width": 2
}

x = extract_barpress_features(df, params=params)
barpress_feature_names = x.columns.tolist()
rat_features = ['sex', 'age', 'weight', 'rat_type']
base_columns = ['id', 'category', 'data', 'data_start_index', 'data_end_index', 'file_name', 'cohort']

df['sex'] = df['sex'].map({'M': 1, 'F': 0})
df['rat_type'] = df['rat_type'].map({'long-evans': 1, 'sprague-dawley': 0})
df['data_start_index'] = df['data_index'].apply(lambda x: x[0])
df['data_end_index'] = df['data_index'].apply(lambda x: x[1])

ML_feature_names = pickle.load(open('output/data/ML_feature_names.pkl', 'rb'))
expected_feature_names = barpress_feature_names + rat_features
if expected_feature_names != ML_feature_names:
    raise ValueError('Application feature names do not match the saved ML feature contract.')

# df_app_ML_unscaled = pd.concat([df[['id', 'category', 'data']], x], axis=1)

df_app_ML = pd.concat([df[base_columns + rat_features], x], axis=1)


with open(f'output/data/df_app.pkl', 'wb') as f:
    pickle.dump(df_app, f)
    
with open(f'output/data/df_app_ML.pkl', 'wb') as f:
    pickle.dump(df_app_ML, f)
