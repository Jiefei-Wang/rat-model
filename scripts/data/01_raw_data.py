# This script process the raw data and saves it to the output/data directory.
import pickle
import os
import pandas as pd
import numpy as np

from modules.read_data import  read_cohort_type1, read_cohort_type2
from modules.data_management import manage_data
from modules.feature_extraction import extract_barpress_features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df_raw1 = read_cohort_type1('data/cohort1')
df_raw2 = read_cohort_type2('data/cohort2/WS EXT')
df_raw3 = read_cohort_type1('data/cohort3')
df_raw1['cohort'] = 1
df_raw2['cohort'] = 2
df_raw3['cohort'] = 3
df_raw1['rat_type'] = "sprague-dawley"
df_raw2['rat_type'] = "sprague-dawley"
df_raw3['rat_type'] = "long-evans"


df_raw = pd.concat([df_raw1, df_raw2, df_raw3], axis=0).reset_index(drop=True)

rat_meta = pd.read_excel("data/Profiling-Weights_age_sex.xlsx")
rat_meta['id'] = rat_meta['id'].astype(str)
rat_meta['cohort'] = rat_meta['cohort'].astype(int)

df_raw = df_raw.merge(rat_meta, on=['id', 'cohort'], how='inner')
# reset id so that different cohorts have different id spaces
df_raw = df_raw.rename(columns={'id': 'within_cohort_id'})
df_raw['id'] = "C" + df_raw['cohort'].astype(str) + ":" + df_raw['within_cohort_id'].astype(str)


df_raw.columns
len(df_raw)
# 186 recordings

truncate_size = 3
max_press = 400
standardize = False
min_press_len = 10
df = manage_data(df_raw, 
                  truncate_size=truncate_size,
                  max_press=max_press,
                  standardize=standardize,
                  min_press_len=min_press_len)
len(df)
# 27360 bar presses

# recode: M=1, F=0
df['sex'] = df['sex'].map({'M': 1, 'F': 0})
df['rat_type'] = df['rat_type'].map({'long-evans': 1, 'sprague-dawley': 0})

rat_ids = df['id'].unique().tolist()
len(rat_ids)
# 63 rats

# keep n_test rats for testing
n_test = 10
train_ids, test_ids = train_test_split(rat_ids, test_size=n_test, random_state=42)

# basic df data split
row_train = df[df['id'].isin(train_ids)].reset_index(drop=True)
row_test = df[df['id'].isin(test_ids)].reset_index(drop=True)
(len(row_train), len(row_test))
# (19749, 3404)


# df_barpress_train = row_train[['category', 'data']]


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

x = extract_barpress_features(df, params=params) 

barpress_features_names = ['total_press_duration', 'max_force', 'pk_widths_max', 'pk_widths_mean', 'pk_widths_min',  'pk_sharp_mean', 'pk_sharp_min', 'skewness', 'kurtosis', 'avg_first_5', 'avg_last_5']

assert set(barpress_features_names).issubset(set(x.columns.tolist())), "Some ML feature names are not in the extracted feature names."


# standardize the features
scaler = StandardScaler()
scaler.fit(x) 
x_scaled = scaler.transform(x)
# convert back to dataframe
x_scaled = pd.DataFrame(x_scaled, columns=x.columns.tolist())



rat_features = ['sex', 'age', 'weight', 'rat_type']
ML_feature_names = barpress_features_names + rat_features


df_ML_unscaled = pd.concat([df[['id', 'category', 'data'] + rat_features], x], axis=1)
df_ML = pd.concat([df[['id', 'category', 'data'] + rat_features], x_scaled], axis=1)




df_ML['category'] = df_ML['category'].astype('category')
df_ML['category'] = df_ML['category'].cat.reorder_categories(['FR1', 'EXT'], ordered=True)
df_ML['label'] = df_ML['category']
df_ML_train = df_ML[df_ML['id'].isin(train_ids)].reset_index(drop=True)
df_ML_test = df_ML[df_ML['id'].isin(test_ids)].reset_index(drop=True)

df_ML.shape
# (23153, 19)




## save to output/data using pickle
output_base = 'output/data' 
if not os.path.exists(output_base):
    os.makedirs(output_base)

with open(f'{output_base}/df_raw1.pkl', 'wb') as f:
    pickle.dump(df_raw1, f)
    

with open(f'{output_base}/df_raw2.pkl', 'wb') as f:
    pickle.dump(df_raw2, f)

with open(f'{output_base}/df_raw.pkl', 'wb') as f:
    pickle.dump(df_raw, f)


with open(f"{output_base}/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open(f"{output_base}/df_ML_unscaled.pkl", "wb") as f:
    pickle.dump(df_ML_unscaled, f)

# with open(f'{output_base}/df_barpress_train.pkl', 'wb') as f:
#     pickle.dump(df_barpress_train, f)
    

with open(f'{output_base}/row_train.pkl', 'wb') as f:
    pickle.dump(row_train, f)
# with open(f'{output_base}/row_valid.pkl', 'wb') as f:
#     pickle.dump(row_valid, f)
with open(f'{output_base}/row_test.pkl', 'wb') as f:
    pickle.dump(row_test, f)


with open(f'{output_base}/df_ML.pkl', 'wb') as f:
    pickle.dump(df_ML, f)
with open(f'{output_base}/df_ML_train.pkl', 'wb') as f:
    pickle.dump(df_ML_train, f)
with open(f'{output_base}/df_ML_test.pkl', 'wb') as f:
    pickle.dump(df_ML_test, f)
    

with open(f'{output_base}/barpress_features_names.pkl', 'wb') as f:
    pickle.dump(barpress_features_names, f)
    
with open(f'{output_base}/ML_feature_names.pkl', 'wb') as f:
    pickle.dump(ML_feature_names, f)

    