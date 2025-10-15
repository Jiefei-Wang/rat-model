# This script process the raw data and saves it to the output/data directory.
import pickle
import os
import pandas as pd
import numpy as np

from modules.read_data import read_data
from modules.data_management import manage_data
from modules.feature_extraction import convert_to_features
from sklearn.model_selection import train_test_split

df_raw1 = read_data('data/01 Sucrose FR1 vs EXT 8_2024')
df_raw2 = read_data('data/FR1 vs EXT 09_25')
df_raw2.id = df_raw2.id + 1000

df_raw = pd.concat([df_raw1, df_raw2], axis=0).reset_index(drop=True)

len(df_raw)
# 142 recordings

truncate_size = 3
max_press = 1000
standardize = False
min_press_len = 10
df = manage_data(df_raw, 
                  truncate_size=truncate_size,
                  max_press=max_press,
                  standardize=standardize,
                  min_press_len=min_press_len)
len(df)
# 23153 bar presses

rat_ids = df['id'].unique().tolist()
len(rat_ids)
# 41 rats

# keep n_test rats for testing
n_test = 5
train_ids, test_ids = train_test_split(rat_ids, test_size=n_test, random_state=42)

# basic df data split
row_train = df[df['id'].isin(train_ids)].reset_index()
row_test = df[df['id'].isin(test_ids)].reset_index()
(len(row_train), len(row_test))
# (19749, 3404)

# df_barpress_train = row_train[['category', 'data']]


# feature dataset 

params = {
    "distance":16,
    "height":18,
    "plateau_size":None,
    "prominence":8.6,
    "rel_height":0.5,
    "threshold":None,
    "width":14,
    "wlen":None
}

x = convert_to_features(df, params=params) 


feature_names = x.columns.tolist()
df_ML = pd.concat([df[['id', 'category', 'data']], x], axis=1)
df_ML['category'] = df_ML['category'].astype('category')
df_ML['category'] = df_ML['category'].cat.reorder_categories(['FR1', 'EXT'], ordered=True)
df_ML['label'] = df_ML['category'].cat.codes
df_ML_train = df_ML[df_ML['id'].isin(train_ids)].reset_index()
df_ML_test = df_ML[df_ML['id'].isin(test_ids)].reset_index()



# Train test split - 90:5:5 (train:validation:test)
# row_train0, row_test = train_test_split(df.index, test_size=0.05, random_state=42, stratify= df[['id', 'category']])

# df_tmp = df.loc[row_train0]
# row_train, row_valid = train_test_split(row_train0, test_size=5/95, random_state=42, stratify= df_tmp[['id', 'category']])


## Check if row_train, row_valid, row_test cover all rows
# assert set(row_train)| set(row_valid)| set(row_test) == set(df.index), "Row splits do not cover all rows in the dataframe."





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
    

with open(f'{output_base}/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

    