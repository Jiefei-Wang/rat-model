# This script process the raw data and saves it to the output/data directory.
import pickle
import os
import pandas as pd

from modules.read_data import read_data
from modules.data_management import manage_data
from modules.feature_extraction import convert_to_features
from sklearn.model_selection import train_test_split

df_raw = read_data('data/01 Sucrose FR1 vs EXT 8_2024')

## save the data to output/processed_data

truncate_size = 3
chunk_size = 1
max_press = 1000
standardize = False
df = manage_data(df_raw, 
                  truncate_size=truncate_size,
                  chunk_size=chunk_size, 
                  max_press=max_press,
                  standardize=standardize)

x = convert_to_features(df) 


# Train test split
row_train, row_test = train_test_split(df.index, test_size=0.05, random_state=42, stratify= df[['id', 'category']])
df_tmp = df.loc[row_train]
row_train, row_valid = train_test_split(row_train, test_size=0.05, random_state=42, stratify= df_tmp[['id', 'category']])


## Check if row_train, row_valid, row_test cover all rows
assert set(row_train)| set(row_valid)| set(row_test) == set(df.index), "Row splits do not cover all rows in the dataframe."


feature_names = x.columns.tolist()
df_ML = pd.concat([df[['category', 'data']], x], axis=1)
df_ML['category'] = df_ML['category'].astype('category')
df_ML['category'] = df_ML['category'].cat.reorder_categories(['FR1', 'EXT'], ordered=True)
df_ML['label'] = df_ML['category'].cat.codes





## save to output/data using pickle
output_base = 'output/data' 
if not os.path.exists(output_base):
    os.makedirs(output_base)
    
with open(f'{output_base}/df_raw.pkl', 'wb') as f:
    pickle.dump(df_raw, f)

with open(f'{output_base}/df_ML.pkl', 'wb') as f:
    pickle.dump(df_ML, f)
with open(f'{output_base}/row_train.pkl', 'wb') as f:
    pickle.dump(row_train, f)
with open(f'{output_base}/row_valid.pkl', 'wb') as f:
    pickle.dump(row_valid, f)
with open(f'{output_base}/row_test.pkl', 'wb') as f:
    pickle.dump(row_test, f)


with open(f'{output_base}/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
