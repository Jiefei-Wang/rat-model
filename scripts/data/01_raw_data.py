# This script process the raw data and saves it to the output/data directory.

from modules.read_data import read_data
from modules.data_management import manage_data
from modules.feature_extraction import convert_to_features
import pickle
import os

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
x,y = convert_to_features(df) 

## save to output/data using pickle
output_base = 'output/data' 
if not os.path.exists(output_base):
    os.makedirs(output_base)
with open(f'{output_base}/df.pkl', 'wb') as f:
    pickle.dump(df, f)
with open(f'{output_base}/x_y.pkl', 'wb') as f:
    pickle.dump((x,y), f)
with open(f'{output_base}/df_raw.pkl', 'wb') as f:
    pickle.dump(df_raw, f)

