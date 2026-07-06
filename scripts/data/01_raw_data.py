# This script process the raw data and saves it to the output/data directory.
import pickle
import os
import pandas as pd
import sys

# use project root as the import path
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from modules.read_data import  read_cohort_type1, read_cohort_type2
from modules.data_management import manage_data
from modules.feature_extraction import extract_barpress_features

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
# if within_cohort_id ends with M or F, remove it
df_raw['within_cohort_id'] = df_raw['within_cohort_id'].apply(lambda x: x[:-1] if x[-1] in ['M', 'F'] else x)
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

# limit to first 30 minutes of data
time_cutoff = 30*60*100
df['data_start_index'] = df['data_index'].apply(lambda x: x[0])
df['data_end_index'] = df['data_index'].apply(lambda x: x[1])
df = df[df['data_end_index'].apply(lambda x: x <= time_cutoff)].reset_index(drop=True)
len(df)
# 24539

# recode variables
df['sex'] = df['sex'].map({'M': 1, 'F': 0})
df['rat_type'] = df['rat_type'].map({'long-evans': 1, 'sprague-dawley': 0})
df['label'] = df['category'].map({'FR1': 0, 'EXT': 1})

# weight by id and category. Each weight is the inverse of the number of presses for that id and category
weights = df.groupby(['id', 'category']).size().reset_index(name='press_count')
weights['sample_weight'] = 1 / weights['press_count']
weights = weights[['id', 'category', 'sample_weight']]
df = df.merge(weights, on=['id', 'category'], how='left')
# normalize the sample weight so that the mean sample weight is 1
df['sample_weight'] = df['sample_weight'] / df['sample_weight'].mean()



rat_ids = sorted(df['id'].unique().tolist())
len(rat_ids)
# 63 rats

# keep 3 rats from each cohort for testing
n_test_per_cohort = 3
rat_cohorts = df[['id', 'cohort']].drop_duplicates().reset_index(drop=True)
test_ids = (
    rat_cohorts
    .groupby('cohort', group_keys=False)
    .sample(n=n_test_per_cohort, random_state=42)['id']
    .tolist()
)
train_ids = sorted(set(rat_ids) - set(test_ids))
test_ids = sorted(test_ids)

# basic df data split
peak_train = df[df['id'].isin(train_ids)].reset_index(drop=True)
peak_test = df[df['id'].isin(test_ids)].reset_index(drop=True)
(len(peak_train), len(peak_test))
# without time filter
# (23116, 4244)
# with time filter
# (21282, 3257)


# df_barpress_train = row_train[['category', 'data']]


# feature dataset 

params = {
    "distance":10,
    "height":20,
    "width":5,
    "prominence":10
}

x = extract_barpress_features(df, params=params) 

barpress_features_names = x.columns.tolist()
# ['total_press_duration', 'max_force', 'pk_num', 'pk_widths_max', 'pk_widths_mean', 'pk_widths_min', 'pk_sharp_max', 'pk_sharp_mean', 'pk_sharp_min', 'skewness', 'kurtosis', 'force_variation_rate', 'avg_first_5', 'avg_last_5']


# # standardize the features
# # scale it before splitting to train and test set for simplicity. 
# scaler = StandardScaler()
# scaler.fit(x) 
# x_scaled = scaler.transform(x)
# # convert back to dataframe
# x_scaled = pd.DataFrame(x_scaled, columns=x.columns.tolist())



rat_features = ['sex', 'age', 'weight', 'rat_type']
ML_feature_names = barpress_features_names + rat_features


df_ML = pd.concat([df[['id', 'category', 'data', 'data_start_index', 'data_end_index', 'file_name', 'cohort', 'label', 'sample_weight'] + rat_features], x], axis=1)
# df_ML = pd.concat([df[['id', 'category', 'data', 'data_start_index', 'data_end_index', 'file_name', 'cohort', 'label', 'sample_weight'] + rat_features], x_scaled], axis=1)



df_ML_train = df_ML[df_ML['id'].isin(train_ids)].reset_index(drop=True)
df_ML_test = df_ML[df_ML['id'].isin(test_ids)].reset_index(drop=True)

df_ML.shape
# (24539, 28)


## save to output/data using pickle
output_base = 'output/data' 
if not os.path.exists(output_base):
    os.makedirs(output_base)

with open(f'{output_base}/df_raw1.pkl', 'wb') as f:
    pickle.dump(df_raw1, f)
with open(f'{output_base}/df_raw2.pkl', 'wb') as f:
    pickle.dump(df_raw2, f)
with open(f'{output_base}/df_raw3.pkl', 'wb') as f:
    pickle.dump(df_raw3, f)

with open(f'{output_base}/df_raw.pkl', 'wb') as f:
    pickle.dump(df_raw, f)


# with open(f"{output_base}/scaler.pkl", "wb") as f:
#     pickle.dump(scaler, f)

# with open(f"{output_base}/df_ML_unscaled.pkl", "wb") as f:
#     pickle.dump(df_ML_unscaled, f)

# with open(f'{output_base}/df_barpress_train.pkl', 'wb') as f:
#     pickle.dump(df_barpress_train, f)
    

with open(f'{output_base}/peak_train.pkl', 'wb') as f:
    pickle.dump(peak_train, f)
# with open(f'{output_base}/row_valid.pkl', 'wb') as f:
#     pickle.dump(row_valid, f)
with open(f'{output_base}/peak_test.pkl', 'wb') as f:
    pickle.dump(peak_test, f)


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

    
