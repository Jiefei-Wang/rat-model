# This script splits the data into training, validation, and test sets.
from sklearn.model_selection import train_test_split
from modules.Data import data_from_pickle
import pandas as pd
import os

df_raw, df, x, y = data_from_pickle()


#split the row indices of the data into training and test sets
row_train, row_test = train_test_split(df.index, test_size=0.05, random_state=42, stratify= df[['id', 'category']])


# a= df.loc[row_tmp, 'category'].value_counts(normalize=True)
# b=df.loc[row_test, 'category'].value_counts(normalize=True)
# pd.concat([a, b], axis=1)


# check 1: row_train and row_test does not overlap
assert len(set(row_train) & set(row_test)) == 0

# check 2: row_train and row_test cover all data
assert len(set(row_train) | set(row_test)) == len(df)



df_train = df.loc[row_train].reset_index(drop=False)
df_test = df.loc[row_test].reset_index(drop=False)

x_train = x.loc[row_train].reset_index(drop=True)
x_test = x.loc[row_test].reset_index(drop=True)
y_train = y.loc[row_train].reset_index(drop=True)
y_test = y.loc[row_test].reset_index(drop=True)

# save the data to output/data
output_base = 'output/data'
if not os.path.exists(output_base):
    os.makedirs(output_base)
    
df_train.to_pickle(f'{output_base}/df_train.pkl')
df_test.to_pickle(f'{output_base}/df_test.pkl')
x_train.to_pickle(f'{output_base}/x_train.pkl')
x_test.to_pickle(f'{output_base}/x_test.pkl')
y_train.to_pickle(f'{output_base}/y_train.pkl')
y_test.to_pickle(f'{output_base}/y_test.pkl')