
# This script prepares further data split for neural network training.
# nn_text is the same as df_test

from modules.Data import data_from_pickle_train_valid_test
from sklearn.model_selection import train_test_split
import os

df_train, df_test, x_train, x_test, y_train, y_test = data_from_pickle_train_valid_test()

# Turn the category column into a code
# FR1 = 0, EXT = 1
df_train['category'] = df_train['category'].astype('category')
df_train['category'] = df_train['category'].cat.reorder_categories(['FR1', 'EXT'], ordered=True)
df_train['label'] = df_train['category'].cat.codes

df_test['category'] = df_test['category'].astype('category')
df_test['category'] = df_test['category'].cat.reorder_categories(['FR1', 'EXT'], ordered=True)
df_test['label'] = df_test['category'].cat.codes

row_train, row_valid = train_test_split(df_train.index, test_size=0.05, random_state=42, stratify= df_train[['id', 'category']])

nn_train = df_train.loc[row_train].reset_index(drop=True)
nn_valid = df_train.loc[row_valid].reset_index(drop=True)
nn_test = df_test.reset_index(drop=True)


output_base = 'output/data'
if not os.path.exists(output_base):
    os.makedirs(output_base)
    
nn_train.to_pickle(f'{output_base}/nn_train.pkl')
nn_valid.to_pickle(f'{output_base}/nn_valid.pkl')
nn_test.to_pickle(f'{output_base}/nn_test.pkl')
