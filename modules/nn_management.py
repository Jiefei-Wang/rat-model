from sklearn.model_selection import train_test_split
from modules.data_management import manage_data
import numpy as np

def get_train_test_valid(df, test_size=0.10, val_size=0.05):
    row_trainval, row_test = train_test_split(
        df.index,
        test_size=test_size,
        stratify=df['category'],
        random_state=42
    )
    row_train, row_val = train_test_split(
        row_trainval,
        test_size=val_size,
        stratify=df.loc[row_trainval, 'category'],
        random_state=42
    )

    df_train = df.loc[row_train].copy()
    df_val = df.loc[row_val].copy()
    df_test = df.loc[row_test].copy()
    return df_train, df_val, df_test


def get_NN_data(df_raw):
    truncate_size = 3
    chunk_size = 1
    max_press = 1000
    standardize = False

    # Process the full raw dataframe before splitting
    df2 = manage_data(df_raw, truncate_size, chunk_size, max_press, standardize)

    # Ensure label encoding before splitting
    df2['category'] = df2['category'].astype('category')
    df2['label'] = df2['category'].cat.codes

    # Split before exploding to avoid leakage
    df_train, df_val, df_test = get_train_test_valid(df2)

    # Explode and process each split separately
    def process_split(df_split):
        df_split = df_split.explode('data')
        df_split.dropna(subset=['data'], inplace=True)
        df_split['data'] = df_split['data'].apply(lambda x: np.array(x, dtype=np.float32))
        df_split['category'] = df_split['category'].astype('category')
        df_split['label'] = df_split['category'].cat.codes
        return df_split

    df_train = process_split(df_train)
    df_val = process_split(df_val)
    df_test = process_split(df_test)

    return df_train, df_val, df_test
