import pandas as pd
import numpy as np


def manage_data(df, truncate_size, max_press, standardize, min_press_len = 10, max_press_len = 500):
    df2 = truncate_data(df, truncate_size)
    df3 = df2.explode(['data', 'data_index']).reset_index(drop=True)
    df4 = cap_max_press(df3, max_press)
    df5 = standardize_data(df4) if standardize else df4
    # filter out the short presses
    df6 = df5[df5['data'].apply(len) >= min_press_len]
    df7 = df6[df6['data'].apply(len) <= max_press_len]
    df7.reset_index(drop=True, inplace=True)
    return df7

## cutting off the first n elements of the data
def truncate_data(df, length):
    df = df.copy()
    df['data'] = df['data'].apply(lambda x: x[length:])
    df['data_index'] = df['data_index'].apply(lambda x: x[length:])
    return df

# mean and std standardization
def standardize_data(df):
    df = df.copy()
    df['data_mean'] = df['data'].apply(lambda x: np.mean(x))
    df['data_std'] = df['data'].apply(lambda x: np.std(x))
    df['data'] = df.apply(lambda x: [(i - x.data_mean)/x.data_std for i in x.data], axis=1)
    df = df.drop(columns=['data_mean', 'data_std'])
    return df


## make sure the maximum press value is capped at max_press
def cap_max_press(df, max_press):
    df = df.copy()
    df['data'] = df['data'].apply(lambda x: [min(z, max_press) for z in x])
    return df


# def chunk_data(df, chunk_size):
#     if chunk_size <= 0:
#         raise ValueError("Chunk size must be a positive integer")
    
#     df = df.copy()
#     df['data'] = df['data'].apply(lambda x: chunk_data_fun(x, chunk_size))
#     df['data_index'] = df['data_index'].apply(lambda x: chunk_data_fun(x, chunk_size))
#     df = df.explode(['data', 'data_index']).reset_index(drop=True)
#     return df


# def chunk_data_fun(list_of_list, chunk_size):
#     """
#     _summary_

#     Args:
#         list_of_list (List): A list of lists, each list is a bar press data
#         chunk_size (num): how many bar presses in each chunk?

#     Returns:
#         List: A list of lists of lists, each list at the second level is a chunk of bar press data
#     """
#     chunks = [list_of_list[i:i + chunk_size] for i in range(0, len(list_of_list), chunk_size)]
    
#     # chunks = [concate_data(chunk) for chunk in chunks]
#     return chunks
