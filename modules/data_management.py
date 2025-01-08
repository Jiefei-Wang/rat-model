import pandas as pd
import numpy as np


def manage_data(df, truncate_size, chunk_size, max_press, standardize):
    df2 = truncate_data(df, truncate_size)
    df3 = chunk_data(df2, chunk_size) 
    df4 = cap_max_press(df3, max_press)
    df5 = standardize_data(df4) if standardize else df4
    return df5

def standardize_data(df):
    df = df.copy()
    df['data_cb'] = df['data'].apply(lambda x: [i for j in x for i in j])
    df['data_mean'] = df['data_cb'].apply(lambda x: np.mean(x))
    df['data_std'] = df['data_cb'].apply(lambda x: np.std(x))
    df['data'] = df.apply(lambda x: [[(j - x.data_mean)/x.data_std for j in i] for i in x.data], axis=1)
    df = df.drop(columns=['data_cb', 'data_mean', 'data_std'])
    return df


## make sure the maximum press value is capped at max_press
def cap_max_press(df, max_press):
    df = df.copy()
    df['data'] = df['data'].apply(lambda x: [[min(z, max_press) for z in y] for y in x])
    return df


def chunk_data(df, chunk_size):
    """
    _summary_

    Args:
        df (DataFrame): A DataFrame containing bar press data
        chunk_size (num): how many bar presses in each chunk?

    Returns:
        DataFrame: A DataFrame containing chunks of bar press data
    """
    if chunk_size <= 0:
        raise ValueError("Chunk size must be a positive integer")
    
    df = df.copy()
    df['data'] = df['data'].apply(lambda x: chunk_data_fun(x, chunk_size))
    df = df.explode('data').reset_index(drop=True)
    return df


def chunk_data_fun(list_of_list, chunk_size):
    """
    _summary_

    Args:
        list_of_list (List): A list of lists, each list is a bar press data
        chunk_size (num): how many bar presses in each chunk?

    Returns:
        List: A list of lists of lists, each list at the second level is a chunk of bar press data
    """
    chunks = [list_of_list[i:i + chunk_size] for i in range(0, len(list_of_list), chunk_size)]
    
    # chunks = [concate_data(chunk) for chunk in chunks]
    return chunks

## cutting off the first n elements of the data
def truncate_data(df, length):
    df = df.copy()
    df['data'] = df['data'].apply(lambda x: x[length:])
    return df