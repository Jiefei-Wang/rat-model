import pandas as pd

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


def concate_data(list_of_lists):
    concatenated_list = []
    # Iterate over the list of lists
    for i, sublist in enumerate(list_of_lists):
        concatenated_list.extend(sublist)  # Add the elements of the sublist to the result list
        if i < len(list_of_lists) - 1:
            concatenated_list.append(0)  # Add 0 between the sublists
            
    return concatenated_list


def truncate_or_padding(data, length=100):
    if len(data) > length:
        return data[:length]
    else:
        return data + [0]*(length-len(data))
    