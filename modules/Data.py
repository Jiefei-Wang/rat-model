import pickle
def data_from_pickle(base = 'output/data' ):
    with open(f'{base}/df_raw.pkl', 'rb') as f:
        df_raw = pickle.load(f)
    with open(f'{base}/df.pkl', 'rb') as f:
        df = pickle.load(f)
    with open(f'{base}/x_y.pkl', 'rb') as f:
        x, y = pickle.load(f)
    return df_raw, df, x, y

def data_from_pickle_train_valid_test(base = 'output/data'):
    with open(f'{base}/df_train.pkl', 'rb') as f:
        df_train = pickle.load(f)
    with open(f'{base}/df_test.pkl', 'rb') as f:
        df_test = pickle.load(f)
    with open(f'{base}/x_train.pkl', 'rb') as f:
        x_train = pickle.load(f)
    with open(f'{base}/x_test.pkl', 'rb') as f:
        x_test = pickle.load(f)
    with open(f'{base}/y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    with open(f'{base}/y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)
    
    return df_train, df_test, x_train, x_test, y_train, y_test


def data_from_pickle_nn(base = 'output/data'):
    """
    nn_test is the same as df_test in the data_from_pickle_train_valid_test function
    """
    with open(f'{base}/nn_train.pkl', 'rb') as f:
        nn_train = pickle.load(f)
    with open(f'{base}/nn_valid.pkl', 'rb') as f:
        nn_valid = pickle.load(f)
    with open(f'{base}/nn_test.pkl', 'rb') as f:
        nn_test = pickle.load(f)
    with open(f'{base}/features_train.pkl', 'rb') as f:
        features_train = pickle.load(f)
    with open(f'{base}/features_valid.pkl', 'rb') as f:
        features_valid = pickle.load(f)
    with open(f'{base}/features_test.pkl', 'rb') as f:
        features_test = pickle.load(f)
    
    
    return nn_train, nn_valid, nn_test, features_train, features_valid, features_test