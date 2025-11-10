## This must be set before loading scipy
## Otherwise, Ctrl+C will cause crash
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import numpy as np
from scipy.signal import find_peaks,peak_widths,peak_prominences
from scipy.stats import skew, kurtosis
import pandas as pd

def max_mean_min(arr):
    if len(arr)>0:
        return np.max(arr), np.mean(arr), np.min(arr)
    else:
        return 0.0, 0.0, 0.0

# Function to calculate the duration of each bar press
def calculate_total_press_duration(x, params): 
    return len(x)  # Duration of each bar press

# Function to calculate the max force of each bar press
def calculate_max_force(x, params): 
    return max(x)  # Max force of each bar press

def calculate_peak_features(cell, params):
    peak_params = ['height', 'threshold', 'distance', 'prominence', 'width', 'wlen', 'rel_height', 'plateau_size']
    args = {key: params.get(key) for key in peak_params if key in params}
    
    peaks, _ = find_peaks(
        cell,
        **args)

    prominences, _, _ = peak_prominences(cell, peaks)
    widths, _, _, _ = peak_widths(cell, peaks, rel_height=0.5)

    num_peaks = float(len(peaks))
    pk_sharp  = prominences/widths

    return peaks, num_peaks, widths, pk_sharp


def calculate_peak_vally_features(x, params):
    x = np.array(x)
    # x_inverse = max(x)-x
    peaks, pk_num, pk_widths, pk_sharp = calculate_peak_features(x, params)
    # valleys, val_num, val_widths, val_sharp = calculate_peak_features(x_inverse, params)

    features = {
        "pk_num": pk_num,
        # "val_num": val_num,
        "pk_widths": pk_widths,
        # "val_widths": val_widths,
        "pk_sharp": pk_sharp,
        # "val_sharp": val_sharp
    }
    
    ## aggregate features if there are multiple peaks/valleys
    new_features = {}
    for k, v in features.items():
        if isinstance(v, (list, np.ndarray)):
            new_features[k+'_max'], new_features[k+'_mean'], new_features[k+'_min'] = max_mean_min(v)
        else:
            new_features[k] = v
    
    return new_features

def calculate_skewness(x, params):
    return skew(x) 

def calculate_kurtosis(x, params):
    return kurtosis(x) 

# maximum rate of change
def calculate_force_variation_rate(x, params):
    return max(np.abs(np.diff(x)))

def calculate_avg_first_5(x, params):
    return np.mean(x[:5])

def calculate_avg_last_5(x, params):
    return np.mean(x[-5:])

# Feature extraction function
def convert_to_features(df, params={}):
    
    ## name and function mapping    
    feature_list = {
        "total_press_duration": calculate_total_press_duration,
        "max_force": calculate_max_force,
        "peak_valley": calculate_peak_vally_features,
        "skewness": calculate_skewness,
        "kurtosis": calculate_kurtosis,
        "force_variation_rate": calculate_force_variation_rate,
        "avg_first_5": calculate_avg_first_5,
        "avg_last_5": calculate_avg_last_5
    }

    feature_df = {}
    # Extract features for each bar press
    for feature_name, feature_func in feature_list.items():
        features = [feature_func(x, params) for x in df['data']]
        
        if isinstance(features[0], dict):
            keys = features[0].keys()
            for key in keys:
                feature_df[key] = [feature[key] for feature in features]
        else:
            feature_df[feature_name] = features
            
    feature_df2 = pd.DataFrame(feature_df)

    feature_df2['sex'] = df['sex']
        
    return feature_df2
