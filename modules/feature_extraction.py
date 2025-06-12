## This must be set before loading scipy
## Otherwise, Ctrl+C will cause crash
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
import pandas as pd

# Increase the number of displayed columns
pd.set_option('display.max_columns', None)

# Increase the width of the display
pd.set_option('display.width', None)

# Function to calculate valley sharpness
def calculate_valley_sharpness(x):
    return x.apply(lambda presses: [calculate_valley_sharpness_for_press(press) for press in presses])  # Valley sharpness for each bar press

# Function to calculate valley sharpness for a single bar press
def calculate_valley_sharpness_for_press(press):
    inverted_press = -np.array(press)  # Invert the signal to detect valleys
    valleys, _ = find_peaks(inverted_press)  # Find valleys (peaks in the inverted signal)
    
    sharpness = []
    for valley in valleys:
        # Calculate the slope before and after the valley
        if valley > 0 and valley < len(press) - 1:
            left_slope = press[valley] - press[valley - 1]
            right_slope = press[valley] - press[valley + 1]
            sharpness.append(abs(left_slope) + abs(right_slope))
    
    # Return the average sharpness of the detected valleys (or 0 if no valleys detected)
    return np.mean(sharpness) if sharpness else 0


# Function to calculate the duration of each bar press
def calculate_total_press_duration(x): 
    return x.apply(lambda presses: [len(press) for press in presses])  # Duration of each bar press

# Function to calculate the max force of each bar press
def calculate_max_force(x): 
    return x.apply(lambda presses: [max(press) for press in presses])  # Max force of each bar press

# Function to calculate the number of peaks in each bar press
def calculate_peaks(x): 
    return x.apply(lambda presses: [len(find_peaks(press)[0]) for press in presses])  # Number of peaks for each bar press

# Function to calculate the max peak duration in each bar press
def calculate_max_peak_duration(x):
    return x.apply(lambda presses: [max(calculate_peak_durations(press, 20)) if calculate_peak_durations(press, 20) else 0 for press in presses])  # Max peak duration for each bar press

# Function to calculate peak durations for each bar press
def calculate_peak_durations(bar_press_forces, threshold):
    peaks, _ = find_peaks(bar_press_forces)
    peak_durations = []
    
    for peak in peaks:
        peak_value = bar_press_forces[peak]
        lower_bound = peak_value - threshold
        upper_bound = peak_value + threshold

        # Find the start of the peak
        start = peak
        while start > 0 and bar_press_forces[start] >= lower_bound:
            start -= 1
        
        # Find the end of the peak
        end = peak
        while end < len(bar_press_forces) - 1 and bar_press_forces[end] >= lower_bound:
            end += 1
        
        # Calculate duration
        duration = end - start
        peak_durations.append(duration)

    return peak_durations

# Function to calculate peak sharpness of each bar press
def calculate_peak_sharpness(x):
    return x.apply(lambda presses: [calculate_peak_sharpness_for_press(press) for press in presses])

def calculate_peak_sharpness_for_press(press):
    peaks, _ = find_peaks(press)  # Detect peaks in the press force signal
    sharpness = []
    for peak in peaks:
        if peak > 0 and peak < len(press) - 1:
            left_slope = press[peak] - press[peak - 1]  
            right_slope = press[peak] - press[peak + 1]  
            sharpness.append(abs(left_slope) + abs(right_slope))  
    return np.mean(sharpness) if sharpness else 0  

# Function to calculate skewness of each bar press
def calculate_skewness(x):
    return x.apply(lambda presses: [skew(press) for press in presses])  # Skewness for each bar press

# Function to calculate kurtosis of each bar press
def calculate_kurtosis(x):
    return x.apply(lambda presses: [kurtosis(press) for press in presses])  # Kurtosis for each bar press

# Function to calculate force variation rate (maximum rate of change) of each bar press
def calculate_force_variation_rate(x):
    return x.apply(lambda presses: [max(np.abs(np.diff(press))) for press in presses])  # Max rate of change (force variation rate)

# Function to calculate percentiles of a list
def calculate_percentiles(arr, p):
    return np.percentile(arr, p)  # Return percentile of list 

# Function to add percentile columns to the dataframe
def add_percentile_columns(df, array_column, percentiles):
    df = df.copy()
    for p in percentiles:
        df[f'{array_column}_{p}'] = df[array_column].apply(lambda x: calculate_percentiles(x, p))
    return df


# Functions to calculate the average of the first and last 5 values
def calculate_avg_first_5(x):
    return x.apply(lambda presses: [np.mean(press[:5]) if len(press) >= 5 else np.mean(press) for press in presses])  # Average of first 5 values

def calculate_avg_last_5(x):
    return x.apply(lambda presses: [np.mean(press[-5:]) if len(press) >= 5 else np.mean(press) for press in presses])  # Average of last 5 values


# Mapping of features and their corresponding functions
def get_feature_list():
    feature_list = {
        "total_press_duration": {"func": calculate_total_press_duration, "params": {}},
        "max_force": {"func": calculate_max_force, "params": {}},
        "num_of_peaks": {"func": calculate_peaks, "params": {}},
        "max_peak_duration": {"func": calculate_max_peak_duration, "params": {}},
        "skewness": {"func": calculate_skewness, "params": {}},
        "kurtosis": {"func": calculate_kurtosis, "params": {}},
        "force_variation_rate": {"func": calculate_force_variation_rate, "params": {}},
        "valley_sharpness": {"func": calculate_valley_sharpness, "params": {}},
        "peak_sharpness": {"func": calculate_peak_sharpness, "params": {}},
        "avg_first_5": {"func": calculate_avg_first_5, "params": {}},  # Added avg_first_5
        "avg_last_5": {"func": calculate_avg_last_5, "params": {}}    # Added avg_last_5
    }
    return feature_list

# Feature extraction function
def convert_to_features(df):
    chunk_size = len(df['data'][0]) 
    
    ## name and function mapping    
    feature_list = get_feature_list()

    df3 = df.copy()
    
    # Extract features for each bar press
    for feature_name, feature in feature_list.items():
        df3[feature_name] = feature["func"](df3['data'], **feature["params"])
        
    if chunk_size == 1:
        # If chunk_size == 1, we only need the first value for each list (single bar press)
        for feature_name in feature_list.keys():
            df3[feature_name] = df3[feature_name].apply(lambda x: x[0])
        
        transformed_feature = [feature_name for feature_name in feature_list.keys()]
    else:
        # If chunk_size > 1, we need to turn the lists into features
        for feature_name in feature_list.keys():
            df3 = add_percentile_columns(df3, feature_name, [25, 50, 75])
            df3[f'range_{feature_name}'] = abs(df3[f'{feature_name}_25'] - df3[f'{feature_name}_75'])
        
        transformed_feature = [f'{feature_name}_{p}' for feature_name in feature_list.keys() for p in [25, 50, 75]] + [f'range_{feature_name}' for feature_name in feature_list.keys()]
        
    features = ['sex'] + transformed_feature
        
    # Select features and the target
    X = df3[features]
    return X
