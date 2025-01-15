from scipy.signal import find_peaks
import numpy as np
from scipy.stats import skew, kurtosis

# Feature extraction function
def convert_to_features(df, chunk_size):
    df3 = df.copy()
    
    # Extract features for each bar press (duration, max force, num of peaks, max duration, skewness, kurtosis, sharpness)
    df3['duration'] = calculate_duration(df3['data'])  # Duration of each bar press
    df3['max_force'] = calculate_max_force(df3['data'])  # Max force of each bar press
    df3['num_of_peaks'] = calculate_peaks(df3['data'])  # Number of peaks for each bar press
    df3['max_duration'] = calculate_max_duration(df3['data'])  # Max duration of each peak for each bar press
    df3['skewness'] = calculate_skewness(df3['data'])  # Skewness for each bar press
    df3['kurtosis'] = calculate_kurtosis(df3['data'])  # Kurtosis for each bar press
    df3['sharpness'] = calculate_sharpness(df3['data'])  # Sharpness for each bar press
    
    # If chunk_size == 1, we only need the first value for each list (single bar press)
    if chunk_size == 1:
        df3['duration'] = df3['duration'].apply(lambda x: x[0])
        df3['max_force'] = df3['max_force'].apply(lambda x: x[0])
        df3['num_of_peaks'] = df3['num_of_peaks'].apply(lambda x: x[0])
        df3['max_duration'] = df3['max_duration'].apply(lambda x: x[0])
        df3['skewness'] = df3['skewness'].apply(lambda x: x[0])
        df3['kurtosis'] = df3['kurtosis'].apply(lambda x: x[0])
        df3['sharpness'] = df3['sharpness'].apply(lambda x: x[0])
        features = ['sex', 'duration', 'max_force', 'num_of_peaks', 'max_duration', 'skewness', 'kurtosis', 'sharpness']
        
    else:
        # Calculate percentiles for each feature across all bar presses
        df3 = add_percentile_columns(df3, 'duration', [25, 50, 75])
        df3 = add_percentile_columns(df3, 'max_force', [25, 50, 75])
        df3 = add_percentile_columns(df3, 'num_of_peaks', [25, 50, 75])
        df3 = add_percentile_columns(df3, 'max_duration', [25, 50, 75])
        df3 = add_percentile_columns(df3, 'skewness', [25, 50, 75])
        df3 = add_percentile_columns(df3, 'kurtosis', [25, 50, 75])
        df3 = add_percentile_columns(df3, 'sharpness', [25, 50, 75])
        
        # Calculate ranges (difference between 75th and 25th percentiles)
        df3['range_duration'] = abs(df3['duration_25'] - df3['duration_75'])
        df3['range_force'] = abs(df3['max_force_25'] - df3['max_force_75'])
        df3['range_peaks'] = abs(df3['num_of_peaks_25'] - df3['num_of_peaks_75'])
        df3['range_max_duration'] = abs(df3['max_duration_25'] - df3['max_duration_75'])
        df3['range_skewness'] = abs(df3['skewness_25'] - df3['skewness_75'])
        df3['range_kurtosis'] = abs(df3['kurtosis_25'] - df3['kurtosis_75'])
        df3['range_sharpness'] = abs(df3['sharpness_25'] - df3['sharpness_75'])
        
        features = [
            'sex',
            'duration_25', 'duration_50', 'duration_75', 
            'max_force_25', 'max_force_50', 'max_force_75', 
            'num_of_peaks_25', 'num_of_peaks_50', 'num_of_peaks_75', 
            'max_duration_25', 'max_duration_50', 'max_duration_75', 
            'skewness_25', 'skewness_50', 'skewness_75', 
            'kurtosis_25', 'kurtosis_50', 'kurtosis_75',
            'sharpness_25', 'sharpness_50', 'sharpness_75',
            'range_duration', 'range_force', 'range_peaks', 'range_max_duration',
            'range_skewness', 'range_kurtosis', 'range_sharpness'
        ]
        
    # Select features and the target
    X = df3[features]
    y = df3['category']
    return X, y


# Function to calculate the duration of each bar press
def calculate_duration(x): 
    return x.apply(lambda presses: [len(press) for press in presses])  # Duration of each bar press

# Function to calculate the max force of each bar press
def calculate_max_force(x): 
    return x.apply(lambda presses: [max(press) for press in presses])  # Max force of each bar press

# Function to calculate the number of peaks in each bar press
def calculate_peaks(x): 
    return x.apply(lambda presses: [len(find_peaks(press)[0]) for press in presses])  # Number of peaks for each bar press

# Function to calculate the max duration of peaks in each bar press
def calculate_max_duration(x):
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

# Function to calculate skewness of each bar press
def calculate_skewness(x):
    return x.apply(lambda presses: [skew(press) for press in presses])  # Skewness for each bar press

# Function to calculate kurtosis of each bar press
def calculate_kurtosis(x):
    return x.apply(lambda presses: [kurtosis(press) for press in presses])  # Kurtosis for each bar press

# Function to calculate sharpness (maximum rate of change) of each bar press
def calculate_sharpness(x):
    return x.apply(lambda presses: [max(np.abs(np.diff(press))) for press in presses])  # Max rate of change (sharpness)

# Function to calculate percentiles of a list
def calculate_percentiles(arr, p):
    return np.percentile(arr, p)  # Return percentile of list 

# Function to add percentile columns to the dataframe
def add_percentile_columns(df, array_column, percentiles):
    df = df.copy()
    for p in percentiles:
        df[f'{array_column}_{p}'] = df[array_column].apply(lambda x: calculate_percentiles(x, p))
    return df
