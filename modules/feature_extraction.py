## This must be set before loading scipy
## Otherwise, Ctrl+C will cause crash
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

from scipy.signal import find_peaks
import numpy as np
from scipy.stats import skew, kurtosis

## features: existing (duration, max_force, num_of_peaks, max_duration) + skewness, kurtosis, sharpness
def convert_to_features(df, chunk_size, peak_distance=3):
    df3 = df.copy()
    
    # Extract features for each bar press (duration, max force, num of peaks, skewness, kurtosis, force variation rate, valley sharpness, peak duration)
    df3['total_press_duration'] = calculate_total_press_duration(df3['data'])  # Total duration of each bar press
    df3['max_force'] = calculate_max_force(df3['data'])  # Max force of each bar press
    df3['num_of_peaks'] = calculate_num_of_peaks(df3['data'])  # Number of peaks for each bar press
    df3['skewness'] = calculate_skewness(df3['data'])  # Skewness for each bar press
    df3['kurtosis'] = calculate_kurtosis(df3['data'])  # Kurtosis for each bar press
    df3['force_variation_rate'] = calculate_force_variation_rate(df3['data'])  # Force variation rate for each bar press
    df3['valley_sharpness'] = calculate_valley_sharpness(df3['data'])  # Valley sharpness for each bar press
    df3['peak_duration'] = calculate_peak_duration(df3['data'])  # Peak duration for each bar press

    ## for a chunk of size 1, flattening the list
    if chunk_size == 1:
        # Flatten all features for chunk_size=1
        df3['total_press_duration'] = df3['total_press_duration'].apply(lambda x: x[0])
        df3['max_force'] = df3['max_force'].apply(lambda x: x[0])
        df3['num_of_peaks'] = df3['num_of_peaks'].apply(lambda x: x[0])
        df3['skewness'] = df3['skewness'].apply(lambda x: x[0])
        df3['kurtosis'] = df3['kurtosis'].apply(lambda x: x[0])
        df3['force_variation_rate'] = df3['force_variation_rate'].apply(lambda x: x[0])
        df3['valley_sharpness'] = df3['valley_sharpness'].apply(lambda x: x[0])
        df3['peak_duration'] = df3['peak_duration'].apply(lambda x: x[0])

        features = ['sex', 'total_press_duration', 'max_force', 'num_of_peaks', 'skewness', 'kurtosis', 'force_variation_rate', 'valley_sharpness', 'peak_duration']
    else:
        # Calculate percentiles for each feature across all bar presses
        df3 = add_percentile_columns(df3, 'total_press_duration', [25, 50, 75])
        df3 = add_percentile_columns(df3, 'max_force', [25, 50, 75])
        df3 = add_percentile_columns(df3, 'num_of_peaks', [25, 50, 75])
        df3 = add_percentile_columns(df3, 'skewness', [25, 50, 75])
        df3 = add_percentile_columns(df3, 'kurtosis', [25, 50, 75])
        df3 = add_percentile_columns(df3, 'force_variation_rate', [25, 50, 75])
        df3 = add_percentile_columns(df3, 'valley_sharpness', [25, 50, 75])
        df3 = add_percentile_columns(df3, 'peak_duration', [25, 50, 75])
        
        # Calculate ranges (difference between 75th and 25th percentiles)
        df3['range_total_press_duration'] = abs(df3['total_press_duration_25'] - df3['total_press_duration_75'])
        df3['range_max_force'] = abs(df3['max_force_25'] - df3['max_force_75'])
        df3['range_num_of_peaks'] = abs(df3['num_of_peaks_25'] - df3['num_of_peaks_75'])
        df3['range_skewness'] = abs(df3['skewness_25'] - df3['skewness_75'])
        df3['range_kurtosis'] = abs(df3['kurtosis_25'] - df3['kurtosis_75'])
        df3['range_force_variation_rate'] = abs(df3['force_variation_rate_25'] - df3['force_variation_rate_75'])
        df3['range_valley_sharpness'] = abs(df3['valley_sharpness_25'] - df3['valley_sharpness_75'])
        df3['range_peak_duration'] = abs(df3['peak_duration_25'] - df3['peak_duration_75'])
        
        features = [
            'sex',
            'total_press_duration_25', 'total_press_duration_50', 'total_press_duration_75', 
            'max_force_25', 'max_force_50', 'max_force_75', 
            'num_of_peaks_25', 'num_of_peaks_50', 'num_of_peaks_75', 
            'skewness_25', 'skewness_50', 'skewness_75', 
            'kurtosis_25', 'kurtosis_50', 'kurtosis_75',
            'force_variation_rate_25', 'force_variation_rate_50', 'force_variation_rate_75',
            'valley_sharpness_25', 'valley_sharpness_50', 'valley_sharpness_75',
            'peak_duration_25', 'peak_duration_50', 'peak_duration_75',
            'range_total_press_duration', 'range_max_force', 'range_num_of_peaks', 'range_skewness', 'range_kurtosis', 
            'range_force_variation_rate', 'range_valley_sharpness', 'range_peak_duration'
        ]
        
    # Select features and the target
    X = df3[features]
    y = df3['category']
    return X, y


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
def calculate_num_of_peaks(x): 
    return x.apply(lambda presses: [len(find_peaks(press)[0]) for press in presses])  # Number of peaks for each bar press

# Function to calculate peak duration (duration when force is at or near maximum)
def calculate_peak_duration(x):
    return x.apply(lambda presses: [calculate_peak_duration_for_press(press) for press in presses])  # Peak duration for each bar press

# Function to calculate the duration for each peak (Maximum Peak Duration)
def calculate_peak_duration_for_press(press):
    max_force = max(press)  # Identify maximum force applied during the press
    peak_durations = []  # List to store durations of each peak
    
    # A peak is considered when the force is at or near the maximum force
    threshold = max_force * 0.95  # Define the threshold for detecting peaks (95% of the max force)
    peak_start = None  # Variable to track the start of a peak

    # Loop through the force data to find and track all peaks
    for i, force in enumerate(press):
        if force >= threshold:  # If the force is at or near the maximum
            if peak_start is None:  # Start of a new peak
                peak_start = i
        elif peak_start is not None:  # End of a peak (force falls below the threshold)
            peak_durations.append(i - peak_start)  # Append the duration of the peak
            peak_start = None  # Reset peak start for the next peak

    # If the peak ends at the last force value, account for it
    if peak_start is not None:
        peak_durations.append(len(press) - peak_start)
    
    # If there were multiple peaks, return the maximum duration of any peak
    if peak_durations:
        return max(peak_durations)  # Return the maximum duration of any peak
    else:
        return 0  # If no peaks were detected, return 0

# Function to calculate skewness of each bar press
def calculate_skewness(x):
    return x.apply(lambda presses: [skew(press) for press in presses])  # Skewness for each bar press

# Function to calculate kurtosis of each bar press
def calculate_kurtosis(x):
    return x.apply(lambda presses: [kurtosis(press) for press in presses])  # Kurtosis for each bar press

# Function to calculate force variation rate (maximum rate of change) of each bar press
def calculate_force_variation_rate(x):
    return x.apply(lambda presses: [max(np.abs(np.diff(press))) for press in presses])  # Max rate of change (force variation rate)

# Function to calculate valley sharpness for each bar press
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

# Function to calculate percentiles of a list
def calculate_percentiles(arr, percentiles):
    return [np.percentile(arr, p) for p in percentiles]

# Function to add percentiles columns to the dataframe
def add_percentile_columns(df, column, percentiles):
    for p in percentiles:
        df[f'{column}_{p}'] = df[column].apply(lambda x: np.percentile(x, p))
    return df

