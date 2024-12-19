from scipy.signal import find_peaks
import numpy as np

## features: duration, max_force, num_of_peaks, max_duration
def convert_to_features(df, chunk_size):
    df3=df.copy()
    df3['duration'] = calculate_duration(df3['data'])
    df3['max_force'] = calculate_max_force(df3['data'])
    df3['num_of_peaks'] = calculate_peaks(df3['data'])
    df3['max_duration'] = calculate_max_duration(df3['data'])

    ## for a chunk of size 1, we need to flatten the list
    if chunk_size == 1:
        df3['duration'] = df3['duration'].apply(lambda x: x[0])
        df3['max_force'] = df3['max_force'].apply(lambda x: x[0])
        df3['num_of_peaks'] = df3['num_of_peaks'].apply(lambda x: x[0])
        df3['max_duration'] = df3['max_duration'].apply(lambda x: x[0])
        features = ['gender','duration', 'max_force', 'num_of_peaks', 'max_duration']
        
    else:
        df3 = add_percentile_columns(df3, 'duration', [25, 50, 75])
        df3 = add_percentile_columns(df3, 'max_force', [25, 50, 75])
        df3 = add_percentile_columns(df3, 'num_of_peaks', [25, 50, 75])
        df3 = add_percentile_columns(df3, 'max_duration', [25, 50, 75])
        df3['range_duration'] = abs(df3['duration_25'] - df3['duration_75'])
        df3['range_force'] = abs(df3['max_force_25'] - df3['max_force_75'])
        df3['range_peaks'] = abs(df3['num_of_peaks_25'] - df3['num_of_peaks_75'])
        df3['range_max_duration'] = abs(df3['max_duration_25'] - df3['max_duration_75'])
        
        features = [
            'gender',
            'duration_25', 'duration_50', 'duration_75', 
            'max_force_25', 'max_force_50', 'max_force_75', 
            'num_of_peaks_25', 'num_of_peaks_50', 'num_of_peaks_75', 
            'max_duration_25', 'max_duration_50', 'max_duration_75', 
            'range_duration', 'range_force', 'range_peaks', 'range_max_duration']
        
    X = df3[features]
    y = df3['category']
    return X, y



def calculate_peak_durations(bar_press_forces, threshold):
    threshold = 10
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


def calculate_duration(x): 
    return x.apply(lambda presses: [len(press) for press in presses])

def calculate_max_force(x): 
    return x.apply(lambda presses: [max(press) for press in presses])

def calculate_peaks(x): 
    return x.apply(lambda presses: [len(find_peaks(press)[0].tolist()) for press in presses])

def calculate_max_duration(x):
   return x.apply(
   lambda presses: [max(calculate_peak_durations(press, 20)) if calculate_peak_durations(press, 20) else 0 for press in presses]
)

def calculate_percentiles(arr, p):
    a = np.percentile(arr, p)
    return a.tolist()
def add_percentile_columns(df, array_column, percentiles):
    df = df.copy()
    for p in percentiles:
        df[f'{array_column}_{p}'] = df[array_column].apply(lambda x: calculate_percentiles(x, p))
    return df


