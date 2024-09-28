from scipy.signal import find_peaks
import numpy as np

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


