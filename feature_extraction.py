from scipy.signal import find_peaks
from read_data import calculate_peak_durations
import numpy as np
rolling_windows_values = 3

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

def add_percentile_columns(df, array_column, percentiles):
    def calculate_percentiles(arr, p):
        a = np.percentile(arr, p)
        return a.tolist()
    for p in percentiles:
        df[f'{array_column}_{p}'] = df[array_column].apply(lambda x: calculate_percentiles(x, p))
    return df


