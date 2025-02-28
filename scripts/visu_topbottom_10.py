import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from modules.read_data import read_data

# Load the dataset
df_train = read_data('data/01 Sucrose FR1 vs EXT 8_2024')

# List to hold all features for each bar press across all subjects
bar_press_features = []

# Loop through all subjects in df_train
for idx, rat in df_train.iterrows():
    bar_press_index = rat['bar_press_index']
    subject_id = rat['file']  # Identifies the subject
    raw_data = rat['raw_data']  # Raw data for the subject

    previous_end = 0  # Initialize the end of the previous peak for overlap checking

    for i, (start, end) in enumerate(bar_press_index):
        # Define the start and end of the peak with a 2-unit margin before and after
        peak_start = start + 2
        peak_end = end + 2
        
        # Extract the raw data for the current peak
        peak_data = raw_data[peak_start:peak_end]
        
        # Calculate Total Press Duration (duration of peak)
        total_press_duration = peak_end - peak_start
        
        # Calculate Max Force (maximum value within the peak data)
        max_force = np.max(peak_data)
        
        # Calculate Number of Peaks (number of local maxima within the peak data)
        num_of_peaks = len([i for i in range(1, len(peak_data)-1) if peak_data[i-1] < peak_data[i] > peak_data[i+1]])
        
        # Calculate Valley Sharpness (min value - mean value)
        valley_sharpness = np.min(peak_data) - np.mean(peak_data)
        
        # Calculate Kurtosis (measure of peak sharpness)
        kurtosis = stats.kurtosis(peak_data)
        
        # Calculate Skewness (measure of asymmetry)
        skewness = stats.skew(peak_data)
        
        # Calculate Force Variation Rate (rate of change of force within the peak)
        force_variation_rate = np.std(np.diff(peak_data))  # Standard deviation of the difference between consecutive values
        
        # Calculate Peak Duration (duration of the peak)
        peak_duration = peak_end - peak_start
        
        # Append the feature values along with the subject id, bar press index, and start-end times
        bar_press_features.append((subject_id, i+1, total_press_duration, max_force, num_of_peaks, skewness, kurtosis, force_variation_rate, valley_sharpness, peak_duration, peak_start, peak_end))
        
        # Update the end of the current peak for the next iteration
        previous_end = peak_end

# Sort the bar press features by each feature (ascending order) to extract top and bottom 10
sorted_by_total_press_duration = sorted(bar_press_features, key=lambda x: x[2])  # Sort by total press duration
sorted_by_max_force = sorted(bar_press_features, key=lambda x: x[3])  # Sort by max force
sorted_by_num_of_peaks = sorted(bar_press_features, key=lambda x: x[4])  # Sort by number of peaks
sorted_by_skewness = sorted(bar_press_features, key=lambda x: x[5])  # Sort by skewness
sorted_by_kurtosis = sorted(bar_press_features, key=lambda x: x[6])  # Sort by kurtosis
sorted_by_force_variation_rate = sorted(bar_press_features, key=lambda x: x[7])  # Sort by force variation rate
sorted_by_valley_sharpness = sorted(bar_press_features, key=lambda x: x[8])  # Sort by valley sharpness
sorted_by_peak_duration = sorted(bar_press_features, key=lambda x: x[9])  # Sort by peak duration

# Function to print the top 10 and bottom 10 for each feature
def print_feature_info(feature_name, top_10, bottom_10):
    print(f"Top 10 {feature_name}:")
    for subject, press, *features in top_10:
        print(f"Subject {subject} - Bar Press {press}: {feature_name}={features[0]}, Start={features[1]}, End={features[2]}")
    
    print(f"\nBottom 10 {feature_name}:")
    for subject, press, *features in bottom_10:
        print(f"Subject {subject} - Bar Press {press}: {feature_name}={features[0]}, Start={features[1]}, End={features[2]}")

# Print feature information
print_feature_info("Total Press Duration", sorted_by_total_press_duration[-10:], sorted_by_total_press_duration[:10])
print_feature_info("Max Force", sorted_by_max_force[-10:], sorted_by_max_force[:10])
print_feature_info("Number of Peaks", sorted_by_num_of_peaks[-10:], sorted_by_num_of_peaks[:10])
print_feature_info("Skewness", sorted_by_skewness[-10:], sorted_by_skewness[:10])
print_feature_info("Kurtosis", sorted_by_kurtosis[-10:], sorted_by_kurtosis[:10])
print_feature_info("Force Variation Rate", sorted_by_force_variation_rate[-10:], sorted_by_force_variation_rate[:10])
print_feature_info("Valley Sharpness", sorted_by_valley_sharpness[-10:], sorted_by_valley_sharpness[:10])
print_feature_info("Peak Duration", sorted_by_peak_duration[-10:], sorted_by_peak_duration[:10])

# Function to plot the raw data for the selected bar presses and features
def plot_feature_data(df, selected_bars, title, feature_name):
    for subject, press, total_press_duration, max_force, num_of_peaks, skewness, kurtosis, force_variation_rate, valley_sharpness, peak_duration, start, end in selected_bars:
        # Locate the raw data for the corresponding subject
        subject_data = df[df['file'] == subject].iloc[0]['raw_data']
        y = subject_data[start:end]
        
        # Plot the raw data
        plt.figure(figsize=(10, 6))
        plt.plot(y)
        plt.title(f'{title} - Subject {subject} Bar Press {press} ({feature_name})')
        
        # Add vertical lines to indicate the start and end of the peak
        plt.axvline(0, color='r', linestyle='--', label="Start")
        plt.axvline(len(y), color='r', linestyle='--', label="End")
        
        plt.xlabel('Samples')
        plt.ylabel('Raw Data')
        plt.grid(True)
        plt.legend()
        
        # Show the plot
        plt.show()

# Plot the data for the top and bottom 10 peaks of each feature
plot_feature_data(df_train, sorted_by_total_press_duration[-10:], "Top 10 Total Press Duration", "Total Press Duration")
plot_feature_data(df_train, sorted_by_total_press_duration[:10], "Bottom 10 Total Press Duration", "Total Press Duration")

plot_feature_data(df_train, sorted_by_max_force[-10:], "Top 10 Max Force", "Max Force")
plot_feature_data(df_train, sorted_by_max_force[:10], "Bottom 10 Max Force", "Max Force")

plot_feature_data(df_train, sorted_by_num_of_peaks[-10:], "Top 10 Number of Peaks", "Number of Peaks")
plot_feature_data(df_train, sorted_by_num_of_peaks[:10], "Bottom 10 Number of Peaks", "Number of Peaks")

plot_feature_data(df_train, sorted_by_skewness[-10:], "Top 10 Skewness", "Skewness")
plot_feature_data(df_train, sorted_by_skewness[:10], "Bottom 10 Skewness", "Skewness")

plot_feature_data(df_train, sorted_by_kurtosis[-10:], "Top 10 Kurtosis", "Kurtosis")
plot_feature_data(df_train, sorted_by_kurtosis[:10], "Bottom 10 Kurtosis", "Kurtosis")

plot_feature_data(df_train, sorted_by_force_variation_rate[-10:], "Top 10 Force Variation Rate", "Force Variation Rate")
plot_feature_data(df_train, sorted_by_force_variation_rate[:10], "Bottom 10 Force Variation Rate", "Force Variation Rate")

plot_feature_data(df_train, sorted_by_valley_sharpness[-10:], "Top 10 Valley Sharpness", "Valley Sharpness")
plot_feature_data(df_train, sorted_by_valley_sharpness[:10], "Bottom 10 Valley Sharpness", "Valley Sharpness")

plot_feature_data(df_train, sorted_by_peak_duration[-10:], "Top 10 Peak Duration", "Peak Duration")
plot_feature_data(df_train, sorted_by_peak_duration[:10], "Bottom 10 Peak Duration", "Peak Duration")

