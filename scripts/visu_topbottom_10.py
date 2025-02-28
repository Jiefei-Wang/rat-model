import os
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from modules.read_data import read_data

# Define output path
output_path = 'output'

# Load the dataset
df_train = read_data('data/01 Sucrose FR1 vs EXT 8_2024')

# List to hold all features for each bar press across all subjects
bar_press_features = []

# Set padding values (number of samples before and after bar press event)
padding_before = 50
padding_after = 50

# Loop through all subjects in df_train
for idx, rat in df_train.iterrows():
    bar_press_index = rat['bar_press_index']
    subject_id = rat['file']  # Identifies the subject
    raw_data = rat['raw_data']  # Raw data for the subject

    for i, (start, end) in enumerate(bar_press_index): #to observe peaks more closely
        peak_start = start + 2
        peak_end = end + 2

        peak_data = raw_data[peak_start:peak_end]
        total_press_duration = peak_end - peak_start
        max_force = np.max(peak_data)
        num_of_peaks = len([j for j in range(1, len(peak_data)-1) if peak_data[j-1] < peak_data[j] > peak_data[j+1]])
        valley_sharpness = np.min(peak_data) - np.mean(peak_data)
        kurtosis = stats.kurtosis(peak_data)
        skewness = stats.skew(peak_data)
        force_variation_rate = np.std(np.diff(peak_data))
        max_peak_duration = sum(1 for j in range(1, len(peak_data)) if peak_data[j] > peak_data[j-1])
        peak_sharpness = np.mean([
            abs(peak_data[j] - peak_data[j-1]) + abs(peak_data[j] - peak_data[j+1]) 
            for j in range(1, len(peak_data)-1) if peak_data[j-1] < peak_data[j] > peak_data[j+1]
        ])

        bar_press_features.append((subject_id, i+1, total_press_duration, max_force, num_of_peaks, skewness, kurtosis, 
                                   force_variation_rate, valley_sharpness, max_peak_duration, peak_sharpness, peak_start, peak_end))

# Sort the features by each metric
features = {
    "Total Press Duration": sorted(bar_press_features, key=lambda x: x[2]),
    "Max Force": sorted(bar_press_features, key=lambda x: x[3]),
    "Number of Peaks": sorted(bar_press_features, key=lambda x: x[4]),
    "Skewness": sorted(bar_press_features, key=lambda x: x[5]),
    "Kurtosis": sorted(bar_press_features, key=lambda x: x[6]),
    "Force Variation Rate": sorted(bar_press_features, key=lambda x: x[7]),
    "Valley Sharpness": sorted(bar_press_features, key=lambda x: x[8]),
    "Max Peak Duration": sorted(bar_press_features, key=lambda x: x[9]),
    "Peak Sharpness": sorted(bar_press_features, key=lambda x: x[10])
}

# Function to save plots with extra padding
def save_plot_with_padding(raw_data, subject, press, title, feature_name, rank, category, start, end):
    # Apply padding
    start_with_padding = max(start - padding_before, 0)
    end_with_padding = min(end + padding_after, len(raw_data))

    # Get the data with padding
    y_with_padding = raw_data[start_with_padding:end_with_padding]
    
    feature_folder = os.path.join(output_path, feature_name.replace(" ", "_"))
    os.makedirs(os.path.join(feature_folder, category), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(y_with_padding)
    plt.title(f'{title} - Subject {subject} Bar Press {press} ({feature_name})')
    plt.axvline(padding_before, color='r', linestyle='--', label="Start")
    plt.axvline(len(y_with_padding) - padding_after, color='r', linestyle='--', label="End")
    plt.xlabel('Samples')
    plt.ylabel('Raw Data')
    plt.grid(True)
    plt.legend()
    
    save_path = os.path.join(feature_folder, category, f"{rank}_subject_{subject}_press_{press}.png")
    plt.savefig(save_path)
    plt.close()

# Generate and save plots for the top and bottom 10 of each feature
for feature_name, sorted_data in features.items():
    top_10 = sorted_data[-10:]
    bottom_10 = sorted_data[:10]

    for rank, (subject, press, *_, start, end) in enumerate(top_10, 1):
        subject_data = df_train[df_train['file'] == subject].iloc[0]['raw_data']
        save_plot_with_padding(subject_data, subject, press, "Top 10", feature_name, rank, "Top_10", start, end)

    for rank, (subject, press, *_, start, end) in enumerate(bottom_10, 1):
        subject_data = df_train[df_train['file'] == subject].iloc[0]['raw_data']
        save_plot_with_padding(subject_data, subject, press, "Bottom 10", feature_name, rank, "Bottom_10", start, end)

print("Plots have been saved successfully.")
