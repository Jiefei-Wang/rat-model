import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from modules.read_data import read_data
from modules.data_management import manage_data
from modules.feature_extraction import convert_to_features
import shutil

# Define output path
output_path = 'output/features'

# Load the dataset
df_train = read_data('data/01 Sucrose FR1 vs EXT 8_2024')

## clean the data
truncate_size = 3
chunk_size = 1
max_press = 1000
standardize = False
df2 = manage_data(df_train, 
                  truncate_size=truncate_size,
                  chunk_size=chunk_size, 
                  max_press=max_press,
                  standardize=standardize)

df2.columns
# ['id', 'category', 'file', 'sex', 'data', 'data_index', 'raw_data',
#        'low_force_mask', 'raw_bar_press', 'raw_bar_press_mask',
#        'raw_bar_press_index', 'constant_value_masks']

## turn data into features
X, y = convert_to_features(df2)

## combine df2 and X
df3 = pd.concat([df2, X], axis=1)

# Set padding values (number of samples before and after bar press event)
padding_before = 50
padding_after = 50

df3.columns
# ['id', 'category', 'file', 'sex', 'data', 'data_index', 'raw_data',
#        'low_force_mask', 'raw_bar_press', 'raw_bar_press_mask',
#        'raw_bar_press_index', 'constant_value_masks', 'sex',
#        'total_press_duration', 'max_force', 'num_of_peaks',
#        'max_peak_duration', 'skewness', 'kurtosis', 'force_variation_rate',
#        'valley_sharpness', 'peak_sharpness', 'avg_first_5', 'avg_last_5']

# Sort the features by each metric (including the new ones)
features = {
    "Total Press Duration": df3.sort_values(by="total_press_duration"),
    "Max Force": df3.sort_values(by="max_force"),
    "Number of Peaks": df3.sort_values(by="num_of_peaks"),
    "Skewness": df3.sort_values(by="skewness"),
    "Kurtosis": df3.sort_values(by="kurtosis"),
    "Force Variation Rate": df3.sort_values(by="force_variation_rate"),
    "Valley Sharpness": df3.sort_values(by="valley_sharpness"),
    "Max Peak Duration": df3.sort_values(by="max_peak_duration"),
    "Peak Sharpness": df3.sort_values(by="peak_sharpness"),
    "Average First 5": df3.sort_values(by="avg_first_5"),  # Added avg_first_5
    "Average Last 5": df3.sort_values(by="avg_last_5")    # Added avg_last_5
}

# Function to save plots with extra padding
def save_plot_with_padding(output_dir, rank, row, feature_name, title):
    index = row['index']
    subject = row['id']
    raw_data = row['raw_data']
    start = row['data_index'][0][0]
    end = row['data_index'][0][1]
    
    # Apply padding
    start_with_padding = max(start - padding_before, 0)
    end_with_padding = min(end + padding_after, len(raw_data))
    
    press_start_index = start - start_with_padding
    press_end_index = end - start_with_padding 

    # Get the data with padding
    y_with_padding = raw_data[start_with_padding:end_with_padding]
    
    plt.figure(figsize=(10, 6))
    plt.plot(y_with_padding)
    plt.title(f'{title} - Subject {subject}_index_{index} ({feature_name})')
    plt.axvline(press_start_index, color='r', linestyle='--', label="Start")
    plt.axvline(press_end_index, color='r', linestyle='--', label="End")
    plt.xlabel('Samples')
    plt.ylabel('Raw Data')
    plt.grid(True)
    plt.legend()
    
    save_path = os.path.join(output_dir, f"{rank}_subject_{subject}_index_{index}.png")
    plt.savefig(save_path)
    plt.close()

def save_plots(df, feature_name, title, output_dir):
    ## delete the folder if exist
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    df = df.reset_index(drop=False)
    for idx in range(len(df)):
        row = df.iloc[idx]
        save_plot_with_padding(output_dir, idx, row, feature_name, title)

output_path_base = 'output/features'

# Generate and save plots for the top and bottom 10 of each feature
for feature_name, sorted_data in features.items():
    print(f"Generating plots for {feature_name}...")
    top_10 = sorted_data.iloc[-10:]
    bottom_10 = sorted_data.iloc[:10]
    
    top10_dir = os.path.join(output_path_base, feature_name, "Top10")
    bottom10_dir = os.path.join(output_path_base, feature_name, "Bottom10")
    
    save_plots(top_10, feature_name, "Top10", top10_dir)
    save_plots(bottom_10, feature_name, "Bottom10", bottom10_dir)

print("Plots have been saved successfully.")
