import matplotlib.pyplot as plt
from modules.read_data import read_data
from modules.data_management import manage_data
from modules.feature_extraction import convert_to_features

# Read the dataset
df_train = read_data('data/01 Sucrose FR1 vs EXT 8_2024')

# Parameters for data management
truncate_size = 3
chunk_size = 1
max_press = 80
standardize = False

# Assuming df_train is your training dataset
df2 = manage_data(df_train, 
                  truncate_size=truncate_size,
                  chunk_size=chunk_size, 
                  max_press=max_press,
                  standardize=standardize)

# Generate the features DataFrame using the convert_to_features function
X, y = convert_to_features(df2, chunk_size)  # X contains the features (including num_of_peaks), y contains the target

# Access the features DataFrame (X)
df3 = X  # X contains the feature data after conversion

# Sort the number of peaks values in ascending order
sorted_num_of_peaks = df3['num_of_peaks'].sort_values()

# Get the top 5 number of peaks values (highest)
top_5_num_of_peaks = sorted_num_of_peaks.tail(5)  # Get the last 5 entries from sorted num_of_peaks (highest values)

# Get the bottom 5 number of peaks values (lowest)
bottom_5_num_of_peaks = sorted_num_of_peaks.head(5)  # Get the first 5 entries from sorted num_of_peaks (lowest values)

# Create the line plot for number of peaks
plt.figure(figsize=(10, 6))

# Plot bottom 5 number of peaks values
plt.plot(bottom_5_num_of_peaks.index, bottom_5_num_of_peaks.values, marker='o', color='red', label='Bottom 5 Num of Peaks')

# Plot top 5 number of peaks values
plt.plot(top_5_num_of_peaks.index, top_5_num_of_peaks.values, marker='o', color='green', label='Top 5 Num of Peaks')

# Adding titles and labels for the number of peaks plot
plt.title('Top 5 and Bottom 5 Number of Peaks for Bar Presses')
plt.xlabel('Bar Press Index')
plt.ylabel('Number of Peaks')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
