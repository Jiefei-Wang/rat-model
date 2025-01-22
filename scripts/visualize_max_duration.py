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
X, y = convert_to_features(df2, chunk_size)  # X contains the features (including max_duration), y contains the target

# Access the features DataFrame (X)
df3 = X  # X contains the feature data after conversion

# Sort the max duration values in ascending order
sorted_max_duration = df3['max_duration'].sort_values()

# Get the top 5 max duration values (highest)
top_5_max_duration = sorted_max_duration.tail(5)  # Get the last 5 entries from sorted max_duration (highest values)

# Get the bottom 5 max duration values (lowest)
bottom_5_max_duration = sorted_max_duration.head(5)  # Get the first 5 entries from sorted max_duration (lowest values)

# Create the line plot for max duration
plt.figure(figsize=(10, 6))

# Plot bottom 5 max duration values
plt.plot(bottom_5_max_duration.index, bottom_5_max_duration.values, marker='o', color='red', label='Bottom 5 Max Duration')

# Plot top 5 max duration values
plt.plot(top_5_max_duration.index, top_5_max_duration.values, marker='o', color='blue', label='Top 5 Max Duration')

# Adding titles and labels for the max duration plot
plt.title('Top 5 and Bottom 5 Max Duration for Bar Presses')
plt.xlabel('Bar Press Index')
plt.ylabel('Max Duration')  # Update with appropriate units (e.g., seconds or ms)
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
