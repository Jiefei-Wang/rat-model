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
X, y = convert_to_features(df2, chunk_size)  # X contains the features (including valley_sharpness), y contains the target

# Access the features DataFrame (X)
df3 = X  # X contains the feature data after conversion

# Sort the valley sharpness values in ascending order
sorted_valley_sharpness = df3['valley_sharpness'].sort_values()

# Get the top 5 valley sharpness values (highest)
top_5_valley_sharpness = sorted_valley_sharpness.tail(5)  # Get the last 5 entries from sorted valley_sharpness (highest values)

# Get the bottom 5 valley sharpness values (lowest)
bottom_5_valley_sharpness = sorted_valley_sharpness.head(5)  # Get the first 5 entries from sorted valley_sharpness (lowest values)

# Create the line plot for valley sharpness
plt.figure(figsize=(10, 6))

# Plot bottom 5 valley sharpness values
plt.plot(bottom_5_valley_sharpness.index, bottom_5_valley_sharpness.values, marker='o', color='red', label='Bottom 5 Valley Sharpness')

# Plot top 5 valley sharpness values
plt.plot(top_5_valley_sharpness.index, top_5_valley_sharpness.values, marker='o', color='blue', label='Top 5 Valley Sharpness')

# Adding titles and labels for the valley sharpness plot
plt.title('Top 5 and Bottom 5 Valley Sharpness for Bar Presses')
plt.xlabel('Bar Press Index')
plt.ylabel('Valley Sharpness')  # No specific units for valley sharpness, but you can add one if relevant
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
