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
X, y = convert_to_features(df2, chunk_size)  # X contains the features (including skewness and kurtosis), y contains the target

# Access the features DataFrame (X)
df3 = X  # X contains the feature data after conversion

# Sort the kurtosis values in ascending order
sorted_kurtosis = df3['kurtosis'].sort_values()

# Get the top 5 kurtosis values (highest)
top_5_kurtosis = sorted_kurtosis.tail(5)  # Get the last 5 entries from sorted kurtosis (highest values)

# Get the bottom 5 kurtosis values (lowest)
bottom_5_kurtosis = sorted_kurtosis.head(5)  # Get the first 5 entries from sorted kurtosis (lowest values)

# Create the line plot for kurtosis
plt.figure(figsize=(10, 6))

# Plot bottom 5 kurtosis values
plt.plot(bottom_5_kurtosis.index, bottom_5_kurtosis.values, marker='o', color='green', label='Bottom 5 Kurtosis')

# Plot top 5 kurtosis values
plt.plot(top_5_kurtosis.index, top_5_kurtosis.values, marker='o', color='purple', label='Top 5 Kurtosis')

# Adding titles and labels for kurtosis plot
plt.title('Top 5 and Bottom 5 Kurtosis for Bar Presses')
plt.xlabel('Bar Press Index')
plt.ylabel('Kurtosis Value')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
