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
X, y = convert_to_features(df2, chunk_size)  # X contains the features (including force_variation_rate), y contains the target

# Access the features DataFrame (X)
df3 = X  # X contains the feature data after conversion

# Sort the force variation rate values in ascending order
sorted_force_variation_rate = df3['force_variation_rate'].sort_values()

# Get the top 5 force variation rate values (highest)
top_5_force_variation_rate = sorted_force_variation_rate.tail(5)  # Get the last 5 entries from sorted force_variation_rate (highest values)

# Get the bottom 5 force variation rate values (lowest)
bottom_5_force_variation_rate = sorted_force_variation_rate.head(5)  # Get the first 5 entries from sorted force_variation_rate (lowest values)

# Create the line plot for force variation rate
plt.figure(figsize=(10, 6))

# Plot bottom 5 force variation rate values
plt.plot(bottom_5_force_variation_rate.index, bottom_5_force_variation_rate.values, marker='o', color='red', label='Bottom 5 Force Variation Rate')

# Plot top 5 force variation rate values
plt.plot(top_5_force_variation_rate.index, top_5_force_variation_rate.values, marker='o', color='blue', label='Top 5 Force Variation Rate')

# Adding titles and labels for the force variation rate plot
plt.title('Top 5 and Bottom 5 Force Variation Rate for Bar Presses')
plt.xlabel('Bar Press Index')
plt.ylabel('Force Variation Rate')  # Update with appropriate units (grams per second or other)
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
