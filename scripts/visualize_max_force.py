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

# Data Management for df_train
df2 = manage_data(df_train, 
                  truncate_size=truncate_size,
                  chunk_size=chunk_size, 
                  max_press=max_press,
                  standardize=standardize)

# Generate the features DataFrame using the convert_to_features function
X, y = convert_to_features(df2, chunk_size)  # X contains the features (including max force), y contains the target

# Access the features DataFrame (X)
df3 = X  # X contains the feature data after conversion

# Sort the max force values in ascending order
sorted_max_force = df3['max_force'].sort_values()

# Get the top 5 max force values (highest)
top_5_max_force = sorted_max_force.tail(5)  # Get the last 5 entries from sorted max force (highest values)

# Get the bottom 5 max force values (lowest)
bottom_5_max_force = sorted_max_force.head(5)  # Get the first 5 entries from sorted max force (lowest values)

# Create the line plot for max force in grams
plt.figure(figsize=(10, 6))

# Plot bottom 5 max force values
plt.plot(bottom_5_max_force.index, bottom_5_max_force.values, marker='o', color='red', label='Bottom 5 Max Force')

# Plot top 5 max force values
plt.plot(top_5_max_force.index, top_5_max_force.values, marker='o', color='green', label='Top 5 Max Force')

# Adding titles and labels for the max force plot
plt.title('Top 5 and Bottom 5 Max Force for Bar Presses')
plt.xlabel('Bar Press Index')
plt.ylabel('Max Force (gms)')  
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
