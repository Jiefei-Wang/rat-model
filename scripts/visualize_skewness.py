import matplotlib.pyplot as plt
from modules.read_data import read_data
from modules.data_management import manage_data
from modules.feature_extraction import convert_to_features

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
X, y = convert_to_features(df2, chunk_size)  # X contains the features (including skewness), y contains the target

# Access the skewness feature from the X DataFrame
df3 = X  # X contains the feature data after conversion

# Sort the skewness values in ascending order
sorted_skewness = df3['skewness'].sort_values()


# Get the top 5 and bottom 5 skewness values
top_5_skewness = sorted_skewness.tail(5)
bottom_5_skewness = sorted_skewness.head(5)

# Create the line plot
plt.figure(figsize=(10, 6))

# Plot bottom 5 skewness values
plt.plot(bottom_5_skewness.index, bottom_5_skewness.values, marker='o', color='red', label='Bottom 5')

# Plot top 5 skewness values
plt.plot(top_5_skewness.index, top_5_skewness.values, marker='o', color='blue', label='Top 5')

# Adding titles and labels
plt.title('Top 5 and Bottom 5 Skewness for Bar Presses')
plt.xlabel('Bar Press Index')
plt.ylabel('Skewness Value')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
