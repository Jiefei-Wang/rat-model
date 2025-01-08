from modules.read_data import read_data

# Read the data
df_train = read_data('data/01 Sucrose FR1 vs EXT 8_2024')

# Function to filter out values between 5 and 20, replacing them with 0
def filter_peaks_in_range(barpress_data):
    """
    This function takes in a list of barpress data (a list of values),
    and removes any values between 5 and 20 grams by replacing them with 0.
    """
    return [value if value < 5 or value > 20 else 0 for value in barpress_data]

# Apply the function to the 'data' column, where each entry is a list of barpress data
df_train['filtered_data'] = df_train['data'].apply(lambda x: [filter_peaks_in_range(barpress) for barpress in x])

# Print the filtered data
print(df_train['filtered_data'])

# Save to Excel in the 'output' folder
output_path = 'output/df_train_filtered.xlsx'
df_train.to_excel(output_path, index=False)