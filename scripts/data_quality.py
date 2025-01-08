from modules.read_data import read_data 

# Read the data
df_train = read_data('data/01 Sucrose FR1 vs EXT 8_2024')

# Function to filter out values between 5 and 20, ensuring no peak above 20 after encountering values in that range
def filter_peaks_in_range(barpress_data):
    """
    This function processes a list of bar press data, ensuring that:
    - No value between 5 and 20 causes any values to peak above 20.
    - Any value above 20 following a value between 5 and 20 is removed.
    """
    # Track if we've encountered a value between 5 and 20
    found_peak_range = False
    filtered_data = []

    for value in barpress_data:
        if 5 <= value <= 20:
            # When a value between 5 and 20 is found, it remains in the data
            filtered_data.append(value)
            found_peak_range = True
        elif found_peak_range and value > 20:
            # If a peak above 20 occurs after a value between 5 and 20, it's removed
            continue
        else:
            # Otherwise, just keep the value
            filtered_data.append(value)

    return filtered_data

# Apply the function to each list of barpress data in the 'data' column
df_train['filtered_data'] = df_train['data'].apply(filter_peaks_in_range)

# Print the filtered data
print(df_train['filtered_data'])

# Save to Excel in the 'output' folder
output_path = 'output/df_train_filtered.xlsx'
df_train.to_excel(output_path, index=False)
