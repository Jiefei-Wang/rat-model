import pandas as pd
from modules.read_data import read_data

df_train = read_data('data/01 Sucrose FR1 vs EXT 8_2024')

# Function to check if the rat's bar press data has never reached 20
def has_never_reached_20(bar_press_data):
    # Check if any sequence of bar presses has a value >= 20
    for press_sequence in bar_press_data:
        if max(press_sequence) >= 20:  # If the max value in the sequence is >= 20, it's not problematic
            return False
    return True  # If no sequence has reached 20, it's problematic

# Creates a list to store problematic rats
problematic_rats = []

# Iterate over each row in the DataFrame
for idx, row in df_train.iterrows():
    # This checks if this rat's bar press data has never reached 20
    if has_never_reached_20(row['data']):
        problematic_rats.append(row)

# Creates a new DataFrame with problematic rats
df_problematic = pd.DataFrame(problematic_rats)

# Saves to the existing output folder
output_path = 'output\\problematic_rats.xlsx'

# Saves the DataFrame with problematic rats to an Excel file in the output folder
df_problematic.to_excel(output_path, index=False) 
##Note:An empty excel sheet indicates all rats have at least one sequence where the bar press value has reached 20 or higher.


