import matplotlib.pyplot as plt
from modules.read_data import read_data
from modules.data_management import manage_data
import itertools


df_train = read_data('data/01 Sucrose FR1 vs EXT 8_2024')

# Flatten the nested lists in df_train['data']
df_train['data'] = df_train['data'].apply(lambda x: list(itertools.chain(*x)) if isinstance(x, list) and isinstance(x[0], list) else x)

#make each bar press from df_train a single row
df_train = df_train.explode('data')



# Extract sequence lengths from df_train['data'] - Raw data
sequence_lengths = df_train['data'].apply(len)

# Plot histogram
plt.figure(figsize=(10, 5))
plt.hist(sequence_lengths, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel("Sequence Length (Bar Presses)")
plt.ylabel("Frequency")
plt.title("Distribution of Bar Press Sequence Lengths")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

df_train = read_data('data/01 Sucrose FR1 vs EXT 8_2024')
truncate_size = 3
chunk_size = 1
max_press = 1000
standardize = False
df2 = manage_data(df_train, 
                  truncate_size=truncate_size,
                  chunk_size=chunk_size, 
                  max_press=max_press,
                  standardize=standardize)

# Modify df2['data'] to remove nested lists
df2['data'] = df2['data'].apply(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)

# Extract sequence lengths from df2['data']
sequence_lengths = df2['data'].apply(len)

# Plot histogram
plt.figure(figsize=(10, 5))
plt.hist(sequence_lengths, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel("Sequence Length (Bar Presses)")
plt.ylabel("Frequency")
plt.title("Distribution of Bar Press Sequence Lengths (After Data Management)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



print(df_train['data'].apply(type).value_counts())  # Ensure all entries are lists
print(df_train['data'].apply(len).describe())  # Get summary stats for sequence lengths
