import numpy as np
import torch
from modules.data_management import manage_data
import matplotlib.pyplot as plt
from modules.read_data import read_data

# Load data using manage_data function to apply truncation and chunking
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

#flatten the data columns
df3=df2[['category','data' ]].copy().explode('data')

# Convert raw_data to NumPy arrays if they are still lists 
df3['data'] = df3['data'].apply(lambda x: np.array(x, dtype=np.float32) if isinstance(x, list) else x)

# Check if data is a NumPy array
print(isinstance(df3['data'].iloc[0], np.ndarray))

# Find sequence lengths without loading all into memory
sequence_lengths = df3['data'].apply(len)

# Find max sequence length without excessive memory usage
max_seq_length = sequence_lengths.max()

# Find longest sequence ID
longest_seq_idx = sequence_lengths.idxmax()
longest_seq_data = df_train.loc[longest_seq_idx, 'data']

# Print key information
print(f"Maximum sequence length: {max_seq_length}")
print(f"Longest Sequence ID: {df_train.loc[longest_seq_idx, 'id']}")
print(f"Length: {len(longest_seq_data)}")
print(f"First 50 values of longest sequence: {longest_seq_data[:50]}")

# Plot histogram of sequence lengths efficiently
plt.figure(figsize=(10, 5))
plt.hist(sequence_lengths, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(max_seq_length, color='r', linestyle='dashed', linewidth=2, label=f"Max Length: {max_seq_length}")

# Set x-axis ticks at intervals of 20
plt.xticks(np.arange(0, max(sequence_lengths) + 20, 40))

plt.xlabel("Sequence Length")
plt.ylabel("Frequency")
plt.title("Distribution of Sequence Lengths in Data")
plt.legend()
plt.show()

# Test if raw data is a NumPy array
print(isinstance(df3['data'].iloc[0], np.ndarray))
print(df3['data'].iloc[0].dtype)
df3['data']

#10 neurons, single layer, increase model size for flexibility later...
#in category - fr1 is no frustration, ext for frustration group (use fr1 as reference)

#Define the RNN Model