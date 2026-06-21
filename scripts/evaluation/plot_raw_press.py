
import pickle
import pandas as pd
import os

# plot raw press data for all rats
with open(f'output/data/df_app.pkl', 'rb') as f:
    df_app = pickle.load(f)
    
output_dir = 'output/plots/raw_press/c3_pr'
os.makedirs(output_dir, exist_ok=True)


import matplotlib.pyplot as plt
for idx in range(len(df_app)):
    row = df_app.iloc[idx]
    data = row['raw_data']
    data_index = row['data_index']
    plt.figure(figsize=(50, 4))
    plt.plot(data)
    
    # color range of bar press using data_index
    #  [(0, 66), (318, 402), (1056, 1217)]
    for start, end in data_index:
        plt.axvspan(start, end, color='red', alpha=0.3)
    
    plt.title(f"Raw Press Data - {row['id']}")
    plt.xlabel("Time")
    plt.ylabel("Force")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{row['file_name']}.png", dpi=300)
    plt.close()