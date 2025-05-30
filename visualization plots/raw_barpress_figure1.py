import pandas as pd
import matplotlib.pyplot as plt
from modules.read_data import read_data
from modules.data_management import manage_data

# Step 1: Load and filter Subject 10M data from session 14h40m.
df_train = read_data('data/01 Sucrose FR1 vs EXT 8_2024')
subject_df = df_train[df_train['file'].str.contains("14h40m.Subject 10M", na=False)]

# Step 2: Preprocess with given parameters
truncate_size = 3
chunk_size = 1
max_press = 1000
standardize = False
df2 = manage_data(df_train, truncate_size, chunk_size, max_press, standardize)

# Step 3: Extract raw signal and bar press regions
raw_force = subject_df.iloc[0]['raw_data']
bar_press_indices = subject_df.iloc[0]['raw_bar_press_index']

# Step 4: Skip first 3 presses, take next 5 (label them as Press 1–5)
selected_presses = bar_press_indices[3:8]

# Step 5: Plot the raw force trace
plt.figure(figsize=(8, 6))
plt.plot(raw_force, color='black', label='Raw Force')

# Highlight and label each selected bar press
for idx, (start, end) in enumerate(selected_presses, start=1):
    plt.axvspan(start, end, color='red', alpha=0.3, label='Bar Press' if idx == 1 else None)
    plt.text((start + end) // 2, max(raw_force) * 0.9, f"Press {idx}",
             ha='center', va='center', color='black', fontsize=10, fontweight='bold')

# Step 6: Add threshold line at 20g
plt.axhline(y=20, color='gray', linestyle='--', linewidth=2, label='Threshold (20g)')

# Plot formatting
plt.title("Bar Press Force Trace: A Subject's Behavioral Response", fontweight='bold', fontsize=14)
plt.xlabel("Sample Index (100 Hz)", fontweight='bold', fontsize=12)
plt.ylabel("Force (gms)", fontweight='bold', fontsize=12)
plt.legend(loc="upper right")
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("output/raw_barpress_subject10M_press1to5_threshold.png", dpi=300)
plt.show()
