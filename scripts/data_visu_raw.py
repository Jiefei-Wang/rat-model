import os
import numpy as np
from modules.read_data import read_data
import matplotlib.pyplot as plt
from modules.visualization import plot_squareish_heatmap

df_train = read_data('data/01 Sucrose FR1 vs EXT 8_2024')

plt.rcParams['font.size'] = 4
plt.tight_layout()
figsize3_1 = (15, 5)
figsize2_1 = (10, 5)

def label_consecutive_trues(arr):
    """
    Labels consecutive runs of True in arr with 1, 2, 1, 2, ...
    Returns a new list with the same length as arr, 
    where False items become 0.
    """
    result = []
    run_count = 0
    current_label = 0
    in_run = False
    for val in arr:
        if val:  
            if not in_run:
                run_count += 1
                current_label = 1 if (run_count % 2) == 1 else 2
                in_run = True
            result.append(current_label)
        else: 
            in_run = False
            result.append(0)
    return result




for i in range(4):
    print(f"Processing {i+1}/{len(df_train)}")
    rat = df_train.iloc[i]
    file = rat['file']
    raw_data = rat['raw_data']
    low_force_mask = rat['low_force_mask']
    constant_force_mask = rat['constant_force_mask']
    bar_press_mask = rat['bar_press_mask']

    ## create folder if not exist
    os.makedirs(f'output/barpress_raw/{file}', exist_ok=True)

    ## raw -> low_force -> filtered
    raw_data2 = np.array(raw_data)
    raw_data2[low_force_mask] = 0
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize3_1)
    # Plot each dataset on its own axes
    plot_squareish_heatmap(raw_data, cmap='YlOrRd', title="Raw Data", ax=axes[0])
    plot_squareish_heatmap(low_force_mask, cmap='YlOrRd', title="Low Force Mask", ax=axes[1], legend= False)
    plot_squareish_heatmap(raw_data2, cmap='YlOrRd', title="Low Force Filtered Data", ax=axes[2])

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'output/barpress_raw/{file}/1.raw_data_to_low_force_filter.png', dpi=300)
    plt.close()

    # low force filtered -> constant force -> filtered
    raw_data3 = raw_data2.copy()
    raw_data3[constant_force_mask] = 0
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize3_1)
    # Plot each dataset on its own axes
    plot_squareish_heatmap(raw_data2, cmap='YlOrRd', title="Low Force Filtered Data", ax=axes[0])
    plot_squareish_heatmap(constant_force_mask, cmap='YlOrRd', title="Constant Force Mask", ax=axes[1], legend= False)
    plot_squareish_heatmap(raw_data3, cmap='YlOrRd', title="Constant Force Filtered Data", ax=axes[2])

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'output/barpress_raw/{file}/2.low_force_to_constant_force_filter.png', dpi=300)
    plt.close()

    # constant force filtered -> bar press
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize2_1)
    # Plot each dataset on its own axes
    plot_squareish_heatmap(raw_data3, cmap='YlOrRd', title="Constant Force Filtered Data", ax=axes[0])
    plot_squareish_heatmap(label_consecutive_trues(bar_press_mask), cmap='YlOrRd', title="Bar Press Mask", ax=axes[1], legend= False)

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'output/barpress_raw/{file}/3.constant_force_to_bar_press.png', dpi=300)
    plt.close()












