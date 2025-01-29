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
figsize1_1 = (5, 5)

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
    bar_press_mask = rat['bar_press_mask']
    bar_press_index = rat['bar_press_index']
    constant_value_masks = rat['constant_value_masks']
    bar_presses_removed = rat['bar_presses_removed']
    
    bar_press_removed_mask = [False] * len(raw_data)
    for j, value_mask in zip(bar_press_index, constant_value_masks):
        if value_mask[0]:
            bar_press_removed_mask[j[0]:j[1]] = [True] * (j[1] - j[0])

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

    # filtered -> bar press
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize3_1)
    # Plot each dataset on its own axes
    plot_squareish_heatmap(raw_data2, cmap='YlOrRd', title="Constant Force Filtered Data", ax=axes[0])
    plot_squareish_heatmap(label_consecutive_trues(bar_press_mask), cmap='YlOrRd', title="Bar Press Mask", ax=axes[1], legend= False)
    plot_squareish_heatmap(bar_press_removed_mask, cmap='YlOrRd', title="Removed Bar Presses due to flat", ax=axes[2])

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'output/barpress_raw/{file}/2.low_force_filter_to_bar_press.png', dpi=300)
    plt.close()
    
    ## plot bad bar presses
    os.makedirs(f'output/barpress_raw/{file}/bad_presses', exist_ok=True)
    bad_press_masks = [k for k in constant_value_masks if k[0]]
    for count, (press, mask) in enumerate(zip(bar_presses_removed, bad_press_masks), start=1):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize1_1)
        plt.plot(press)
        plt.axvline(mask[1][0], color='r', linestyle='--')
        plt.axvline(mask[1][1], color='r', linestyle='--')
        plt.tight_layout()
        plt.savefig(f'output/barpress_raw/{file}/bad_presses/{count}.png', dpi=300)
        plt.close()
    
    
    
    


# df2 = df_train[df_train['file'] == '!2024-07-17_12h46m.Subject 12F']
# rat = df2.iloc[0]
# rat.raw_data
# bar_press_mask = rat['bar_press_mask']

# index = []
# for i in range(len(raw_data)-1):
#     if not constant_force_mask[i] and constant_force_mask[i+1]:
#         index.append(i)
#     if constant_force_mask[i] and not constant_force_mask[i+1]:
#         index.append(i)
        

# ## plot
# for i in range(0, len(index), 2):
#     start = index[i]
#     end = index[i+1]
#     y = raw_data[start-50:end+50]
#     plt.plot(y)
#     ## add vertical line to indicate the start and end of the bar press
#     plt.axvline(50, color='r', linestyle='--')
#     plt.axvline(50+end-start, color='r', linestyle='--')
#     ## save to output/temp
#     plt.savefig(f'output/temp/{i}.png', dpi=300)
#     plt.close()

    
# idx = 171011
# y = raw_data[idx-100: idx+100]

# plt.plot(y)
# plt.show()
# len(rat.data)









