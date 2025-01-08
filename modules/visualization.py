import os
import matplotlib.pyplot as plt


def make_boxplot(image_dir, df, prob_cols, model_titles, by_col):
    os.makedirs(image_dir, exist_ok=True)
    rats = df['id'].unique()
    for rat in rats:
        rat_data = df[df['id'] == rat]
        make_single_boxplot(rat_data, prob_cols, model_titles, by_col)
        # Adjust the overall title and layout
        plt.suptitle(f'Box Plots for Rat {rat}')
        
        # Save the figure
        plt.savefig(f"{image_dir}/rat_{rat}.png")
        plt.close()
    
def make_single_boxplot(df, prob_cols, model_titles, by_col):
    # Group by 'file' and create boxplots for each group
    fig, ax = plt.subplots(1, len(prob_cols), figsize=(15, 5))
    
    for i, prob_col in enumerate(prob_cols):
        df.boxplot(column=prob_col, by=by_col, ax=ax[i])
        ax[i].set_title(model_titles[i])
        ax[i].set_xlabel('category')
        ax[i].set_ylabel('Probability')
    
    # Adjust the overall title and layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])



def make_bar_press_plot(output_folder, dt, max_press_num = None):
    os.makedirs(output_folder, exist_ok=True)
    files = dt['file'].unique()
    for file in files:
        rat = dt[dt['file'] == file].copy()
        ## combine all barpresses into one list
        barpresses = [j for i in rat['data'] for j in i ]
        if max_press_num:
            barpresses = barpresses[0:max_press_num]
        ## find out the boundary of each barpress
        boundaries = [0]
        for i in barpresses:
            boundaries.append(len(i)+boundaries[-1])
        labels = [f'{i+1}' for i in range(len(boundaries))]
        barpress = [j for i in barpresses for j in i]
        plt.plot(barpress)
        ## add tick on x axis for each bar press
        plt.xticks(boundaries, labels)
        plt.savefig(f'{output_folder}/{file}.png')
        plt.close()

