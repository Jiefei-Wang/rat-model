import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

# Visualization Functions for Top 10 and Bottom 10
def plot_top_bottom(df, feature, title):
    """Plot top 10 and bottom 10 for a given feature."""
    # Get top 10 and bottom 10 for the feature
    top_10 = df.nlargest(10, feature)
    bottom_10 = df.nsmallest(10, feature)

    # Combine the top and bottom data for visualization
    top_bottom = pd.concat([top_10, bottom_10])

    # Create bar plot for top 10 and bottom 10 entries
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_bottom.index, y=top_bottom[feature], palette="viridis")
    plt.title(f"{title} (Top 10 and Bottom 10)", fontsize=16)
    plt.xlabel('Index', fontsize=14)
    plt.ylabel(feature, fontsize=14)
    plt.xticks(rotation=90)
    plt.show()


def generate_visualizations(df):
    """Generate bar graphs for top 10 and bottom 10 entries of duration, max_force, and num_of_peaks."""
    # Plot for 'duration'
    plot_top_bottom(df, 'duration', 'Duration')

    # Plot for 'max_force'
    plot_top_bottom(df, 'max_force', 'Max Force')

    # Plot for 'num_of_peaks'
    plot_top_bottom(df, 'num_of_peaks', 'Number of Peaks')


# Boxplot Functions
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


# Bar Press Plot Functions
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



def plot_squareish_heatmap(data, cmap='YlOrRd', title="Data", ax=None, legend = True):
    """
    Plot a heatmap from a 1D list of values, arranging them in a grid
    that is as close to square as possible. Allows plotting on a 
    specified Axes object to facilitate subplot layouts.

    Parameters
    ----------
    data : list or 1D array-like
        The 1D data to visualize.
    cmap : str, optional
        Colormap to use for the heatmap (default is 'YlOrRd').
    title : str, optional
        Title for the heatmap. Default is 'Data'.
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot. If None, uses the current axes (gca).

    Returns
    -------
    im : matplotlib.image.AxesImage
        The image object created by imshow (useful for colorbar, etc.).
    ax : matplotlib.axes.Axes
        The axes on which the heatmap was drawn.
    """
    if ax is None:
        ax = plt.gca()  # get current active Axes

    n = len(data)
    if n == 0:
        print("No data provided. Nothing to plot.")
        return None, ax
    
    # 1) Determine the grid shape to be as square as possible:
    nrows = int(math.floor(math.sqrt(n)))
    ncols = int(math.ceil(n / nrows))
    
    # 2) Pad data if necessary so that the grid is fully rectangular
    needed_length = nrows * ncols
    padded_data = list(data)  # Make a copy as a list
    if needed_length > n:
        padded_data.extend([0] * (needed_length - n))
    
    # 3) Reshape into 2D (row-major order)
    matrix = []
    idx = 0
    for _ in range(nrows):
        row = padded_data[idx : idx + ncols]
        matrix.append(row)
        idx += ncols
    
    # 4) Plot the matrix as a heatmap on the provided axes
    im = ax.imshow(matrix, aspect='auto', cmap=cmap)
    
    # 5) Add colorbar tied to this image
    if legend:
        plt.colorbar(im, ax=ax, label="Value")
    
    # 6) Add labels and title
    ax.set_title(f"{title} Heatmap\n"
                 "Time proceeds left to right, then top to bottom")
    ax.set_xlabel("Column Index")
    ax.set_ylabel("Row Index")

    return im, ax







# Example usage:
# Assuming `df` is your DataFrame containing the data column with the bar press force data as lists
# and `df_features` is the DataFrame with calculated features like 'duration', 'max_force', 'num_of_peaks'

# Convert the data to features (duration, max_force, num_of_peaks, max_duration)
# df_features = convert_to_features(df, chunk_size=1)

# Generate the visualizations for the top 10 and bottom 10 of the selected features
# generate_visualizations(df_features)

# Boxplot example (save boxplots for each rat to a folder)
# make_boxplot("boxplots_output", df, prob_cols=['duration', 'max_force', 'num_of_peaks'], model_titles=["Duration", "Max Force", "Num of Peaks"], by_col='category')

# Bar press plot example (save bar press plots to a folder)
# make_bar_press_plot("barpress_plots", df, max_press_num=100)
