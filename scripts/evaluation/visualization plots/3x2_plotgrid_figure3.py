#NOTE: Run all plot subject scripts before running this script to ensure the images are generated.
# This script combines the results of different subjects into a 3x2 grid plot.
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#File paths
non_frustrated_files = [
    "output/results/FR1_Subject10M_GRU_vs_RF.png",
    "output/results/FR1_Subject15F_GRU_vs_RF.png",
    "output/results/FR1_Subject22F_GRU_vs_RF.png"
]

frustrated_files = [
    "output/results/EXT_Subject10M_GRU_vs_RF.png",
    "output/results/EXT_Subject15F_GRU_vs_RF.png",
    "output/results/EXT_Subject22F_GRU_vs_RF.png"
]

#Combine files row-wise
all_files = [non_frustrated_files, frustrated_files]

#Plot Grid
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))

for row_idx, row in enumerate(all_files):
    for col_idx, file_path in enumerate(row):
        ax = axes[row_idx, col_idx]
        img = mpimg.imread(file_path)
        ax.imshow(img)
        ax.axis('off')

#Add row labels
axes[0, 1].set_title("Non-Frustrated Session", fontsize=16, pad=20)
axes[1, 1].set_title("Frustrated Session", fontsize=16, pad=20)

plt.tight_layout()
plt.savefig("output/results/Combined_3x2_Subjects_Categorized.png", dpi=300)
plt.show()
