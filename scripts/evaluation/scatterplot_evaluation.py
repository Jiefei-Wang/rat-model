# In order to run this script, you must have already run the evaluation script to save test probabilities.
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load saved test probabilities
prob_path = "output/evaluation/test_model_probabilities.csv"
df = pd.read_csv(prob_path)

pred_cols = [c for c in ["LG_prob", "GB_prob", "GRU_prob", "GRU_F_prob"] if c in df.columns]
print("Columns found:", list(df.columns))
print("Using probability columns:", pred_cols)

if len(pred_cols) < 2:
    raise ValueError("Need at least 2 probability columns to plot.")

# Keep label column
df_plot = df[pred_cols + ["true_label"]].dropna()

# Split classes
df_ext = df_plot[df_plot["true_label"] == 1]   # EXT = frustrated
df_fr1 = df_plot[df_plot["true_label"] == 0]   # FR1 = non-frustrated

# Compute correlations
corr = df_plot[pred_cols].corr(method="pearson")
r2 = corr ** 2

print("\n=== Pearson correlation (r) ===")
print(corr.round(4))
print("\n=== R-squared (r^2) ===")
print(r2.round(4))

os.makedirs("output/evaluation", exist_ok=True)
corr.to_csv("output/evaluation/prediction_correlations.csv")
r2.to_csv("output/evaluation/prediction_r2.csv")

# Custom scatter matrix
n = len(pred_cols)
fig, axes = plt.subplots(n, n, figsize=(9, 9))

for i in range(n):
    for j in range(n):
        ax = axes[i, j]

        if i == j:
            # Overlapping histograms split by class
            ax.hist(
                df_fr1[pred_cols[i]], bins=40,
                color="blue", edgecolor="black",
                alpha=0.5, label="FR1"
            )
            ax.hist(
                df_ext[pred_cols[i]], bins=40,
                color="red", edgecolor="black",
                alpha=0.5, label="EXT"
            )
            ax.set_xlim(0, 1)
        else:
            # EXT (red, behind)
            ax.scatter(
                df_ext[pred_cols[j]],
                df_ext[pred_cols[i]],
                color="red",
                alpha=0.4,
                s=8,
                zorder=1
            )
            # FR1 (blue, in front)
            ax.scatter(
                df_fr1[pred_cols[j]],
                df_fr1[pred_cols[i]],
                color="blue",
                alpha=0.4,
                s=8,
                zorder=2
            )
            # Decision boundary lines
            ax.axhline(0.5, color='gray', linestyle='--', linewidth=1)
            ax.axvline(0.5, color='gray', linestyle='--', linewidth=1)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        if i == n - 1:
            ax.set_xlabel(pred_cols[j])
        if j == 0:
            ax.set_ylabel(pred_cols[i])

# Legend
handles = [
    plt.Line2D([0], [0], marker='o', color='w', label='FR1', markerfacecolor='blue', markersize=6),
    plt.Line2D([0], [0], marker='o', color='w', label='EXT', markerfacecolor='red', markersize=6)
]
fig.legend(handles=handles, loc="upper right")

plt.suptitle("Scatter Plot Matrix: Test-Set Model Probabilities", y=1.02)
plt.tight_layout()

out_png = "output/evaluation/scatter_matrix_test_probabilities.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.show()
print(f"\nSaved scatter matrix to: {out_png}")