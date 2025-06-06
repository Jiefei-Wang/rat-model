## NOTE: This script will only work if all 3 neural network models are trained.
#The script extracts the ROC data saved from each model's output and plots them for comparison.
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

# Define model info: file paths and colors
model_info = {
    'RNN':  ('output/rnn/RNN_roc_data.pkl',  'red'),
    'LSTM': ('output/lstm/LSTM_roc_data.pkl', 'blue'),
    'GRU':  ('output/gru/GRU_roc_data.pkl',  'green'),
}

# Set fancy plot style
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(10, 7))
plt.rcParams.update({'font.size': 14})

# Plot each ROC curve
for model_name, (pkl_path, color) in model_info.items():
    if not os.path.exists(pkl_path):
        print(f"File not found: {pkl_path}")
        continue

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    y_true = data['y_true']
    y_scores = data['y_scores']

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    model_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, color=color, linewidth=2.5,
             label=f"{model_name} (AUC = {model_auc:.3f})")

# Add reference diagonal line
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1.5)

# labels and layout
plt.xlabel("False Positive Rate", fontsize=14, weight='bold')
plt.ylabel("True Positive Rate", fontsize=14, weight='bold')
plt.title("ROC Curve Comparison of Neural Network Models", fontsize=16, weight='bold')
plt.grid(True, linestyle='--', alpha=0.6)

# Legend formatting
plt.legend(loc="lower right", fontsize=12, frameon=True, fancybox=True, framealpha=0.9)

# Save and display
plt.tight_layout()
plt.savefig("output/roc_comparison.png", dpi=300)
plt.show()
