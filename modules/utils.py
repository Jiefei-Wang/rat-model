import torch
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curve(train_losses, valid_losses, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='#0077B6', linewidth=2)
    plt.plot(valid_losses, label='Validation Loss', color='#F77F00', linewidth=2)
    min_val_epoch = np.argmin(valid_losses)
    plt.axvline(min_val_epoch, color='gray', linestyle='--', linewidth=1)
    plt.text(min_val_epoch + 1, valid_losses[min_val_epoch] + 0.01,
             f"Early stop? Epoch {min_val_epoch + 1}", color='gray')
    plt.title("Learning Curves", fontsize=18, fontweight='bold')
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curve.png'), dpi=300)
    plt.show()