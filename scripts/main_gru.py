import torch
import os
import wandb
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from modules.models import GRUModel
from modules.data_management import manage_data
from modules.read_data import read_data
from modules.dataset import PressDataset, collate_fn
from modules.utils import save_model, plot_learning_curve
from modules.train import mytrain, myvalidate

# Step 1: Load and preprocess data
df_train = read_data('data/01 Sucrose FR1 vs EXT 8_2024')

# Parameters for data processing
truncate_size = 3
chunk_size = 1
max_press = 1000
standardize = False

df2 = manage_data(df_train, truncate_size, chunk_size, max_press, standardize)

# Prepare dataset (convert categorical labels to numeric)
df3 = df2[['category', 'data']].copy().explode('data')
df3['data'] = df3['data'].apply(lambda x: np.array(x, dtype=np.float32) if isinstance(x, list) else x)
df3['category'] = df3['category'].astype('category')
df3['label'] = df3['category'].cat.codes

# Step 2: Create train-validation-test split using df3 index
row_trainval, row_test = train_test_split(df3.index, test_size=0.10, stratify=df3['category'], random_state=42)
row_train, row_val = train_test_split(row_trainval, test_size=0.05, stratify=df3.loc[row_trainval, 'category'], random_state=42)

df_train = df3.loc[row_train]
df_val = df3.loc[row_val]
df_test = df3.loc[row_test]

# Create datasets
train_dataset = PressDataset(df_train)
valid_dataset = PressDataset(df_val)
test_dataset = PressDataset(df_test)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

# Step 3: Initialize model, loss function, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRUModel(input_size=1, hidden_size=32, num_layers=3).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Step 4: wandb initialization
wandb.init(
    project="rat-frustration-gru",
    name="gru-model-run",
    config={
        "model": "GRU",
        "input_size": 1,
        "hidden_size": 32,
        "num_layers": 3,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "loss_fn": "CrossEntropyLoss",
        "batch_size": 64,
        "epochs": 100,
        "truncate_size": truncate_size,
        "chunk_size": chunk_size,
        "max_press": max_press,
        "standardize": standardize
    }
)

# Step 5: Training and Validation Loop
epochs = 100
train_losses = []
valid_losses = []
best_valid_loss = float('inf')
output_dir = "output/gru"
os.makedirs(output_dir, exist_ok=True)

for epoch in range(epochs):
    train_loss = mytrain(model, train_loader, criterion, optimizer, device)
    valid_loss = myvalidate(model, valid_loader, criterion, device)

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    wandb.log({
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "epoch": epoch + 1
    })

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        save_model(model, epoch, best_valid_loss, output_dir)
        print(f"Saved best model at epoch {epoch+1}")

    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

# Step 6: Plot learning curves
plot_learning_curve(train_losses, valid_losses, output_dir)

# Step 7: Evaluate the model on the test set & calculate AUC score
model.eval()
y_true, y_scores = [], []

with torch.no_grad():
    for batch_x, batch_y, lengths in test_loader:
        batch_x, batch_y, lengths = batch_x.to(device), batch_y.to(device), lengths.to(device)
        logits = model(batch_x, lengths)
        probs = torch.softmax(logits, dim=1)[:, 1]
        y_scores.extend(probs.cpu().numpy())
        y_true.extend(batch_y.cpu().numpy())

auc = roc_auc_score(y_true, y_scores)
print(f"\nAUC Score on Test Set: {auc:.4f}")
wandb.log({"AUC": auc})

# Step 8: Visualizations (ROC Curve, Confusion Matrix)
# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_scores)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
plt.show()

# Confusion Matrix
y_pred = [1 if score >= 0.5 else 0 for score in y_scores]
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['FR1', 'EXT'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.show()

