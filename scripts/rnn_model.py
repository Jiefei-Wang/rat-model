import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import os
from modules.data_management import manage_data
from modules.read_data import read_data

# Step 1: Load and preprocess data
df_train = read_data('data/01 Sucrose FR1 vs EXT 8_2024')
truncate_size = 3
chunk_size = 1
max_press = 1000
standardize = False

df2 = manage_data(df_train, truncate_size, chunk_size, max_press, standardize)

df3 = df2[['category', 'data']].copy().explode('data')
df3['data'] = df3['data'].apply(lambda x: np.array(x, dtype=np.float32) if isinstance(x, list) else x)
df3['category'] = df3['category'].astype('category')
df3['label'] = df3['category'].cat.codes

# Step 2: Custom Dataset
class PressDataset(Dataset):
    def __init__(self, dataframe, max_length=100):
        self.data = dataframe['data'].tolist()
        self.labels = dataframe['label'].tolist()
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        label = self.labels[idx]
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        else:
            sequence = np.pad(sequence, (0, self.max_length - len(sequence)))
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded_seqs = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    return padded_seqs, torch.tensor(labels), lengths

# Step 3: RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=10, num_layers=1, num_classes=2):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        x = x.unsqueeze(-1)
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(packed_input)
        out = self.fc(hidden[-1])
        return out

# Step 4: Data Splits and Loaders
dataset = PressDataset(df3)
train_valid_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.1, random_state=42)
train_idx, valid_idx = train_test_split(train_valid_idx, test_size=0.05, random_state=42)

train_data = torch.utils.data.Subset(dataset, train_idx)
valid_data = torch.utils.data.Subset(dataset, valid_idx)
test_data = torch.utils.data.Subset(dataset, test_idx)

generator = torch.Generator(device='cpu')
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate_fn, generator=generator)
valid_loader = DataLoader(valid_data, batch_size=64, shuffle=False, collate_fn=collate_fn, generator=generator)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=collate_fn, generator=generator)

# Step 5: Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNNModel(hidden_size=10, num_layers=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Step 6: Training Functions
def mytrain(model, train_loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0
    n_samples = 0
    for batch_x, batch_y, lengths in train_loader:
        batch_x, batch_y, lengths = batch_x.to(device), batch_y.to(device), lengths.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x, lengths)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        n_samples += batch_x.size(0)
        epoch_loss += loss.item() * batch_x.size(0)
    return epoch_loss / n_samples

def myvalidate(model, valid_loader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    n_samples = 0
    with torch.no_grad():
        for batch_x, batch_y, lengths in valid_loader:
            batch_x, batch_y, lengths = batch_x.to(device), batch_y.to(device), lengths.to(device)
            outputs = model(batch_x, lengths)
            loss = criterion(outputs, batch_y)
            n_samples += batch_x.size(0)
            epoch_loss += loss.item() * batch_x.size(0)
    return epoch_loss / n_samples

# Step 7: Training Loop with Loss Tracking
epochs = 100
train_losses = []
valid_losses = []

os.makedirs("output/rnn", exist_ok=True)

for epoch in range(epochs):
    train_loss = mytrain(model, train_loader, criterion, optimizer, device)
    valid_loss = myvalidate(model, valid_loader, criterion, device)
    torch.save(model.state_dict(), f'output/rnn/rnn_model_{epoch}.pth')
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

# Step 8: Learning Curve Plot
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='training', color='#0077B6', linewidth=2)
plt.plot(valid_losses, label='validation', color='#F77F00', linewidth=2)

min_val_epoch = np.argmin(valid_losses)
plt.axvline(min_val_epoch, color='gray', linestyle='--', linewidth=1)
plt.text(min_val_epoch + 1, valid_losses[min_val_epoch] + 0.01,
         f"Early stop? Epoch {min_val_epoch+1}", color='gray')

plt.title("The Learning Curves", fontsize=18, fontweight='bold')
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("output/rnn/learning_curves.png", dpi=300)
plt.show()

# Step 9: Evaluation on Test Set
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

# Step 10: Visualizations
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
plt.show()

# Confusion Matrix
y_pred = [1 if score >= 0.5 else 0 for score in y_scores]
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['FR1', 'EXT'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
