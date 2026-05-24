import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import re
from sklearn.ensemble import RandomForestClassifier
from modules.data_management import manage_data
from modules.nn_models import GRUModel
from modules.read_data import read_data
from modules.feature_extraction import extract_barpress_features

# === Parameters ===
truncate_size = 3
chunk_size = 1
max_press = 1000
standardize = False
window_size = 5

# === Load full dataset ===
df_all = read_data('data/01 Sucrose FR1 vs EXT 8_2024')

# === NEW: Exclude test subject from training ===
file_for_test = "13h36m.Subject 15F"
df_all_exclude_text = df_all[~df_all['file'].str.contains(file_for_test, na=False)]
df_all_processed = manage_data(df_all_exclude_text, truncate_size, chunk_size, max_press, standardize)
X_all, y_all = extract_barpress_features(df_all_processed)

# === Map labels: EXT = 0 (frustration), FR1 = 1 ===
label_map = {'EXT': 0, 'FR1': 1}
y_all = [label_map[label] for label in y_all]

# === Train Random Forest ===
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_all, y_all)

# === Preprocess Subject 10M for testing ===
df_subject = df_all[df_all['file'].str.contains(file_for_test, na=False)]
df_subject_processed = manage_data(df_subject, truncate_size, chunk_size, max_press, standardize)
X_subject_rf, _ = extract_barpress_features(df_subject_processed)

# === Predict EXT probabilities using RF (class 0 = EXT) ===
rf_probs = rf.predict_proba(X_subject_rf)[:, 0]
rf_rolling = np.convolve(rf_probs, np.ones(window_size)/window_size, mode='valid')

# === Load GRU model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=32, num_layers=3, batch_first=True)
        self.fc = nn.Linear(32, 2)
    def forward(self, x, lengths):
        x = x.unsqueeze(-1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return self.fc(h[-1])

model = GRUModel().to(device)

# === Auto-load best GRU model ===
model_dir = "output/gru"
pattern = r"best_model_epoch_(\d+)_loss_([\d\.]+)\.pth"
best_loss = float("inf")
best_model_file = None

for filename in os.listdir(model_dir):
    match = re.match(pattern, filename)
    if match:
        loss = float(match.group(2))
        if loss < best_loss:
            best_loss = loss
            best_model_file = filename

model_path = os.path.join(model_dir, best_model_file)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Predict with GRU ===
gru_probs = []
sequences = df_subject_processed['data'].tolist()
with torch.no_grad():
    for seq in sequences:
        seq = np.array(seq, dtype=np.float32).flatten()
        if len(seq) == 0:
            continue
        if len(seq) < 100:
            seq = np.pad(seq, (0, 100 - len(seq)))
        else:
            seq = seq[:100]
        tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
        length = torch.tensor([min(len(seq), 100)])
        logits = model(tensor, length)
        prob = torch.softmax(logits, dim=1)[0, 0].item()  # EXT class = 0
        gru_probs.append(prob)

gru_rolling = np.convolve(gru_probs, np.ones(window_size)/window_size, mode='valid')

# === Plot Combined ===
plt.figure(figsize=(10, 6))
plt.scatter(range(len(gru_probs)), gru_probs, color='red', alpha=0.5, label='GRU Raw Probability')
plt.scatter(range(len(rf_probs)), rf_probs, color='blue', alpha=0.5, label='RF Raw Probability')
plt.plot(range(window_size-1, len(gru_probs)), gru_rolling, color='darkred', linewidth=2.5, label='GRU Rolling Avg')
plt.plot(range(window_size-1, len(rf_probs)), rf_rolling, color='darkblue', linewidth=2.5, label='RF Rolling Avg')

plt.title("Non-Frustrated - Subject 15F")
plt.xlabel("Bar Press Sequences")
plt.ylabel("Probability of Frustration")
plt.yticks(np.arange(0.0, 1.1, 0.2))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("output/results/FR1_Subject15F_GRU_vs_RF.png", dpi=300)
plt.show()
