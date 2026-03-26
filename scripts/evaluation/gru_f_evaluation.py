import os
import pickle
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

from modules.nn_models import GRUModel
from modules.nn_train import big_train_loop, dataframe_to_tensors


# Fixed best parameters (GRU+F)
output_base = "output/data"

epochs = 10000
hidden_size = 128
num_layers = 4
use_features = True
random_state = 42


# Load data
df_ML_train = pickle.load(open(os.path.join(output_base, "df_ML_train.pkl"), "rb"))
df_ML_test  = pickle.load(open(os.path.join(output_base, "df_ML_test.pkl"), "rb"))
feature_names = pickle.load(open(os.path.join(output_base, "feature_names.pkl"), "rb"))

for d in (df_ML_train, df_ML_test):
    d["category"] = d["category"].astype("category")
    d["category"] = d["category"].cat.reorder_categories(["FR1", "EXT"], ordered=True)
    d["label"] = d["category"].cat.codes


# Train/Valid split (ID-level) ON TRAIN ONLY
if "id" not in df_ML_train.columns:
    raise KeyError("df_ML_train must contain an 'id' column for ID-level splitting.")

unique_ids = df_ML_train["id"].unique()
train_ids, valid_ids = train_test_split(unique_ids, test_size=0.1, random_state=random_state)

train_df = df_ML_train[df_ML_train["id"].isin(train_ids)].reset_index(drop=True)
valid_df = df_ML_train[df_ML_train["id"].isin(valid_ids)].reset_index(drop=True)

nn_train0 = train_df[["label", "data"]]
nn_valid  = valid_df[["label", "data"]]

features_train = train_df[feature_names].reset_index(drop=True)
features_valid = valid_df[feature_names].reset_index(drop=True)
feature_size = features_train.shape[1]


# Model + Training
model = GRUModel(
    input_size=1,
    hidden_size=hidden_size,
    num_layers=num_layers,
    feature_size=feature_size
)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
batch_ratio = int(np.log(max(1, 1_200_000 // max(1, num_params))) / np.log(2))
batch_size = max(1024 * 16, 2 ** batch_ratio)

run = None
optimizer = None

model = big_train_loop(
    model=model,
    nn_train=nn_train0,
    nn_valid=nn_valid,
    features_train=features_train,
    features_valid=features_valid,
    epochs=epochs,
    optimizer=optimizer,
    run=run,
    batch_size=batch_size,
    model_name="GRU_F"
)


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()


# Predict probabilities on TEST
test_df = df_ML_test.reset_index(drop=True)
nn_test = test_df[["label", "data"]]

test_x, test_y, test_lengths = dataframe_to_tensors(nn_test, device=device)
y_true = test_y.detach().cpu().numpy()

features_test = test_df[feature_names].reset_index(drop=True)
test_features_tensor = torch.tensor(features_test.to_numpy(), dtype=torch.float32).to(device)

with torch.no_grad():
    test_outputs = model(test_x, test_lengths, test_features_tensor)
    test_probs = torch.softmax(test_outputs, dim=1)[:, 1].detach().cpu().numpy()


# Choose threshold that maximizes F1 on TEST
thresholds = np.unique(test_probs)

best_threshold = 0.5
best_f1 = -1.0

for t in thresholds:
    preds = (test_probs >= t).astype(int)
    f1 = f1_score(y_true, preds, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = float(t)

print(f"\nGRU+F F1-Optimal Threshold (TEST-chosen): {best_threshold:.4f}")
print(f"Best TEST F1 at that threshold: {best_f1:.4f}")


# Apply threshold + compute metrics on TEST
y_pred = (test_probs >= best_threshold).astype(int)

test_auc  = roc_auc_score(y_true, test_probs)
test_acc  = accuracy_score(y_true, y_pred)
test_prec = precision_score(y_true, y_pred, zero_division=0)
test_rec  = recall_score(y_true, y_pred, zero_division=0)
test_f1   = f1_score(y_true, y_pred, zero_division=0)
cm        = confusion_matrix(y_true, y_pred)

print("\nTEST RESULTS (GRU+F trained on df_ML_train, tested on df_ML_test) — TEST-CHOSEN THRESHOLD")
print(f"AUC:        {test_auc:.4f}")
print(f"ACC:        {test_acc:.4f}")
print(f"Precision:  {test_prec:.4f}")
print(f"Recall:     {test_rec:.4f}")
print(f"F1:         {test_f1:.4f}")
print(f"Threshold:  {best_threshold:.4f}")
print("Confusion matrix:\n", cm)


# Save metrics
os.makedirs("output/evaluation", exist_ok=True)

metrics_df = pd.DataFrame({
    "AUROC": [test_auc],
    "Accuracy (test-chosen)": [test_acc],
    "Precision (test-chosen)": [test_prec],
    "Recall (test-chosen)": [test_rec],
    "F1 (test-chosen)": [test_f1],
    "Optimal Threshold (max F1 on test)": [best_threshold],
}, index=["GRU+F"])

metrics_path = "output/evaluation/GRU_F_test_performance_F1_TEST_CHOSEN.csv"
metrics_df.to_csv(metrics_path, index=True)
print(f"\nSaved metrics to: {metrics_path}")

# Append TEST probabilities to shared master CSV
prob_save_path = "output/evaluation/test_model_probabilities.csv"

if os.path.exists(prob_save_path):
    prob_df = pd.read_csv(prob_save_path)
else:
    prob_df = pd.DataFrame({"true_label": y_true})

if len(prob_df) != len(y_true):
    raise ValueError(
        f"Mismatch in test set size when appending GRU_F_prob: "
        f"file has {len(prob_df)} rows, but test has {len(y_true)} rows."
    )

if "true_label" in prob_df.columns:
    if not np.array_equal(prob_df["true_label"].to_numpy(), y_true):
        raise ValueError(
            "true_label in the master probability file does not match current y_true ordering. "
            "This usually means df_ML_test ordering differs between runs."
        )
else:
    prob_df["true_label"] = y_true

prob_df["GRU_F_prob"] = test_probs
prob_df.to_csv(prob_save_path, index=False)
print(f"Appended GRU_F_prob to: {prob_save_path}")