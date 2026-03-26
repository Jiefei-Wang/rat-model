import os
import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
)


# Load Data
output_base = "output/data"

df_ML_train = pickle.load(open(os.path.join(output_base, "df_ML_train.pkl"), "rb"))
df_ML_test  = pickle.load(open(os.path.join(output_base, "df_ML_test.pkl"), "rb"))
feature_names = pickle.load(open(os.path.join(output_base, "feature_names.pkl"), "rb"))

# Ensure consistent label coding (match your Logistic script)
for d in (df_ML_train, df_ML_test):
    d["category"] = d["category"].astype("category")
    d["category"] = d["category"].cat.reorder_categories(["FR1", "EXT"], ordered=True)
    d["label"] = d["category"].cat.codes

X_train = df_ML_train[feature_names].values
y_train = df_ML_train["label"].values

X_test = df_ML_test[feature_names].values
y_test = df_ML_test["label"].values


# Train GB model 
model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=3,
    subsample=0.8,
    min_samples_split=2,
    min_samples_leaf=3,
    random_state=42
)

model.fit(X_train, y_train)


# Predict probabilities on TEST
y_prob_test = model.predict_proba(X_test)[:, 1]


# Find F1-optimal threshold on TEST (test-tuned)
thresholds = np.unique(y_prob_test)

best_threshold = 0.5
best_f1 = -1.0

for t in thresholds:
    y_pred_t = (y_prob_test >= t).astype(int)
    f1 = f1_score(y_test, y_pred_t, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = float(t)

print(f"\nGB F1-Optimal Threshold (TEST-tuned): {best_threshold:.4f}")
print(f"Best TEST F1 at that threshold: {best_f1:.4f}")

# Apply chosen threshold
y_pred_test = (y_prob_test >= best_threshold).astype(int)

# Compute Test Metrics (thresholded metrics are test-tuned)
metrics = {
    "AUROC": roc_auc_score(y_test, y_prob_test),
    "Accuracy (test-tuned)": accuracy_score(y_test, y_pred_test),
    "Precision (test-tuned)": precision_score(y_test, y_pred_test, zero_division=0),
    "Recall (test-tuned)": recall_score(y_test, y_pred_test, zero_division=0),
    "F1 (test-tuned)": f1_score(y_test, y_pred_test, zero_division=0),
    "Optimal Threshold (max F1 on test)": best_threshold
}

metrics_df = pd.DataFrame(metrics, index=["Gradient Boosting"]).T

print("\n=== Test metrics (TEST-tuned threshold) ===")
print(metrics_df)

# Save metrics
os.makedirs("output/evaluation", exist_ok=True)
metrics_df.to_csv("output/evaluation/GB_test_performance_F1_TEST_TUNED.csv")

# Save test probabilities for correlation/scatter
prob_save_path = "output/evaluation/test_model_probabilities.csv"

if os.path.exists(prob_save_path):
    prob_df = pd.read_csv(prob_save_path)
else:
    prob_df = pd.DataFrame({"true_label": y_test})

# Safety checks
if len(prob_df) != len(y_test):
    raise ValueError("Mismatch in test set size when saving probabilities.")

if "true_label" in prob_df.columns:
    if not np.array_equal(prob_df["true_label"].to_numpy(), y_test):
        raise ValueError("true_label in saved file does not match current y_test ordering.")

prob_df["GB_prob"] = y_prob_test
prob_df.to_csv(prob_save_path, index=False)

print(f"\nSaved probabilities to: {prob_save_path}")