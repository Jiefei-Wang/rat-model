from sklearn.metrics import f1_score, recall_score, precision_score,roc_auc_score
import numpy as np

def calc_metrics(y_true, y_prob, sample_weight=None):
    thresholds = np.unique(y_prob)
    best_threshold = 0.5
    best_f1 = -1.0

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred_t, zero_division=0, sample_weight=sample_weight)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)

    y_pred_t = (y_prob >= best_threshold).astype(int)
    f1 = f1_score(y_true, y_pred_t, zero_division=0, sample_weight=sample_weight)
    recall = recall_score(y_true, y_pred_t, zero_division=0, sample_weight=sample_weight)
    precision = precision_score(y_true, y_pred_t, zero_division=0, sample_weight=sample_weight)
    auc = roc_auc_score(y_true, y_prob, sample_weight=sample_weight)

    return {
        "threshold": round(best_threshold, 3),
        "f1_score": round(f1, 3),
        "recall": round(recall, 3),
        "precision": round(precision, 3),
        "auc": round(auc, 3),
    }