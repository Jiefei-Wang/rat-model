# We assume the chunk size is 1 in this script!!

# scripts/sweep/peak_param_sweep.py
import os, pickle, argparse
from types import SimpleNamespace
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import wandb

from modules.feature_extraction import calculate_peak_vally_features

wandb.init()

num_folds = 10
output_base = 'output/data' 
wandb_config = {
    'prominence': 3.0,
    'height': 5.0,
    'distance': 5,
    'width': None,
    'wlen': None,
    'threshold': None,
    'rel_height': None,
    'plateau_size': None
}

wandb_config = dict(wandb.config)
cfg = SimpleNamespace(**wandb_config)


peak_train = pickle.load(open(os.path.join(output_base, "peak_train.pkl"), "rb"))



y_all = peak_train['label'].values
barpresses = peak_train["data"] 
sample_weights = peak_train['sample_weight'].values

features_list = [calculate_peak_vally_features(cell, wandb_config) for cell in barpresses]

features_list = pd.DataFrame(features_list)

scaler = StandardScaler()
scaler.fit(features_list) 
features_list = scaler.transform(features_list)

# k-fold cross validation
gb = xgb.XGBClassifier(
    tree_method="hist",
    random_state=42
)
groups = peak_train['id'].values
kf = GroupKFold(n_splits=num_folds, shuffle=True, random_state=42)  
cross_val_results = cross_val_score(
    gb,
    features_list,
    y_all,
    cv=kf,
    groups=groups,
    scoring="roc_auc",
    params={"sample_weight": sample_weights}
)
mean_auc = float(cross_val_results.mean())
wandb.log({"val_auc": mean_auc, 'wandb_config': wandb_config})

