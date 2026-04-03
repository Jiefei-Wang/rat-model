import os
import pickle
import wandb
from types import SimpleNamespace
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GroupKFold

wandb.init()

num_folds = 10
output_base = "output/data"

wandb_config = {}
wandb_config = dict(wandb.config)
wandb_config["n_estimators"] = wandb_config.get("n_estimators", 100)
wandb_config["max_depth"] = wandb_config.get("max_depth", None)
wandb_config["min_samples_split"] = wandb_config.get("min_samples_split", 2)
wandb_config["min_samples_leaf"] = wandb_config.get("min_samples_leaf", 1)
cfg = SimpleNamespace(**wandb_config)

df_ML_train = pickle.load(open(os.path.join(output_base, "df_ML_train.pkl"), "rb"))
feature_names = pickle.load(open(os.path.join(output_base, "feature_names.pkl"), "rb"))

x_train = df_ML_train.loc[:, feature_names].values
y_train = df_ML_train.loc[:, "category"].values

model = RandomForestClassifier(
    n_estimators=int(cfg.n_estimators),
    max_depth=None if cfg.max_depth is None else int(cfg.max_depth),
    min_samples_split=int(cfg.min_samples_split),
    min_samples_leaf=int(cfg.min_samples_leaf),
    random_state=42,
)

groups = df_ML_train["id"].values
kf = GroupKFold(n_splits=num_folds, shuffle=True, random_state=42)
cross_val_results = cross_val_score(model, x_train, y_train, cv=kf, groups=groups)

mean_auc = float(cross_val_results.mean())

wandb.log({"val_auc": mean_auc, "wandb_config": wandb_config})
