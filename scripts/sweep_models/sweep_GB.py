import os, pickle,wandb
from tracemalloc import start
from types import SimpleNamespace
from sklearn.model_selection import cross_val_score, GroupKFold
import xgboost as xgb

wandb.init()

num_folds = 10
output_base = 'output/data' 

wandb_config = dict(wandb.config)

# defaults for XGBClassifier sweep params
wandb_config["n_estimators"] = wandb_config.get("n_estimators", 100)
wandb_config["max_depth"] = wandb_config.get("max_depth", 3)
wandb_config["min_child_weight"] = wandb_config.get("min_child_weight", 1)
wandb_config["subsample"] = wandb_config.get("subsample", 1.0)
wandb_config["colsample_bytree"] = wandb_config.get("colsample_bytree", 1.0)
wandb_config["reg_lambda"] = wandb_config.get("reg_lambda", 1.0)

cfg = SimpleNamespace(**wandb_config)




df_ML_train = pickle.load(open(os.path.join(output_base, "df_ML_train.pkl"), "rb"))
feature_names = pickle.load(open(os.path.join(output_base, "feature_names.pkl"), "rb"))


x_train = df_ML_train.loc[:, feature_names].values
y_train = df_ML_train.loc[:, 'label'].values


model = xgb.XGBClassifier(
    n_estimators=int(cfg.n_estimators),
    max_depth=int(cfg.max_depth),
    min_child_weight=float(cfg.min_child_weight),
    subsample=float(cfg.subsample),
    colsample_bytree=float(cfg.colsample_bytree),
    reg_lambda=float(cfg.reg_lambda),
    tree_method="hist",
    random_state=42
)

    
groups = df_ML_train['id'].values
kf = GroupKFold(n_splits=num_folds, shuffle=True, random_state=42)  

cross_val_results = cross_val_score(
    model,
    x_train,
    y_train,
    cv=kf,
    groups=groups,
    scoring="roc_auc",
)
mean_auc = float(cross_val_results.mean())

wandb.log({"val_auc": mean_auc, 'wandb_config': wandb_config})


