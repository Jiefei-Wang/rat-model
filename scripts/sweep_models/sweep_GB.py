import os, pickle,wandb
from types import SimpleNamespace
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.ensemble import GradientBoostingClassifier

wandb.init()


num_folds = 10
output_base = 'output/data' 

wandb_config = {}
wandb_config = dict(wandb.config)
wandb_config['n_estimators'] = wandb_config.get('n_estimators', 100)
wandb_config['subsample'] = wandb_config.get('subsample', 1.0)
wandb_config['max_depth'] = wandb_config.get('max_depth', 3)
wandb_config['min_samples_split'] = wandb_config.get('min_samples_split', 2)
wandb_config['min_samples_leaf'] = wandb_config.get('min_samples_leaf', 1)
cfg = SimpleNamespace(**wandb_config)



df_ML_train = pickle.load(open(os.path.join(output_base, "df_ML_train.pkl"), "rb"))
feature_names = pickle.load(open(os.path.join(output_base, "feature_names.pkl"), "rb"))


x_train = df_ML_train.loc[:, feature_names].values
y_train = df_ML_train.loc[:, 'category'].values


model = GradientBoostingClassifier(
    n_estimators=int(cfg.n_estimators), 
    subsample=float(cfg.subsample), 
    max_depth=int(cfg.max_depth), 
    min_samples_split=int(cfg.min_samples_split), 
    min_samples_leaf=int(cfg.min_samples_leaf))
    
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


