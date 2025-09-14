import os, pickle, wandb
from types import SimpleNamespace
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GroupKFold


wandb.init()


num_folds = 10
output_base = 'output/data' 

wandb_config = {}
wandb_config = dict(wandb.config)
wandb_config['C'] = wandb_config.get('C', 0)
cfg = SimpleNamespace(**wandb_config)


df_ML_train = pickle.load(open(os.path.join(output_base, "df_ML_train.pkl"), "rb"))
feature_names = pickle.load(open(os.path.join(output_base, "feature_names.pkl"), "rb"))


x_train = df_ML_train.loc[:, feature_names].values
y_train = df_ML_train.loc[:, 'category'].values


if cfg.C ==0 :
    model = LogisticRegression(max_iter=1000)
else:
    model = LogisticRegression(max_iter=1000,penalty = 'l1', C=float(cfg.C), solver='liblinear')
    
    


groups = df_ML_train['id'].values
kf = GroupKFold(n_splits=num_folds, shuffle=True, random_state=42)  
cross_val_results = cross_val_score(model, x_train, y_train, cv=kf, groups=groups)

mean_auc = float(cross_val_results.mean())

wandb.log({"val_auc": mean_auc, 'wandb_config': wandb_config})


