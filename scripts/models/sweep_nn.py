import os, pickle, wandb
import torch
# from muon import MuonWithAuxAdam
from sklearn.model_selection import train_test_split

from modules.nn_models import GRUModel
from modules.nn_train import big_train_loop



wandb_config = {}
run = None
wandb.init()
wandb_config = dict(wandb.config)
run = wandb.run

num_folds = 10
output_base = 'output/data' 

hidden_size = wandb_config.get('hidden_size', 1)
num_layers = wandb_config.get('num_layers', 1)
use_features = wandb_config.get('use_features', True)
epochs = wandb_config.get('epochs', 1000)

df_ML_train = pickle.load(open(os.path.join(output_base, "df_ML_train.pkl"), "rb"))
feature_names = pickle.load(open(os.path.join(output_base, "feature_names.pkl"), "rb"))


x_train = df_ML_train[feature_names].values
y_train = df_ML_train['category'].values


# id level train valid split
unique_ids = df_ML_train['id'].unique()
train_ids, valid_ids = train_test_split(unique_ids, test_size=0.1, random_state=42)

nn_train0 = df_ML_train[df_ML_train['id'].isin(train_ids)].reset_index(drop=True)[['label', 'data']]
nn_valid = df_ML_train[df_ML_train['id'].isin(valid_ids)].reset_index(drop=True)[['label', 'data']]
if use_features:
    features_train = df_ML_train[df_ML_train['id'].isin(train_ids)].reset_index(drop=True)[feature_names]
    features_valid = df_ML_train[df_ML_train['id'].isin(valid_ids)].reset_index(drop=True)[feature_names]
    feature_size = features_train.shape[1]
else:
    features_train = None
    features_valid = None
    feature_size = 0



model = GRUModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers, feature_size=feature_size)
optimizer = None
model = big_train_loop(
    model=model,
    nn_train=nn_train0,
    nn_valid=nn_valid,
    features_train=features_train,
    features_valid=features_valid,
    epochs = epochs,
    optimizer=optimizer,
    run =run
    )

