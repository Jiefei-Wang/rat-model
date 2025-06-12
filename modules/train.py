import wandb
from modules.Data import data_from_pickle
from sklearn.metrics import roc_auc_score
import numpy as np


if False:
    model_func= logistic_model
    config = {
        "penalty": "l2",
        "C": 1.0,
        'avg_last_5': True,
        'peak_sharpness': False
    }



def train_traditional_model(model_func):
    with wandb.init() as run:
        config = wandb.config
        # Assuming you have a function to get your data
        df_raw, df_ML, row_train, row_valid, row_test,feature_names = data_from_pickle()
        
        features = [key for key in feature_names if config.get(key, True)]
        
        x_train = df_ML.loc[row_train, feature_names][features].values
        y_train = df_ML.loc[row_train, 'category'].values
        x_valid = df_ML.loc[row_valid, feature_names][features] .values
        y_valid = df_ML.loc[row_valid, 'category'].values
        
        
        # Build and train the model
        predictions = model_func(y_train, x_train, x_valid, config)
        auc = roc_auc_score(y_valid.tolist(), predictions)
        # Log the AUC
        wandb.log({"auc": auc})
