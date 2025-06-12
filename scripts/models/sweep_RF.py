import wandb
import numpy as np
from modules.model import random_forest_model
from modules.train import train_traditional_model
from modules.Data import data_from_pickle



df_raw, df_ML, row_train, row_valid, row_test,feature_names = data_from_pickle()


sweep_config = {
    "method": "grid",
    "metric": {"goal": "maximize", "name": "auc"},
}


parameters = {
    "n_estimators" : {'values': [i for i in range(10, 201, 10)]},
    "max_depth" : {'values': [i for i in range(1, 21)]},
    "min_samples_split": {'values': [2, 5, 10]},
    "min_samples_leaf": {'values': [1, 2, 4]},
}

parameters.update({key: {'values': [True, False]} for key in feature_names})


sweep_config = sweep_config.copy()
sweep_config['parameters'] = parameters

sweep_id = wandb.sweep(sweep_config, project="RF_sweep")

## create a bat file in scripts/sweep/logistic.bat to run the sweep
with open("scripts/sweep/random_forest.py", "w") as f:
    f.write(f"""
import wandb
from modules.model import random_forest_model
from modules.train import train_traditional_model
wandb.agent('{sweep_id}', function=lambda: train_traditional_model(random_forest_model))
""")

