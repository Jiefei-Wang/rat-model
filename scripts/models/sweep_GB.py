import wandb
import numpy as np
from modules.tradition_models import gradient_boosting_model
from modules.tradition_train import train_traditional_model
from modules.Data import data_from_pickle



df_raw, df_ML, row_train, row_valid, row_test,feature_names = data_from_pickle()


sweep_config = {
    "method": "grid",
    "metric": {"goal": "maximize", "name": "auc"},
}


parameters = {
    "n_estimators" : {'values': [i for i in range(10, 201, 10)]},
    "subsample" : {'values': [i/10 for i in range(1, 11)]},
    "max_depth": {'values': [i for i in range(10, 201, 10)]},
    "min_samples_split": {'values': [2, 5, 10]},
    "min_samples_leaf": {'values': [1, 2, 4]},
}

sweep_config = sweep_config.copy()
sweep_config['parameters'] = parameters

model = "gradient_boosting_model"
project = model
sweep_id = wandb.sweep(sweep_config, project=project)

with open("scripts/sweep/template.py", "r") as f:
    template = f.read()

code = template.format(
    model=model,
    sweep_id=sweep_id,
    project=project
)
## create a bat file in scripts/sweep/logistic.bat to run the sweep
with open(f"scripts/sweep/{model}.py", "w") as f:
    f.write(code)
