import wandb
import numpy as np
from modules.tradition_models import logistic_model
from modules.tradition_train import train_traditional_model
from modules.Data import data_from_pickle



df_raw, df_ML, row_train, row_valid, row_test,feature_names = data_from_pickle()


sweep_config = {
    "method": "grid",
    "metric": {"goal": "maximize", "name": "auc"},
}

parameters = {
    "C" : {'values': np.logspace(0.01, 2, 20).tolist()},
}

parameters.update({key: {'values': [True, False]} for key in feature_names})


parameters.update({
    "penalty" : {
        "value": "l2"
    }})

sweep_config = sweep_config.copy()
sweep_config['parameters'] = parameters

model="logistic_model"
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
with open("scripts/sweep/logistic.py", "w") as f:
    f.write(code)
