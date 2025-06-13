import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import torch
from modules.Data import data_from_pickle
from modules.nn_train import big_train_loop
from modules.nn_models import GRUModel, LSTMModel, RNNModel
import wandb



sweep_config = {
    "method": "grid",
    "metric": {"goal": "maximize", "name": "best_valid_auc"},
}

parameters = {
    "hidden_size" : {'values': [4,8,16,32,64, 128, 256, 384]},
    "num_layers" : {'values': [1,2,3,4,5,6]},
    "use_features" : {'values': [True, False]},
    "epochs" : {'value': 10000},
}

sweep_config['parameters'] = parameters

model_list = {
    "GRU": "GRUModel",
    "LSTM": "LSTMModel",
    "RNN": "RNNModel"
}

for key, model_class in model_list.items():
    project = f"{key}_sweep"
    sweep_id = wandb.sweep(sweep_config, project=project)

    with open("scripts/sweep/nn_template.py", "r") as f:
        template = f.read()

    code = template.format(
        model_class=model_class,
        sweep_id=sweep_id,
        project = project
    )
    
    ## create a bat file in scripts/sweep/logistic.bat to run the sweep
    with open(f"scripts/sweep/{key}.py", "w") as f:
        f.write(code)


