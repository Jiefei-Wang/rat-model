import wandb

project = "rat-model-LG"

sweep_config = {
    "method": "grid",
    "metric": {"goal": "maximize", "name": "val_auc"},
    "program": "scripts.sweep_models.sweep_LG",
    "command": ["${env}", "${interpreter}", "-m", "${program}"],
}

parameters = {
    "C" : {'values': [i * 0.1 for i in range(0, 100)]}
}
sweep_config['parameters'] = parameters


sweep_id = wandb.sweep(sweep=sweep_config, project=project)