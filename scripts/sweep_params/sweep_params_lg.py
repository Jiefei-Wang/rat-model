import wandb

project = "rat-model-lg"

sweep_config = {
    "method": "grid",
    "metric": {"goal": "maximize", "name": "val_auc"},
    "program": "scripts.sweep_models.sweep_LG",
    "command": ["${env}", "${interpreter}", "-m", "${program}"],
}

parameters = {
    "C" : {'values': [i * 0.001 for i in range(1, 200)]}
}
sweep_config['parameters'] = parameters


sweep_id = wandb.sweep(sweep=sweep_config, project=project)