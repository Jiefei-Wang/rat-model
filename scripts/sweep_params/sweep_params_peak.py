import wandb


project = "rat-model-peaks"
sweep_config = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_auc"},
    "program": "scripts.sweep_models.sweep_peak",
    "command": ["${env}", "${interpreter}", "-m", "${program}"],
}

parameters = {
    "prominence" : {'values': [i * 0.2 for i in range(1, 51)]},
    "height" : {'values': [i for i in range(1, 51)]},
    "distance" : {'values': [i for i in range(1, 51)]},
    "width" : {'values': [None] + [i for i in range(1, 21)]},
    "wlen" : {'values': [None]},
    "threshold" : {'values': [None] + [i * 0.2 for i in range(1, 101)]},
    "rel_height" : {'values': [i * 0.1 for i in range(1, 11)]},
    "plateau_size" : {'values': [None] + [i for i in range(1, 11)]},
}
sweep_config['parameters'] = parameters


sweep_id = wandb.sweep(sweep=sweep_config, project=project)