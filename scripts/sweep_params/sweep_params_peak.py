import wandb


project = "rat-model-peaks"
sweep_config = {
    "method": "grid",
    "metric": {"goal": "maximize", "name": "val_auc"},
    "program": "-m scripts.sweep_models.sweep_peak"
}

parameters = {
    "prominence" : {'values': [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]},
    "height" : {'values': [3.0, 5.0, 8.0, 10.0, 15.0, 20.0]},
    "distance" : {'values': [1, 3, 5, 10, 20, 50]},
    "width" : {'values': [None, 1.0, 2.0, 3.0]},
    "wlen" : {'values': [None]},
    "threshold" : {'values': [None, 0.0, 0.5, 1.0, 2.0]},
    "rel_height" : {'values': [0.5, 0.6, 0.7, 0.8]},
    "plateau_size" : {'values': [None, 1.0, 2.0, 3.0, 5.0]},
}
sweep_config['parameters'] = parameters


sweep_id = wandb.sweep(sweep=sweep_config, project=project)