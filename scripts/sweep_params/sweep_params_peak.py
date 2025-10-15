import wandb


project = "rat-model-peaks"
sweep_config = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "val_auc"},
    "program": "scripts.sweep_models.sweep_peak",
    "command": ["${env}", "${interpreter}", "-m", "${program}"],
}

parameters = {
    "prominence" : {'values': [i * 0.2 for i in range(1, 50)]},
    "height" : {'values': [i for i in range(1, 50)]},
    "distance" : {'values': [i for i in range(1, 50)]},
    "width" : {'values': [None] + [i for i in range(1, 20)]},
    "wlen" : {'values': [None]},
    "threshold" : {'values': [None] + [i * 0.2 for i in range(1, 100)]},
    "rel_height" : {'values': [i * 0.1 for i in range(1, 10)]},
    "plateau_size" : {'values': [None] + [i for i in range(1, 10)]},
}
sweep_config['parameters'] = parameters


sweep_id = wandb.sweep(sweep=sweep_config, project=project)