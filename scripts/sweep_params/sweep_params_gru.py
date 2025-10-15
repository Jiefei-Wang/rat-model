import wandb

project = "rat-model-gru"

sweep_config = {
    "method": "grid",
    "metric": {"goal": "maximize", "name": "best_valid_auc"},
    "program": "scripts.sweep_models.sweep_gru",
    "command": ["${env}", "${interpreter}", "-m", "${program}"],
}


parameters = {
    "hidden_size" : {'values': [4,8,16,32,64, 128, 256, 384]},
    "num_layers" : {'values': [1,2,3,4,5,6]},
    "use_features" : {'values': [True, False]},
    "epochs" : {'value': 10000},
}

sweep_config['parameters'] = parameters

# Initialize sweep by passing in config.
sweep_id = wandb.sweep(sweep=sweep_config, project=project)

