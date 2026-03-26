import wandb

project = "rat-model-GB"

sweep_config = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "val_auc"},
    "program": "scripts.sweep_models.sweep_GB",
    "command": ["${env}", "${interpreter}", "-m", "${program}"],
}

parameters = {
    "n_estimators": {"values": [60, 80, 100, 120, 140, 160, 180, 200]},
    "subsample": {"values": [0.6, 0.7, 0.8, 0.9, 1.0]},
    "max_depth": {"values": [2, 3, 4, 5]},
    "min_samples_split": {"values": [2, 4, 6, 8]},
    "min_samples_leaf": {"values": [1, 2, 3, 4]},
}

sweep_config["parameters"] = parameters


sweep_id = wandb.sweep(sweep=sweep_config, project=project)

