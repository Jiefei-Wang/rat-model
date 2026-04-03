import wandb

project = "rat-model-gb"

sweep_config = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "val_auc"},
    "program": "scripts.sweep_models.sweep_GB",
    "command": ["${env}", "${interpreter}", "-m", "${program}"],
}

parameters = {
    "n_estimators": {"values": [i for i in range(10, 201, 10)]},
    "subsample": {"values": [i * 0.1 for i in range(1, 11)]},
    "max_depth": {"values": [i for i in range(10, 201, 10)]},
    "min_samples_split": {"values": [2, 5, 10]},
    "min_samples_leaf": {"values": [1, 2, 4]},
}
sweep_config["parameters"] = parameters

sweep_id = wandb.sweep(sweep=sweep_config, project=project)
