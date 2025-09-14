import wandb

def main():
    exec(open("scripts/models/sweep_nn.py").read())



sweep_config = {
    "method": "grid",
    "metric": {"goal": "maximize", "name": "final_valid_auc"},
}

parameters = {
    "hidden_size" : {'values': [4,8,16,32,64, 128, 256, 384]},
    "num_layers" : {'values': [1,2,3,4,5,6]},
    "use_features" : {'values': [True, False]},
    "epochs" : {'value': 10000},
}

sweep_config['parameters'] = parameters

# Initialize sweep by passing in config.
sweep_id = wandb.sweep(sweep=sweep_config, project="rat-model-gru")

# Start sweep job.
wandb.agent(sweep_id, function=main)
