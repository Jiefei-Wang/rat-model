import wandb

project = "rat-model-gb"

sweep_config = {
    "method": "grid",
    "metric": {"goal": "maximize", "name": "val_auc"},
    "program": "scripts.sweep_models.sweep_GB",
    "command": ["${env}", "${interpreter}", "-m", "${program}"],
}
parameters = {
    # include shallow->deep transitions
    "max_depth": {"values": [1, 3, 5, 7]},              
    # fine near low values + stronger regularization
    "min_child_weight": {"values": [1, 3, 5, 7]},  
    # low/mid/high sampling        
    "subsample": {"values": [0.6, 0.75, 0.9, 1.0]},     
    # feature sampling sensitivity      
    "colsample_bytree": {"values": [0.6, 0.75, 0.9, 1.0]},
    # weak/default/strong L2    
    "reg_lambda": {"values": [0, 0.1, 0.5, 1.0, 3, 10.0]},               
}

sweep_config["parameters"] = parameters

sweep_id = wandb.sweep(sweep=sweep_config, project=project)
