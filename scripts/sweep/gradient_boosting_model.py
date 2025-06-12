
import wandb
from modules.model_tradition import gradient_boosting_model
from modules.train import train_traditional_model
import tempfile
import os
# Create a temporary directory
temp_dir = tempfile.mkdtemp()
os.environ["WANDB_DIR"] = temp_dir
wandb.agent('67mb4np8', function=lambda: train_traditional_model(gradient_boosting_model), project = "gradient_boosting_model")
