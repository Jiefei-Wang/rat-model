
import wandb
from modules.tradition_models import random_forest_model
from modules.tradition_train import train_traditional_model
import tempfile
import os
# Create a temporary directory
temp_dir = tempfile.mkdtemp()
os.environ["WANDB_DIR"] = temp_dir
wandb.agent('issyc7tm', function=lambda: train_traditional_model(random_forest_model), project = "RF_sweep")
