
import wandb
from modules.tradition_models import {model}
from modules.tradition_train import train_traditional_model
import tempfile
import os
# Create a temporary directory
temp_dir = tempfile.mkdtemp()
os.environ["WANDB_DIR"] = temp_dir
wandb.agent('{sweep_id}', function=lambda: train_traditional_model({model}), project = "{project}")
