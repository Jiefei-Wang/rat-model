from modules.nn_models import RNNModel
from modules.nn_train import outter_train_loop
import tempfile
import os
import wandb

# Create a temporary directory
temp_dir = tempfile.mkdtemp()
os.environ["WANDB_DIR"] = temp_dir
wandb.agent('3qpu3416', function=lambda: outter_train_loop(model_class=RNNModel), project = "RNN_sweep")

