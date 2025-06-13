from modules.nn_models import {model_class}
from modules.nn_train import outter_train_loop
import tempfile
import os
import wandb

api = wandb.Api()
project = api.project("{project}")
# get the latest sweep for the project
sweeps = project.sweeps()
sweep_id = sweeps[0].id



# Create a temporary directory
temp_dir = tempfile.mkdtemp()
os.environ["WANDB_DIR"] = temp_dir
wandb.agent(sweep_id, function=lambda: outter_train_loop(model_class={model_class}), project = "{project}")

