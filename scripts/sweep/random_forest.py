
import wandb
from modules.model import random_forest_model
from modules.train import train_traditional_model
wandb.agent('rfpoa6v1', function=lambda: train_traditional_model(random_forest_model))
