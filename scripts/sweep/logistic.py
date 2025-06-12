
import wandb
from modules.model import logistic_model
from modules.train import train_traditional_model
wandb.agent('lu7qjn98', function=lambda: train_traditional_model(logistic_model))
