
import wandb
from modules.tradition_models import logistic_model
from modules.tradition_train import train_traditional_model
wandb.agent('lu7qjn98', function=lambda: train_traditional_model(logistic_model))
