import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'


import torch
from modules.Data import data_from_pickle
from modules.nn_train import big_train_loop
from modules.nn_models import GRUModel, LSTMModel, RNNModel
import wandb
from modules.nn_train import outter_train_loop

model_class=GRUModel
epochs = 10000
hidden_size=8
num_layer=2
use_features = True
outter_train_loop(model_class=model_class, hidden_size=hidden_size, num_layers=num_layer, epochs=epochs, use_features=use_features)


