import torch
from modules.Data import data_from_pickle_nn
from modules.nn_train import big_train_loop
from modules.models import GRUModel
from modules.read_data import read_data

output_dir = "output/gru"

nn_train, nn_valid, nn_test = data_from_pickle_nn()



model = GRUModel(input_size=1, hidden_size=32, num_layers=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Trains the model, saves the model, and logs the training process to Weights & Biases
big_train_loop(
    model_name='GRU',
    model=model,
    nn_train=nn_train,
    nn_valid=nn_valid,
    nn_test=nn_test,
    epochs = 500,
    batch_size=1024*64,
    output_dir = output_dir)

