import torch
from modules.Data import data_from_pickle_nn
from modules.nn_train import big_train_loop
from modules.models import GRUModel, LSTMModel, RNNModel

epochs = 5000
nn_train, nn_valid, nn_test = data_from_pickle_nn()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


GRU = GRUModel(input_size=1, hidden_size=32, num_layers=3)

#Trains the model, saves the model, and logs the training process to Weights & Biases
GRU, train_info, prediction = big_train_loop(
    model=GRU,
    nn_train=nn_train,
    nn_valid=nn_valid,
    nn_test=nn_test,
    epochs = epochs)




LSTM = LSTMModel(input_size=1, hidden_size=32, num_layers=3)

#Trains the model, saves the model, and logs the training process to Weights & Biases
LSTM, train_info, prediction = big_train_loop(
    model=LSTM,
    nn_train=nn_train,
    nn_valid=nn_valid,
    nn_test=nn_test,
    epochs = epochs)



RNN = RNNModel(input_size=1, hidden_size=32, num_layers=3)
#Trains the model, saves the model, and logs the training process to Weights & Biases
RNN, train_info, prediction = big_train_loop(
    model=RNN,
    nn_train=nn_train,
    nn_valid=nn_valid,
    nn_test=nn_test,
    epochs = epochs)
