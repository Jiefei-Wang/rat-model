import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'


import torch
from modules.Data import data_from_pickle_nn
from modules.nn_train import big_train_loop
from modules.models import GRUModel, LSTMModel, RNNModel

nn_train, nn_valid, nn_test, features_train, features_valid, features_test = data_from_pickle_nn()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



epochs = 8000
# we reduce the number of epochs for large models
param_cutoff = 64
over_curoff_epoch = 3000

hidden_size_list = [4,8,16,32,64, 128,256]
num_layers = [1,2,3,4,5,6]




# model = GRUModel(input_size=1, hidden_size=32, num_layers=3)
# model, train_info, prediction = big_train_loop(
#     model=model,
#     nn_train=nn_train,
#     nn_valid=nn_valid,
#     nn_test=nn_test,
#     epochs = epochs)


model_list = {
    "GRU": GRUModel,
    "LSTM": LSTMModel,
    "RNN": RNNModel
}


hidden_size=128
num_layer=5
model_name = "GRU"
model_class = model_list[model_name]


results = []
for hidden_size in hidden_size_list:
    for num_layer in num_layers:
        for model_name, model_class in model_list.items():
            model = model_class(input_size=1, hidden_size=hidden_size, num_layers=num_layer)
            param_num = hidden_size * num_layer
            train_epochs = epochs
            if param_num > param_cutoff:
                train_epochs = over_curoff_epoch
            model, train_info, prediction = big_train_loop(
                model=model,
                nn_train=nn_train,
                nn_valid=nn_valid,
                nn_test=nn_test,
                epochs = train_epochs)
            


# features_train=features_train,
# features_valid=features_valid,
# features_test=features_test,

model = model_class(input_size=1, hidden_size=hidden_size, num_layers=num_layer, manual_feature_size=features_train.shape[1] )