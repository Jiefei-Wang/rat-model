import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import torch
from modules.Data import data_from_pickle_nn
from modules.nn_train import big_train_loop
from modules.models import GRUModel, LSTMModel, RNNModel

epochs = 5000
nn_train, nn_valid, nn_test = data_from_pickle_nn()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




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


hidden_size_list = [4,8,16,32,64]
num_layers = [1,2,3,4]
results = []
for hidden_size in hidden_size_list:
    for num_layer in num_layers:
        for model_name, model_class in model_list.items():
            model = model_class(input_size=1, hidden_size=hidden_size, num_layers=num_layer)
            model, train_info, prediction = big_train_loop(
                model=model,
                nn_train=nn_train,
                nn_valid=nn_valid,
                nn_test=nn_test,
                epochs = epochs)
            # Find the row with best validation loss
            train_info = train_info[train_info['valid_loss'] == train_info['valid_loss'].min()].iloc[0].to_dict()
            results.append({
                "model": model_name,
                "hidden_size": hidden_size,
                "num_layers": num_layer
            }|train_info)
            


