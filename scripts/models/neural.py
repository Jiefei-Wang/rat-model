import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'


import torch
from modules.Data import data_from_pickle_nn
from modules.nn_train import big_train_loop
from modules.nn_models import GRUModel, LSTMModel, RNNModel
import wandb




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


sweep_config = {
    'method': 'random'
    }

metric = {
    'name': 'loss',
    'goal': 'minimize'   
    }

sweep_config['metric'] = metric

parameters_dict = {
    'hidden_size': {
        'values': [4,8,16,32,64, 128,256]
        },
    'num_layers': {
        'values': [1,2,3,4,5,6]
        },
    'features': {
          'values': [True, False]
        },
    }

sweep_config['parameters'] = parameters_dict


import pprint
pprint.pprint(sweep_config)



sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")


def train_outter(model_class, config):
    with wandb.init(config=config):
        config = wandb.config
        hidden_size = config.hidden_size
        num_layers = config.num_layers
        use_features = config.features

        nn_train, nn_valid, nn_test, features_train, features_valid, features_test = data_from_pickle_nn()
        
        features_size = features_train.shape[1] if use_features else 0
        
        if use_features:
            model = model_class(input_size=1, hidden_size=hidden_size, num_layers=num_layers, manual_feature_size=features_size)
        else:
            model = model_class(input_size=1, hidden_size=hidden_size, num_layers=num_layers)
            features_train = None
            features_valid = None
            features_test = None
            
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        big_train_loop(
                model=model,
                nn_train=nn_train,
                nn_valid=nn_valid,
                nn_test=nn_test,
                features_train=features_train,
                features_valid=features_valid,
                features_test=features_test,
                epochs = train_epochs)
        

wandb.agent(sweep_id, train_outter, count=5)

        


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