import torch
from modules.nn_train import big_train_loop
from modules.models import GRUModel
from modules.read_data import read_data

output_dir = "output/gru"
df_raw = read_data('data/01 Sucrose FR1 vs EXT 8_2024')
model = GRUModel(input_size=1, hidden_size=32, num_layers=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Trains the model, saves the model, and logs the training process to Weights & Biases
big_train_loop(
    model_name='GRU',
    model=model,
    df_raw=df_raw,
    epochs = 500,
    batch_size=1024*64,
    output_dir = output_dir)

