
import os
import torch
import torch.nn.functional as nnf
from torch import nn
from torch import optim
from RNN_model import Rat_RNN1, Rat_RNN2
from RNN_model import performance_matrics, train, convert_to_ML_data

from read_data import read_data
from data_management import manage_data


df_train = read_data('data/01 Sucrose FR1 vs EXT 8_2024')

category_mapping = {"FR1": 1, "EXT": 0}
truncate_size = 1
chunk_size = 1
max_press = 80
standardize = True
df2 = manage_data(df_train, truncate_size, chunk_size, max_press, standardize)
df2["category"] = df2["category"].map(category_mapping)


## find length of each element in the data column
# df_train2['data'].apply(lambda x: [len(i) for i in x]).explode().mean()
# df_train2['data'].apply(lambda x: [len(i) for i in x]).explode().hist(bins=100)
# plt.show()

length = 100
df_ML = convert_to_ML_data(df2, length, chunk_size)





from torch.utils.data import TensorDataset, DataLoader
## split the data into training and testing
from sklearn.model_selection import train_test_split
dt_split_train, dt_split_test = train_test_split(df_ML, test_size=0.05, random_state=43)

## create a dataloader for training
## X: Batch x Channel x Time
batch_size = 20000
train_x = torch.tensor(dt_split_train["data"].tolist())
train_y = torch.tensor(dt_split_train["category"].tolist())
my_dataset = TensorDataset(train_x, train_y)
dataloader = DataLoader(my_dataset, batch_size)

# device = 'cpu'
# device = 'mps'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
    
    
model = Rat_RNN1(chunk_size, 30, hidden_layers=2, dropout=0.2)
model = model.to(device)
loss_fn = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.0001)

auc_train_list = []
auc_eval_list = []
epochs = []
import time
## use datetime
base_name = time.strftime("%Y%m%d%H%M%S")
base_folder = f'output/RNN/model_{base_name}'
os.makedirs(base_folder, exist_ok=True)
for i in range(0, 100000):
    ## save models
    loss = train(model, dataloader, loss_fn, optimizer)
    if i % 100 == 0:
        auc_train = performance_matrics(model, dt_split_train)
        auc_eval = performance_matrics(model, dt_split_test)
        print(f"Epoch {i}, loss: {loss:.3f},Train AUC: {auc_train:.2f}, Eval AUC: {auc_eval:.2f}")
        auc_train_list.append(auc_train)
        auc_eval_list.append(auc_eval)
        epochs.append(i)
        
    if i % 1000 == 0:
        model.save(f'{base_folder}/{i}.pth')
        

## plot the auc vs epochs
import matplotlib.pyplot as plt
plt.plot(range(len(auc_train_list)), auc_train_list, label='train')
plt.plot(range(len(auc_train_list)), auc_eval_list, label='eval')
plt.legend()
plt.show()





performance_matrics(model, dt_test)




# epochs = 10000
# learning_rate = 0.0001
# loss_fn = nn.MSELoss()
# # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)