
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from sklearn.metrics import roc_auc_score
import os
import pickle

class Rat_base(nn.Module):
    def save(self, path):
        state = self.state_dict()
        # Save the class type (self.__class__) along with the state and args
        with open(path, 'wb') as f:
            pickle.dump((self.__class__, state, self.args), f)
        
    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            # Load the class type, state, and args
            class_type, state, args = pickle.load(f)
        # Dynamically create an instance of the loaded class type
        model = class_type(*args)
        model.load_state_dict(state)
        return model

class Rat_RNN1(Rat_base):
    def __init__(self, input_size, hidden_size, hidden_layers=1, dropout=0):
        self.args = (input_size, hidden_size, hidden_layers, dropout)
        # This just calls the base class constructor
        super().__init__()
        # Neural network layers assigned as attributes of a Module subclass
        # have their parameters registered for training automatically.
        self.rnn = torch.nn.RNN(
            input_size, hidden_size, 
            num_layers = hidden_layers,
            nonlinearity='relu', 
            dropout= dropout, 
            batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        # The RNN also returns its hidden state but we don't use it.
        # While the RNN can also take a hidden state as input, the RNN
        # gets passed a hidden state initialized with zeros by default.
        
        # Batch x Channel x Time -> Batch x Time x Channel
        x2 = x.permute(0, 2, 1)
        h = self.rnn(x2)[0]
        # Batch x Time x hidden_size -> Batch x hidden_size
        prob = h[ :, -1, :]
        # Batch x hidden_size -> Batch x 1
        prob2 = self.linear(prob)
        ## for the last layer, to probability
        prob3 = nnf.sigmoid(prob2[:,0])
        
        return prob3
    

# model = Rat_RNN1(2,3,1)
# x = torch.randn(4, 2, 10)
# model(x)


class Rat_RNN2(Rat_base):
    def __init__(self, input_size, hidden_size, hidden_layers=1,dropout=0):
        self.args = (input_size, hidden_size, hidden_layers, dropout)
        # This just calls the base class constructor
        super().__init__()
        # Neural network layers assigned as attributes of a Module subclass
        # have their parameters registered for training automatically.
        self.rnn = torch.nn.RNN(
            1, hidden_size, 
            num_layers = hidden_layers,
            nonlinearity='tanh', 
            batch_first=True)
        self.linear1 = torch.nn.Linear(hidden_size, 1)
        self.linear2 = torch.nn.Linear(input_size, 1)

    def forward(self, x):
        # The RNN also returns its hidden state but we don't use it.
        # While the RNN can also take a hidden state as input, the RNN
        # gets passed a hidden state initialized with zeros by default.
        
        # Batch x input_size x Time -> (Batch x input_size) x 1 x Time
        x2 = x.view(-1, 1, x.shape[2])
        # (Batch x input_size) x 1 x Time -> (Batch x input_size) x Time x 1
        x2 = x2.permute(0, 2, 1)
        # (Batch x input_size) x Time x 1 -> (Batch x input_size) x Time x hidden_size
        h = self.rnn(x2)[0]
        # (Batch x input_size) x Time x hidden_size -> (Batch x input_size) x 1 x hidden_size
        h_last = h[:,-1,:]
        # (Batch x input_size) x hidden_size -> (Batch x input_size) x 1
        l1= self.linear1(h_last)
        # (Batch x input_size) x 1 -> Batch x input_size
        l1_unroll = l1.view(x.shape[0], -1)
        ## sort the last layer
        l1_sort = torch.sort(l1_unroll, dim=1)[0]
        # Batch x input_size-> Batch x 1
        l2 = self.linear2(l1_sort)
        # Batch x 1 -> Batch
        l2 = l2.squeeze()
        
        ## logit transform to probability
        prob = nnf.sigmoid(l2)
        return prob
    


def performance_matrics(model, dt):
    device = next(model.parameters()).device.type
    model.eval()
    data = torch.tensor(dt["data"].tolist())
    data = data.to(device)
    output = model(data)
    predictions = output.detach().tolist()
    
    target = dt["category"].tolist()
    
    roc_auc = roc_auc_score(target, predictions)
    return roc_auc



def train(model, dataloader, loss_fn, optimizer):
    ## get model device
    device = next(model.parameters()).device.type
    model.train()
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        target = target.type(torch.float32)
        output = model(data)
        loss = loss_fn(output, target) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def concate_data(list_of_lists):
    concatenated_list = []
    # Iterate over the list of lists
    for i, sublist in enumerate(list_of_lists):
        concatenated_list.extend(sublist)  # Add the elements of the sublist to the result list
        if i < len(list_of_lists) - 1:
            concatenated_list.append(0)  # Add 0 between the sublists
            
    return concatenated_list


def convert_to_ML_data(df, length, chunk_size):
    df = df.copy()
    df['data'] = df['data'].apply(lambda x: truncate_or_padding(x, length))
    ## remove rows that the length of data is less than chunk_size
    df = df[df['data'].apply(lambda x: len(x) == chunk_size)]
    return df


def truncate_or_padding(data, length=100):
    for i in range(len(data)):
        data[i] = truncate_or_padding_elt(data[i], length)
    return data

def truncate_or_padding_elt(data, length=100):
    if len(data) > length:
        return data[:length]
    else:
        return data + [0]*(length-len(data))



