# https://medium.com/@VersuS_/coding-a-recurrent-neural-network-rnn-from-scratch-using-pytorch-a6c9fc8ed4a7
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch import tensor
import numpy as np

from utils import *
from model import *
    
files = [
    "./work/Threshold 10/Frustrated/(10) Frustrated Rat 1", 
    "./work/Threshold 10/Frustrated/(10) Frustrated Rat 2",
    "./work/Threshold 10/Non-Frustrated/(10) Non - Frustrated Rat 1",
    "./work/Threshold 10/Non-Frustrated/(10) Non - Frustrated Rat 2"
]

frustrated = [1, 1, 0, 0]
rats = [1,2,1,2]


mydata = [] 
for file in files:
    data = extractData(file)
    data = lessThan5(data)
    data = process_data(data)
    mydata.append(data)


## to pandas dataframe
import pandas as pd
df = pd.DataFrame({
    "data": mydata,
    "frustrated": frustrated,
    "rat": rats
})



def chunk_data(data, chunk_size=100):
    """
    data: List of data
    chunk_size: Size of each chunk
    """
    chunk = []
    for i in range(0, len(data), chunk_size):
        if i + chunk_size > len(data):
            break
        chunk.append(data[i:i+chunk_size])
    return chunk

## chunk data so that each row has a limited number of bar presses
df2 = df.copy()
df2["chunk"] = df2.apply(lambda x: chunk_data(x["data"], chunk_size=1), axis=1)
df2.drop("data", axis=1, inplace=True)
df2 = df2.explode("chunk")
df2.shape

## concatenate all bar presses within a chunk
df3 = df2.copy()
df3["concate"] = df3.apply(lambda x: concate_data(x["chunk"]), axis=1)
df3["n_points"] = df3.apply(lambda x: len(x["concate"]), axis=1)
df3.drop("chunk", axis=1, inplace=True)

## histogram of the number of data points
# df3["n_points"].hist(bins=100)
# plt.show()


## truncate or padding the data so each row has the same number of data points
df4 = df3.copy()
df4["data"] = df4.apply(lambda x: truncate_or_padding(x["concate"]), axis=1)
df4.drop("concate", axis=1, inplace=True)
    


# ## for each row, concatenate all the bar presses into a single list with 0 padding
# df2 = df.copy()
# df2["concate"] = df2.apply(lambda x: concate_data(x["data"]), axis=1)
# df2.drop("data", axis=1, inplace=True)



# ## chunk data so that each row has a limited number of bar presses
# df3 = df2.copy()
# df3["chunked"] = df3.apply(lambda x: chunk_data(x["concate"], chunk_size=40), axis=1)
# df3.drop("concate", axis=1, inplace=True)
# df3 = df3.explode("chunked")
# df3.shape



## split the data into training and testing
from sklearn.model_selection import train_test_split
train, test = train_test_split(df4, test_size=0.2, random_state=42)


## create a dataloader for training
batch_size = 200
train_len = torch.tensor(train["n_points"].tolist())
train_x = torch.tensor(train["data"].tolist())
train_y = torch.tensor(train["frustrated"].tolist())
my_dataset = TensorDataset(train_x, train_y, train_len)
dataloader = DataLoader(my_dataset, batch_size)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = RNN(input_size=1, hidden_size=10, output_size=1)
model.to(device)

epochs = 10000
learning_rate = 0.0001
loss_fn = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

train_model(model, dataloader, epochs, optimizer=optimizer, loss_fn=loss_fn)

## save model
torch.save(model.state_dict(), "model3.pth")
## load model
model = RNN(input_size=1, hidden_size=10, output_size=1)
model.load_state_dict(torch.load("model3.pth"))


def make_predictions(model, x, n_pre = None):
    if n_pre is None:
        n_pre = x.shape[1]
        
    model.eval()
    model.reset_state()
    predictions = np.zeros((x.shape[0], x.shape[1]))
    for c in range(n_pre):
        out = model(x[:, c].reshape(x.shape[0],1))
        pred = out.detach().numpy().flatten()
        predictions[:, c] = pred
        
    for c in range(n_pre, x.shape[1]):
        current_x = predictions[:, c-1].reshape(predictions.shape[0],1)
        out = model(tensor(current_x).to(model.device.type).type(torch.float32))
        pred = out.detach().numpy().flatten()
        predictions[:, c] = pred
    
    return predictions


## logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

def performance_matrics(model, dt_train, dt_test):
    y_train = dt_train["frustrated"]
    x_train = torch.tensor(dt_train["chunked"].tolist())
    y_test = dt_test["frustrated"]
    x_test = torch.tensor(dt_test["chunked"].tolist())
    
    predictions = make_predictions(model, x_train)
    x_train_hidden = model.hidden_state.detach().numpy()
    
    predictions = make_predictions(model, x_test)
    x_test_hidden = model.hidden_state.detach().numpy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(x_train_hidden)
    X_test_scaled = scaler.transform(x_test_hidden)
    logistic = LogisticRegression(max_iter=1000)
    logistic.fit(X_train_scaled, y_train)
    
    y_pred = logistic.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"ROC AUC: {roc_auc:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    
performance_matrics(model, train, test)
performance_matrics(model, train, train)



n_pre = 10
x_test = torch.tensor(test["chunked"].tolist())
predictions = make_predictions(model, x_test, n_pre)
x_test_hidden = model.hidden_state.detach().numpy()
## take the first sample as example
import matplotlib.pyplot as plt
k = 0
plt.plot(predictions[k])
plt.plot(x_test[k].numpy())
## legend
plt.legend(["Predictions", "True"])
plt.show()