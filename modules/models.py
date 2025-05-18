import torch
import torch.nn as nn

# GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=3, num_classes=2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        x = x.unsqueeze(-1)  # (batch, seq_len, input_size=1)
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed_input)
        out = self.fc(hidden[-1])
        return out


# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=3, num_classes=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        x = x.unsqueeze(-1)  # (batch, seq_len, input_size=1)
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed_input)
        out = self.fc(hidden[-1])
        return out


# RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=3, num_classes=2):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        x = x.unsqueeze(-1)
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(packed_input)
        out = self.fc(hidden[-1])
        return out
