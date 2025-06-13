import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils # Ensure rnn_utils is imported if not already

# GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=3, num_classes=2, feature_size=0): # Added feature_size
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.feature_size = feature_size  # Store feature_size

        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)

        # Adjust FC layer input size based on manual features
        fc_input_dim = hidden_size + self.feature_size
        
        self.fc = nn.Linear(fc_input_dim, num_classes)
        
        self.name = f"GRU"
        # Update params to include feature_size if it's a defining characteristic
        self.params = f"{input_size}_{hidden_size}_{num_layers}_{feature_size}"

    def forward(self, x, lengths, manual_features=None): # Added manual_features argument
        packed_input = rnn_utils.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed_input)
        
        # Use the hidden state from the last layer
        rnn_out = hidden[-1]

        if self.feature_size > 0:
            if manual_features is None:
                raise ValueError("feature_size > 0 but manual_features were not provided to the forward method.")
            if manual_features.shape[1] != self.feature_size:
                raise ValueError(f"Expected manual_features to have {self.feature_size} features, but got {manual_features.shape[1]}.")
            # Ensure manual_features is on the same device as rnn_out
            manual_features = manual_features.to(rnn_out.device)
            # Concatenate RNN output with manual features
            combined_features = torch.cat((rnn_out, manual_features), dim=1)
            out = self.fc(combined_features)
        else:
            # If no manual features are used
            out = self.fc(rnn_out)
            
        return out

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=3, num_classes=2, feature_size=0): # Added feature_size
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.feature_size = feature_size

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        
        fc_input_dim = hidden_size + self.feature_size

        self.fc = nn.Linear(fc_input_dim, num_classes)
        self.name = f"LSTM"
        self.params = f"{input_size}_{hidden_size}_{num_layers}_{feature_size}"

    def forward(self, x, lengths, manual_features=None): # Added manual_features argument
        packed_input = rnn_utils.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed_input)
        
        rnn_out = hidden[-1]

        if self.feature_size > 0:
            if manual_features is None:
                raise ValueError("feature_size > 0 but manual_features were not provided to the forward method.")
            if manual_features.shape[1] != self.feature_size:
                raise ValueError(f"Expected manual_features to have {self.feature_size} features, but got {manual_features.shape[1]}.")
            manual_features = manual_features.to(rnn_out.device)
            combined_features = torch.cat((rnn_out, manual_features), dim=1)
            out = self.fc(combined_features)
        else:
            out = self.fc(rnn_out)
            
        return out


# RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=3, num_classes=2, feature_size=0): # Added feature_size
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.feature_size = feature_size

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        fc_input_dim = hidden_size + self.feature_size
            
        self.fc = nn.Linear(fc_input_dim, num_classes)
        self.name = f"RNN"
        self.params = f"{input_size}_{hidden_size}_{num_layers}_{feature_size}"

    def forward(self, x, lengths, manual_features=None): # Added manual_features argument
        packed_input = rnn_utils.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(packed_input)
        
        rnn_out = hidden[-1]

        if self.feature_size > 0:
            if manual_features is None:
                raise ValueError("feature_size > 0 but manual_features were not provided to the forward method.")
            if manual_features.shape[1] != self.feature_size:
                raise ValueError(f"Expected manual_features to have {self.feature_size} features, but got {manual_features.shape[1]}.")
            manual_features = manual_features.to(rnn_out.device)
            combined_features = torch.cat((rnn_out, manual_features), dim=1)
            out = self.fc(combined_features)
        else:
            out = self.fc(rnn_out)
            
        return out