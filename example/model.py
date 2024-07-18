import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim

class RNN(nn.Module):
    """
    Basic RNN block. This represents a single layer of RNN
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        """
        input_size: Number of features of your input vector
        hidden_size: Number of hidden neurons
        output_size: Number of features of your output vector
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.first_run = True
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.hidden_state = None
    
    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns computed output and tanh(i2h + h2h)
        Inputs
        ------
        x: Input vector
        hidden_state: Previous hidden state
        Outputs
        -------
        out: Linear output (without activation because of how pytorch works)
        hidden_state: New hidden state matrix
        """
        if self.first_run:
            self.hidden_state = torch.zeros(x.shape[0], self.hidden_size, requires_grad=False)
            self.first_run = False
        
        x = self.i2h(x)
        self.hidden_state = self.h2h(self.hidden_state)
        self.hidden_state = torch.tanh(x + self.hidden_state)
        out = self.h2o(self.hidden_state)
        return out
    
    def reset_state(self):
        self.first_run = True
        
    @property
    def device(self):
        return next(self.parameters()).device
    

def train_model(model: RNN, dataloader: DataLoader, epochs: int, optimizer: optim.Optimizer, loss_fn: nn.Module) -> None:
    """
    Trains the model for the specified number of epochs
    Inputs
    ------
    model: RNN model to train
    dataloader: Iterable DataLoader
    epochs: Number of epochs to train the model
    optiimizer: Optimizer to use for each epoch
    loss_fn: Function to calculate loss
    """
    
    device = model.device.type
    train_losses = {}
    
    model.train()
    print("=> Starting training")
    for epoch in range(epochs):
        epoch_losses = list()
        for X, Y, n_points in dataloader:
            model.reset_state()
            # send tensors to device
            X, Y = X.to(device), Y.to(device)

            # 2. clear gradients
            model.zero_grad()

            loss = 0
            for c in range(X.shape[1]-1):
                if sum(c < n_points - 1).item() == 0:
                    break
                out = model(X[:, c].reshape(X.shape[0],1))
                ## only calculate loss for points where c < n_points - 1 
                l = loss_fn(out[c < n_points - 1, 0], X[c < n_points - 1, c])
                loss += l

            # 4. Compte gradients gradients
            loss.backward()

            # 5. Adjust learnable parameters
            # clip as well to avoid vanishing and exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
        
            epoch_losses.append(loss.detach().item() / X.shape[1])

        train_losses[epoch] = torch.tensor(epoch_losses).mean()
        print(f'=> epoch: {epoch + 1}, loss: {train_losses[epoch]}')
