import torch
import os
import wandb
import torch.nn.utils.rnn as rnn_utils
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import gc
import tempfile
import os

import datetime
import pandas as pd
from tqdm import tqdm

from modules.Data import data_from_pickle


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss >= (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class contextManager:
    def __init__(self, run):
        self.run = run
    
    def __del__(self):
        self.run.finish()


def dataframe_to_tensors(dataframe, device=None):
    """
    Convert dataframe with variable-length sequences to padded tensors.
    
    Args:
        dataframe: DataFrame with 'data' (sequences) and 'label' columns
        max_length: Maximum sequence length (default 100)
        device: Target device for tensors
    
    Returns:
        tuple: (padded_data, labels, lengths) as tensors
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract sequences and labels
    sequences = dataframe['data'].tolist()
    labels = dataframe['label'].tolist()
    
    # Truncate sequences to max_length and calculate actual lengths
    truncated_sequences = []
    actual_lengths = []
    
    for seqs in sequences:
        seq_min_length = min([len(seq) for seq in seqs])  
        seqs = [seq[:seq_min_length] for seq in seqs] 
        seqs_tensor = torch.tensor(seqs, dtype=torch.float32)
        # switch 0 and 1 dimensions
        # to make Time * input_size
        seqs_tensor = seqs_tensor.permute(1, 0)
        truncated_sequences.append(seqs_tensor)
        actual_lengths.append(seq_min_length)
    
    # Pad sequences to max_length
    padded_sequences = rnn_utils.pad_sequence(
        truncated_sequences,
        batch_first=True,
        padding_value=0.0
    )
    
    # Convert to tensors and move to device
    padded_tensor = padded_sequences.to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
    actual_lengths = torch.tensor(actual_lengths, dtype=torch.long).to('cpu') # Keep lengths on CPU for packing
    
    return padded_tensor, labels_tensor, actual_lengths




def big_train_loop(model,
                   nn_train, nn_valid,
                   features_train=None, features_valid=None,
                   device=None,
                   epochs=100,
                   use_features = None,
                   run = None
                   ):
    ## Free memory
    gc.collect()
    torch.cuda.empty_cache()
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    stopper = EarlyStopper(patience=500, min_delta=0)
    
    base_path = tempfile.mkdtemp()
    os.makedirs(base_path, exist_ok=True)
    
    if run is None:
        run = wandb.init(
            project=model.name,
            name = f"{model.params}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                'use_features': use_features
            }
        )
        runContext = contextManager(run)
    
    
    config = run.config
    use_features = config.get("use_features", False) 
    
    model = model.to(device)
    run.watch(model, log_freq=100, log="all")
    
    # Convert all data to tensors and load directly to GPU
    train_x, train_y, train_lengths = dataframe_to_tensors(nn_train, device=device)
    valid_x, valid_y, valid_lengths = dataframe_to_tensors(nn_valid, device=device)
    # test_x, test_y, test_lengths = dataframe_to_tensors(nn_test, device=device)
    
    train_y_cpu = train_y.cpu().numpy()
    valid_y_cpu = valid_y.cpu().numpy()
    # test_y_cpu = test_y.cpu().numpy()
    
    if use_features:
        features_train = torch.tensor(features_train.to_numpy(), dtype=torch.float32).to(device)
        features_valid = torch.tensor(features_valid.to_numpy(), dtype=torch.float32).to(device)
    
    lr = 0.001
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_valid_auc = 0
    for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
        # Update progress bar with current losses after each epoch
        # Training
        model.train()
        optimizer.zero_grad()
        train_outputs = model(train_x, train_lengths, features_train)
        train_loss = criterion(train_outputs, train_y)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        
        # Validation
        model.eval()
        with torch.no_grad():
            valid_outputs = model(valid_x, valid_lengths, features_valid)
            valid_loss = criterion(valid_outputs, valid_y)
        
        # Log training and validation metrics every 10 epochs
        with torch.no_grad():
            train_probs = torch.softmax(train_outputs, dim=1)[:, 1]
            train_auc = roc_auc_score(train_y_cpu, train_probs.detach().cpu().numpy())
            
            valid_probs = torch.softmax(valid_outputs, dim=1)[:, 1]
            valid_auc = roc_auc_score(valid_y_cpu, valid_probs.cpu().numpy())
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                run.log({
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "train_auc": train_auc,
                    "valid_auc": valid_auc,
                    "epoch": epoch + 1
                })
            
            
        saved = False
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            if valid_auc>=0.7:
                model_path = os.path.join(base_path, "best.pth")
                torch.save(model.state_dict(), model_path)
                artifact = wandb.Artifact('best_model', type='model')
                artifact.add_file(model_path)
                run.log_artifact(artifact)
                saved = True
        
        tqdm.write(f"T Loss: {train_loss:.4f}, V Loss: {valid_loss:.4f}, T AUC: {train_auc:.4f}, V AUC: {valid_auc:.4f} counter: {stopper.counter} {'(Saved)' if saved else ''}")
        
        # Early stopping
        if stopper.early_stop(valid_loss.item()):
            tqdm.write(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Save final model
    model_path = os.path.join(base_path, "final.pth")
    torch.save(model.state_dict(), model_path)
    artifact = wandb.Artifact('final', type='model')
    artifact.add_file(model_path)
    run.log_artifact(artifact)
    
    run.log({
            "best_valid_auc": best_valid_auc
        })
        
    return model



def outter_train_loop(model_class, hidden_size=None, num_layers=None, epochs=None, use_features=None):
    df_raw, df_ML, row_train, row_valid, row_test,feature_names = data_from_pickle()

    nn_train = df_ML.loc[row_train, ['label', 'data']]
    nn_valid = df_ML.loc[row_valid, ['label', 'data']]
    features_train = df_ML.loc[row_train, feature_names]
    features_valid = df_ML.loc[row_valid, feature_names]

    
    if "WANDB_SWEEP_ID" in os.environ:
        run = wandb.init()
        config = wandb.config
        runContext = contextManager(run)
        hidden_size=config['hidden_size']
        num_layers=config['num_layers']
        use_features=config['use_features']
        epochs=config['epochs']
    else:
        run = None
        
    feature_size = features_train.shape[1] if use_features else 0
    model = model_class(input_size=1, hidden_size=hidden_size, num_layers=num_layers, feature_size=feature_size)
    model = big_train_loop(
        model=model,
        nn_train=nn_train,
        nn_valid=nn_valid,
        features_train=features_train,
        features_valid=features_valid,
        epochs = epochs,
        use_features=use_features,
        run = run)
    return model
