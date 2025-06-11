import torch
import os
import wandb
import torch.nn.utils.rnn as rnn_utils
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import gc

import datetime
import pandas as pd
from tqdm import tqdm

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


class wandbLogger:
    def __init__(self, project, name, config):
        """
        Initialize the wandb logger with given parameters.
        """
        self.project = project
        self.name = name
        self.config = config
        
    def creat_logger(self):
        """
        Create a wandb run with the specified project, name, and config.
        """
        wandb.init(
            project=self.project,
            name=self.name,
            config=self.config,
            reinit=True
        )
        
    
    def log(self, data):
        """
        Log data to wandb.
        """
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")
        
        wandb.log(data)
    
    def is_exists(self):
        """
        Check if a run with the same name already exists in the project.
        
        Returns:
            bool: True if run exists, False otherwise
        """
        try:
            api = wandb.Api()
            runs = api.runs(f"{api.default_entity}/{self.project}")
            for run in runs:
                if run.name == self.name:
                    return True
        except Exception as e:
            return False
        return False
    
    def __del__(self):
        """
        Finish the wandb run when the logger is deleted.
        """
        wandb.finish()
    





def big_train_loop(model,
                   nn_train, nn_valid, nn_test,
                   features_train=None, features_valid=None, features_test=None,
                   output_base="output/nn",
                   device=None,
                   epochs=100, skip_if_exists=True):
    ## Free memory
    gc.collect()
    torch.cuda.empty_cache()
    
    
    model_name = model.name
    model_params = model.params
    
    project_name=f"rat-frustration-{model_name}"
    run_name = model_params
    
    run_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = wandbLogger(
        project=project_name,
        name=run_name,
        config={
            "model": model_name,
            "input_size": 1,
            "hidden_size": model.hidden_size,
            "num_layers": model.num_layers,
            "optimizer": "Adam",
            "loss_fn": "CrossEntropyLoss",
            "epochs": epochs,
            "datetime": run_datetime
        }
    )
    
    
    if skip_if_exists and logger.is_exists():
        print(f"Run {run_name} already exists in project {project_name}. Skipping training.")
        return None, None, None
    
    logger.creat_logger()
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    output_dir = os.path.join(output_base, f"{model_name}_{model_params}") 
    
    
    # Convert all data to tensors and load directly to GPU
    train_x, train_y, train_lengths = dataframe_to_tensors(nn_train, device=device)
    valid_x, valid_y, valid_lengths = dataframe_to_tensors(nn_valid, device=device)
    test_x, test_y, test_lengths = dataframe_to_tensors(nn_test, device=device)
    
    train_y_cpu = train_y.cpu().numpy()
    valid_y_cpu = valid_y.cpu().numpy()
    test_y_cpu = test_y.cpu().numpy()
    
    
    if (features_train is not None):
        features_train = torch.tensor(features_train.to_numpy(), dtype=torch.float32).to(device)
        features_valid = torch.tensor(features_valid.to_numpy(), dtype=torch.float32).to(device)
        features_test = torch.tensor(features_test.to_numpy(), dtype=torch.float32).to(device)
        manual_feature_size = features_train.shape[1]
    else:
        features_train = None
        features_valid = None
        features_test = None
        manual_feature_size = 0
    
    

    lr = 0.001
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs(output_dir, exist_ok=True)
    best_valid_loss = float('inf')
    
    epoch_list = []
    train_loss_list = []
    valid_loss_list = []
    train_auc_list = []
    valid_auc_list = []
    
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
        
        saved = False
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            model_path = os.path.join(output_dir, "best.pth")
            torch.save(model.state_dict(), model_path)
            saved = True
        
        # Log training and validation metrics every 10 epochs
        with torch.no_grad():
            train_probs = torch.softmax(train_outputs, dim=1)[:, 1]
            train_auc = roc_auc_score(train_y_cpu, train_probs.detach().cpu().numpy())
            
            valid_probs = torch.softmax(valid_outputs, dim=1)[:, 1]
            valid_auc = roc_auc_score(valid_y_cpu, valid_probs.cpu().numpy())
            
            epoch_list.append(epoch)
            train_loss_list.append(train_loss.item())
            valid_loss_list.append(valid_loss.item())
            train_auc_list.append(train_auc)
            valid_auc_list.append(valid_auc)
            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.log({
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "train_auc": train_auc,
                    "valid_auc": valid_auc,
                    "epoch": epoch + 1
                })
                
            tqdm.write(f"T Loss: {train_loss:.4f}, V Loss: {valid_loss:.4f}, T AUC: {train_auc:.4f}, V AUC: {valid_auc:.4f} {'(Saved)' if saved else ''}")

    
    # Save final model
    final_model_path = os.path.join(output_dir, f"final.pth")
    torch.save(model.state_dict(), final_model_path)
    
    
    ## Evaluate on Test Set
    model.eval()
    with torch.no_grad():
        test_logits = model(test_x, test_lengths, features_test)
        test_probs = torch.softmax(test_logits, dim=1)[:, 1]
        
    test_auc = roc_auc_score(test_y_cpu, test_probs.cpu().numpy())
    logger.log({
        "Test AUC": test_auc,
        "Best Valid Loss": best_valid_loss,
        "Best Valid AUC": max(valid_auc_list)
        })

    ## making a dataframe for loss, auc
    train_info = pd.DataFrame({
        "epoch": epoch_list,
        "train_loss": train_loss_list,
        "valid_loss": valid_loss_list,
        "train_auc": train_auc_list,
        "valid_auc": valid_auc_list
    })
    
    prediction = test_probs.cpu().numpy()
    return model, train_info, prediction
