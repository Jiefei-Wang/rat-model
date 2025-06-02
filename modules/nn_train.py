import torch
import os
import wandb
import matplotlib.pyplot as plt
import pickle
import torch.nn.utils.rnn as rnn_utils
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay

from modules.utils import save_model, plot_learning_curve
import datetime

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
                   nn_train, nn_valid, nn_test,
                   output_base="output/nn",
                   device=None,
                   epochs=100):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model_name = model.name
    
    output_dir = os.path.join(output_base, model_name) 
    
    
    # Convert all data to tensors and load directly to GPU
    train_x, train_y, train_lengths = dataframe_to_tensors(nn_train, device=device)
    valid_x, valid_y, valid_lengths = dataframe_to_tensors(nn_valid, device=device)
    test_x, test_y, test_lengths = dataframe_to_tensors(nn_test, device=device)
    
    train_y_cpu = train_y.cpu().numpy()
    valid_y_cpu = valid_y.cpu().numpy()
    test_y_cpu = test_y.cpu().numpy()
    

    lr = 0.001
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    run_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.init(
        project=f"rat-frustration-{model_name}",
        name=f"{model_name}_{run_datetime}",
        config={
            "model": model_name,
            "input_size": 1,
            "hidden_size": model.hidden_size,
            "num_layers": model.num_layers,
            "learning_rate": lr,
            "optimizer": "Adam",
            "loss_fn": "CrossEntropyLoss",
            "epochs": epochs,
            "datetime": run_datetime
        }
    )
    
    os.makedirs(output_dir, exist_ok=True)
    best_valid_loss = float('inf')
    
    epoch_list = []
    train_loss_list = []
    valid_loss_list = []
    train_auc_list = []
    valid_auc_list = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        train_outputs = model(train_x, train_lengths)
        train_loss = criterion(train_outputs, train_y)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        
        # Validation
        model.eval()
        with torch.no_grad():
            valid_outputs = model(valid_x, valid_lengths)
            valid_loss = criterion(valid_outputs, valid_y)
        
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_model(model, epoch, best_valid_loss, output_dir)
            print(f"Saved best model at epoch {epoch+1}")
        
        # Log training and validation metrics every 10 epochs
        if (epoch + 1) % 10 == 0:
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
                
                wandb.log({
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "train_auc": train_auc,
                    "valid_auc": valid_auc,
                    "epoch": epoch + 1
                })

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Train AUC: {train_auc:.4f}, Valid AUC: {valid_auc:.4f}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, f"{model_name}_final.pth")
    torch.save(model.state_dict(), final_model_path)
    
    
    ## Evaluate on Test Set
    model.eval()
    with torch.no_grad():
        test_logits = model(test_x, test_lengths)
        test_probs = torch.softmax(test_logits, dim=1)[:, 1]
        
    test_auc = roc_auc_score(test_y_cpu, test_probs.cpu().numpy())
    wandb.log({"Test AUC": test_auc})

    return model, epoch_list, train_loss_list, valid_loss_list, train_auc_list, valid_auc_list, test_auc
