import torch
import torch.nn.utils.rnn as rnn_utils
import numpy as np

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
        device: Target device for tensors
    
    Returns:
        tuple: (padded_data, labels, lengths) as tensors
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract sequences and labels
    sequences = dataframe['data'].tolist()
    labels = dataframe['label'].tolist()
    
    actual_lengths = []
    
    actual_lengths=[len(seq) for seq in sequences]
    max_len = np.max(actual_lengths)
    sequences_tensor = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
    
    # Pad sequences to max_length: batch_size x Time
    padded_sequences = rnn_utils.pad_sequence(
        sequences_tensor,
        batch_first=True,
        padding_value=0.0
    )
    
    # shape is (batch_size, Time, 1)
    padded_sequences = padded_sequences.unsqueeze(-1)  
    
    # Convert to tensors and move to device
    padded_tensor = padded_sequences.to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
    actual_lengths = torch.tensor(actual_lengths, dtype=torch.long).to('cpu') # Keep lengths on CPU for packing
    
    return padded_tensor, labels_tensor, actual_lengths


def create_data_loader(x, y, lengths, features=None, batch_size=32, shuffle=True):
    """Create a DataLoader for batched training"""
    if features is not None:
        dataset = torch.utils.data.TensorDataset(x, y, lengths, features)
    else:
        dataset = torch.utils.data.TensorDataset(x, y, lengths)
    
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        pin_memory=True if x.device.type == 'cpu' else False
    )


def big_train_loop(model,
                   nn_train, nn_valid,
                   features_train=None, features_valid=None,
                   device=None,
                   epochs=100,
                   optimizer=None,
                   run=None,
                   batch_size=32,
                   output_dir="output/evaluation",
                   model_name="model"):

    import gc
    import os
    import matplotlib.pyplot as plt
    import torch
    from sklearn.metrics import roc_auc_score
    from tqdm import tqdm

    gc.collect()
    torch.cuda.empty_cache()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stopper = EarlyStopper(patience=500, min_delta=0)

    # Save to output/evaluation instead of temp directory
    os.makedirs(output_dir, exist_ok=True)

    best_model_path = os.path.join(output_dir, f"{model_name}_best_model_auc.pt")

    model = model.to(device)

    # Convert datasets
    train_x, train_y, train_lengths = dataframe_to_tensors(nn_train, device=device)
    valid_x, valid_y, valid_lengths = dataframe_to_tensors(nn_valid, device=device)

    valid_y_cpu = valid_y.cpu().numpy()

    use_features = features_train is not None
    if use_features:
        features_train = torch.tensor(features_train.to_numpy(), dtype=torch.float32).to(device)
        features_valid = torch.tensor(features_valid.to_numpy(), dtype=torch.float32).to(device)

    train_loader = create_data_loader(train_x, train_y, train_lengths, features_train, batch_size, shuffle=True)

    lr = 0.001
    criterion = torch.nn.CrossEntropyLoss()

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if run is not None:
        run.watch(model, log_freq=100, log="all")

    best_valid_auc = -float("inf")
    best_epoch = -1

    # store AUC curves
    train_auc_curve = []
    valid_auc_curve = []

    for epoch in tqdm(range(epochs), desc=f"Training {model_name}", unit="epoch"):

      
        # Training
        model.train()
        total_train_loss = 0
        train_outputs_list = []
        train_targets_list = []

        for batch_data in train_loader:

            if use_features:
                batch_x, batch_y, batch_lengths, batch_features = batch_data
            else:
                batch_x, batch_y, batch_lengths = batch_data
                batch_features = None

            optimizer.zero_grad()

            outputs = model(batch_x, batch_lengths, batch_features)
            loss = criterion(outputs, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_train_loss += loss.item()

            train_outputs_list.append(outputs.detach())
            train_targets_list.append(batch_y)

        avg_train_loss = total_train_loss / len(train_loader)

        all_train_outputs = torch.cat(train_outputs_list, dim=0)
        all_train_targets = torch.cat(train_targets_list, dim=0)

        
        # Validation
        model.eval()

        with torch.no_grad():

            valid_outputs = model(valid_x, valid_lengths, features_valid)
            valid_loss = criterion(valid_outputs, valid_y)

            train_probs = torch.softmax(all_train_outputs, dim=1)[:,1]
            train_auc = roc_auc_score(
                all_train_targets.cpu().numpy(),
                train_probs.cpu().numpy()
            )

            valid_probs = torch.softmax(valid_outputs, dim=1)[:,1]
            valid_auc = roc_auc_score(
                valid_y_cpu,
                valid_probs.cpu().numpy()
            )

        train_auc_curve.append(train_auc)
        valid_auc_curve.append(valid_auc)

      
        # Save best checkpoint
        if valid_auc > best_valid_auc:

            best_valid_auc = float(valid_auc)
            best_epoch = epoch + 1

            torch.save(model.state_dict(), best_model_path)

        tqdm.write(
            f"[{model_name}] Epoch {epoch+1} | "
            f"T Loss {avg_train_loss:.4f} | "
            f"V Loss {valid_loss:.4f} | "
            f"T AUC {train_auc:.4f} | "
            f"V AUC {valid_auc:.4f} | "
            f"Best AUC {best_valid_auc:.4f} (epoch {best_epoch})"
        )

        # Early stopping
        if stopper.early_stop(valid_loss.item()):
            tqdm.write(f"[{model_name}] Early stopping at epoch {epoch+1}")
            break

    # Reload best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    print(f"\n[{model_name}] Loaded best model from epoch {best_epoch} with validation AUC {best_valid_auc:.4f}")

    # Plot AUC curve
    plt.figure(figsize=(7,5))

    plt.plot(train_auc_curve, label="Train AUC")
    plt.plot(valid_auc_curve, label="Validation AUC")

    plt.axvline(best_epoch-1, linestyle="--", color="red", label="Best AUC epoch")

    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title(f"{model_name} — Training vs Validation AUC")
    plt.legend()

    plot_path = os.path.join(output_dir, f"{model_name}_auc_training_curve.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"[{model_name}] AUC curve saved to: {plot_path}")
    print(f"[{model_name}] Best model saved to: {best_model_path}")

    return model