import torch
import os
import wandb
import matplotlib.pyplot as plt
import pickle
import torch.nn.utils.rnn as rnn_utils
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay

from modules.utils import save_model, plot_learning_curve

def dataframe_to_tensors(dataframe, max_length=100, device=None):
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
        lengths = [len(seq) for seq in seqs]
        seqs = [seq[:max_length] for seq in seqs]  
        # pad 0 if the sequence is shorter than max_length
        for seq in seqs:
            if len(seq) < max_length:
                seq.extend([0.0] * (max_length - len(seq)))
        truncated_sequences.append(seqs)
        actual_lengths.append(lengths)
    
    sequences_tensor = torch.tensor(truncated_sequences, dtype=torch.float32)
    
    # Convert to tensors and move to device
    sequences_tensor = sequences_tensor.to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
    actual_lengths = torch.tensor(actual_lengths, dtype=torch.long).to(device)
    
    return sequences_tensor, labels_tensor, actual_lengths

def big_train_loop(model_name, model,
                   nn_train, nn_valid, nn_test,
                   output_dir,
                   device=None,
                   epochs=100,
                   batch_size=64):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    
    # Convert all data to tensors and load directly to GPU
    train_x, train_y, train_lengths = dataframe_to_tensors(nn_train, max_length=100, device=device)
    valid_x, valid_y, valid_lengths = dataframe_to_tensors(nn_valid, max_length=100, device=device)
    test_x, test_y, test_lengths = dataframe_to_tensors(nn_test, max_length=100, device=device)

    lr = 0.001
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    wandb.init(
        project=f"rat-frustration-{model_name}",
        name=f"model-run-{model.hidden_size}-{model.num_layers}",
        config={
            "model": model_name,
            "input_size": 1,
            "hidden_size": model.hidden_size,
            "num_layers": model.num_layers,
            "learning_rate": lr,
            "optimizer": "Adam",
            "loss_fn": "CrossEntropyLoss",
            "batch_size": batch_size,
            "epochs": epochs        }
    )
    
    os.makedirs(output_dir, exist_ok=True)
    train_losses, valid_losses = [], []
    best_valid_loss = float('inf')
    
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
        
        train_losses.append(train_loss.item())
        valid_losses.append(valid_loss.item())

        wandb.log({
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "epoch": epoch + 1
        })

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_model(model, epoch, best_valid_loss, output_dir)
            print(f"Saved best model at epoch {epoch+1}")

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

    plot_learning_curve(train_losses, valid_losses, output_dir)    # Test AUC evaluation
    model.eval()
    with torch.no_grad():
        test_logits = model(test_x, test_lengths)
        test_probs = torch.softmax(test_logits, dim=1)[:, 1]
        
    y_true_test = test_y.cpu().numpy()
    y_scores_test = test_probs.cpu().numpy()
    
    test_auc = roc_auc_score(y_true_test, y_scores_test)
    print(f"\nAUC Score on Test Set: {test_auc:.4f}")
    wandb.log({"Test AUC": test_auc})

    # Train AUC evaluation
    with torch.no_grad():
        train_logits = model(train_x, train_lengths)
        train_probs = torch.softmax(train_logits, dim=1)[:, 1]
        
    y_true_train = train_y.cpu().numpy()
    y_scores_train = train_probs.cpu().numpy()
    
    train_auc = roc_auc_score(y_true_train, y_scores_train)
    print(f"AUC Score on Full Training Set: {train_auc:.4f}")
    wandb.log({"Train AUC": train_auc})

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_true_test, y_scores_test)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {test_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.show()

    # Plot Confusion Matrix
    y_pred = [1 if score >= 0.5 else 0 for score in y_scores_test]
    cm = confusion_matrix(y_true_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['FR1', 'EXT'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.show()

    # Save ROC data to pickle file
    roc_data = {'y_true': y_true_test, 'y_scores': y_scores_test}
    roc_path = os.path.join(output_dir, f'{model_name}_roc_data.pkl')
    with open(roc_path, 'wb') as f:
        pickle.dump(roc_data, f)
    print(f"Saved ROC data for {model_name} to {roc_path}")

    return model, test_auc, train_auc
