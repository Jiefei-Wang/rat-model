import torch
import os
import wandb
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay

from modules.dataset import PressDataset, collate_fn
from modules.nn_management import get_NN_data
from modules.utils import save_model, plot_learning_curve

def mytrain(model, train_loader, criterion, optimizer, device):
    model.train()
    epoch_loss, n_samples = 0.0, 0
    for batch_x, batch_y, lengths in train_loader:
        batch_x, batch_y, lengths = batch_x.to(device), batch_y.to(device), lengths.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x, lengths)
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        n_samples += batch_x.size(0)
        epoch_loss += loss.item() * batch_x.size(0)
    return epoch_loss / n_samples

def myvalidate(model, valid_loader, criterion, device):
    model.eval()
    epoch_loss, n_samples = 0.0, 0
    with torch.no_grad():
        for batch_x, batch_y, lengths in valid_loader:
            batch_x, batch_y, lengths = batch_x.to(device), batch_y.to(device), lengths.to(device)
            outputs = model(batch_x, lengths)
            loss = criterion(outputs, batch_y)
            n_samples += batch_x.size(0)
            epoch_loss += loss.item() * batch_x.size(0)
    return epoch_loss / n_samples

def big_train_loop(df_raw, model_name, model, output_dir,
                   device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                   epochs=100,
                   batch_size=64):

    model = model.to(device)
    df_train, df_val, df_test = get_NN_data(df_raw)
    train_dataset = PressDataset(df_train)
    valid_dataset = PressDataset(df_val)
    test_dataset = PressDataset(df_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

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
            "epochs": epochs
        }
    )

    os.makedirs(output_dir, exist_ok=True)
    train_losses, valid_losses = [], []
    best_valid_loss = float('inf')

    for epoch in range(epochs):
        train_loss = mytrain(model, train_loader, criterion, optimizer, device)
        valid_loss = myvalidate(model, valid_loader, criterion, device)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

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

    plot_learning_curve(train_losses, valid_losses, output_dir)

    # Test AUC evaluation
    model.eval()
    y_true_test, y_scores_test = [], []
    with torch.no_grad():
        for batch_x, batch_y, lengths in test_loader:
            batch_x, batch_y, lengths = batch_x.to(device), batch_y.to(device), lengths.to(device)
            logits = model(batch_x, lengths)
            probs = torch.softmax(logits, dim=1)[:, 1]
            y_scores_test.extend(probs.cpu().numpy())
            y_true_test.extend(batch_y.cpu().numpy())

    test_auc = roc_auc_score(y_true_test, y_scores_test)
    print(f"\nAUC Score on Test Set: {test_auc:.4f}")
    wandb.log({"Test AUC": test_auc})

    # Train AUC evaluation
    y_true_train, y_scores_train = [], []
    with torch.no_grad():
        for batch_x, batch_y, lengths in train_loader:
            batch_x, batch_y, lengths = batch_x.to(device), batch_y.to(device), lengths.to(device)
            logits = model(batch_x, lengths)
            probs = torch.softmax(logits, dim=1)[:, 1]
            y_scores_train.extend(probs.cpu().numpy())
            y_true_train.extend(batch_y.cpu().numpy())

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
