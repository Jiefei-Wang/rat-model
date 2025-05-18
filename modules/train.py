import torch

def mytrain(model, train_loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0
    n_samples = 0
    for batch_x, batch_y, lengths in train_loader:
        batch_x, batch_y, lengths = batch_x.to(device), batch_y.to(device), lengths.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x, lengths)
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        n_samples += batch_x.size(0)
        epoch_loss += loss.item() * batch_x.size(0)
    return epoch_loss / n_samples

def myvalidate(model, valid_loader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    n_samples = 0
    with torch.no_grad():
        for batch_x, batch_y, lengths in valid_loader:
            batch_x, batch_y, lengths = batch_x.to(device), batch_y.to(device), lengths.to(device)
            outputs = model(batch_x, lengths)
            loss = criterion(outputs, batch_y)
            n_samples += batch_x.size(0)
            epoch_loss += loss.item() * batch_x.size(0)
    return epoch_loss / n_samples  
