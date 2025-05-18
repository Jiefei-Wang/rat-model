import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.utils.rnn as rnn_utils

# Custom Dataset for press data
class PressDataset(Dataset):
    def __init__(self, dataframe):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing 'data' (sequences) and 'label' (targets)
        """
        self.data = dataframe['data'].tolist()
        self.labels = dataframe['label'].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the data sample.
        
        Returns:
            tuple: (data, label) where 'data' is a tensor containing the sequence and 'label' is the class label.
        """
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# Collate function to pad sequences of varying lengths
def collate_fn(batch):
    """
    Args:
        batch (list): List of tuples (data, label) where data is a sequence and label is the target.
    
    Returns:
        tuple: (padded_sequences, labels, lengths) where:
            - padded_sequences is a tensor of padded sequences
            - labels is a tensor of class labels
            - lengths is a tensor containing the lengths of each sequence in the batch
    """
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    
    # Pad the sequences to the same length
    padded_seqs = rnn_utils.pad_sequence(sequences, batch_first=True)
    
    return padded_seqs, torch.tensor(labels), lengths
