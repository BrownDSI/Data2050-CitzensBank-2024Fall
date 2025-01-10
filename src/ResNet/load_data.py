import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


# Function to convert triplets to DataLoader
def triplets_to_dataloader(triplets, batch_size):
    anchors = np.stack([triplet[0] for triplet in triplets]).astype(np.float32)
    positives = np.stack([triplet[1] for triplet in triplets]).astype(np.float32)
    negatives = np.stack([triplet[2] for triplet in triplets]).astype(np.float32)
    
    anchors = anchors.transpose((0, 3, 1, 2))  # Ensure images have shape (batch_size, 3, 224, 224)
    positives = positives.transpose((0, 3, 1, 2))
    negatives = negatives.transpose((0, 3, 1, 2))
    
    dataset = TensorDataset(
        torch.tensor(anchors, dtype=torch.float32),
        torch.tensor(positives, dtype=torch.float32),
        torch.tensor(negatives, dtype=torch.float32)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)