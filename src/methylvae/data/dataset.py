# ==============================================================================
# Script:           dataset.py
# Purpose:          Methylation Dataset for PyTorch Lightning
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             12/31/2025
# ==============================================================================

import torch

from torch.utils.data import Dataset

class MethylDataset(Dataset):
    def __init__(self, adata):
        self.adata = adata

    def __len__(self):
        return self.adata.n_obs
    
    def __getitem__(self, idx):

        # Handle sparse matrices
        x = self.adata.X[idx]
        x = x.toarray().squeeze()
        x = torch.tensor(x, dtype = torch.float32)
        return {"methylation_data": x}
    
# [END]