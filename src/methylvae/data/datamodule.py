# ==============================================================================
# Script:           datamodule.py
# Purpose:          Methylation DataModule for PyTorch Lightning
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             12/31/2025
# ==============================================================================

import anndata as ad
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from methylvae.data.dataset import MethylDataset

class MethylDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_adata_path: str,
                 val_adata_path: str,
                 test_adata_path: str, 
                 batch_size: int = 128,
                 num_workers: int = 2):
        super().__init__()
        
        self.train_adata_path = train_adata_path
        self.val_adata_path = val_adata_path
        self.test_adata_path = test_adata_path

        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage = None):

        # Load the pre-computed AnnData object splits
        train_adata = ad.read_h5ad(self.train_adata_path)
        val_adata = ad.read_h5ad(self.val_adata_path)
        test_adata = ad.read_h5ad(self.test_adata_path)

        # Create Datasets
        self.train_dataset = MethylDataset(train_adata)
        self.val_dataset = MethylDataset(val_adata)
        self.test_dataset = MethylDataset(test_adata)

    def train_dataloader(self):
        if not hasattr(self, "train_dataset"):
            self.setup()
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = self.num_workers,
            pin_memory = True,
            persistent_workers = False
        )
    
    def val_dataloader(self):
        if not hasattr(self, "val_dataset"):
            self.setup()
        return DataLoader(
            self.val_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = self.num_workers,
            pin_memory = True,
            persistent_workers = False
        )
    
    def test_dataloader(self):
        if not hasattr(self, "test_dataset"):
            self.setup()
        return DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = self.num_workers,
            pin_memory = True,
            persistent_workers = False
        )

# [END]