# =============================================================================
# Script:           train.py
# Purpose:          Trains a single run
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             03/26/2026
#
# Configurations:   betaVAE_train.yaml
# =============================================================================

import gc
import os
import torch
import optuna
import wandb
import pytorch_lightning as pl
from pathlib import Path

from MethylCDM.utils.utils import resolve_path
from MethylCDM.models.betaVAE import BetaVAE
from MethylCDM.data.methylation_datamodule import MethylDataModule
from MethylCDM.constants import BETAVAE_CHECKPOINT_DIR
from MethylCDM.utils.training_utils import (
    configure_callbacks, 
    configure_loggers
)

def run_training(config, run_name, seed):
    try:
        pl.seed_everything(seed, workers=True)

        # 1. Setup Model & Data (using your existing classes)
        # Note: derive decoder dims as before
        from .betaVAE_objective import _derive_decoder_dims 
        encoder_dims = config['encoder_dims']
        
        model = BetaVAE(
            input_dim     = config['input_dim'],
            latent_dim    = config['latent_dim'],
            encoder_dims  = encoder_dims,
            decoder_dims  = _derive_decoder_dims(encoder_dims),
            beta          = config['beta'],
            input_dropout = config['input_dropout'],
            num_cycles    = config['num_cycles'],
            lr            = config['lr']
        )

        datamodule = MethylDataModule(
            train_adata_path = config['train_adata_path'],
            val_adata_path   = config["val_adata_path"],
            test_adata_path  = config["test_adata_path"],
            batch_size       = config['batch_size'],
            num_workers      = config['num_workers']
        )

        # 2. Setup Callbacks/Loggers (Pass None for Trial)
        # Assuming you've updated these to handle trial=None
        # Initialize callbacks and loggers
        checkpoint_dir = config.get('checkpoint_dir', '')
        checkpoint_dir = resolve_path(checkpoint_dir, BETAVAE_CHECKPOINT_DIR)
        checkpoint_dir = os.path.join(checkpoint_dir, study_name, f"trial_{trial.number}")
        Path(checkpoint_dir).mkdir(parents = True, exist_ok = True)
        
        callbacks = configure_callbacks(None, checkpoint_dir)
        logger = configure_loggers(None, run_name)[0]

        # 3. Trainer
        trainer = pl.Trainer(
            max_epochs        = config['max_epochs'],
            callbacks         = callbacks,
            logger            = logger,
            accelerator       = "auto",
            devices           = 1
        )

        trainer.fit(model, datamodule)
        return trainer.callback_metrics.get('val_loss').item()

    finally:
        wandb.finish()
        torch.cuda.empty_cache()
        gc.collect()