# =============================================================================
# Script:           train.py
# Purpose:          Trains a single run
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             03/26/2026
#
# Configurations:   betaVAE_train.yaml
# =============================================================================

import pytorch_lightning as pl

from pathlib import Path
from typing import Dict

from methylvae.data.datamodule import MethylDataModule
from methylvae.models.betaVAE import BetaVAE
from methylvae.utils.training_utils import configure_callbacks, configure_loggers
from methylvae.utils.utils import resolve_path
from methylvae.constants import BETAVAE_CHECKPOINT_DIR


def train(config: Dict, 
          run_name: str, 
          seed: int, 
          trial = None, 
          study_name = None):
    """
    Reusable training function used by both single runs and Optuna sweeps.
    """

    pl.seed_everything(seed, workers = True)

    # Initialize the model from the user-provided configurations
    encoder_dims = config['encoder_dims']
    model = BetaVAE(
        input_dim     = config['input_dim'],
        latent_dim    = config['latent_dim'],
        encoder_dims  = encoder_dims,
        decoder_dims  = list(reversed(encoder_dims[:-1])),
        beta          = config['beta'],
        input_dropout = config['input_dropout'],
        num_cycles    = config['num_cycles'],
        lr            = config['lr']
    )

    # Initialize the datamodule using the user-provided configurations
    datamodule = MethylDataModule(
        train_adata_path = config['train_adata_path'],
        val_adata_path   = config["val_adata_path"],
        test_adata_path  = config["test_adata_path"],
        batch_size       = config['batch_size'],
        num_workers      = config['num_workers']
    )

    # Initialize the checkpoints directory
    checkpoint_dir = config.get('checkpoint_dir', '')
    checkpoint_dir = resolve_path(checkpoint_dir, BETAVAE_CHECKPOINT_DIR)

    if trial is not None and study_name is not None:
        checkpoint_dir = f"{checkpoint_dir}/{study_name}/trial_{trial.number}"
    else:
        checkpoint_dir = f"{checkpoint_dir}/{run_name}"

    Path(checkpoint_dir).mkdir(parents = True, exist_ok = True)
    
    # Initialize callbacks and loggers
    callbacks = configure_callbacks(None, checkpoint_dir)
    logger = configure_loggers(None, run_name)[0]

    # Initialize the trainer
    trainer = pl.Trainer(
        max_epochs        = config['max_epochs'],
        callbacks         = callbacks,
        logger            = logger,
        accelerator       = "auto",
        devices           = 1,
        num_nodes         = 1,
        strategy          = "auto",
        deterministic     = False,
        log_every_n_steps = 1
    )

    trainer.fit(model, datamodule)
    return trainer.callback_metrics

# [END]