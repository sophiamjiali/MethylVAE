# ==============================================================================
# Script:           betaVAE_objective.py
# Purpose:          Defines the Optuna objective function for training.
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             12/31/2025
#
# Configurations:   betaVAE.yaml
#
# Notes:            Compatible with an Optuna hyperparameter sweep
# ==============================================================================

import os
import torch
import optuna
import wandb
import lightning.pytorch as pl
from pathlib import Path

from MethylCDM.utils.utils import resolve_path
from MethylCDM.models.betaVAE import BetaVAE
from MethylCDM.data.methylation_datamodule import MethylDataModule
from MethylCDM.constants import BETAVAE_CHECKPOINT_DIR
from MethylCDM.utils.training_utils import (
    configure_callbacks, 
    configure_loggers
)

def objective(trial, study_name, config):
    """
    Objective function for an an Optuna hyperparameter optimization trial.

    Parameters
    ----------
    trial (Trial): Optuna Trial object to sample hyperparameter values
    study_name (str): name of the study
    config (Dict): Dictionary containing hyperparameter sweep value ranges

    Returns
    -------
    (float): validation loss to be minimized by Optuna
    """

    # Wrap the trial to prune it if it fails
    try:
        # Suggest hyperparameters
        latent_dim = trial.suggest_int("latent_dim", *config['latent_dim'])
        beta = trial.suggest_float("beta", *config['beta'], log = True)
        lr = trial.suggest_float("lr", *config['lr'], log = True)
        batch_size = trial.suggest_int("batch_size", *config['batch_size'])

        # Initialize the model
        model = BetaVAE(
            input_dim = config['input_dim'],
            latent_dim = latent_dim,
            encoder_dims = config['encoder_dims'],
            decoder_dims = config['decoder_dims'],
            beta = beta,
            lr = lr
        )

        # Initialize the DataModule
        datamodule = MethylDataModule(
            train_adata_path = config['train_adata_path'],
            val_adata_path = config["val_adata_path"],
            test_adata_path = config["test_adata_path"],
            batch_size = batch_size,
            num_workers = config['num_workers']
        )

        # Initialize callbacks and loggers
        checkpoint_dir = config.get('checkpoint_dir', '')
        checkpoint_dir = resolve_path(checkpoint_dir, BETAVAE_CHECKPOINT_DIR)
        checkpoint_dir = os.path.join(checkpoint_dir, study_name)
        Path(checkpoint_dir).mkdir(parents = True, exist_ok = True)

        callbacks = configure_callbacks(trial, checkpoint_dir)
        logger = configure_loggers(trial, study_name)[0]

        # Initialize the Trainer and train the model
        trainer = pl.Trainer(
            max_epochs = config['max_epochs'],
            callbacks = callbacks,
            logger = logger,
            accelerator = "auto",
            devices = "auto"
        )
        trainer.fit(model, datamodule)

        # Fetch validation metrics
        val_loss = trainer.callback_metrics['val_loss'].item()
        wandb.finish()

        # Clean up to prevent memory leaks
        del model
        del datamodule
        del trainer
        torch.cuda.empty_cache()

        return val_loss
    
    except Exception as e:
        print(f"Trial {trial.number} failed with error {e}")
        wandb.finish()
        raise optuna.TrialPruned()
