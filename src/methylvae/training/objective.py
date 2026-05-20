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

import gc
import os
import torch
import optuna
import wandb
import pytorch_lightning as pl
from pathlib import Path

from MethylVAE.utils.utils import resolve_path
from MethylVAE.models.betaVAE import BetaVAE
from MethylCDM.data.methylation_datamodule import MethylDataModule
from MethylVAE.constants import BETAVAE_CHECKPOINT_DIR
from MethylVAE.utils.training_utils import (
    configure_callbacks, 
    configure_loggers
)

def objective(trial, study_name, config):
    """
    Objective function for an an Optuna hyperparameter optimization trial.

    Parameters
    ----------
    trial      : optuna.Trial — Optuna Trial object for sampling hyperparameters
    study_name : str          — Name of the Optuna study (used for W&B and checkpointing)
    config     : dict         — Sweep configuration dictionary. Expected keys:

    # Fixed
    input_dim        (int)          : Number of CpG probes
    train_adata_path (str)          : Path to training AnnData
    val_adata_path   (str)          : Path to validation AnnData
    test_adata_path  (str)          : Path to test AnnData
    max_epochs       (int)          : Maximum training epochs
    num_workers      (int)          : DataLoader workers

    # Swept — each value is a [low, high] range unless noted
    latent_dim       ([int, int])   : e.g. [64, 256]; sampled as int
    beta             ([float,float]): e.g. [0.001, 0.01]; log-uniform
    lr               ([float,float]): e.g. [1e-4, 1e-2]; log-uniform
    input_dropout    ([float,float]): e.g. [0.1, 0.3]; uniform
    num_cycles       (list[int])    : e.g. [2, 4]; categorical
    encoder_dims     (list[list])   : list of candidate dim sequences, categorical
                                      e.g. [[1024,256,128],[2048,512,128],[2048,1024,256,128]]
    batch_size       (list[int])    : powers-of-2 candidates e.g. [64, 128, 256]

    # Optional
    checkpoint_dir   (str)          : override default checkpoint root

    Returns
    -------
    float : Validation loss to be minimised by Optuna
    """

    # Wrap the trial to prune it if it fails
    try:
        # Suggest hyperparameters
        latent_dim = trial.suggest_int("latent_dim", *config['latent_dim'])
        beta = trial.suggest_float("beta", *config['beta'], log = True)
        num_cycles = trial.suggest_categorical("num_cycles", config['num_cycles'])
        input_dropout = trial.suggest_float("input_dropout", *config['input_dropout'])
        encoder_dims_idx = trial.suggest_categorical("encoder_dims_idx", list(range(len(config['encoder_dims']))))
        encoder_dims = config['encoder_dims'][encoder_dims_idx]
        decoder_dims = _derive_decoder_dims(encoder_dims)
        lr = trial.suggest_float("lr", *config['lr'], log = True)
        batch_size = trial.suggest_categorical("batch_size", config['batch_size'])

        pl.seed_everything(trial.number, workers = True)

        # Initialize the model
        model = BetaVAE(
            input_dim     = config['input_dim'],
            latent_dim    = latent_dim,
            encoder_dims  = encoder_dims,
            decoder_dims  = decoder_dims,
            beta          = beta,
            input_dropout = input_dropout,
            num_cycles    = num_cycles,
            lr            = lr
        )

        # Initialize the DataModule
        datamodule = MethylDataModule(
            train_adata_path = config['train_adata_path'],
            val_adata_path   = config["val_adata_path"],
            test_adata_path  = config["test_adata_path"],
            batch_size       = batch_size,
            num_workers      = config['num_workers']
        )

        # Initialize callbacks and loggers
        checkpoint_dir = config.get('checkpoint_dir', '')
        checkpoint_dir = resolve_path(checkpoint_dir, BETAVAE_CHECKPOINT_DIR)
        checkpoint_dir = os.path.join(checkpoint_dir, study_name, f"trial_{trial.number}")
        Path(checkpoint_dir).mkdir(parents = True, exist_ok = True)

        callbacks = configure_callbacks(trial, checkpoint_dir)
        logger = configure_loggers(trial, study_name)[0]

        # Log the full hyperparameter set to W&B run config
        if wandb.run is not None:
            wandb.config.update({
                "latent_dim":    latent_dim,
                "beta":          beta,
                "num_cycles":    num_cycles,
                "input_dropout": input_dropout,
                "encoder_dims":  encoder_dims,
                "decoder_dims":  decoder_dims,
                "lr":            lr,
                "batch_size":    batch_size,
            })

        # Initialize the Trainer and train the model
        trainer = pl.Trainer(
            max_epochs        = config['max_epochs'],
            callbacks         = callbacks,
            logger            = logger,
            accelerator       = "auto",
            devices           = 1,
            num_nodes         = 1,
            strategy          = "auto",
            deterministic     = False, # For runtime performance
            log_every_n_steps = 1
        )
        trainer.fit(model, datamodule)

        # Fetch validation metrics
        val_loss = trainer.callback_metrics.get('val_loss')
        mean_post_var_raw = trainer.callback_metrics.get('mean_posterior_var')
        mean_post_var = mean_post_var_raw.item() if mean_post_var_raw is not None else 0.0
        
        if val_loss is None:
            raise optuna.exceptions.TrialPruned(
                f"Trial {trial.number}: val_loss not found in callback_metrics"
            )
        val_loss = val_loss.item()

        # Penalise trials where posterior var is critically low
        if mean_post_var < 0.1:
            return float('inf')   # Force Optuna to discard this trial

        trial.report(val_loss, step = config['max_epochs'])
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned(
                f"Trial {trial.number}: pruned at final epoch."
            )

        return val_loss
    
    except optuna.exceptions.TrialPruned:
        raise

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        raise

    finally:
        wandb.finish()
        for obj in ['model', 'datamodule', 'trainer']:
            if obj in dir():
                del obj
        torch.cuda.empty_cache()
        gc.collect()


def _derive_decoder_dims(encoder_dims: list) -> list:
    """
    Derive decoder hidden dims as the mirror of the encoder.
    """
    
    return list(reversed(encoder_dims[:-1]))
    