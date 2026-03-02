# ==============================================================================
# Script:           training_utils.py
# Purpose:          Defines callbacks for Beta-VAE model training and sweeps
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             01/01/2026
# ==============================================================================

import os
import wandb
from lightning.pytorch.callbacks import (
    ModelCheckpoint, 
    EarlyStopping,
    LearningRateMonitor
)
from optuna.integration import PyTorchLightningPruningCallback
from lightning.pytorch.loggers import WandbLogger

def configure_callbacks(trial = None, checkpoint_dir = None):
    """
    Configures and returns a list of callbacks for Beta-VAE
    model training. Enables pruning of unpromising trials in
    Optuna.
    """

    # Model checkpoint to save the state of the optimal model
    checkpoint_path = os.path.join(
        checkpoint_dir, "best-{epoch:02d}-{val_loss:.4f}"
    )
    checkpoint_callback = ModelCheckpoint(
        monitor = "val_loss",
        mode = "min",
        save_top_k = 1,
        filename = checkpoint_path
    )

    # Stops training early if validation loss doesn't improve
    early_stop_callback = EarlyStopping(
        monitor = 'val_loss',
        patience = 20,
        min_delta = 1e-4,
        mode = 'min'
    )

    # Monitors 
    lr_monitor = LearningRateMonitor(
        logging_interval = 'step'
    )

    callbacks = [checkpoint_callback, early_stop_callback, lr_monitor]

    # Pruning of unpromising runs
    if trial is not None:
        pruning_callback = PyTorchLightningPruningCallback(
            trial, monitor = 'val_loss'
        )
        callbacks.append(pruning_callback)

    return callbacks


def configure_loggers(trial = None, study_name = None):

    if trial is not None:

        # Start a new run with this trial if in a sweep
        wandb.init(
            project = study_name,
            name = f"trial_{trial.number}",
            reinit = True,
            config = trial.params
        )
        wandb_logger = WandbLogger(experiment = wandb.run)

    else:

        # Else, start a regular run that isn't in a sweep
        wandb_logger = WandbLogger(experiment = "betaVAE",
                                   name = "single_run")

    return [wandb_logger]