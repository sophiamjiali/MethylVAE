# ==============================================================================
# Script:           logger.py
# Purpose:          Defines the logger for Beta-VAE model training and sweeps
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             01/01/2026
# ==============================================================================

import os
import wandb
from pytorch_lightning.loggers import WandbLogger


def configure_loggers(trial = None, study_name = None):

    # Start a new run with this trial if in a sweep
    if trial is not None:
        run = wandb.init(
            project = "MethylVAE",
            group = study_name,
            name    = f"trial_{trial.number}",
            config  = trial.params
        )

    # Else, start a regular run that isn't in a sweep
    else:
        run = wandb.init(
            project = "MethylVAE",
            group = study_name if study_name is not None else "betaVAE",
            name    = "single_run",
            config = {}
        )

    wandb_logger = WandbLogger(
        experiment = run,
        log_model  = False
    )

    return [wandb_logger]

# [END]