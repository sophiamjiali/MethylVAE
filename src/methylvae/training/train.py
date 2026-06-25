# ==============================================================================
# Script:           train.py
# Purpose:          Trains a single BetaVAE run.
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             03/26/2026
#
# Expects a flat merged config dict produced by config.merge_configs().
# All keys are documented in config.py _MERGE_MAP and merge_configs().
# ==============================================================================

import lightning.pytorch as pl

from pathlib import Path
from typing import Dict

from methylvae.data.datamodule import MethylDataModule
from methylvae.models.betaVAE import BetaVAE
from methylvae.training.callbacks import configure_callbacks
from methylvae.logging.logger import configure_loggers


def train(config: Dict,
          run_name: str,
          seed: int,
          trial = None,
          study_name = None) -> Dict:
    """
    Reusable training function for single runs and Optuna sweeps.

    Parameters
    ----------
    config     : Flat merged config dict from merge_configs(). Must contain all
                 keys listed below. Sweep entry points override swept keys before
                 passing here.
    run_name   : Unique run identifier for logging and checkpointing.
    seed       : Random seed.
    trial      : Optuna Trial, if called from a sweep. None for single runs.
    study_name : Optuna study name for checkpoint subdirectory. None for single runs.

    Required config keys
    --------------------
    From data.yaml    : input_dim, train_adata_path, val_adata_path,
                        test_adata_path, num_workers
    From training.yaml: max_epochs, batch_size, lr, gradient_clip_val,
                        early_stopping_patience, early_stopping_min_delta, seed
    From loss.yaml    : beta, free_bits, num_cycles
    Swept / fixed     : latent_dim, encoder_dims, input_dropout

    Returns
    -------
    Dict : trainer.callback_metrics at end of training.
    """

    pl.seed_everything(seed, workers=True)

    # --- Model ----------------------------------------------------------------

    encoder_dims = config["encoder_dims"]
    model = BetaVAE(
        input_dim       = config["input_dim"],
        latent_dim      = config["latent_dim"],
        encoder_dims    = encoder_dims,
        decoder_dims    = list(reversed(encoder_dims[:-1])),
        beta            = config["beta"],
        free_bits       = config.get("free_bits", 0.5),
        decoder_dropout = config['decoder_dropout'],
        input_dropout   = config["input_dropout"],
        num_cycles      = config["num_cycles"],
        lr              = config["lr"],
        mu_reg_weight   = config["mu_reg_weight"]
    )

    # --- Data -----------------------------------------------------------------

    datamodule = MethylDataModule(
        train_adata_path = config["train_adata_path"],
        val_adata_path   = config["val_adata_path"],
        test_adata_path  = config["test_adata_path"],
        batch_size       = config["batch_size"],
        num_workers      = config["num_workers"],
    )

    # --- Checkpoint directory -------------------------------------------------

    checkpoint_dir = config["paths.checkpoint_dir"]
    if trial is not None and study_name is not None:
        checkpoint_dir = Path(checkpoint_dir) / study_name / f"trial_{trial.number}"
    else:
        checkpoint_dir = Path(checkpoint_dir) / run_name

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # --- Callbacks & logger ---------------------------------------------------

    callbacks = configure_callbacks(
        trial                    = trial,
        checkpoint_dir           = str(checkpoint_dir),
        early_stopping_patience = int(config.get("early_stopping_patience", 20)),
        early_stopping_min_delta = float(config.get("early_stopping_min_delta", 1e-5)),
    )
    logger = configure_loggers(trial=trial, study_name=study_name or run_name)[0]

    # --- Trainer --------------------------------------------------------------

    trainer = pl.Trainer(
        max_epochs        = config["max_epochs"],
        callbacks         = callbacks,
        logger            = logger,
        accelerator       = "cpu",
        devices           = 1,
        num_nodes         = 1,
        deterministic     = False,
        log_every_n_steps = 1,
        gradient_clip_val = config.get("gradient_clip_val", 1.0),
        gradient_clip_algorithm = "norm",
        check_val_every_n_epoch = 1,
        enable_checkpointing = False,
        enable_progress_bar = False
    )

    trainer.fit(model, datamodule = datamodule)
    return trainer.callback_metrics

# [END]