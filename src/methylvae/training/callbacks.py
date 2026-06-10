# ==============================================================================
# Script:           callbacks.py
# Purpose:          Defines callbacks for Beta-VAE model training and sweeps.
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             01/01/2026
# ==============================================================================

from typing import Optional
import optuna
import torch

from optuna.integration import PyTorchLightningPruningCallback
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

def configure_callbacks(trial: Optional[optuna.trial.Trial] = None,
                        checkpoint_dir: Optional[str] = None,
                        early_stopping_patience: int = 20,
                        early_stopping_min_delta: float = 1e-5) -> list:
    """
    Configures and returns callbacks for BetaVAE training.

    Parameters
    ----------
    trial                    : Optuna Trial, if running a sweep. Adds pruning callback.
    checkpoint_dir           : Directory for model checkpoints.
    early_stopping_patience  : Epochs without val_loss improvement before stopping.
    early_stopping_min_delta : Minimum improvement threshold for early stopping.
    """

    # checkpoint_callback = ModelCheckpoint(
    #     dirpath           = checkpoint_dir,
    #     filename          = "best-{epoch:02d}-{val_loss:.4f}",
    #     monitor           = "val_loss",
    #     mode              = "min",
    #     save_top_k        = 0,
    #     save_last         = False,
    #     save_weights_only = False,
    # )

    early_stop_callback = EarlyStopping(
        monitor   = "val_loss",
        mode      = "min",
        patience  = int(early_stopping_patience),
        min_delta = float(early_stopping_min_delta),
        strict = False,
        check_on_train_epoch_end = False,
        check_finite = True
    )
    early_stop_callback.best_score = torch.tensor(float('inf'))

    lr_monitor = LearningRateMonitor(logging_interval="step")

    callbacks = [
        # checkpoint_callback,
        early_stop_callback,
        lr_monitor,
        GradientNormCallback(log_every_n_steps=50),
    ]

    if trial is not None:
        pruning_callback = PyTorchLightningPruningCallback(
            trial = trial, monitor = "val_loss"
        )
        callbacks.append(pruning_callback)

    return callbacks


# ------------------------------------------------------------------------------

class GradientNormCallback(pl.Callback):
    """
    Logs the total L2 gradient norm after each optimiser step.

    Encoder collapse in high-dimensional methylation VAEs manifests as
    vanishing encoder gradients that are invisible in standard loss logs.
    Logging every N steps to limit overhead.

    Parameters
    ----------
    log_every_n_steps : Frequency of gradient norm logging. Default 50.
    """

    def __init__(self, log_every_n_steps: int = 50):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_after_backward(self, trainer, pl_module):
        if trainer.global_step % self.log_every_n_steps != 0:
            return
        total_norm = sum(
            p.grad.detach().data.norm(2).item() ** 2
            for p in pl_module.parameters()
            if p.grad is not None
        ) ** 0.5
        pl_module.log("grad_norm", total_norm)

# [END]