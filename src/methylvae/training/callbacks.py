# ==============================================================================
# Script:           callbacks.py
# Purpose:          Defines callbacks for Beta-VAE model training and sweeps
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             01/01/2026
# ==============================================================================

from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    EarlyStopping,
    LearningRateMonitor
)

def configure_callbacks(trial = None, checkpoint_dir = None):
    """
    Configures and returns a list of callbacks for Beta-VAE
    model training. Enables pruning of unpromising trials in
    Optuna.
    """

    # Model checkpoint to save the state of the optimal model
    checkpoint_callback = ModelCheckpoint(
        dirpath           = checkpoint_dir,
        filename          = "best-{epoch:02d}-{val_loss:.4f}",
        monitor           = "val_loss",
        mode              = "min",
        save_top_k        = 1,
        save_last         = True,
        save_weights_only = True
    )

    # Stops training early if validation loss doesn't improve
    early_stop_callback = EarlyStopping(
        monitor   = 'val_loss',
        patience  = 30,
        min_delta = 1e-5,
        mode      = 'min'
    )

    # Monitors 
    lr_monitor = LearningRateMonitor(logging_interval = 'step')

    callbacks = [
        checkpoint_callback, 
        early_stop_callback, 
        lr_monitor,
        GradientNormCallback(log_every_n_steps = 50)
    ]

    # Pruning of unpromising runs
    if trial is not None:
        pruning_callback = PyTorchLightningPruningCallback(
            trial, monitor = 'val_loss'
        )
        callbacks.append(pruning_callback)

    return callbacks

# -----------------------------------------------------------------------------

class GradientNormCallback(pl.Callback):
    """
    Logs the total gradient norm to W&B after each optimiser step.

    Gradient norm monitoring is important for high-dimensional methylation
    VAEs: encoder collapse (a mode where z_mu/z_logvar outputs degenerate)
    manifests as vanishing encoder gradients that are otherwise invisible
    in the standard loss logs. Logging every N steps to avoid overhead.

    Parameters
    ----------
    log_every_n_steps : int
        Frequency of gradient norm logging. Default 50.
    """
    def __init__(self, log_every_n_steps: int = 50):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_after_backward(self, trainer, pl_module):
        if trainer.global_step % self.log_every_n_steps != 0:
            return
        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                total_norm += p.grad.detach().data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        pl_module.log('grad_norm', total_norm)

# [END]