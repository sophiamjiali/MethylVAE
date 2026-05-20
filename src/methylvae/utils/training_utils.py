# ==============================================================================
# Script:           training_utils.py
# Purpose:          Defines callbacks for Beta-VAE model training and sweeps
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             01/01/2026
# ==============================================================================

import os
import wandb
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    EarlyStopping,
    LearningRateMonitor,
    GradientAccumulationScheduler
)
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl

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
        SpikeDetectionCallback(spike_factor = 2.0),
        GradientNormCallback(log_every_n_steps = 50)
    ]

    # Pruning of unpromising runs
    if trial is not None:
        pruning_callback = PyTorchLightningPruningCallback(
            trial, monitor = 'val_loss'
        )
        callbacks.append(pruning_callback)

    return callbacks


def configure_loggers(trial = None, study_name = None):

    # Start a new run with this trial if in a sweep
    if trial is not None:
        run = wandb.init(
            project = study_name,
            name    = f"trial_{trial.number}",
            reinit  = True,
            config  = trial.params
        )

    # Else, start a regular run that isn't in a sweep
    else:
        run = wandb.init(
            project = study_name if study_name is not None else "betaVAE",
            name    = "single_run",
            reinit  = True
        )

    wandb_logger = WandbLogger(
        experiment = run,
        log_model  = False
    )

    return [wandb_logger]

# -----------------------------------------------------------------------------

class SpikeDetectionCallback(pl.Callback):
    """
    Monitors validation loss for anomalous spikes during cyclical beta
    annealing. At the start of each annealing cycle, the KL weight resets
    to 0, which can briefly destabilise training and cause val_loss spikes
    that are transient — not indicative of a failing trial.

    Rather than triggering early stopping on these spikes (which would
    prematurely terminate otherwise healthy trials), this callback logs
    a W&B alert so spikes are visible in the dashboard for post-hoc
    inspection.

    Parameters
    ----------
    spike_factor : float
        Ratio of current val_loss to best val_loss above which a spike
        is flagged. Default 2.0 (i.e. flag if loss doubles vs. best).
    """
    def __init__(self, spike_factor: float = 2.0):
        super().__init__()
        self.spike_factor = spike_factor
        self.best_val_loss = float('inf')

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get('val_loss')
        if val_loss is None:
            return

        val_loss = val_loss.item()

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
        elif val_loss > self.spike_factor * self.best_val_loss:
            msg = (
                f"Epoch {trainer.current_epoch}: val_loss spike detected. "
                f"Current: {val_loss:.5f}, Best: {self.best_val_loss:.5f} "
                f"(ratio: {val_loss / self.best_val_loss:.2f}x). "
                f"May be a cyclical annealing restart artefact."
            )
            print(f"[SpikeDetectionCallback] WARNING — {msg}")
            if wandb.run is not None:
                wandb.alert(title="val_loss spike", text=msg,
                            level=wandb.AlertLevel.WARN)


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