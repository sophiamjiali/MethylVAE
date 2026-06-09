# ==============================================================================
# Script:           objective.py
# Purpose:          Optuna objective function for BetaVAE hyperparameter sweeps.
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             12/31/2025
# ==============================================================================

import gc
import torch
import optuna
import wandb

from methylvae.training.train import train


def objective(trial, study_name: str, config: dict, mini: bool = False):
    """
    Optuna objective for one hyperparameter trial.

    Parameters
    ----------
    trial      : Optuna Trial object.
    study_name : Study name string, used for run naming and checkpointing.
    config     : Merged flat config dict from merge_configs_with_search_space().
                 Fixed keys (input_dim, paths, max_epochs, free_bits, etc.) are
                 passed through. Swept keys are overridden below from
                 config["search_space"].

    Returns
    -------
    float : val_loss (minimised). Returns inf on posterior variance collapse.
    """

    try:
        search_space = config["search_space"]
        trial_config = {k: v for k, v in config.items() if k != "search_space"}

        # --- Hyperparameter sampling ------------------------------------------

        trial_config["latent_dim"] = trial.suggest_categorical(
            "latent_dim", search_space["latent_dim"]
        )
        trial_config["beta"] = trial.suggest_float(
            "beta", *search_space["beta"], log=True
        )
        trial_config["lr"] = trial.suggest_float(
            "lr", *search_space["lr"], log=True
        )

        trial_config['max_epochs'] = search_space['max_epochs']
        trial_config['free_bits'] = search_space['free_bits']
        trial_config['gradient_clip_val'] = search_space['gradient_clip_val']
        trial_config['n_startup_trials'] = search_space['n_startup_trials']
        trial_config['early_stopping_patience'] = search_space['early_stopping']['patience']
        trial_config['early_stopping_min_delta'] = search_space['early_stopping']['min_delta']

        if mini:
            trial_config['encoder_dims'] = search_space['encoder_dims']
            trial_config["num_cycles"] = search_space['num_cycles']
            trial_config["batch_size"] = search_space['batch_size']
            trial_config["input_dropout"] = search_space['input_dropout']
        else:

            encoder_idx = trial.suggest_categorical(
                "encoder_dims_idx",
                list(range(len(search_space["encoder_dims"])))
            )
            trial_config["encoder_dims"] = search_space["encoder_dims"][encoder_idx]
            trial_config["num_cycles"] = trial.suggest_categorical(
                "num_cycles", search_space["num_cycles"]
            )
            trial_config["batch_size"] = trial.suggest_categorical(
                "batch_size", search_space["batch_size"]
            )
            trial_config["input_dropout"] = trial.suggest_float(
                "input_dropout", *search_space["input_dropout"]
            )

        # --- Training ---------------------------------------------------------

        metrics = train(
            config     = trial_config,
            run_name   = f"{study_name}_trial_{trial.number}",
            seed       = trial_config["seed"],
            trial      = trial,
            study_name = study_name,
        )

        # --- Metric extraction ------------------------------------------------

        val_loss = metrics.get("val_loss")
        if val_loss is None:
            raise optuna.exceptions.TrialPruned("val_loss missing from callback_metrics")
        val_loss = float(val_loss)

        val_post_var = metrics.get("val_post_var")
        val_post_var = float(val_post_var) if val_post_var is not None else 0.0

        if val_post_var < 0.4:
            return float("inf")

        trial.report(val_loss, step=trial.number)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return val_loss

    except optuna.exceptions.TrialPruned:
        raise

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        raise

    finally:
        wandb.finish()
        torch.cuda.empty_cache()
        gc.collect()

# [END]