# ==============================================================================
# Script:           betaVAE_objective.py
# Purpose:          Defines the Optuna objective function for training.
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             12/31/2025
# ==============================================================================

import gc
import torch
import optuna
import wandb

from methylvae.training.train import train


def objective(trial, study_name, sweep_config):
    """
    Objective function for an an Optuna hyperparameter optimization trial.
    """

    # Wrap the trial to prune it if it fails
    try:

        # Resolve the sweep configurations into runtime values
        config = dict(sweep_config)
        config['latent_dim'] = trial.suggest_int(
            'latent_dim', *sweep_config['latent_dim']
        )
        config['beta'] = trial.suggest_float(
            'beta', *sweep_config['beta'], log = True
        )
        config['lr'] = trial.suggest_float(
            'lr', *sweep_config['lr'], log = True
        )
        config['input_dropout'] = trial.suggest_float(
            'input_dropout', *sweep_config['input_dorpout']
        )

        config['num_cycles'] = trial.suggest_categorical(
            'num_cycles', sweep_config['num_cycles']
        )
        encoder_idx = trial.suggest_categorical(
            'encoder_dims_idx', list(range(len(sweep_config['encoder_dims'])))
        )
        config['encoder_dims'] = sweep_config['encoder_dims'][encoder_idx]
        config['batch_size'] = trial.suggest_categorical(
            'batch_size', sweep_config['batch_size']
        )

        # Train the model
        metrics = train(
            config = config,
            run_name = f"{study_name}_trial_{trial.number}",
            seed = config['seed'],
            trial = trial,
            study_name = study_name
        )

        val_loss = metrics.get('val_loss')
        mean_post_var = metrics.get('mean_posterior_var')

        if val_loss is None:
            raise optuna.exceptions.TrialPruned(
                "val_loss missing"
            )
        
        val_loss = val_loss.item()

        mean_post_var = (
            mean_post_var.item()
            if mean_post_var is not None
            else 0.0
        )

        # Penalize trials where posterior variance is critically low
        if mean_post_var < 0.1:
            return float("inf")
        
        # Perform Optuna pruning
        trial.report(val_loss, step=sweep_config["max_epochs"])

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