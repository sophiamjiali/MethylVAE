#!/usr/bin/env python3
# ==============================================================================
# Script:           train_betaVAE.py
# Purpose:          Entry-point to train a betaVAE using a pancancer dataset
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             12/31/2025
#
# Configurations:   betaVAE.yaml
#
# Notes:            Begins an Optuna hyperparameter sweep
# ==============================================================================

from datetime import datetime
from pathlib import Path
import os
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import argparse

from MethylCDM.utils.utils import init_environment, load_config, resolve_path
from MethylCDM.training.betaVAE_objective import objective
from MethylCDM.constants import BETAVAE_SWEEP_DIR

def main():

    # -----| Environment Initialization |-----

    # Parse the arguments provided to the entry-point script
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_pipeline", type = str, required = True)
    parser.add_argument("--config_train", type = str, required = True)
    parser.add_argument("--verbose", type = bool, default = False)
    args = parser.parse_args()

    if args.verbose:
        print("=" * 50)
        print(f"~~~~~| Beginning BetaVAE Hyperparameter Sweep")
        print("=" * 50)
        print("\n")

    # Load the relevant configuration files 
    pipeline_cfg = load_config(args.config_pipeline)
    train_cfg = load_config(args.config_train)

    # Initialize the environment for reproducible analysis
    init_environment(pipeline_cfg)

    # -----| Sweep Initialization |-----

    # Define the Optuna Sweep Study
    experiment_dir = train_cfg.get('experiment_dir', '')
    experiment_dir = resolve_path(experiment_dir, BETAVAE_SWEEP_DIR)
    Path(experiment_dir).mkdir(parents = True, exist_ok = True)
    db_path = os.path.join(experiment_dir, "betaVAE_hyperparam_sweep")
    experiment_storage = "sqlite:///{}.db".format(db_path)

    study_name = f"betaVAE_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study = optuna.create_study(
        storage = experiment_storage,
        study_name = study_name,
        direction = "minimize",
        sampler = TPESampler(),
        pruner = MedianPruner(n_warmup_steps = 20, n_startup_trials = 5)
    )

    # Perform the sweep
    study.optimize(lambda trial: objective(trial, study_name, train_cfg), 
                   timeout = 86400, n_trials = 100)

    # Display the results
    if args.verbose:
        print("=" * 50)
        print(f"Hyperparameter Sweep Results")
        print("=" * 50)
        print("\n")
    
        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        best_trial = study.best_trial
        print("  Value: ", best_trial.value)
        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

    if args.verbose:
        print("\n")
        print("=" * 50)
        print(f"~~~~~| Completed BetaVAE Hyperparameter Sweep")
        print("=" * 50)


if __name__ == "__main__":
    main()