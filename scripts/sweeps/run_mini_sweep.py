#!/usr/bin/env python3
# ==============================================================================
# Script:           run_mini_sweep.py
# Purpose:          Entry-point for a short Optuna mini-sweep (10 trials).
#                   Run before the full sweep to validate config ranges.
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
#
# Usage:
#   python run_mini_sweep.py \
#     --config_data    config/data.yaml         \
#     --config_train   config/training.yaml     \
#     --config_loss    config/loss.yaml         \
#     --config_search  config/search_space.yaml
# ==============================================================================

import argparse
from pathlib import Path

from methylvae.utils.config import (
    load_config, 
    merge_configs_with_search_space
)
from methylvae.training.objective import objective
from methylvae.tuning.study import get_or_create_study_name, build_study


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",           required=True)
    parser.add_argument("--config_dir",     required=True)
    parser.add_argument("--trial_seed",     type=int, default=0)

    args = parser.parse_args()

    config = merge_configs_with_search_space(
        load_config(f"{args.config_dir}/base.yaml"),
        load_config(f"{args.config_dir}/data.yaml"),
        load_config(f"{args.config_dir}/loss.yaml"),
        search_space=load_config(f"{args.config_dir}/search_space.yaml")
    )

    experiment_dir = config["paths.experiment_dir"]
    Path(experiment_dir).mkdir(parents=True, exist_ok=True) 

    study_name = get_or_create_study_name(
        experiment_dir, 
        prefix="mini", 
        name=args.name
    )
    storage    = f"sqlite:///{experiment_dir}/{study_name}.db"

    study = build_study(
        storage          = storage,
        study_name       = study_name,
        n_startup_trials = config['search_space']['n_startup_trials'],
        seed             = args.trial_seed,
    )

    study.optimize(
        lambda trial: objective(trial, study_name, config),
        n_trials = 10,
    )


if __name__ == "__main__":
    main()

# [END]