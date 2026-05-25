#!/usr/bin/env python3
# ==============================================================================
# Script:           run_full_sweep.py
# Purpose:          Entry-point for a full Optuna BetaVAE sweep (SLURM array).
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
#
# Usage (single node, testing):
#   python run_full_sweep.py \
#     --config_data    config/data.yaml       \
#     --config_train   config/training.yaml   \
#     --config_loss    config/loss.yaml       \
#     --config_search  config/search_space.yaml
#
# Usage (SLURM array, one trial per job):
#   sbatch slurm/betaVAE_sweep.sh  (calls this script with the same flags)
# ==============================================================================

import argparse
from pathlib import Path

from methylvae.utils.config import load_config, merge_configs_with_search_space, resolve_path
from methylvae.training.objective import objective
from methylvae.constants import BETAVAE_SWEEP_DIR
from methylvae.tuning.study import get_or_create_study_name, build_study


def parse_args():
    parser = argparse.ArgumentParser()

    # CHANGED: four config flags replacing config_pipeline + config_train.
    parser.add_argument("--config_data",    required=True)
    parser.add_argument("--config_train",   required=True)
    parser.add_argument("--config_loss",    required=True)
    parser.add_argument("--config_search",  required=True,
                        help="Path to search_space.yaml")

    parser.add_argument("--trial_seed",        type=int,  default=0)
    parser.add_argument("--n_startup_trials",  type=int,  default=10)
    parser.add_argument("--report_only",       action="store_true")
    parser.add_argument("--study_name",        type=str,  default=None)
    parser.add_argument("--verbose",           action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    # CHANGED: all four configs loaded and merged. search_space is attached
    # as config["search_space"] for objective.py to consume.
    config = merge_configs_with_search_space(
        load_config(args.config_data),
        load_config(args.config_train),
        load_config(args.config_loss),
        search_space=load_config(args.config_search),
    )

    experiment_dir = resolve_path(
        config.get("experiment_dir", ""),
        BETAVAE_SWEEP_DIR,
        build_path=True,
    )
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)

    study_name = (
        args.study_name
        if args.study_name
        else get_or_create_study_name(experiment_dir)
    )

    storage = f"sqlite:///{experiment_dir}/{study_name}.db"

    study = build_study(
        storage          = storage,
        study_name       = study_name,
        n_startup_trials = args.n_startup_trials,
        seed             = args.trial_seed,
    )

    if args.report_only:
        print(study.best_trial if study.trials else "No trials yet.")
        return

    study.optimize(
        lambda trial: objective(trial, study_name, config),
        n_trials = 1,
        timeout  = 86400,
    )


if __name__ == "__main__":
    main()

# [END]