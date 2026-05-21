#!/usr/bin/env python3

import argparse
from pathlib import Path

from methylvae.utils.seed import init_environment
from methylvae.utils.config import load_config, resolve_path
from methylvae.training.objective import objective
from methylvae.constants import BETAVAE_SWEEP_DIR
from methylvae.tuning.study import get_or_create_study_name, build_study


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_pipeline", required=True)
    parser.add_argument("--config_train", required=True)
    parser.add_argument("--trial_seed", type=int, default=0)

    args = parser.parse_args()

    pipeline_cfg = load_config(args.config_pipeline)
    train_cfg = load_config(args.config_train)

    init_environment(pipeline_cfg)

    experiment_dir = resolve_path(
        train_cfg.get("experiment_dir", ""),
        BETAVAE_SWEEP_DIR,
        build_path=True
    )

    Path(experiment_dir).mkdir(parents=True, exist_ok=True)

    study_name = get_or_create_study_name(
        experiment_dir,
        prefix="betaVAE_mini"
    )

    storage = f"sqlite:///{experiment_dir}/{study_name}.db"

    study = build_study(
        storage=storage,
        study_name=study_name,
        n_startup_trials=3,
        seed=args.trial_seed
    )

    study.optimize(
        lambda trial: objective(trial, study_name, train_cfg),
        n_trials=10
    )


if __name__ == "__main__":
    main()