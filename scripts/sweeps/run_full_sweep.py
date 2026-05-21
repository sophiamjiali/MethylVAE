#!/usr/bin/env python3

import argparse
from pathlib import Path

from methylvae.utils.seed import init_environment
from methylvae.utils.config import resolve_path, load_config
from methylvae.training.objective import objective
from methylvae.constants import BETAVAE_SWEEP_DIR
from methylvae.tuning.study import get_or_create_study_name, build_study


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_pipeline", required=True)
    parser.add_argument("--config_train", required=True)
    parser.add_argument("--trial_seed", type=int, default=0)
    parser.add_argument("--n_startup_trials", type=int, default=10)
    parser.add_argument("--report_only", action="store_true")
    parser.add_argument("--study_name", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    pipeline_cfg = load_config(args.config_pipeline)
    train_cfg = load_config(args.config_train)

    init_environment(pipeline_cfg)

    experiment_dir = resolve_path(
        train_cfg.get("experiment_dir", ""),
        BETAVAE_SWEEP_DIR,
        build_path=True
    )
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)

    study_name = (
        args.study_name
        if args.study_name
        else get_or_create_study_name(experiment_dir)
    )

    storage = f"sqlite:///{experiment_dir}/{study_name}.db"

    study = build_study(
        storage=storage,
        study_name=study_name,
        n_startup_trials=args.n_startup_trials,
        seed=args.trial_seed
    )

    if args.report_only:
        print(study.best_trial if study.trials else "No trials")
        return

    study.optimize(
        lambda trial: objective(trial, study_name, train_cfg),
        n_trials=1,
        timeout=86400,
    )


if __name__ == "__main__":
    main()