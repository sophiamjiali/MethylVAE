# ==============================================================================
# Script:           study.py
# Purpose:          Optuna study factory + SLURM-safe coordination
# ==============================================================================

import os
from datetime import datetime
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner


def get_or_create_study_name(experiment_dir, 
                             prefix="betaVAE_sweep", 
                             name = "v1"):
    array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID", "local")
    lockfile = os.path.join(experiment_dir, f"study_name_{array_job_id}.txt")

    if os.path.exists(lockfile):
        with open(lockfile) as f:
            return f.read().strip()

    study_name = f"{prefix}_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    tmp = lockfile + ".tmp"
    with open(tmp, "w") as f:
        f.write(study_name)

    os.replace(tmp, lockfile)
    return study_name


def build_study(
    storage: str,
    study_name: str,
    n_startup_trials: int,
    seed: int
):
    return optuna.create_study(
        storage=storage,
        study_name=study_name,
        direction="minimize",
        sampler=TPESampler(
            n_startup_trials=n_startup_trials,
            multivariate=True,
            seed=seed,
        ),
        pruner=MedianPruner(
            n_warmup_steps=20,
            n_startup_trials=n_startup_trials,
            interval_steps=5,
        ),
        load_if_exists=False,
    )