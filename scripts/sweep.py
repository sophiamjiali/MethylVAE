#!/usr/bin/env python3
# ==============================================================================
# Script:           sweep_betaVAE.py
# Purpose:          Entry-point for BetaVAE Optuna hyperparameter sweep.
#                   Designed to be launched as a SLURM job array where each
#                   task runs exactly one trial. All tasks share a single
#                   SQLite storage file for coordinated search via Optuna.
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
#
# Configurations:   pipeline.yaml, betaVAE.yaml
#
# Usage:
#   # Standard (called by SLURM array task):
#   python scripts/sweep_betaVAE.py \
#       --config_pipeline pipeline.yaml \
#       --config_train    betaVAE.yaml \
#       --trial_seed      $SLURM_ARRAY_TASK_ID \
#       --verbose         True
#
#   # Report best results without running a trial (safe on login node):
#   python scripts/sweep_betaVAE.py \
#       --config_pipeline pipeline.yaml \
#       --config_train    betaVAE.yaml \
#       --report_only
# ==============================================================================

import os
import argparse
from datetime import datetime
from pathlib import Path

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from MethylCDM.utils.utils import init_environment, load_config, resolve_path
from MethylCDM.training.betaVAE_objective import objective
from MethylCDM.constants import BETAVAE_SWEEP_DIR


# ==============================================================================
# Argument parsing
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter sweep for BetaVAE."
    )
    parser.add_argument(
        "--config_pipeline", type=str, required=True,
        help="Path to pipeline.yaml."
    )
    parser.add_argument(
        "--config_train", type=str, required=True,
        help="Path to betaVAE.yaml."
    )
    parser.add_argument(
        "--trial_seed", type=int, default=0,
        help="Per-trial random seed. Pass SLURM_ARRAY_TASK_ID."
    )
    parser.add_argument(
        "--verbose", type=bool, default=False,
        help="Print sweep progress and results."
    )
    parser.add_argument(
        "--report_only", action="store_true",
        help="Print best trial results and exit without running a trial."
    )
    parser.add_argument(
        "--n_startup_trials", type=int, default=10,
        help="Random trials before TPE activates. Default 10 for a "
             "7-dimensional search space."
    )
    parser.add_argument(
        "--study_name", type=str, default=None,
        help="Existing study name to load. Required for --report_only. "
             "If not provided, a new study name is generated (for job array use)."
    )
    
    return parser.parse_args()


# ==============================================================================
# Study name coordination across array tasks
# ==============================================================================

def get_or_create_study_name(experiment_dir: str) -> str:
    """
    Returns a consistent study name for all array tasks in the same submission.

    The original train_betaVAE.py generated the study name from datetime.now()
    at process start. In a job array, different tasks start at different times,
    so each task would generate a different name and create separate studies —
    defeating the purpose of shared storage.

    This function writes the study name to a lockfile on first call (whichever
    task starts first) and reads it on all subsequent calls. All tasks in the
    same sweep therefore use the same study name regardless of start time.

    The lockfile is named after the SLURM array job ID so separate sweep
    submissions each get their own study, even if submitted in quick succession.
    """
    array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID", "local")
    lockfile     = os.path.join(experiment_dir, f"study_name_{array_job_id}.txt")

    if os.path.exists(lockfile):
        with open(lockfile) as f:
            return f.read().strip()

    # First task to reach this point creates the study name
    study_name = f"betaVAE_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # Write atomically: write to tmp then rename to avoid partial reads
    tmp = lockfile + ".tmp"
    with open(tmp, "w") as f:
        f.write(study_name)
    os.replace(tmp, lockfile)
    return study_name


# ==============================================================================
# Report
# ==============================================================================

def print_report(study: optuna.Study) -> None:
    
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.PRUNED]
    failed    = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.FAIL]

    print("\n" + "=" * 60)
    print(f"  OPTUNA SWEEP REPORT — {study.study_name}")
    print("=" * 60)
    print(f"  Total trials:     {len(study.trials)}")
    print(f"  Completed:        {len(completed)}")
    print(f"  Pruned:           {len(pruned)}")
    print(f"  Failed:           {len(failed)}")
    print("-" * 60)

    if not completed:
        print("  No completed trials yet.")
        print("=" * 60)
        return

    best = study.best_trial
    print(f"  Best trial:       #{best.number}")
    print(f"  Best val_loss:    {best.value:.5f}")
    print("\n  Best hyperparameters:")
    for k, v in best.params.items():
        print(f"    {k:<20} {v}")
    print("-" * 60)
    print("\n  All completed trials (sorted by val_loss):")
    print(f"  {'#':>4}  {'val_loss':>10}  {'latent_dim':>10}  {'beta':>8}  "
          f"{'lr':>8}  {'enc_idx':>7}  {'dropout':>7}  "
          f"{'cycles':>6}  {'batch':>6}")
    for t in sorted(completed, key=lambda t: t.value):
        p = t.params
        print(f"  {t.number:>4}  {t.value:>10.5f}  "
              f"{p.get('latent_dim','?'):>10}  "
              f"{p.get('beta','?'):>8.5f}  "
              f"{p.get('lr','?'):>8.5f}  "
              f"{p.get('encoder_dims_idx','?'):>7}  "
              f"{p.get('input_dropout','?'):>7.3f}  "
              f"{p.get('num_cycles','?'):>6}  "
              f"{p.get('batch_size','?'):>6}")
    print("=" * 60)


# ==============================================================================
# Main
# ==============================================================================

def main():
    args = parse_args()

    # Load configs — matches your existing two-config pattern
    pipeline_cfg = load_config(args.config_pipeline)
    train_cfg    = load_config(args.config_train)

    # Initialise environment (sets seeds, configures logging, etc.)
    init_environment(pipeline_cfg)

    # Resolve experiment directory and SQLite storage path
    experiment_dir = train_cfg.get("experiment_dir", "")
    experiment_dir = resolve_path(experiment_dir, BETAVAE_SWEEP_DIR)
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)

    if args.study_name:
        study_name = args.study_name
    else:
        study_name = get_or_create_study_name(experiment_dir)
    db_path    = os.path.join(experiment_dir, f"{study_name}.db")
    storage    = f"sqlite:///{db_path}"

    if args.verbose:
        print("=" * 50)
        print(f"~~~~~| BetaVAE Hyperparameter Sweep")
        print("=" * 50)
        print(f"  Study name:  {study_name}")
        print(f"  Storage:     {storage}")
        print(f"  Trial seed:  {args.trial_seed}")

    # Create or load the study.
    # load_if_exists=True is critical for the job array: every task calls
    # create_study() on startup. The first task creates the study; all
    # subsequent tasks load it without error.
    study = optuna.create_study(
        storage   = storage,
        study_name = study_name,
        direction  = "minimize",
        sampler    = TPESampler(
            n_startup_trials = args.n_startup_trials,
            # Model interactions between hyperparameters — encoder_dims and
            # latent_dim interact strongly and should be sampled jointly
            multivariate     = True,
            seed             = args.trial_seed,
        ),
        pruner = MedianPruner(
            n_warmup_steps   = 20,   # epochs before pruning eligible per trial;
                                     # covers one full annealing ramp phase
            n_startup_trials = args.n_startup_trials,
            interval_steps   = 5,    # check pruning every 5 epochs
        ),
        load_if_exists = True,
    )

    # ------------------------------------------------------------------
    # Report-only mode
    # ------------------------------------------------------------------
    if args.report_only:
        print_report(study)
        return

    if args.verbose:
        n_done = len([t for t in study.trials
                      if t.state == optuna.trial.TrialState.COMPLETE])
        print(f"  Completed trials so far: {n_done}\n")

    # ------------------------------------------------------------------
    # Run exactly one trial per SLURM array task.
    # timeout=86400 matches your original script's per-trial safety ceiling.
    # ------------------------------------------------------------------
    study.optimize(
        lambda trial: objective(trial, study_name, train_cfg),
        n_trials  = 1,
        timeout   = 86400,
        callbacks = [
            lambda study, trial: print(
                f"\n[Sweep] Trial #{trial.number} finished — "
                f"val_loss={trial.value:.5f}  state={trial.state.name}\n"
                f"  params: {trial.params}\n"
            )
        ],
    )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    if args.verbose:
        print_report(study)
        print("\n" + "=" * 50)
        print(f"~~~~~| Completed BetaVAE Hyperparameter Sweep")
        print("=" * 50)


if __name__ == "__main__":
    main()