#!/bin/bash
# ==============================================================================
# Script:           sweep_betaVAE.sh
# Purpose:          SLURM job array for BetaVAE Optuna hyperparameter sweep.
#                   Each array task runs one Optuna trial independently on its
#                   own GPU. All tasks share a single SQLite Optuna storage
#                   file for coordinated search.
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
#
# Usage:
#   sbatch sweep_betaVAE.sh
#
# To resume a partially completed sweep after node failures:
#   sbatch sweep_betaVAE.sh
#   Optuna loads the existing study from the SQLite file and assigns each
#   new task a trial not yet run.
#
# To monitor progress from the login node:
#   python scripts/sweep_betaVAE.py \
#       --config_pipeline pipeline.yaml \
#       --config_train betaVAE.yaml \
#       --report_only
# ==============================================================================

#SBATCH --job-name=betaVAE_sweep
#SBATCH --output=/ddn_exa/campbell/sli/methylcdm-project/logs/betaVAE_sweep_%A_%a.out
#SBATCH --error=/ddn_exa/campbell/sli/methylcdm-project/logs/betaVAE_sweep_%A_%a.err
#SBATCH --time=24:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=gpu2,gpu3                 # gpu1 may have insufficient VRAM
#SBATCH --array=0-74%2                       # 75 trials, only 2 at the same time
                                             # Resubmit to add more — Optuna will resume.

# ==============================================================================
# Environment
# ==============================================================================

source ~/miniforge3/etc/profile.d/conda.sh
conda activate methylcdm-env

cd /ddn_exa/campbell/sli/methylcdm-project

mkdir -p logs

echo "=========================================="
echo "Sweep Job ID:  $SLURM_ARRAY_JOB_ID"
echo "Trial index:   $SLURM_ARRAY_TASK_ID"
echo "Node:          $SLURMD_NODENAME"
echo "GPUs:          $CUDA_VISIBLE_DEVICES"
echo "Start time:    $(date)"
echo "=========================================="

nvidia-smi

# ==============================================================================
# W&B
# ==============================================================================

export WANDB_PROJECT="MethylCDM-BetaVAE-Sweep"
# Uncomment if running on nodes without internet access:
# export WANDB_MODE=offline

# ==============================================================================
# SQLite concurrency
# With 50 trials each running for hours, write contention is minimal.
# Increase timeout if you see "database is locked" in logs.
# ==============================================================================

export OPTUNA_SQLITE_TIMEOUT=300

# Tell SLURM that I'm managing the job array
export SLURM_NTASKS=1
export SLURM_NTASKS_PER_NODE=1

# ==============================================================================
# Run — SLURM_ARRAY_TASK_ID is passed as trial_seed for per-trial
# reproducibility without all trials being identically seeded.
# ==============================================================================

srun python scripts/sweep_betaVAE.py \
    --config_pipeline pipeline.yaml \
    --config_train    betaVAE.yaml \
    --trial_seed      $SLURM_ARRAY_TASK_ID \
    --verbose         True

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="