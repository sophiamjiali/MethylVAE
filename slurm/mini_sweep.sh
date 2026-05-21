#!/bin/bash
# ==============================================================================
# Script:           mini_sweep.slurm
# Purpose:          Lightweight Optuna sweep for BetaVAE (debug / compatibility)
#                   Single-node, low-cost hyperparameter exploration.
# ==============================================================================

#SBATCH --job-name=betavae_mini_sweep
#SBATCH --output=/ddn_exa/campbell/sli/methylcdm-project/logs/sweep/mini_%j.out
#SBATCH --error=/ddn_exa/campbell/sli/methylcdm-project/logs/sweep/mini_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ------------------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------------------

source ~/miniforge3/etc/profile.d/conda.sh
conda activate methylcdm-env

cd /ddn_exa/campbell/sli/methylcdm-project

mkdir -p logs/sweep

echo "=========================================="
echo "Mini Sweep Job ID:  $SLURM_JOB_ID"
echo "Node:               $SLURMD_NODENAME"
echo "GPU:                $CUDA_VISIBLE_DEVICES"
echo "Start time:         $(date)"
echo "=========================================="

nvidia-smi

# ------------------------------------------------------------------------------
# Experiment tracking
# ------------------------------------------------------------------------------

export WANDB_PROJECT="MethylCDM-BetaVAE-MiniSweep"
# export WANDB_MODE=offline

# ------------------------------------------------------------------------------
# Optuna safety (optional but consistent with full sweep)
# ------------------------------------------------------------------------------

export OPTUNA_SQLITE_TIMEOUT=300

unset SLURM_NTASKS
unset SLURM_JOB_NAME

# ------------------------------------------------------------------------------
# Execution
# ------------------------------------------------------------------------------

srun python scripts/sweeps/run_mini_sweep.py \
    --config_pipeline pipeline.yaml \
    --config_train betaVAE.yaml \
    --trial_seed 0 \
    --verbose

# ------------------------------------------------------------------------------
# Finish
# ------------------------------------------------------------------------------

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="