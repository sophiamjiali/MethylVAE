#!/bin/bash
# ==============================================================================
# Script:           full_sweep.slurm
# Purpose:          SLURM array for Optuna BetaVAE hyperparameter sweep
# ==============================================================================

#SBATCH --job-name=betavae_sweep
#SBATCH --output=/ddn_exa/campbell/sli/methylcdm-project/logs/sweep/%x_%A_%a.out
#SBATCH --error=/ddn_exa/campbell/sli/methylcdm-project/logs/sweep/%x_%A_%a.err
#SBATCH --time=24:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --array=0-74%2

# ------------------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------------------

source ~/miniforge3/etc/profile.d/conda.sh
conda activate methylcdm-env

cd /ddn_exa/campbell/sli/methylcdm-project

mkdir -p logs/sweep

echo "=========================================="
echo "Sweep Job ID:   $SLURM_ARRAY_JOB_ID"
echo "Trial Index:    $SLURM_ARRAY_TASK_ID"
echo "Node:           $SLURMD_NODENAME"
echo "GPU:            $CUDA_VISIBLE_DEVICES"
echo "Start:          $(date)"
echo "=========================================="

nvidia-smi

# ------------------------------------------------------------------------------
# Experiment tracking
# ------------------------------------------------------------------------------

export WANDB_PROJECT="MethylCDM-BetaVAE-Sweep"
# export WANDB_MODE=offline

# ------------------------------------------------------------------------------
# Optuna / SQLite stability
# ------------------------------------------------------------------------------

export OPTUNA_SQLITE_TIMEOUT=300

# Optional: avoid SLURM misinterpretation in Lightning
unset SLURM_NTASKS
unset SLURM_JOB_NAME

# ------------------------------------------------------------------------------
# Execution
# ------------------------------------------------------------------------------

srun python scripts/sweeps/run_full_sweep.py \
    --config_pipeline pipeline.yaml \
    --config_train betaVAE.yaml \
    --trial_seed $SLURM_ARRAY_TASK_ID \
    --verbose

# ------------------------------------------------------------------------------
# Finish
# ------------------------------------------------------------------------------

echo "=========================================="
echo "End: $(date)"
echo "=========================================="