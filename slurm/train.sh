#!/bin/bash
# ==============================================================================
# Script:           train.sh
# Purpose:          SLURM submission script for a single BetaVAE training run.
#                   Uses fixed parameters defined in betaVAE_train.yaml.
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
#
# Usage:
#   sbatch slurm/train.sh
# ==============================================================================

#SBATCH --job-name=betaVAE_train
#SBATCH --output=/ddn_exa/campbell/sli/methylcdm-project/logs/train/betaVAE_train_%j.out
#SBATCH --error=/ddn_exa/campbell/sli/methylcdm-project/logs/train/betaVAE_train_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=gpu2
#SBATCH --ntasks=1
#SBATCH --nodes=1

# ==============================================================================
# Environment Setup
# ==============================================================================

source ~/miniforge3/etc/profile.d/conda.sh
conda activate methylcdm-env

cd /ddn_exa/campbell/sli/methylcdm-project

mkdir -p logs

# Clean up SLURM variables that can confuse PyTorch Lightning's 
# internal ClusterEnvironment detection.
unset SLURM_NTASKS
unset SLURM_JOB_NAME

echo "=========================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "GPUs:        $CUDA_VISIBLE_DEVICES"
echo "Start time:  $(date)"
echo "=========================================="

nvidia-smi

# ==============================================================================
# Weights & Biases (W&B)
# ==============================================================================

export WANDB_PROJECT="MethylCDM-BetaVAE-Final"
# export WANDB_MODE=offline # Uncomment if nodes lack internet

# ==============================================================================
# Execution
# ==============================================================================

# We use a fixed seed (42) for the final run to ensure reproducibility.
# We call the new train_betaVAE.py script instead of the sweep script.

srun python scripts/train.py \
    --config_pipeline pipeline.yaml \
    --config_train    betaVAE_train.yaml \
    --seed            42 \
    --verbose         True

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="