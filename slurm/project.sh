#!/bin/bash
# =============================================================================
# Script:           project.sh
# Purpose:          SLURM script to generate Beta-VAE embeddings (latent-only).
#                   Outputs lightweight .h5ad files preserving obs and uns.
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
#
# Usage:
#   sbatch scripts/project_betaVAE.sh
# =============================================================================

#SBATCH --job-name=project
#SBATCH --output=/ddn_exa/campbell/sli/methylcdm-project/logs/project/project_%j.out
#SBATCH --error=/ddn_exa/campbell/sli/methylcdm-project/logs/project/project_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=gpu2

# =============================================================================
# Environment Setup
# =============================================================================

source ~/miniforge3/etc/profile.d/conda.sh
conda activate methylcdm-env

cd /ddn_exa/campbell/sli/methylcdm-project

mkdir -p logs

# Clean up SLURM variables that can confuse PyTorch Lightning detection
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
# Execution
# ==============================================================================

# Define your paths here for easy swapping
CHECKPOINT="/ddn_exa/campbell/sli/methylcdm-project/models/beta_vae/betaVAE_sweep_20260306_133219/trial_72/best-epoch=151-val_loss=1.3515.ckpt"
DATA="/ddn_exa/campbell/sli/methylcdm-project/data/training/methylation/pancancer_cohort_adata.h5ad"
OUT_DIR="/ddn_exa/campbell/sli/methylcdm-project/data/embeddings"

# Run projection
# Toggle --split_projects if you want individual TCGA project files
srun python scripts/project.py \
    --checkpoint "$CHECKPOINT" \
    --data_path "$DATA" \
    --output_dir "$OUT_DIR" \
    --batch_size 512 \
    --name "pancancer_latent" \
    --device cuda

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="