#!/bin/bash
# ==============================================================================
# Script:           eval.slurm
# Purpose:          Generate BetaVAE latent embeddings from checkpoint
#                   Outputs .h5ad with obsm["X_embeddings"]
# ==============================================================================

#SBATCH --job-name=betavae_eval
#SBATCH --output=/ddn_exa/campbell/sli/methylcdm-project/logs/eval/%x_%j.out
#SBATCH --error=/ddn_exa/campbell/sli/methylcdm-project/logs/eval/%x_%j.err
#SBATCH --time=04:00:00
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

mkdir -p logs/eval

# Avoid PyTorch Lightning SLURM auto-detection issues
unset SLURM_NTASKS
unset SLURM_JOB_NAME

echo "=========================================="
echo "Eval Job ID:   $SLURM_JOB_ID"
echo "Node:          $SLURMD_NODENAME"
echo "GPU:           $CUDA_VISIBLE_DEVICES"
echo "Start:         $(date)"
echo "=========================================="

# ------------------------------------------------------------------------------
# Execution
# ------------------------------------------------------------------------------

CHECKPOINT="/path/to/checkpoint.ckpt"
DATA="/path/to/pancancer_cohort.h5ad"
OUT_DIR="/ddn_exa/campbell/sli/methylcdm-project/data/embeddings"

srun python scripts/run_eval.py \
    --checkpoint "$CHECKPOINT" \
    --data_path "$DATA" \
    --output_dir "$OUT_DIR" \
    --batch_size 512 \
    --name "pancancer" \
    --device cuda \
    --split_projects

# ------------------------------------------------------------------------------
# Finish
# ------------------------------------------------------------------------------

echo "=========================================="
echo "End: $(date)"
echo "=========================================="