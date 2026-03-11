#!/bin/bash
#SBATCH --job-name=generate_embeddings
#SBATCH --output=/ddn_exa/campbell/sli/methylcdm-project/logs/generate_embeddings/generate_embeddings_%j.out
#SBATCH --error=/ddn_exa/campbell/sli/methylcdm-project/logs/generate_embeddings/generate_embeddings_%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

echo "Job ID: $SLURM_JOB_ID, Node: $SLURMD_NODENAME, GPUs: $CUDA_VISIBLE_DEVICES"
mkdir -p /ddn_exa/campbell/sli/methylcdm-project/logs/generate_embeddings

cd /ddn_exa/campbell/sli/methylcdm-project

# run without activating
conda run -n methylcdm-env python scripts/generate_embeddings.py