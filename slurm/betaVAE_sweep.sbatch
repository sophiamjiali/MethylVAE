#!/bin/bash
#SBATCH --job-name=betaVAE_sweep
#SBATCH --output=/ddn_exa/campbell/sli/methylcdm-project/logs/betaVAE_sweep_%j.out
#SBATCH --error=/ddn_exa/campbell/sli/methylcdm-project/logs/betaVAE_sweep_%j.err
#SBATCH --time=24:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu3

# Activate Conda Environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate methylcdm-gpu

# Navigate to the project directory
cd /ddn_exa/campbell/sli/methylcdm-project

# Run the sweep
srun python scripts/train_betaVAE.py \
    --config_pipeline pipeline.yaml \
    --config_train betaVAE.yaml \
    --verbose True
