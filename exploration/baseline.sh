#!/bin/bash
#SBATCH --output=/cluster/home/t144807uhn/marginal_%j.out
#SBATCH --error=/cluster/home/t144807uhn/marginal_%j.err
#SBATCH --time=6:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sophiamjia.li@mail.utoronto.ca

# Make the project-specific logs directory
mkdir -p /cluster/home/t144807uhn/logs/MethylVAE/mini/$1

# Activate the virtual environment
module load python3/3.12.11
source /cluster/home/t144807uhn/envs/methylvae-env/bin/activate

# Ensure that all commands resolve back to the proper root directory
cd /cluster/home/t144807uhn/MethylVAE

echo "=========================================="
echo "Mini Sweep Job ID:  $SLURM_JOB_ID"
echo "Job Name:           $1"
echo "Node:               $SLURMD_NODENAME"
echo "GPU:                $CUDA_VISIBLE_DEVICES"
echo "Start time:         $(date)"
echo "=========================================="

python exploration/baseline.py \
    /cluster/home/t144807uhn/data/cohorts/pancancer_v2/pancancer_v2_adata.h5ad \
    --latent_dim 128

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="