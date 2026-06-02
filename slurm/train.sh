#!/bin/bash
#SBATCH --output=/cluster/home/t144807uhn/logs/MethylVAE/train/%x/%x_%j.out
#SBATCH --error=/cluster/home/t144807uhn/logs/MethylVAE/train/%x/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sophiamjia.li@mail.utoronto.ca

# Make the project-specific logs directory
mkdir -p /cluster/home/t144807uhn/logs/MethylVAE/train/$1

# Activate the virtual environment
module load python3/3.12.11
source /cluster/home/t144807uhn/envs/methylvae-env/bin/activate

# Ensure that all commands resolve back to the proper root directory
cd /cluster/home/t144807uhn/MethylVAE

# Avoid PyTorch Lightning SLURM mis-detection issues
unset SLURM_NTASKS
unset SLURM_JOB_NAME

echo "=========================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Job Name:   $1"
echo "Node:       $SLURMD_NODENAME"
echo "GPU:        $CUDA_VISIBLE_DEVICES"
echo "Start:      $(date)"
echo "=========================================="

nvidia-smi

export WANDB_PROJECT="MethylVAE-train"

srun python scripts/run_train.py \
    --name $1 \
    --config_data data.yaml \
    --config_train train.yaml \
    --config_loss loss.yaml \
    --latent_dim 128 \
    --encoder_dims 2048 512 128 \
    --input_dropout 0.1 \
    --seed 42 \
    --verbose

echo "=========================================="
echo "End: $(date)"
echo "=========================================="