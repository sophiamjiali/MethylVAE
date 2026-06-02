#!/bin/bash
#SBATCH --output=/cluster/home/t144807uhn/logs/MethylVAE/full/%x/%x_%j.out
#SBATCH --error=/cluster/home/t144807uhn/logs/MethylVAE/full/%x/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sophiamjia.li@mail.utoronto.ca

# Make the project-specific logs directory
mkdir -p /cluster/home/t144807uhn/logs/MethylVAE/full/$1

# Activate the virtual environment
module load python3/3.12.11
source /cluster/home/t144807uhn/envs/methylvae-env/bin/activate

# Ensure that all commands resolve back to the proper root directory
cd /cluster/home/t144807uhn/MethylVAE


echo "=========================================="
echo "Sweep Job ID:   $SLURM_ARRAY_JOB_ID"
echo "Job Name:       $1"
echo "Trial Index:    $SLURM_ARRAY_TASK_ID"
echo "Node:           $SLURMD_NODENAME"
echo "GPU:            $CUDA_VISIBLE_DEVICES"
echo "Start:          $(date)"
echo "=========================================="

nvidia-smi

export WANDB_PROJECT="MethylVAE-full"

export OPTUNA_SQLITE_TIMEOUT=300

unset SLURM_NTASKS
unset SLURM_JOB_NAME

srun python scripts/sweeps/run_full_sweep.py \
    --name $1 \
    --config_data data.yaml \
    --config_train train.yaml \
    --config_loss loss.yaml \
    --config_search config/search_space.yaml \
    --trial_seed 42 \
    --verbose

echo "=========================================="
echo "End: $(date)"
echo "=========================================="