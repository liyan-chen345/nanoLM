#!/bin/bash
#SBATCH -p vision-shared-h200
#SBATCH --qos=lab-free
#SBATCH --account=vision-billf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --job-name=train_diffusion
#SBATCH --output=logs/train_diffusion_%j.out
#SBATCH --error=logs/train_diffusion_%j.err

echo "=== Job started: $(date) ==="
echo "Node list: $SLURM_NODELIST"
nvidia-smi

# -----------------------------
# Micromamba environment setup
# -----------------------------
export MAMBA_ROOT_PREFIX=/data/vision/torralba/selfmanaged/freeman/u/xuanhuim/mamba-root
eval "$(/data/vision/torralba/selfmanaged/freeman/u/xuanhuim/micromamba/bin/micromamba shell hook -s bash)"
micromamba activate tinylm

echo "Micromamba environment activated: $CONDA_DEFAULT_ENV"

# -----------------------------
# Weights & Biases API key
# -----------------------------
export WANDB_API_KEY="3145d19bd177e2949c52a913475d6adde52e10d3"
export WANDB_START_METHOD="thread"

echo "WandB initialized."

# -----------------------------
# Run training
# -----------------------------
python train.py config/train_diffusion.py

echo "=== Job finished: $(date) ==="