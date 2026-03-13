#!/bin/bash
#SBATCH --job-name=csci566_mask2former_r101
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|a40"
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_%x_%j.out

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate loveda-mask2former

REPO_DIR="${REPO_DIR:-/scratch1/xwei6575/github/CSCI566_Project}"
cd "$REPO_DIR"

echo "Working directory: $(pwd)"
echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

python -u train.py \
  --backbone resnet101 \
  --full \
  --epochs 50 \
  --num-workers 4
