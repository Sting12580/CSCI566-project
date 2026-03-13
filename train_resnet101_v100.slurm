#!/bin/bash
#SBATCH --job-name=csci566_r101
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_%x_%j.out

set -euo pipefail

source ~/venvs/loveda-mask2former/bin/activate
cd /scratch1/xwei6575/github/CSCI566_Project/CSCI566-project

python -u train.py --backbone resnet101 --full --epochs 50 --batch-size 1 --num-workers 2
