#!/bin/bash
#SBATCH --job-name=FasterRisk
#SBATCH --time=30-00:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=500G
#SBATCH --output=None

source ~/miniconda3/etc/profile.d/conda.sh
conda activate 474

srun 