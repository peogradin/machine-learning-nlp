#!/bin/bash
#SBATCH --output=logs/a5_%j.out
#SBATCH --error=logs/a5_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1 --exclude=callisto
#SBATCH --job-name=a5
#SBATCH --time=01:50:00
#SBATCH --partition=long

mkdir -p logs


echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

source /data/courses/2025_dat450_dit247/venvs/dat450_venv/bin/activate
python3 -u a5.py

echo "\nJob finished at: $(date)"
