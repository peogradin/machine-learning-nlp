#!/bin/bash

#SBATCH --time=01:50:00
#SBATCH -c 1
#SBATCH --mem-per-cpu=24G 
#SBATCH --gres=gpu:1 --exclude=callisto
#SBATCH --job-name=a2
#SBATCH --output=a2_%j.log
#SBATCH --partition=long

#cd "$SLURM_SUBMIT_DIR"
# Activate virtual environment
#source ../.venv/bin/activate
source /data/courses/2025_dat450_dit247/venvs/dat450_venv/bin/activate

# Run the Python script with specified arguments
python3 -u A2.py