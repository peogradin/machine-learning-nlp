#!/bin/bash

#SBATCH --time=01:50:00
#SBATCH -c 1
#SBATCH --mem-per-cpu=24G 
#SBATCH --gres=gpu:1
#SBATCH --job-name=a2
#SBATCH --output=a2_%j.log
#SBATCH --partition=long

cd "$SLURM_SUBMIT_DIR"
# Activate virtual environment
source ../.venv/Scripts/Activate.ps1

# Run the Python script with specified arguments
python3 -u A2.py