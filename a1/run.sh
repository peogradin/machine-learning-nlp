#!/bin/bash

#SBATCH -c 1
#SBATCH --mem-per-cpu=24G 
#SBATCH --gres=gpu:1
#SBATCH --job-name=a1
#SBATCH --output=a1_%j.log

# Activate virtual environment
source /data/courses/2025_dat450_dit247/venvs/dat450_venv/bin/activate

# Run the Python script with specified arguments
python3 A1_skeleton.py