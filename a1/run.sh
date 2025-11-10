#!/bin/bash

#SBATCH --time=00:50:00
#SBATCH -c 1
#SBATCH --mem-per-cpu=24G 
#SBATCH --gres=gpu:1
#SBATCH --job-name=a1
#SBATCH --output=a1_%j.log
#SBATCH --partition=long

# Activate virtual environment
source ./.venv/bin/activate

# Run the Python script with specified arguments
python3 A1_skeleton.py