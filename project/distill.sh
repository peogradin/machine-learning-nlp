#!/bin/bash
#SBATCH --output=logs/project_distill_%j.out
#SBATCH --error=logs/project_distill_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1 --exclude=callisto
#SBATCH --job-name=distill
#SBATCH --time=01:50:00
#SBATCH --partition=long

mkdir -p logs
set -euo pipefail

OUTPUT_DIR="./outputs"
NUM_EPOCHS=20
SEED=101
TRAIN_FRACTION=0.05
DEVICE="cuda"

TRAIN_ARGS=(
  --output-dir "$OUTPUT_DIR"
  --num-epochs "$NUM_EPOCHS"
  --seed "$SEED"
  --device "$DEVICE"
  --train-fraction "$TRAIN_FRACTION"
)

source /data/courses/2025_dat450_dit247/venvs/dat450_venv/bin/activate
which python3

run_training() {
  python3 -u distillation.py "${TRAIN_ARGS[@]}" "$@"
}

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

run_training "$@"
echo "Job finished at: $(date)"
