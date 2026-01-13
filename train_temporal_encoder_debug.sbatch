#!/bin/bash
#SBATCH --job-name=tabpfn3d_debug
#SBATCH --output=logs/train_debug_%j.out
#SBATCH --error=logs/train_debug_%j.err
#SBATCH --partition=pi_dbertsim
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

# Debug/smoke test script - quick validation
# Usage: sbatch train_temporal_encoder_debug.sbatch

set -e  # Exit on error

echo "=========================================="
echo "DEBUG JOB started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Load Python module
module load sloan/python/3.11.4

# Activate virtual environment
source $HOME/venv_tabpfn3d/bin/activate

# Set working directory
cd $HOME/TabPFN3D
mkdir -p logs

# Print environment info
echo "Python: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Change to temporal encoder directory
cd 04_temporal_encoder

# Run debug training (quick test)
echo "Starting DEBUG training (--debug flag)..."
python train.py --debug

echo "=========================================="
echo "DEBUG JOB finished at: $(date)"
echo "=========================================="
