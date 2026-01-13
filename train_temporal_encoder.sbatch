#!/bin/bash
#SBATCH --job-name=tabpfn3d_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=pi_dbertsim
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# MIT Engaging cluster training script for Temporal Encoder
# Usage: sbatch train_temporal_encoder.sbatch

set -e  # Exit on error

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo "=========================================="

# Load Python module
module load sloan/python/3.11.4

# Activate virtual environment
source $HOME/venv_tabpfn3d/bin/activate

# Set working directory to project root
cd $HOME/TabPFN3D

# Create logs directory if it doesn't exist
mkdir -p logs

# Print environment info
# Note: SLURM automatically sets CUDA_VISIBLE_DEVICES
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; exit(0 if torch.cuda.is_available() else 1)'; then
    echo "CUDA device: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi

# Change to temporal encoder directory
cd 04_temporal_encoder

# Run training (using default config)
echo "Starting training..."
python train.py

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
