#!/bin/bash
#SBATCH --job-name=tabpfn3d_mgpu
#SBATCH --output=logs/train_multigpu_%j.out
#SBATCH --error=logs/train_multigpu_%j.err
#SBATCH --partition=pi_dbertsim
#SBATCH --gres=gpu:4
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Multi-GPU debug script - uses 4 GPUs for parallel training
# Usage: sbatch train_temporal_encoder_multigpu_debug.sbatch

set -e  # Exit on error

echo "=========================================="
echo "MULTI-GPU DEBUG JOB started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs requested: 4"
echo "=========================================="

# Initialize module system (if available)
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi

# Load Python module
module load sloan/python/3.11.4 2>/dev/null || echo "Module system not available, using venv Python directly"

# Activate virtual environment
source $HOME/venv_tabpfn3d/bin/activate

# Set working directory
cd $HOME/TabPFN3D
mkdir -p logs

# Print environment info
echo "Python: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA devices: $(python -c 'import torch; print(torch.cuda.device_count())')"

# List available GPUs
python -c "import torch; [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

# Change to temporal encoder directory
cd 04_temporal_encoder

# Run debug training with multi-GPU
echo ""
echo "Starting MULTI-GPU debug training..."
python -u train.py --debug --multi-gpu

echo "=========================================="
echo "MULTI-GPU DEBUG JOB finished at: $(date)"
echo "=========================================="
