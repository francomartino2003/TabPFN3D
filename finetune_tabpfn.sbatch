#!/bin/bash
#SBATCH --job-name=finetune_tabpfn
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err
#SBATCH --partition=pi_dbertsim
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Fine-tune TabPFN on flattened synthetic datasets
# Evaluates on real datasets periodically
#
# Usage: sbatch finetune_tabpfn.sbatch
# Debug:  sbatch --export=DEBUG=1 finetune_tabpfn.sbatch

set -e  # Exit on error

echo "=========================================="
echo "FINETUNE TABPFN JOB started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Initialize module system
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi

# Load Python module
module load sloan/python/3.11.4 2>/dev/null || echo "Module system not available"

# Activate virtual environment
source $HOME/venv_tabpfn3d/bin/activate

# Set working directory
cd $HOME/TabPFN3D
mkdir -p logs

# Print environment info
echo ""
echo "[Environment Info]"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    echo "Memory: $(python -c 'import torch; print(f\"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB\")')"
fi
echo ""

# Set HuggingFace token (needed for TabPFN weights)
# Export HF_TOKEN in your shell before running, or set it in ~/.bashrc
# export HF_TOKEN=your_token_here
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN not set. TabPFN may fail to download weights."
fi

# Change to finetuning directory
cd 07_finetuning

# Check if debug mode
if [ "${DEBUG:-0}" == "1" ]; then
    echo "Running in DEBUG mode..."
    python -u finetune_tabpfn.py --debug
else
    echo "Running full training..."
    # Config comes from dataset_generator_v2.py (single source of truth)
    python -u finetune_tabpfn.py \
        --n-steps 1000 \
        --batch-size 64 \
        --lr 1e-5 \
        --eval-every 50
fi

echo "=========================================="
echo "FINETUNE TABPFN JOB finished at: $(date)"
echo "=========================================="
