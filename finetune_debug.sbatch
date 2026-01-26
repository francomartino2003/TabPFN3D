#!/bin/bash
#SBATCH --job-name=finetune_debug
#SBATCH --output=finetune_debug_%j.out
#SBATCH --error=finetune_debug_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=pi_dbertsim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

set -e

echo "============================================"
echo "FINETUNE TABPFN - DEBUG RUN"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Initialize module system
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi

# Load Python module
module load sloan/python/3.11.4 2>/dev/null || echo "Module not available"

# Activate virtual environment
source $HOME/venv_tabpfn3d/bin/activate

# Go to project directory
cd $HOME/TabPFN3D

# Check Python and CUDA
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Run debug finetuning
python -u 07_finetuning/finetune_tabpfn.py \
    --n-steps 20 \
    --batch-size 8 \
    --eval-every 10 \
    --device cuda \
    --seed 42

echo ""
echo "End time: $(date)"
echo "============================================"
