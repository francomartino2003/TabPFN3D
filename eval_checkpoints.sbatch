#!/bin/bash
#SBATCH --job-name=eval_checkpoints
#SBATCH --output=eval_checkpoints_%j.out
#SBATCH --error=eval_checkpoints_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=pi_dbertsim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

set -e

echo "============================================"
echo "EVALUATE ALL CHECKPOINTS"
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

# Run evaluation
python -u 07_finetuning/evaluate_checkpoints.py \
    --checkpoint-dir 07_finetuning/checkpoints \
    --output-dir 07_finetuning/eval_results \
    --device cuda \
    --n-estimators 8

echo ""
echo "End time: $(date)"
echo "============================================"
