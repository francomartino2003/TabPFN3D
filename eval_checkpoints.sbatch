#!/bin/bash
#SBATCH --job-name=eval_ckpt
#SBATCH --partition=pi_dbertsim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_ckpt_%j.out
#SBATCH --error=logs/eval_ckpt_%j.out

echo "=========================================="
echo "EVAL CHECKPOINTS JOB started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Load environment
source ~/venv_tabpfn3d/bin/activate

# Print environment info
echo ""
echo "[Environment Info]"
python --version
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# Change to project directory
cd ~/TabPFN3D/05_flattened_benchmark

echo ""
echo "Running checkpoint evaluation..."
python -u evaluate_checkpoints.py --device cuda

echo "=========================================="
echo "EVAL CHECKPOINTS JOB finished at: $(date)"
echo "=========================================="
