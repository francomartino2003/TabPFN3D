#!/bin/bash
#SBATCH --job-name=finetune_v2
#SBATCH --output=logs/finetune_v2_%j.out
#SBATCH --error=logs/finetune_v2_%j.err
#SBATCH --partition=pi_dbertsim
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Fine-tune TabPFN using DAG generator (folder 11)
#
# Usage:
#   sbatch finetune_v2.sbatch
#   sbatch --export=ALL,DEBUG=1 finetune_v2.sbatch
#   sbatch --export=ALL,RESUME=latest finetune_v2.sbatch
#   sbatch --export=ALL,RESUME=checkpoints_v2/checkpoint_step500.pt finetune_v2.sbatch

set -e

echo "=========================================="
echo "FINETUNE V2 (DAG generator) — $(date)"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "=========================================="

# Module + venv
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi
module load sloan/python/3.11.4 2>/dev/null || echo "Module system not available"
source $HOME/venv_tabpfn3d/bin/activate

cd $HOME/TabPFN3D
mkdir -p logs

# Environment info
echo ""
echo "[Environment]"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
python -c 'import torch; torch.cuda.is_available() and print("GPU:", torch.cuda.get_device_name(0))' 2>/dev/null
echo ""

# HuggingFace token
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN not set."
fi

cd 07_finetuning

# Resume logic
RESUME_ARGS=""
echo "RESUME=${RESUME:-not set}"
if [ -n "${RESUME:-}" ]; then
    if [ "$RESUME" = "latest" ]; then
        LATEST=$(ls -t checkpoints_v2/checkpoint_*.pt 2>/dev/null | head -1)
        if [ -n "$LATEST" ]; then
            RESUME_ARGS="--resume $LATEST"
            echo "Resuming from: $LATEST"
        else
            echo "No checkpoint found; starting fresh"
        fi
    else
        RESUME_ARGS="--resume $RESUME"
        echo "Resuming from: $RESUME"
    fi
fi

# Run
if [ "${DEBUG:-0}" == "1" ]; then
    echo "DEBUG MODE"
    python -u finetune_tabpfn_v2.py --debug $RESUME_ARGS
else
    echo "Full training..."
    python -u finetune_tabpfn_v2.py \
        --n-steps 1000 \
        --batch-size 64 \
        --lr 1e-5 \
        --eval-every 1 \
        $RESUME_ARGS
fi

echo "=========================================="
echo "DONE — $(date)"
echo "=========================================="
