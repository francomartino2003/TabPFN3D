#!/bin/bash
#SBATCH --job-name=finetune_v3
#SBATCH --output=logs/finetune_v3_%j.out
#SBATCH --error=logs/finetune_v3_%j.err
#SBATCH --partition=pi_dbertsim
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Fine-tune TabPFN V3 — Temporal Positional Encoding
#
# Changes from V2:
#   - features_per_group=3 (kept as pretrained, same speed/memory)
#   - Feature shuffle disabled, T padded to multiple of 3
#   - Structured group embeddings: feature_emb(j) + sinusoidal_PE(group_idx)
#   - 100% pretrained weights, no new parameters
#
# Usage:
#   sbatch finetune_v3.sbatch
#   sbatch --export=ALL,LR=1e-5,RUN_NAME=temporal_v1 finetune_v3.sbatch
#   sbatch --export=ALL,DEBUG=1 finetune_v3.sbatch
#   sbatch --export=ALL,RESUME=checkpoints_v3/default/checkpoint_step100.pt finetune_v3.sbatch

set -e

echo "=========================================="
echo "FINETUNE V3 (Temporal PE) — $(date)"
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
        LATEST=$(ls -t checkpoints_v3/*/checkpoint_*.pt 2>/dev/null | head -1)
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

# Run name (for parallel experiments)
RUN_NAME="${RUN_NAME:-default}"
LR="${LR:-7e-5}"
LR_MIN="${LR_MIN:-1e-7}"
WARMUP="${WARMUP:-10}"
BATCH="${BATCH:-128}"
STEPS="${STEPS:-1500}"
FREEZE="${FREEZE:-0}"
echo "RUN_NAME=$RUN_NAME  LR=$LR  LR_MIN=$LR_MIN  WARMUP=$WARMUP  BATCH=$BATCH  STEPS=$STEPS  FREEZE=$FREEZE"

# Run
if [ "${DEBUG:-0}" == "1" ]; then
    echo "DEBUG MODE"
    python -u finetune_tabpfn_v3.py --debug --run-name $RUN_NAME --lr $LR --freeze-middle $FREEZE $RESUME_ARGS
else
    echo "Full training..."
    python -u finetune_tabpfn_v3.py \
        --n-steps $STEPS \
        --batch-size $BATCH \
        --lr $LR \
        --lr-min $LR_MIN \
        --warmup-steps $WARMUP \
        --eval-every 1 \
        --run-name $RUN_NAME \
        --freeze-middle $FREEZE \
        $RESUME_ARGS
fi

echo "=========================================="
echo "DONE — $(date)"
echo "=========================================="
