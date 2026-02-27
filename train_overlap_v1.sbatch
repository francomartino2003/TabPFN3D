#!/bin/bash
#SBATCH --job-name=overlap_v1
#SBATCH --output=logs/overlap_v1_%j.out
#SBATCH --error=logs/overlap_v1_%j.err
#SBATCH --partition=pi_dbertsim
#SBATCH --gres=gpu:3
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12

# Overlap TabPFN training with 3-GPU pipeline:
#   GPU 0 → generator (writes .npz batches — group_size=16)
#   GPU 1 → trainer   (overlap model: 1 fresh + 24 pretrained layers)
#   GPU 2 → evaluator (polls last.pt, evals synth+real, saves best.pt)
#
# Usage:
#   sbatch train_overlap_v1.sbatch
#   sbatch --export=ALL,RUN_NAME=ov1,LR_FRESH=1e-4,LR_PRETRAINED=1e-5 train_overlap_v1.sbatch
#
# Resume:
#   sbatch --export=ALL,RUN_NAME=ov1,...,RESUME=07_finetuning/checkpoints_overlap/ov1/last.pt \
#          train_overlap_v1.sbatch

set -e

echo "=========================================="
echo "OVERLAP TRAINING V1 — $(date)"
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
echo "CUDA devices: $(python -c 'import torch; print(torch.cuda.device_count())')"
for i in 0 1 2; do
    python -c "import torch; print(f'  GPU $i: {torch.cuda.get_device_name($i)}')" 2>/dev/null || true
done
echo ""

# ── Configuration (env vars with defaults) ──
RUN_NAME="${RUN_NAME:-overlap_default}"
LR_FRESH="${LR_FRESH:-1e-4}"
LR_PRETRAINED="${LR_PRETRAINED:-1e-5}"
LR_MIN="${LR_MIN:-1e-7}"
WARMUP="${WARMUP:-200}"
BATCH="${BATCH:-92}"
MAX_DATASETS="${MAX_DATASETS:-2000000}"
N_STEPS="${N_STEPS:-100000}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
EVAL_INTERVAL="${EVAL_INTERVAL:-60}"
RESUME="${RESUME:-}"

BATCH_DIR="scratch_batches/${RUN_NAME}"
CKPT_DIR="07_finetuning/checkpoints_overlap/${RUN_NAME}"
LOG_DIR="07_finetuning/logs_overlap/${RUN_NAME}"

echo "RUN_NAME=$RUN_NAME  LR_FRESH=$LR_FRESH  LR_PRETRAINED=$LR_PRETRAINED"
echo "BATCH=$BATCH  N_STEPS=$N_STEPS"
echo "BATCH_DIR=$BATCH_DIR  CKPT_DIR=$CKPT_DIR  LOG_DIR=$LOG_DIR"
echo "RESUME=${RESUME:-(none)}"
echo ""

mkdir -p "$BATCH_DIR" "$CKPT_DIR" "$LOG_DIR"

# ── Launch 3 workers on separate GPUs ──

echo "[1/3] Starting generator on GPU 0 (group_size=16)..."
CUDA_VISIBLE_DEVICES=0 python -u 07_finetuning/worker_generator.py \
    --batch-size $BATCH \
    --max-datasets $MAX_DATASETS \
    --output-dir "$BATCH_DIR" \
    --seed 42 \
    --group-size 16 \
    > "$LOG_DIR/generator.log" 2>&1 &
PID_GEN=$!
echo "  Generator PID=$PID_GEN"

echo "  Waiting 30s for generator to produce initial batches..."
sleep 30

echo "[2/3] Starting trainer on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python -u 07_finetuning/worker_trainer_v2.py \
    --batch-dir "$BATCH_DIR" \
    --checkpoint-dir "$CKPT_DIR" \
    --lr-fresh $LR_FRESH \
    --lr-pretrained $LR_PRETRAINED \
    --lr-min $LR_MIN \
    --warmup-steps $WARMUP \
    --n-steps $N_STEPS \
    --weight-decay $WEIGHT_DECAY \
    --grad-clip $GRAD_CLIP \
    ${RESUME:+--resume "$RESUME"} \
    > "$LOG_DIR/trainer.log" 2>&1 &
PID_TRAIN=$!
echo "  Trainer PID=$PID_TRAIN"

echo "  Waiting 60s for trainer to produce first checkpoint..."
sleep 60

echo "[3/3] Starting evaluator on GPU 2..."
CUDA_VISIBLE_DEVICES=2 python -u 07_finetuning/worker_evaluator_v2.py \
    --checkpoint-dir "$CKPT_DIR" \
    --log-dir "$LOG_DIR" \
    --eval-interval $EVAL_INTERVAL \
    --batch-size $BATCH \
    > "$LOG_DIR/evaluator.log" 2>&1 &
PID_EVAL=$!
echo "  Evaluator PID=$PID_EVAL"

echo ""
echo "All workers launched. PIDs: gen=$PID_GEN train=$PID_TRAIN eval=$PID_EVAL"
echo "Logs: $LOG_DIR/{generator,trainer,evaluator}.log"
echo ""

# Wait for all to finish
wait $PID_GEN
echo "Generator finished (exit $?)"

wait $PID_TRAIN
echo "Trainer finished (exit $?)"

wait $PID_EVAL
echo "Evaluator finished (exit $?)"

echo ""
echo "=========================================="
echo "DONE — $(date)"
echo "=========================================="
