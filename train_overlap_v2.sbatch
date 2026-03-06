#!/bin/bash
#SBATCH --job-name=overlap_v2
#SBATCH --output=logs/overlap_v2_%j.out
#SBATCH --error=logs/overlap_v2_%j.err
#SBATCH --partition=pi_dbertsim
#SBATCH --gres=gpu:3
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12

# Overlap TabPFN v2 training — 3-GPU pipeline:
#
#   GPU 0 → generator   (synthetic datasets, group_size=16, 3:1 aug ratio)
#   GPU 1 → trainer     (MLP encoder fresh + 24 pretrained layers, window=16 stride=8)
#   GPU 2 → evaluator   (polls last.pt, evals synth+real, saves best.pt)
#
# Architecture (build_overlap_model.py):
#   Fresh (Xavier init, trained at LR_FRESH):
#     - Encoder:  MLP(32→64→192, GELU, no bias)
#     - Feature positional embedding projection: Linear(48→192, seed=137)
#   Pretrained (original TabPFN weights, trained at LR_PRETRAINED):
#     - 24 transformer layers (attention between features + attention between
#       items + feedforward MLP with GELU)
#     - y_encoder (label projection)
#     - decoder (Linear(192→384)→GELU→Linear(384→n_classes))
#
# Data pipeline:
#   1. DAG generator → 3D dataset (n, m, T)
#   2. Flatten: (n, m*T)  [feature-major: f0_t0..f0_tT, f1_t0..f1_tT, ...]
#   3. Pad T to multiple of 16 (right zero-pad)
#   4. Per-feature augmentation (log/exp/squash/KDI/kuma, class perm)
#   5. Temporal granularity aug (pooling or interpolation)
#   6. Missing values (per-feature, random rate)
#   7. Re-pad T to multiple of 16 after aug
#   Batch ratio: N/4 originals + 3N/4 augmented copies
#
# Overlap expansion (at forward pass time, in worker_trainer_v2.py):
#   - window=16, stride=8 → n_groups = (T-16)/8 + 1 overlapping windows per feature
#   - Layout: feature-major (all groups from f0, then all groups from f1, ...)
#   - set_temporal_info → temporal_add_embeddings adds:
#       feature_emb(j):  pseudo-random subspace proj Linear(48→192), same for
#                        all groups of feature j  (seed=137, NOT the pretrained 42)
#       sinusoidal_PE(g): standard Vaswani PE per group index within feature j
#
# Usage:
#   sbatch train_overlap_v2.sbatch
#   sbatch --export=ALL,RUN_NAME=ov2,LR_FRESH=1e-4,LR_PRETRAINED=1e-5 train_overlap_v2.sbatch
#
# Resume from best.pt (weights only, new schedule — use --resume-weights-only if added):
#   sbatch --export=ALL,RUN_NAME=ov2,...,RESUME=07_finetuning/checkpoints_overlap/ov2/best.pt \
#          train_overlap_v2.sbatch
#
# Download best.pt from cluster (replace USER and HOST):
#   scp USER@HOST:~/TabPFN3D/07_finetuning/checkpoints_overlap/ov2/best.pt ./
#
# Final evaluation (local, after download):
#   python 07_finetuning/final_evaluation.py best.pt --device cuda

set -e

echo "=========================================="
echo "OVERLAP TRAINING V2 — $(date)"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "=========================================="

# ── Module + venv ──
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi
module load sloan/python/3.11.4 2>/dev/null || echo "Module system not available"
source $HOME/venv_tabpfn3d/bin/activate

cd $HOME/TabPFN3D
mkdir -p logs

# ── Environment info ──
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
RUN_NAME="${RUN_NAME:-ov2}"
LR_FRESH="${LR_FRESH:-1e-4}"
LR_PRETRAINED="${LR_PRETRAINED:-1e-5}"
LR_MIN="${LR_MIN:-1e-7}"
WARMUP="${WARMUP:-200}"
BATCH="${BATCH:-92}"
MAX_DATASETS="${MAX_DATASETS:-2000000}"
# N_STEPS ~= ceil(MAX_DATASETS / BATCH) = ceil(2e6 / 92) ≈ 21740
# Set slightly above so cosine reaches minimum by end of training
N_STEPS="${N_STEPS:-22000}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
EVAL_INTERVAL="${EVAL_INTERVAL:-60}"
RESUME="${RESUME:-}"

BATCH_DIR="scratch_batches/${RUN_NAME}"
CKPT_DIR="07_finetuning/checkpoints_overlap/${RUN_NAME}"
LOG_DIR="07_finetuning/logs_overlap/${RUN_NAME}"

echo "RUN_NAME=$RUN_NAME"
echo "LR_FRESH=$LR_FRESH  LR_PRETRAINED=$LR_PRETRAINED  LR_MIN=$LR_MIN"
echo "BATCH=$BATCH  MAX_DATASETS=$MAX_DATASETS  N_STEPS=$N_STEPS"
echo "WARMUP=$WARMUP  GRAD_CLIP=$GRAD_CLIP  WEIGHT_DECAY=$WEIGHT_DECAY"
echo "BATCH_DIR=$BATCH_DIR"
echo "CKPT_DIR=$CKPT_DIR"
echo "LOG_DIR=$LOG_DIR"
echo "RESUME=${RESUME:-(none)}"
echo ""

mkdir -p "$BATCH_DIR" "$CKPT_DIR" "$LOG_DIR"

# ── Launch 3 workers on separate GPUs ──

echo "[1/3] Starting generator on GPU 0 (group_size=16, window matches WINDOW=16)..."
CUDA_VISIBLE_DEVICES=0 python -u 07_finetuning/worker_generator.py \
    --batch-size "$BATCH" \
    --max-datasets "$MAX_DATASETS" \
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
    --lr-fresh "$LR_FRESH" \
    --lr-pretrained "$LR_PRETRAINED" \
    --lr-min "$LR_MIN" \
    --warmup-steps "$WARMUP" \
    --n-steps "$N_STEPS" \
    --weight-decay "$WEIGHT_DECAY" \
    --grad-clip "$GRAD_CLIP" \
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
    --eval-interval "$EVAL_INTERVAL" \
    --batch-size "$BATCH" \
    > "$LOG_DIR/evaluator.log" 2>&1 &
PID_EVAL=$!
echo "  Evaluator PID=$PID_EVAL"

echo ""
echo "All workers launched."
echo "  Generator  PID=$PID_GEN"
echo "  Trainer    PID=$PID_TRAIN"
echo "  Evaluator  PID=$PID_EVAL"
echo "Logs: $LOG_DIR/{generator,trainer,evaluator}.log"
echo ""

# ── Wait for all workers to finish ──
# Trainer and evaluator finish after generator signals DONE and all batches consumed.
wait $PID_GEN
echo "Generator finished (exit $?)"

wait $PID_TRAIN
echo "Trainer finished (exit $?)"

wait $PID_EVAL
echo "Evaluator finished (exit $?)"

echo ""
echo "=========================================="
echo "DONE — $(date)"
echo "Best checkpoint: $CKPT_DIR/best.pt"
echo "History:         $LOG_DIR/history.json"
echo "=========================================="
