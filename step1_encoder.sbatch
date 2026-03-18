#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Encoder warm-up
#
#  What gets trained:
#    • model.encoder                              (MLP 32→96→96→192)
#    • model.feature_positional_embedding_embeddings  (Linear 48→192)
#    • model.temporal_pe_projection               (Linear 48→192)
#    • model.transformer_encoder.layers[0]        (first TabPFN layer)
#  Everything else is FROZEN.
#
#  Schedule: constant LR=1e-4 for 3000 steps.
#
#  Output:
#    Checkpoints  →  03_finetuning/checkpoints/phase1/
#      last.pt      saved atomically after every training step
#      best.pt      promoted by the evaluator when eval score improves
#    Logs         →  03_finetuning/logs/phase1/
#      generator.log  trainer.log  evaluator.log
#
#  After this job finishes, run step2_finetune.sbatch to continue from
#  03_finetuning/checkpoints/phase1/best.pt.
# ══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=tabpfn_step1
#SBATCH --output=logs/step1_%j.out
#SBATCH --error=logs/step1_%j.err
#SBATCH --partition=pi_dbertsim
#SBATCH --gres=gpu:3
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12

set -e

echo "════════════════════════════════════════"
echo "  STEP 1: Encoder warm-up — $(date)"
echo "  Job: $SLURM_JOB_ID   Node: $SLURM_NODELIST"
echo "════════════════════════════════════════"

# ── Environment ──
if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
module load sloan/python/3.11.4 2>/dev/null || echo "Module system not available"
source $HOME/venv_tabpfn3d/bin/activate

cd $HOME/TabPFN3D
mkdir -p logs

echo ""
echo "[Environment]"
echo "Python:  $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "GPUs:    $(python -c 'import torch; print(torch.cuda.device_count())')"
for i in 0 1 2; do
    python -c "import torch; print(f'  GPU $i: {torch.cuda.get_device_name($i)}')" 2>/dev/null || true
done
echo ""

# ── Tunable hyperparameters ──
LR="${LR:-1e-4}"           # constant learning rate (phase 1)
N_STEPS="${N_STEPS:-3000}" # training steps
BATCH="${BATCH:-92}"       # datasets per batch
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
EVAL_INTERVAL="${EVAL_INTERVAL:-60}"  # evaluator poll interval (seconds)

# ── Fixed paths ──
BATCH_DIR="scratch_batches/phase1"
CKPT_DIR="03_finetuning/checkpoints/phase1"
LOG_DIR="03_finetuning/logs/phase1"

echo "[Config]"
echo "  LR=$LR  N_STEPS=$N_STEPS  BATCH=$BATCH"
echo "  WEIGHT_DECAY=$WEIGHT_DECAY  GRAD_CLIP=$GRAD_CLIP"
echo "  BATCH_DIR=$BATCH_DIR"
echo "  CKPT_DIR=$CKPT_DIR    ← last.pt + best.pt saved here"
echo "  LOG_DIR=$LOG_DIR"
echo ""

mkdir -p "$BATCH_DIR" "$CKPT_DIR" "$LOG_DIR"

# ── GPU 0: Generator ──
echo "[1/3] Generator → GPU 0"
CUDA_VISIBLE_DEVICES=0 python -u 03_finetuning/worker_generator.py \
    --batch-size    "$BATCH" \
    --max-datasets  2000000 \
    --output-dir    "$BATCH_DIR" \
    --seed          42 \
    --group-size    8 \
    > "$LOG_DIR/generator.log" 2>&1 &
PID_GEN=$!
echo "  PID=$PID_GEN"

echo "  Waiting 30s for generator to produce initial batches..."
sleep 30

# ── GPU 1: Trainer (phase 1) ──
echo "[2/3] Trainer (phase1) → GPU 1"
CUDA_VISIBLE_DEVICES=1 python -u 03_finetuning/worker_trainer_v2.py \
    --mode          phase1 \
    --lr            "$LR" \
    --n-steps       "$N_STEPS" \
    --batch-dir     "$BATCH_DIR" \
    --checkpoint-dir "$CKPT_DIR" \
    --weight-decay  "$WEIGHT_DECAY" \
    --grad-clip     "$GRAD_CLIP" \
    > "$LOG_DIR/trainer.log" 2>&1 &
PID_TRAIN=$!
echo "  PID=$PID_TRAIN"

echo "  Waiting 60s for trainer to produce first checkpoint..."
sleep 60

# ── GPU 2: Evaluator ──
echo "[3/3] Evaluator → GPU 2"
CUDA_VISIBLE_DEVICES=2 python -u 03_finetuning/worker_evaluator_v2.py \
    --checkpoint-dir "$CKPT_DIR" \
    --log-dir        "$LOG_DIR" \
    --eval-interval  "$EVAL_INTERVAL" \
    --batch-size     "$BATCH" \
    > "$LOG_DIR/evaluator.log" 2>&1 &
PID_EVAL=$!
echo "  PID=$PID_EVAL"

echo ""
echo "Workers running:"
echo "  Generator  PID=$PID_GEN"
echo "  Trainer    PID=$PID_TRAIN"
echo "  Evaluator  PID=$PID_EVAL"
echo "Logs: $LOG_DIR/{generator,trainer,evaluator}.log"
echo ""

# ── Wait ──
wait $PID_GEN;   echo "Generator  finished (exit $?)"
wait $PID_TRAIN; echo "Trainer    finished (exit $?)"
wait $PID_EVAL;  echo "Evaluator  finished (exit $?)"

echo ""
echo "════════════════════════════════════════"
echo "  STEP 1 DONE — $(date)"
echo ""
echo "  Best checkpoint:  $CKPT_DIR/best.pt"
echo "  Last checkpoint:  $CKPT_DIR/last.pt"
echo "  Trainer log:      $LOG_DIR/trainer.log"
echo "  Eval history:     $LOG_DIR/history.json"
echo ""
echo "  Next: sbatch step2_finetune.sbatch"
echo "        (will resume from $CKPT_DIR/best.pt)"
echo "════════════════════════════════════════"
