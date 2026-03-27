#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  Benchmark — our finetuned model (phase1 + phase2)
#
#  Evaluates 4 configurations on all PFN-eligible datasets:
#    D1_e8       — phase1/best.pt,  8 ensemble iterations
#    D2_e1       — phase2/best.pt,  1 ensemble iteration
#    D2_e8       — phase2/best.pt,  8 ensemble iterations
#    D2_e8_prep  — phase2/best.pt,  8 ensemble iterations + step-repeat
#
#  Each config × dataset: 30 random seeds → averaged acc and auc.
#
#  UCR and UEA run sequentially on the same GPU.
#  Results saved to 04_evaluation/results/benchmark_ours/
#    ucr_benchmark_ours.csv
#    uea_benchmark_ours.csv
#
#  Override checkpoints:
#    PHASE1=path/to/phase1/best.pt PHASE2=path/to/phase2/best.pt sbatch benchmark_ours.sbatch
#  Override runs:
#    N_RUNS=10 sbatch benchmark_ours.sbatch
# ══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=tabpfn_bench
#SBATCH --output=logs/bench_%j.out
#SBATCH --error=logs/bench_%j.err
#SBATCH --partition=pi_dbertsim
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -e

echo "════════════════════════════════════════"
echo "  Benchmark — $(date)"
echo "  Job: $SLURM_JOB_ID   Node: $SLURM_NODELIST"
echo "════════════════════════════════════════"

# ── Environment ──────────────────────────────────────────────────────────────
if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
module load sloan/python/3.11.4 2>/dev/null || echo "[warn] module system not available"
source $HOME/venv_tabpfn3d/bin/activate

cd $HOME/TabPFN3D
mkdir -p logs

echo ""
echo "[Environment]"
echo "  Python  : $(which python)"
echo "  PyTorch : $(python -c 'import torch; print(torch.__version__)')"
echo "  GPU     : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")')"
echo ""

# ── Config ───────────────────────────────────────────────────────────────────
PHASE1="${PHASE1:-03_finetuning/checkpoints/phase1/best.pt}"
PHASE2="${PHASE2:-03_finetuning/checkpoints/phase2/best.pt}"
DEVICE="${DEVICE:-cuda}"
N_RUNS="${N_RUNS:-30}"

echo "[Config]"
echo "  Phase-1 ckpt : $PHASE1"
echo "  Phase-2 ckpt : $PHASE2"
echo "  Device       : $DEVICE"
echo "  N runs       : $N_RUNS"
echo ""

# ── Sanity checks ────────────────────────────────────────────────────────────
for F in "$PHASE1" "$PHASE2"; do
    if [ ! -f "$F" ]; then
        echo "ERROR: checkpoint not found: $F"
        exit 1
    fi
done

# ── UCR ──────────────────────────────────────────────────────────────────────
echo "════════════════════════════"
echo "  [1/2] UCR benchmark — $(date)"
echo "════════════════════════════"
python -u 04_evaluation/benchmark_ours.py \
    --phase1-ckpt "$PHASE1" \
    --phase2-ckpt "$PHASE2" \
    --device      "$DEVICE" \
    --collection  ucr \
    --n-runs      "$N_RUNS"

# ── UEA ──────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════"
echo "  [2/2] UEA benchmark — $(date)"
echo "════════════════════════════"
python -u 04_evaluation/benchmark_ours.py \
    --phase1-ckpt "$PHASE1" \
    --phase2-ckpt "$PHASE2" \
    --device      "$DEVICE" \
    --collection  uea \
    --n-runs      "$N_RUNS"

echo ""
echo "════════════════════════════════════════"
echo "  Benchmark DONE — $(date)"
echo "  Results: 04_evaluation/results/benchmark_ours/"
echo "════════════════════════════════════════"
