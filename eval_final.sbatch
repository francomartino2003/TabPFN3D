#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  Final Evaluation — UCR + UEA benchmarks
#
#  Runs final_evaluation.py with the phase2 best checkpoint.
#  UCR and UEA are evaluated sequentially on a single GPU.
#  Results saved to 04_evaluation/results/<ckpt_stem>/.
#
#  Override checkpoint:  CKPT=path/to/other.pt sbatch eval_final.sbatch
# ══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=tabpfn_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --partition=pi_dbertsim
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -e

echo "════════════════════════════════════════"
echo "  Final Evaluation — $(date)"
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
echo "GPU:     $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")')"
echo ""

# ── Config ──
CKPT="${CKPT:-03_finetuning/checkpoints/phase2/best.pt}"
DEVICE="${DEVICE:-cuda}"

echo "[Config]"
echo "  Checkpoint: $CKPT"
echo "  Device:     $DEVICE"
echo ""

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT"
    exit 1
fi

# ── UCR ──
echo "[1/2] UCR evaluation"
python -u 04_evaluation/final_evaluation.py "$CKPT" \
    --device     "$DEVICE" \
    --collection ucr

# ── UEA ──
echo "[2/2] UEA evaluation"
python -u 04_evaluation/final_evaluation.py "$CKPT" \
    --device     "$DEVICE" \
    --collection uea

echo ""
echo "════════════════════════════════════════"
echo "  Evaluation DONE — $(date)"
echo "  Results: 04_evaluation/results/$(basename $CKPT .pt)/"
echo "════════════════════════════════════════"
