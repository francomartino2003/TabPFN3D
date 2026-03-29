#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  Quick eval — finetuned vs vanilla TabPFN (same 8-iter schedule)
#
#  Schedule: squash odd iters, pool K=16 S=8 on iters {2,3,6,7} if T > 96
#  1 run per dataset, 112 PFN-filtered datasets.
#  Results saved to 04_evaluation/results/quick_eval.csv
# ══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=quick_eval
#SBATCH --output=logs/quick_eval_%j.out
#SBATCH --error=logs/quick_eval_%j.err
#SBATCH --partition=pi_dbertsim
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -e

echo "════════════════════════════════════════"
echo "  Quick eval — $(date)"
echo "  Job: $SLURM_JOB_ID   Node: $SLURM_NODELIST"
echo "════════════════════════════════════════"

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

python -u 04_evaluation/quick_eval.py

echo ""
echo "════════════════════════════════════════"
echo "  DONE — $(date)"
echo "  Results: 04_evaluation/results/quick_eval.csv"
echo "════════════════════════════════════════"
