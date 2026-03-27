#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  Benchmark TabPFN on all PFN-eligible UCR and UEA datasets.
#
#  Three variants, each run N_RUNS times with different seeds and averaged:
#    e1           — TabPFN with n_estimators=1  (standard sklearn pipeline)
#    e8           — TabPFN with n_estimators=8  (standard sklearn pipeline)
#    e8_ours_inf  — pretrained TabPFN + our inference pipeline
#                   (per-channel normalisation, channel/class permutation,
#                    odd-iter squashing, no SVD, no feature subsampling)
#
#  Prereq:  01_real_data/datasets_summary.csv and 01_real_data/data/ must exist.
#           Run 'python 01_real_data/download.py' locally first, then rsync data.
#
#  Outputs (loaded by 04_evaluation/final_evaluation.py):
#    01_real_data/benchmark_results/ucr_benchmark_tabpfn.csv
#    01_real_data/benchmark_results/uea_benchmark_tabpfn.csv
#
#  UCR and UEA are run as separate sequential steps so a timeout in one
#  does not lose results already saved for the other.  The script saves
#  results incrementally (one row per dataset) so it is safe to interrupt
#  and resume — already-finished datasets are skipped automatically.
#
#  Usage:
#    sbatch benchmark_tabpfn.sbatch
#    sbatch --export=ALL,N_RUNS=10 benchmark_tabpfn.sbatch    # quick test
# ══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=benchmark_tabpfn
#SBATCH --output=logs/benchmark_tabpfn_%j.out
#SBATCH --error=logs/benchmark_tabpfn_%j.err
#SBATCH --partition=pi_dbertsim
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4

set -e

echo "════════════════════════════════════════"
echo "  BENCHMARK TabPFN (e1 / e8 / e8_ours_inf) — $(date)"
echo "  Job: $SLURM_JOB_ID   Node: $SLURM_NODELIST"
echo "════════════════════════════════════════"

# ── Environment ──
if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
module load sloan/python/3.11.4 2>/dev/null || echo "Module system not available"
source $HOME/venv_tabpfn3d/bin/activate

cd $HOME/TabPFN3D
mkdir -p logs 01_real_data/benchmark_results

echo ""
echo "[Environment]"
echo "Python:  $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:    $(python -c 'import torch; print(torch.cuda.is_available())')"
python -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}')" 2>/dev/null || true
echo ""

# ── Tunable ──
N_RUNS="${N_RUNS:-30}"
DEVICE="cuda"

echo "[Config]  N_RUNS=$N_RUNS  DEVICE=$DEVICE"
echo ""

# ── Prereq check ──
if [ ! -f 01_real_data/datasets_summary.csv ]; then
    echo "ERROR: 01_real_data/datasets_summary.csv not found."
    echo "       Run 'python 01_real_data/download.py' locally and rsync the data."
    exit 1
fi

# ── UCR ──
echo "════════════════════════════════════════"
echo "  Step 1/2 — UCR datasets  ($(date))"
echo "════════════════════════════════════════"
python -u 01_real_data/benchmark_tabpfn.py \
    --ucr-only \
    --device  "$DEVICE" \
    --n-runs  "$N_RUNS"
echo "UCR done  ($(date))"
echo ""

# ── UEA ──
echo "════════════════════════════════════════"
echo "  Step 2/2 — UEA datasets  ($(date))"
echo "════════════════════════════════════════"
python -u 01_real_data/benchmark_tabpfn.py \
    --uea-only \
    --device  "$DEVICE" \
    --n-runs  "$N_RUNS"
echo "UEA done  ($(date))"
echo ""

echo "════════════════════════════════════════"
echo "  DONE — $(date)"
echo "  Results:"
echo "    01_real_data/benchmark_results/ucr_benchmark_tabpfn.csv"
echo "    01_real_data/benchmark_results/uea_benchmark_tabpfn.csv"
echo "════════════════════════════════════════"
