#!/bin/bash
#SBATCH --job-name=bench_timing
#SBATCH --output=logs/benchmark_timing_%j.out
#SBATCH --error=logs/benchmark_timing_%j.err
#SBATCH --partition=pi_dbertsim
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

# Benchmark finetuning timing: batch generation + synth eval + real eval
#
# Usage:
#   sbatch benchmark_timing.sbatch
#   sbatch --export=ALL,BATCH=96,REPEATS=10 benchmark_timing.sbatch

set -e

echo "=========================================="
echo "BENCHMARK TIMING — $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi
module load sloan/python/3.11.4 2>/dev/null || true
source $HOME/venv_tabpfn3d/bin/activate

cd $HOME/TabPFN3D
mkdir -p logs

BATCH="${BATCH:-96}"
REPEATS="${REPEATS:-5}"
echo "BATCH=$BATCH  REPEATS=$REPEATS"
echo ""

cd 07_finetuning
python -u benchmark_timing.py --batch-size $BATCH --n-repeats $REPEATS

echo ""
echo "=========================================="
echo "DONE — $(date)"
echo "=========================================="
