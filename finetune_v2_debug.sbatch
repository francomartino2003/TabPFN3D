#!/bin/bash
#SBATCH --job-name=ft_v2_debug
#SBATCH --output=logs/finetune_v2_debug_%j.out
#SBATCH --error=logs/finetune_v2_debug_%j.err
#SBATCH --partition=pi_dbertsim
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Quick debug run (10 steps, batch 4)
set -e

if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
module load sloan/python/3.11.4 2>/dev/null || true
source $HOME/venv_tabpfn3d/bin/activate

cd $HOME/TabPFN3D/07_finetuning

echo "Debug finetune V2 — $(date)"
python -u finetune_tabpfn_v2.py --debug --device cuda

echo "Done — $(date)"
