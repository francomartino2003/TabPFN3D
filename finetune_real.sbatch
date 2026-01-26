#!/bin/bash
#SBATCH --job-name=finetune_tabpfn
#SBATCH --output=finetune_real_%j.out
#SBATCH --error=finetune_real_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=sched_mit_hill
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

echo "============================================"
echo "FINETUNE TABPFN - FULL TRAINING"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Go to project directory
cd $HOME/TabPFN3D

# Activate conda environment
source ~/.bashrc
conda activate tabpfn3d

# Check Python and CUDA
echo "Python: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Run full finetuning
python 07_finetuning/finetune_tabpfn.py \
    --n-steps 1000 \
    --batch-size 64 \
    --eval-every 50 \
    --lr 1e-5 \
    --device cuda \
    --seed 42

echo ""
echo "End time: $(date)"
echo "============================================"
