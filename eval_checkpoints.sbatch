#!/bin/bash
#SBATCH --job-name=eval_checkpoints
#SBATCH --output=eval_checkpoints_%j.out
#SBATCH --error=eval_checkpoints_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=sched_mit_hill
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

echo "============================================"
echo "EVALUATE ALL CHECKPOINTS"
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

# Run evaluation
python 07_finetuning/evaluate_checkpoints.py \
    --checkpoint-dir 07_finetuning/checkpoints \
    --output-dir 07_finetuning/eval_results \
    --device cuda \
    --n-estimators 8

echo ""
echo "End time: $(date)"
echo "============================================"
