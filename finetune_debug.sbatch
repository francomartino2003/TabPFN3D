#!/bin/bash
#SBATCH --job-name=finetune_debug
#SBATCH --output=finetune_debug_%j.out
#SBATCH --error=finetune_debug_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=sched_mit_hill
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

echo "============================================"
echo "FINETUNE TABPFN - DEBUG RUN"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load modules
module load cuda/12.1
module load anaconda3/2023.09

# Activate environment
source activate tabpfn3d

# Go to project directory
cd /home/fhernand/TabPFN3D

# Run debug finetuning
python 07_finetuning/finetune_tabpfn.py \
    --n-steps 20 \
    --batch-size 8 \
    --eval-every 10 \
    --device cuda \
    --seed 42

echo ""
echo "End time: $(date)"
echo "============================================"
