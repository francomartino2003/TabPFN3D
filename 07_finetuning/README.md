# 07 - TabPFN Finetuning

Fine-tune TabPFN on flattened synthetic 3D time series datasets.

## Overview

This module uses the synthetic data generator from `06_generator_experiments` to create training data for finetuning TabPFN. The generator creates diverse time series with:
- Random neural network architectures
- Diverse activation functions (identity, log, sigmoid, abs, sin, tanh, etc.)
- Per-node noise
- IID sampling mode
- Post-hoc quantization for classification

## Usage

### Train
```bash
# Full training (1000 steps, batch size 64, eval every 50 steps)
python finetune_tabpfn.py --n-steps 1000 --batch-size 64 --eval-every 50

# Debug mode (quick test)
python finetune_tabpfn.py --debug

# With custom learning rate
python finetune_tabpfn.py --lr 3e-5 --n-steps 500
```

### Evaluate Checkpoints
```bash
# Evaluate all checkpoints on all real datasets
python evaluate_checkpoints.py --checkpoint-dir checkpoints --device cuda
```

## Configuration

Key parameters in `FinetuneConfig`:
- `lr`: Learning rate (default: 1e-5)
- `batch_size`: Number of datasets per gradient update (default: 64)
- `n_steps`: Number of training steps (default: 1000)
- `eval_every`: Evaluate every N steps (default: 50)

## Output

- `checkpoints/`: Model checkpoints at each evaluation step
- `logs/`: Training curves, evaluation results

## Data Constraints

Both synthetic and real datasets must satisfy:
- n_samples ≤ 1000
- n_features × length ≤ 500 (flattened features)
- n_classes ≤ 10
