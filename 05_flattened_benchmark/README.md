# 05 - Flattened 3D Benchmark

This module evaluates whether fine-tuning TabPFN on synthetic temporal data improves its performance on real temporal datasets (when both are flattened to 2D).

## Motivation

The TabPFN paper mentions fine-tuning for specific domains (e.g., medical datasets). Our hypothesis is that we can fine-tune TabPFN toward "temporal datasets in flattened form" using synthetic data from our generator, then evaluate on real flattened temporal datasets.

If we see improvement on real datasets after fine-tuning on synthetic, it validates that:
1. The synthetic generator produces useful temporal priors
2. We can tune generator hyperparameters independently (without the encoder)

Once validated, we can move to the encoder-based approach for larger 3D datasets where flattening isn't feasible.

## Data Constraints

For compatibility with TabPFN and memory limits:

| Constraint | Value | Reason |
|------------|-------|--------|
| n_samples | ≤ 1000 | TabPFN CPU limit / MPS memory |
| n_features × length | ≤ 500 | TabPFN feature limit |
| n_classes | ≤ 10 | TabPFN class limit |

**55 real datasets** meet all criteria (from 181 total UCR/UEA datasets).

## Experimental Pipeline

1. **Identify eligible datasets**: Find real 3D datasets meeting all constraints
2. **Baseline benchmark**: Evaluate TabPFN directly on flattened real datasets
3. **Generate synthetic data**: Create synthetic 3D datasets of similar distribution
4. **Visualize & compare**: Qualitative comparison of real vs synthetic
5. **Fine-tune TabPFN**: Train on flattened synthetic data
6. **Re-evaluate**: Test fine-tuned model on real datasets
7. **Compare**: Measure performance delta

## Structure

```
05_flattened_benchmark/
├── README.md
├── __init__.py
├── analyze_flattenable_datasets.py    # Identify eligible datasets
├── benchmark_baseline.py              # TabPFN baseline on real data
├── benchmark_synthetic.py             # TabPFN on synthetic data
├── generate_synthetic_matching.py     # Generate synthetic matching real distributions
├── visualize_datasets.py              # Visualize real datasets
├── finetune_tabpfn.py                 # Fine-tune TabPFN on synthetic
└── results/
    ├── baseline_benchmark_latest.csv   # Real data benchmark results
    ├── synthetic_benchmark_latest.csv  # Synthetic benchmark results
    ├── synthetic_datasets/             # Generated .npz files
    ├── synthetic_visualizations/       # PNG visualizations
    ├── real_visualizations/            # Real data visualizations
    ├── finetune_checkpoints/           # Training checkpoints
    └── finetune_logs/                  # Training logs & curves
```

## Usage

### Step 1: Analyze eligible datasets

```bash
python analyze_flattenable_datasets.py
```

Outputs:
- `usable_datasets.json`: Datasets meeting all constraints
- `usable_datasets_summary.csv`: Summary statistics

### Step 2: Run baseline benchmark on real data

```bash
python benchmark_baseline.py
```

Evaluates TabPFN on all eligible real datasets. Uses CPU with `ignore_pretraining_limits=True` to handle datasets with >1000 samples.

### Step 3: Generate synthetic datasets

```bash
python generate_synthetic_matching.py --n-datasets 20 --visualize-all --seed 42
```

Generates synthetic 3D datasets with distributions matching the real data:
- Similar n_samples, n_features, length, n_classes distributions
- Respects the 500 flattened feature limit
- Higher probability for i.i.d. sampling mode

### Step 4: Benchmark synthetic data

```bash
python benchmark_synthetic.py
```

Runs TabPFN on the generated synthetic datasets to verify they're learnable.

### Step 5: Visualize datasets

```bash
python visualize_datasets.py
```

Creates visualizations of real datasets for qualitative comparison with synthetic.

### Step 6: Fine-tune TabPFN

```bash
# Quick debug run
python finetune_tabpfn.py --debug

# Full training
python finetune_tabpfn.py --n-steps 1000 --batch-size 64 --eval-every 50

# Resume from checkpoint
python finetune_tabpfn.py --resume results/finetune_checkpoints/checkpoint_step500.pt
```

**Fine-tuning approach:**
- Generates synthetic datasets on-the-fly (never repeats)
- Flattens to 2D before passing to TabPFN
- Low learning rate (1e-5) to preserve pretrained knowledge
- Batch size = 64 datasets with gradient accumulation
- Cross-entropy loss on test predictions
- Periodic evaluation on real datasets (AUC ROC tracking)

**Key hyperparameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n-steps` | 1000 | Number of optimizer steps |
| `--batch-size` | 64 | Datasets per gradient update |
| `--lr` | 1e-5 | Learning rate |
| `--eval-every` | 50 | Evaluate on real data every N steps |
| `--device` | cpu | Device (cpu/cuda/mps) |

## Results

After fine-tuning, compare:
- **Baseline**: `results/baseline_benchmark_latest.csv`
- **Fine-tuned**: Evaluation metrics in `results/finetune_logs/training_history.json`

Key metrics:
- ROC AUC (main metric)
- Accuracy
- Training curves: `results/finetune_logs/training_curves.png`

## Dependencies

- TabPFN (from `00_TabPFN/`)
- Synthetic generator (from `03_synthetic_generator_3D/`)
- Real data (from `01_real_data/AEON/data/`)
