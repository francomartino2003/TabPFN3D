# 04 - Temporal Encoder for TabPFN

This module extends TabPFN to time series classification by adding a learnable temporal encoder while keeping TabPFN frozen.

## Architecture

```
Input: (n_samples, n_features, n_timesteps)
           |
           v
    [Preprocessor3D]         # Normalization, missing handling
           |
           v
    [TemporalEncoder]        # LEARNABLE: (n, m, t) -> (n, m*16, 128)
           |
           v
    [FrozenTabPFN]           # FROZEN: In-context learning
           |
           v
Output: (n_test, n_classes)  # Predictions for test samples
```

## Key Components

### TemporalEncoder (Perceiver-style)

For each time series feature:
1. **Input Projection**: Each timestep `(1,)` → `(128,)`
2. **Positional Encoding**: Sinusoidal or learned
3. **Cross-Attention**: 16 learnable queries attend to all timesteps
4. **Self-Attention**: Optional layers between the 16 latent queries

Output: `n_features × 16 = K` "virtual features" with 128-dim embeddings

### FrozenTabPFN

The pre-trained TabPFN model with:
- All parameters frozen (`requires_grad=False`)
- `emsize = 128` (embedding dimension)
- `nlayers = 12` (transformer layers)
- Bypasses the original input encoder, uses our temporal embeddings directly

## Training Paradigm

Following TabPFN's training approach:

1. **Batch = Multiple Datasets**: Each batch contains `batch_datasets` complete classification tasks
2. **Train/Test Split**: Each dataset has visible train labels and hidden test labels
3. **Loss on Test Only**: Cross-entropy computed only on test sample predictions
4. **Only Encoder Trained**: TabPFN remains completely frozen

## Quick Start

### Training

```python
from config import get_default_config
from train import train

# Default training
config = get_default_config()
model = train(config)

# Debug run (quick test)
from config import get_debug_config
config = get_debug_config()
model = train(config)
```

### Command Line

```bash
# Default training
python -m 04_temporal_encoder.train

# Debug mode (quick run)
python -m 04_temporal_encoder.train --debug

# With custom config
python -m 04_temporal_encoder.train --config my_config.json

# Resume from checkpoint
python -m 04_temporal_encoder.train --resume checkpoints/checkpoint_step1000.pt
```

### Inference

```python
from model import TemporalTabPFN
from config import FullConfig

# Load trained model
config = FullConfig.load("checkpoints/config.json")
model = TemporalTabPFN(config)

# Load encoder weights
checkpoint = torch.load("checkpoints/best_model.pt")
model.encoder.load_state_dict(checkpoint["encoder_state_dict"])

# Predict
predictions, probabilities = model.predict(X_train, y_train, X_test)
```

## Configuration

All hyperparameters are in `config.py`:

### EncoderConfig
| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 128 | Embedding dimension (must match TabPFN) |
| `n_queries` | 16 | Latent queries per feature |
| `n_layers` | 2 | Self-attention layers after cross-attention |
| `n_heads` | 8 | Attention heads |
| `dropout` | 0.1 | Dropout rate |
| `pos_enc_type` | "learned" | Positional encoding type |

### TrainingConfig
| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 1e-4 | Learning rate |
| `batch_datasets` | 4 | Datasets per batch |
| `n_steps` | 10000 | Total training steps |
| `eval_every` | 500 | Evaluation frequency |
| `warmup_steps` | 500 | LR warmup steps |
| `val_synth_seed` | 42 | Fixed seed for synthetic validation |

### DataConfig
| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_samples_range` | (100, 2000) | Samples per dataset |
| `n_features_range` | (1, 10) | Features per dataset |
| `n_timesteps_range` | (50, 500) | Timesteps per dataset |
| `max_classes` | 10 | Maximum classes |

## File Structure

```
04_temporal_encoder/
├── __init__.py
├── config.py           # All configuration classes
├── encoder.py          # TemporalEncoder, positional encodings
├── preprocessing_3d.py # 3D-adapted preprocessing
├── tabpfn_wrapper.py   # Frozen TabPFN wrapper
├── data_loader.py      # Synthetic and real data loading
├── model.py            # Main TemporalTabPFN model
├── evaluate.py         # Evaluation utilities
├── train.py            # Training loop
└── README.md
```

## Evaluation

The training tracks:

1. **Synthetic Validation**: Fixed set of synthetic datasets (same seed every time)
   - Monitors if the model learns the synthetic prior

2. **Real Validation**: Actual time series classification datasets
   - Monitors transfer to real-world data

Metrics: Accuracy, Cross-Entropy Loss, Macro-F1

## Requirements

- PyTorch
- NumPy
- scikit-learn (for metrics)
- TabPFN (from `00_TabPFN/`)
- Synthetic Generator (from `03_synthetic_generator_3D/`)
- matplotlib (optional, for plotting)
- wandb (optional, for logging)

## Notes

- **Memory**: Each dataset is processed independently, so memory scales with dataset size, not batch size
- **GPU**: Recommended for reasonable training speed
- **TabPFN**: Must have TabPFN model weights downloaded (auto-downloads on first use)

