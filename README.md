# TabPFN 3D - Time Series Classification Extension

Extension of [TabPFN](https://github.com/PriorLabs/tabpfn) to 3D time series classification datasets.

## Overview

TabPFN is a foundation model for tabular data that performs in-context learning in a single forward pass. This project extends TabPFN to handle **time series classification** with shape `(n_samples, n_features, n_timesteps)`.

### Approach

1. **Temporal Encoder**: A Perceiver-style encoder compresses each time series into fixed-size embeddings
2. **Frozen TabPFN**: The pre-trained TabPFN model provides in-context learning capabilities
3. **End-to-end Training**: Train the encoder on synthetic 3D datasets, then fine-tune the full model

```
Input: (n_samples, n_features, n_timesteps)
           │
           ▼
   ┌─────────────────┐
   │ Temporal Encoder │  ← TRAINABLE (Perceiver cross-attention)
   │ (n,m,t)→(n,m*K,d)│
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │  Frozen TabPFN   │  ← Pre-trained, frozen weights
   │  (in-context)    │
   └────────┬────────┘
            │
            ▼
Output: (n_test, n_classes)
```

## Project Structure

```
TabPFN3D/
├── 00_TabPFN/                    # TabPFN source (clone separately, see below)
├── 01_real_data/                 # Real time series datasets (UCR/UEA)
│   ├── AEON/                     # Benchmarks and analysis
│   └── src/                      # Data loading utilities
├── 02_synthetic_generator_2D/    # Original TabPFN-style generator
├── 03_synthetic_generator_3D/    # 3D temporal extension
│   ├── config.py                 # Prior configuration
│   ├── temporal_dag_builder.py   # Unrolled temporal DAG
│   ├── temporal_connections.py   # Rich temporal patterns
│   ├── row_generator_3d.py       # Time-aware propagation
│   └── generator.py              # Main generator
├── 04_temporal_encoder/          # Temporal encoder + TabPFN integration
│   ├── encoder.py                # Perceiver-style temporal encoder
│   ├── tabpfn_wrapper.py         # Frozen TabPFN wrapper
│   ├── model.py                  # TemporalTabPFN model
│   ├── train.py                  # Training loop
│   └── training_config.py        # Configuration
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/francomartino2003/TabPFN3D.git
cd TabPFN3D
```

### 2. Clone TabPFN into `00_TabPFN/`

```bash
git clone https://github.com/PriorLabs/tabpfn.git 00_TabPFN
```

### 3. Install dependencies

```bash
# Create conda environment
conda create -n tabpfn3d python=3.10
conda activate tabpfn3d

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install other dependencies
pip install -r requirements.txt
```

### 4. Download real datasets (optional)

```bash
cd 01_real_data
python src/load_classification_datasets.py
```

This downloads UCR/UEA classification datasets (~180 datasets) using [AEON](https://github.com/aeon-toolkit/aeon).

The benchmark CSVs can be downloaded from [AEON bakeoff results](https://github.com/aeon-toolkit/aeon/tree/main/aeon/benchmarking/results) and placed in `01_real_data/AEON/benchmarks/`.

## Quick Start

### Generate Synthetic 3D Dataset

```python
import sys
sys.path.insert(0, '03_synthetic_generator_3D')

from generator import SyntheticDatasetGenerator3D
from config import PriorConfig3D

generator = SyntheticDatasetGenerator3D(seed=42)
dataset = generator.generate()

print(f"X shape: {dataset.X.shape}")  # (n_samples, n_features, n_timesteps)
print(f"y shape: {dataset.y.shape}")  # (n_samples,)
print(f"Classes: {dataset.n_classes}")
```

### Test the Temporal Encoder

```bash
cd 04_temporal_encoder
python test_encoder.py
python test_full_model.py
python test_training.py
```

### Train the Model

```python
import sys
sys.path.insert(0, '04_temporal_encoder')

from training_config import get_default_config
from train import train

config = get_default_config()
model = train(config)
```

## Modules

### 01 - Real Data (UCR/UEA Archive)

- Downloads and analyzes real time series classification datasets
- Used for validation and final evaluation
- ~180 datasets from UCR/UEA archive

### 02 - Synthetic Generator 2D

- Original TabPFN-style tabular data generator
- Generates diverse datasets from causal DAGs
- Used as reference implementation

### 03 - Synthetic Generator 3D

- Extends 2D generator to time series
- Creates unrolled temporal DAGs with rich temporal connections:
  - Self-connections (AR-like)
  - Cross-connections
  - Multi-skip with decay
  - Conditional lag switching
- Generates datasets with shape `(n, m, t)`

### 04 - Temporal Encoder

- **TemporalEncoder**: Perceiver-style cross-attention
  - Compresses `t` timesteps → `K` latent queries (default K=16)
  - Per-feature encoding: `(n, m, t)` → `(n, m*K, d=128)`
- **FrozenTabPFN**: Wrapper around pre-trained TabPFN
  - Injects temporal embeddings directly into TabPFN
  - Gradients flow through to encoder
- **TemporalTabPFN**: Combined model for training and inference

## Training Pipeline

1. **Phase 1**: Train temporal encoder only (TabPFN frozen)
   - Train on synthetic 3D datasets
   - Validate on fixed synthetic set + real datasets

2. **Phase 2**: Fine-tune entire model (planned)
   - Unfreeze TabPFN
   - Continue training with lower learning rate

## Key Hyperparameters

| Component | Parameter | Default | Description |
|-----------|-----------|---------|-------------|
| Encoder | `d_model` | 128 | Must match TabPFN's emsize |
| Encoder | `n_queries` | 16 | Latent queries per feature |
| Encoder | `n_layers` | 2 | Self-attention layers |
| Training | `batch_datasets` | 4 | Datasets per batch |
| Training | `n_steps` | 10000 | Training steps |
| Data | `n_timesteps_range` | (50, 500) | Time series length |

## Progress

- [x] Analysis of real time series datasets
- [x] 2D synthetic generator (TabPFN-style)
- [x] 3D synthetic generator (temporal extension)
- [x] Temporal encoder (Perceiver-style)
- [x] TabPFN integration (embedding injection)
- [x] Training pipeline
- [ ] Full training run
- [ ] Evaluation on real datasets
- [ ] Fine-tuning phase
- [ ] Comparison with baselines

## References

- [TabPFN Paper](https://arxiv.org/abs/2207.01848) - Hollmann et al., 2022
- [TabPFN v2](https://arxiv.org/abs/2402.04082) - Updated architecture
- [UCR/UEA Archive](https://www.timeseriesclassification.com/) - Time series datasets
- [Perceiver](https://arxiv.org/abs/2103.03206) - Cross-attention architecture

## License

MIT License

## Citation

If you use this work, please cite:

```bibtex
@software{tabpfn3d,
  author = {Franco Martino},
  title = {TabPFN 3D: Time Series Classification Extension},
  year = {2024},
  url = {https://github.com/francomartino2003/TabPFN3D}
}
```
