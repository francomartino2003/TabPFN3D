# TabPFN 3D - Time Series Classification Extension

Extension of [TabPFN](https://github.com/PriorLabs/tabpfn) to 3D time series classification datasets.

## Overview

TabPFN is a foundation model for tabular data that performs in-context learning in a single forward pass. This project extends TabPFN to handle **time series classification** with shape `(n_samples, n_features, n_timesteps)`.

### Approach

1. **Synthetic Data Generation**: Generate diverse 3D datasets with temporal dependencies
2. **Temporal Encoder**: A Perceiver-style encoder compresses each time series into fixed-size embeddings
3. **Frozen TabPFN**: The pre-trained TabPFN model provides in-context learning capabilities
4. **End-to-end Training**: Train the encoder on synthetic 3D datasets, then fine-tune the full model

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
├── 00_TabPFN/                    # TabPFN source (clone separately)
├── 01_real_data/                 # Real time series datasets (UCR/UEA)
│   ├── AEON/                     # Benchmarks and analysis
│   └── src/                      # Data loading utilities
├── 02_synthetic_generator_2D/    # Original TabPFN-style 2D generator
│   ├── config.py                 # Prior configuration
│   ├── dag_builder.py            # DAG construction (topological order)
│   ├── transformations.py        # NN, Tree, Discretization (12 activations)
│   ├── generator.py              # Main generator
│   ├── sanity_checks.py          # Validation suite
│   └── tests.py                  # Unit tests
├── 03_synthetic_generator_3D/    # 3D temporal extension
│   ├── config.py                 # Prior with temporal parameters
│   ├── dag_utils.py              # DAG wrapper
│   ├── temporal_inputs.py        # Noise, Time, State inputs
│   ├── temporal_propagator.py    # Optimized temporal propagation
│   ├── sequence_sampler.py       # IID, sliding window, mixed sampling
│   ├── generator.py              # Main generator
│   ├── sanity_checks.py          # Validation + real data comparison
│   └── discriminator_analysis.py # Synthetic vs real analysis
├── 04_temporal_encoder/          # Temporal encoder + TabPFN integration
│   ├── encoder.py                # Perceiver-style temporal encoder
│   ├── tabpfn_wrapper.py         # Frozen TabPFN wrapper
│   ├── model.py                  # TemporalTabPFN model
│   └── train.py                  # Training loop
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
python src/analyze_all_datasets.py
```

This downloads UCR/UEA classification datasets (~180 datasets) using [AEON](https://github.com/aeon-toolkit/aeon).

## Quick Start

### Generate Synthetic 2D Dataset (Tabular)

```python
import sys
sys.path.insert(0, '02_synthetic_generator_2D')

from generator import SyntheticDatasetGenerator

generator = SyntheticDatasetGenerator(seed=42)
dataset = generator.generate()

print(f"X shape: {dataset.X.shape}")  # (n_samples, n_features)
print(f"y shape: {dataset.y.shape}")  # (n_samples,)
print(f"Classes: {dataset.n_classes}")
```

### Generate Synthetic 3D Dataset (Time Series)

```python
import sys
sys.path.insert(0, '03_synthetic_generator_3D')

from generator import SyntheticDatasetGenerator3D

generator = SyntheticDatasetGenerator3D(seed=42)
dataset = generator.generate()

print(f"X shape: {dataset.X.shape}")  # (n_samples, n_features, n_timesteps)
print(f"y shape: {dataset.y.shape}")  # (n_samples,)
print(f"Classes: {dataset.n_classes}")
print(f"Mode: {dataset.config.sample_mode}")  # iid, sliding_window, or mixed
```

### Run Sanity Checks

```bash
# 2D Generator
cd 02_synthetic_generator_2D
python sanity_checks.py

# 3D Generator
cd 03_synthetic_generator_3D
python sanity_checks.py
```

### Test the Temporal Encoder

```bash
cd 04_temporal_encoder
python test_encoder.py
python test_full_model.py
```

## Modules

### 01 - Real Data (UCR/UEA Archive)

- Downloads and analyzes real time series classification datasets
- Used for validation and final evaluation
- ~180 datasets from UCR/UEA archive

### 02 - Synthetic Generator 2D

Generates diverse tabular datasets from causal DAGs:

- **DAG Construction**: Topological order with controllable density
- **Transformations**: NN (12 activations), Decision Tree, Discretization
- **Post-processing**: Warping, quantization, missing values
- **Validation**: 10 sanity checks including DAG visualization

Key parameters:
- `density_range`: Controls edge density (0.01-0.8)
- `n_roots_range`: Number of input nodes (3-15)
- `activations`: 12 functions from paper (identity, log, sigmoid, etc.)
- `min_samples_per_class`: Minimum samples per class (10)

### 03 - Synthetic Generator 3D

Extends 2D generator for time series with temporal dependencies:

- **Input Types**: Noise (variability), Time (trends), State (memory)
- **Sampling Modes**: IID, Sliding Window, Mixed
- **Target Positions**: within, future_near, future_far, past
- **Optimizations**: Vectorized propagation, complexity limits

Key parameters:
- `max_complexity`: Limits n_samples × T_total × n_nodes (10M)
- `max_roots_fraction`: Roots ≤ 25% of total nodes
- `prob_sliding_window_mode`: 60% (most realistic)

Performance: ~0.8s per dataset (average)

### 04 - Temporal Encoder

- **TemporalEncoder**: Perceiver-style cross-attention
  - Compresses `t` timesteps → `K` latent queries (default K=16)
  - Per-feature encoding: `(n, m, t)` → `(n, m*K, d=128)`
- **FrozenTabPFN**: Wrapper around pre-trained TabPFN
  - Injects temporal embeddings directly
  - Gradients flow through to encoder
- **TemporalTabPFN**: Combined model for training and inference

## Synthetic Data Generation Pipeline

### 2D Pipeline
```
Root Nodes → DAG Propagation → Feature Selection → Post-processing → (X, y)
   (noise)     (NN/Tree/Disc)   (relevant/irrelevant)  (warp/missing)
```

### 3D Pipeline
```
For t = 0 to T:
    Noise(t) + Time(t) + State(t-1) → DAG Propagation → Node Values(t)
                                              ↓
                                    Update State for t+1

Extract Windows → Feature Selection → Post-processing → (X, y)
(IID/Sliding/Mixed)
```

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
| 3D Gen | `max_complexity` | 10M | Computational limit |
| 3D Gen | `density_range` | (0.01, 0.8) | DAG edge density |

## Progress

- [x] Analysis of real time series datasets (180 UCR/UEA)
- [x] 2D synthetic generator with 12 activations
- [x] 3D synthetic generator with temporal dependencies
- [x] Sanity checks with real data comparison
- [x] Discriminator analysis (synthetic vs real)
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
