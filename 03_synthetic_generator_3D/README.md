# 3D Synthetic Dataset Generator with Temporal Dependencies

Generator of 3D synthetic datasets with temporal dependencies for time series classification.

## Overview (v4)

This module generates datasets with shape `(n_samples, n_features, t_timesteps)` where:
- `n_samples`: Number of observations
- `n_features`: Number of features (observed graph nodes)
- `t_timesteps`: Temporal subsequence length

## Architecture (v4)

```
                    ┌──────────────────────────────────────────┐
                    │           Causal DAG                     │
                    │  (constructed with topological order)    │
                    └──────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
              ┌─────┴─────┐                      ┌──────┴──────┐
              │  t = 0    │        ...           │  t = T-1    │
              └─────┬─────┘                      └──────┬──────┘
                    │                                   │
    Inputs:   TIME(0) + MEMORY              TIME(T-1) + MEMORY (same)
                    │                                   │
                    ▼                                   ▼
              Propagation                         Propagation
                    │                                   │
                    └───────────────┬───────────────────┘
                                    │
                              Extract Windows
                                    │
                                    ▼
                              (n, m, t), y
```

## Root Inputs (v4)

### TIME Input (1 node)
- Single deterministic input: `u = t/T` (normalized time in [0, 1])
- Same value for all samples at timestep t
- Provides temporal dependency structure

### MEMORY Inputs (1-8 nodes)
- Vector of dimension 1-8, sampled ONCE per sequence
- Does NOT change within a sequence
- **For IID mode**: Each sample gets its own MEMORY → variability between samples
- **For sliding/mixed**: One MEMORY per long sequence
- Initialization: Normal N(0, σ²) or Uniform U(-a, a)

## Transformations (v4)

Each non-root node has **one** transformation:

| Type | Probability | Description |
|------|-------------|-------------|
| **NN** | 70% | weights × parents + bias → activation → noise |
| **Tree** | 15% | Decision tree over subset of parents |
| **Discretization** | 15% | Distance to prototypes → normalized category |

### Available Activations (8)
```python
['identity', 'tanh', 'sigmoid', 'relu', 'leaky_relu', 'elu', 'softplus', 'sin']
```

### Noise
- Applied ONLY at the end of each transformation
- Scale: 0.001 to 0.05 (log-uniform)

## Feature and Target Selection (v4)

### Target
- **Must be from a discretization node** (categorical = classification)
- Prefers nodes in main subgraph

### Features
- **At least 1 RELEVANT** (from main connected subgraph)
- **At least 1 CONTINUOUS** (not from discretization)
- Distance-weighted selection for remaining features

## Sampling Modes

### IID Mode (55%)
```
Sequence 1: ──────────────────────────── (own MEMORY)
Sequence 2: ──────────────────────────── (own MEMORY)
    ...
Sequence N: ──────────────────────────── (own MEMORY)
```
Each sample is an independent sequence with its own MEMORY.

### Sliding Window Mode (30%)
```
Long sequence: ══════════════════════════════════════════ (shared MEMORY)
Windows:       [───────]
                 [───────]
                   [───────]
                     [───────]
```

### Mixed Mode (15%)
```
Seq 1: ══════════════════════ (MEMORY 1)
        [───] [───] [───]
Seq 2: ══════════════════════ (MEMORY 2)
        [───] [───] [───]
```

## Target Offset (v4)

- **Balanced 50/50 future/past** (not 75% future)
- prob(offset=k) ∝ 1 / (1 + |k|^α)
- Favors offset=0 (within), then ±1, ±2, etc.

| Type | Offset |
|------|--------|
| within | 0 |
| future | > 0 |
| past | < 0 |

## Key Parameters (v4)

### Size Limits
| Parameter | Value | Description |
|-----------|-------|-------------|
| max_samples | 10,000 | Maximum samples |
| max_features | 15 | Maximum features |
| max_t_subseq | 1,000 | Maximum timesteps per window |
| max_T_total | 5,000 | Maximum total timesteps |
| max_classes | 10 | Maximum classes |
| max_complexity | 10,000,000 | n_samples × T_total × n_nodes |

### Root Configuration
| Parameter | Value |
|-----------|-------|
| TIME inputs | 1 (fixed) |
| MEMORY dimension | 1-8 |
| MEMORY noise type | normal or uniform |
| MEMORY σ range | (0.5, 2.0) |

### Graph Structure
| Parameter | Value |
|-----------|-------|
| n_nodes_range | (8, 30) |
| density_range | (0.1, 0.6) |
| Total roots | 1 + memory_dim |

## Usage

```python
from generator import SyntheticDatasetGenerator3D, generate_3d_dataset

# Quick generation
dataset = generate_3d_dataset(seed=42)
X, y = dataset.X, dataset.y  # (n, m, t), (n,)

# With custom configuration
from config import PriorConfig3D
prior = PriorConfig3D(
    max_features=10,
    prob_nn_transform=0.70,
    prob_tree_transform=0.15,
    prob_discretization=0.15,
)
generator = SyntheticDatasetGenerator3D(prior=prior, seed=42)
dataset = generator.generate()

# Multiple datasets
for i, dataset in enumerate(generator.generate_many(100)):
    print(f"Dataset {i}: {dataset.shape}")
```

## Module Structure

```
03_synthetic_generator_3D/
├── config.py              # PriorConfig3D, DatasetConfig3D
├── dag_utils.py           # DAG wrapper, TransformationFactory
├── temporal_inputs.py     # TIME + MEMORY generators
├── temporal_propagator.py # Temporal propagation (no clipping)
├── sequence_sampler.py    # Subsequence extraction
├── feature_selector.py    # Feature (1+ relevant, 1+ cont) and target (disc) selection
├── generator.py           # Main class
├── sanity_checks.py       # Validation
└── README.md
```

## Differences from v3

| Aspect | v3 | v4 |
|--------|----|----|
| TIME inputs | Multiple with activations | 1 (u = t/T only) |
| STATE inputs | Yes (t-k lookups) | **No** |
| MEMORY inputs | No | **Yes (1-8 dims)** |
| Variability source | State noise init | MEMORY sampling |
| NN/Tree/Disc | 40/45/15 | **70/15/15** |
| Noise params | Many (edge, init, etc.) | **Single scale** |
| Target | Any node | **Discretization only** |
| Feature constraints | None | **1+ relevant, 1+ continuous** |
| Future prob | 75% | **50%** |
| Clipping | Yes (-1e6, 1e6) | **No** |
