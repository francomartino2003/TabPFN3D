# 3D Synthetic Dataset Generator with Temporal Dependencies

Generator of 3D synthetic datasets with temporal dependencies for time series classification.

## Overview

This module generates datasets with shape `(n_samples, n_features, t_timesteps)` where:
- `n_samples`: Number of observations
- `n_features`: Number of features (observed graph nodes)
- `t_timesteps`: Temporal subsequence length

## Architecture

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
    Inputs:  Noise + Time(0) + State(init)    Noise + Time(T-1) + State(t-1)
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

## Input Types (Root Nodes)

### Noise Inputs
- Normal N(0, σ²) or Uniform U(-a, a)
- New values at each timestep
- Provide variability between samples

### Time Inputs
Deterministic functions of normalized time `u = t/T`:
- `linear`: u
- `quadratic`: u²
- `cubic`: u³
- `tanh`: tanh(β(2u-1)), β ∈ LogUniform(0.5, 3.0)
- `sin_k`: sin(2πku), k ∈ {1,2,3,5}
- `cos_k`: cos(2πku), k ∈ {1,2,3,5}
- `exp_decay`: exp(-γu), γ ∈ LogUniform(0.5, 5.0)
- `log`: log(u + 0.1)

### State Inputs
- Memory from previous timestep
- At t=0 initialized with noise
- Normalized: `tanh(α · s_{t-1})`
- Allow temporal dependencies (AR-like)

## Transformations

Each non-root node has **one** transformation (same as 2D):

| Type | Description |
|------|-------------|
| **NN** | weights × parents + bias → activation → noise |
| **Tree** | Decision tree over subset of parents |
| **Discretization** | Distance to prototypes → normalized category |

### Available Activations (12)
```python
['identity', 'log', 'sigmoid', 'abs', 'sin', 'tanh', 
 'rank', 'square', 'power', 'softplus', 'step', 'mod']
```

## Sampling Modes

### IID Mode
```
Sequence 1: ────────────────────────────
Sequence 2: ────────────────────────────
    ...
Sequence N: ────────────────────────────
```
Each sample is an independent sequence with different noise.

### Sliding Window Mode
```
Long sequence: ══════════════════════════════════════════
Windows:       [───────]
                 [───────]
                   [───────]
                     [───────]
```
From a long sequence T, multiple windows are extracted (may overlap).

### Mixed Mode
```
Seq 1: ══════════════════════
        [───] [───] [───]
Seq 2: ══════════════════════
        [───] [───] [───]
```
Several long sequences, multiple windows per sequence.

## Target Configuration

The target can be at different positions:
- **within**: Within the feature window
- **future_near**: 1-5 steps after the window
- **future_far**: 6-20 steps ahead
- **past**: Before the window (rare)

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
    prob_classification=1.0,
    prob_sliding_window_mode=0.6,
    max_complexity=5_000_000  # Limit complexity
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
├── dag_utils.py           # DAG wrapper over 2D (uses topological order)
├── temporal_inputs.py     # Input generators (noise, time, state)
├── temporal_propagator.py # Optimized temporal propagation
├── sequence_sampler.py    # Subsequence extraction
├── feature_selector.py    # Feature and target selection
├── generator.py           # Main class
├── sanity_checks.py       # Complete validation + comparison with real data
├── discriminator_analysis.py  # Synthetic vs real analysis
├── visualize_dag.py       # Graph visualization
└── README.md
```

## Key Parameters

### Size Limits
| Parameter | Value | Description |
|-----------|-------|-------------|
| max_samples | 10,000 | Maximum samples |
| max_features | 15 | Maximum features |
| max_t_subseq | 1,000 | Maximum timesteps per window |
| max_T_total | 5,000 | Maximum total timesteps |
| max_classes | 10 | Maximum classes |
| max_complexity | 10,000,000 | n_samples × T_total × n_nodes |

### Graph Structure
| Parameter | Value | Description |
|-----------|-------|-------------|
| n_nodes_range | (12, 300) | DAG nodes |
| density_range | (0.01, 0.8) | Edge density |
| n_roots_range | (3, 40) | Number of roots |
| max_roots_fraction | 0.25 | Roots ≤ 25% of nodes |

### Input Distribution
| Type | Minimum | Description |
|------|---------|-------------|
| Noise | 1 | Variability between samples |
| Time | 1 | Temporal trends |
| State | 1 | Temporal dependencies |

### Mode Probabilities
| Mode | Probability |
|------|-------------|
| IID | 20% |
| Sliding Window | 60% |
| Mixed | 20% |

## Performance Optimizations

The generator includes several optimizations:

1. **Complexity limit**: If `n_samples × T_total × n_nodes > max_complexity`, automatically reduces parameters

2. **Vectorized propagation**: Pre-allocated arrays instead of dictionaries

3. **Timeseries cache**: Timeseries are cached for efficient extraction

4. **Batch processing**: Multiple samples processed in parallel

Typical time: **~0.8s per dataset** (average)

## Sanity Checks

```bash
cd 03_synthetic_generator_3D
python sanity_checks.py
```

Sanity checks include:

1. **Basic Stats**: Shapes, modes, NaN rates
2. **Learnability**: Models beat baseline
3. **Temporal Characteristics**: Autocorrelation, trends
4. **Mode Comparison**: IID vs Sliding vs Mixed
5. **Label Permutation**: No data leakage
6. **Comparison with Real**: Distributions vs UCR/UEA datasets
7. **Difficulty Spectrum**: Variety of difficulties
8. **Input Type Distribution**: Balance noise/time/state

## Comparison with Real Datasets

Check 6 compares with real datasets from PKL:
- n_samples, t_length: similar distributions
- Autocorrelation: synthetics have less AC(1) than real
- Variance: real have more variability

## Differences with 2D Generator

| Aspect | 2D | 3D |
|--------|----|----|
| Shape | (n, m) | (n, m, t) |
| Inputs | Only noise | Noise + Time + State |
| Dependencies | None | Temporal (memory) |
| Target | One node | One node at a timestep |
| Sampling | One propagation | T propagations + extraction |
| Complexity | O(n × nodes) | O(n × T × nodes) |

## Discriminator Analysis

Analysis of synthetic vs real distinguishability:

```bash
python discriminator_analysis.py
```

Generates:
- Dataset features (34 metrics)
- Random Forest classifier to distinguish
- Feature importance
- Visualizations per dataset
