# Synthetic Dataset Generator for TabPFN

Implementation of the tabular synthetic data generator based on the TabPFN paper. Generates diverse datasets to train models that learn to solve tabular problems in general.

## Architecture

```
Root Nodes (Noise) → DAG Transformations → Feature Selection → Post-processing → (X, y)
     ↓                      ↓                     ↓                  ↓
  Normal/Uniform    NN, Tree, Discretize    Relevant/Irrelevant   Warp, Missing
```

## Quick Start

```python
from generator import SyntheticDatasetGenerator

# Generate a dataset
gen = SyntheticDatasetGenerator(seed=42)
dataset = gen.generate()

X, y = dataset.X, dataset.y
print(f"Shape: {X.shape}, Classes: {dataset.n_classes}")
```

## Modules

| File | Function |
|------|----------|
| `config.py` | `PriorConfig` (distributions) and `DatasetConfig` (instance) |
| `dag_builder.py` | DAG construction with topological order and controllable density |
| `transformations.py` | Transformations: NN (with 12 activations), Tree, Discretization |
| `row_generator.py` | Value propagation through the graph |
| `feature_selector.py` | Feature and target selection |
| `post_processing.py` | Warping (Kumaraswamy), quantization, missing values |
| `generator.py` | Main class |
| `sanity_checks.py` | Complete validation of generated datasets |
| `tests.py` | Unit tests |

## Key Parameters

### Size
- **n_samples**: Uniform in [50, 2048]
- **n_features**: Beta(0.95, 8.0) scaled to [1, 160]
- **max_cells**: 75,000 (n_samples × n_features)

### Graph Structure
- **n_nodes**: Log-uniform in [50, 600] (latent nodes)
- **density**: Uniform in [0.01, 0.8] - controls how many additional edges are added
- **n_roots_range**: (3, 15) - number of root nodes (inputs)

### Transformations
Each non-root node has exactly **one** transformation that takes all its parents as input:

| Type | Prob | Description |
|------|------|-------------|
| NN | ~60% | Linear combination + activation + noise |
| Tree | ~25% | Decision tree with subset of parent features |
| Discretization | ~15% | Distance to prototypes → normalized category |

### Activations (12 functions from paper)
```python
['identity', 'log', 'sigmoid', 'abs', 'sin', 'tanh', 
 'rank', 'square', 'power', 'softplus', 'step', 'mod']
```

- **identity**: f(x) = x (linear transformation)
- **log**: f(x) = log(|x| + 1)
- **sigmoid**: f(x) = 1 / (1 + e^(-x))
- **abs**: f(x) = |x|
- **sin**: f(x) = sin(x)
- **tanh**: f(x) = tanh(x)
- **rank**: f(x) = percentile rank
- **square**: f(x) = x²
- **power**: f(x) = |x|^α, α ∈ [0.5, 3]
- **softplus**: f(x) = log(1 + e^x)
- **step**: f(x) = 1 if x > 0, else 0
- **mod**: f(x) = x mod m, m ∈ [0.5, 2]

### Discretization
- Receives vector of parents
- Calculates distance to K prototypes (K ∈ [2, 8])
- Assigns category of closest prototype
- Normalizes: output = category / K (for use in graph)
- Adds Gaussian noise

### Decision Tree
- Selects subset of features from parents (tree_max_features_fraction=0.7)
- Generates tree with depth [2, 5]
- Each node: (feature_idx, threshold, left_val, right_val)

### Noise in Transformations
All transformations add Gaussian noise N(0, σ²) at the end.

### Number of Classes
- Gamma(2.0, 2.0) + offset of 2
- Limited to min(10, n_samples/min_samples_per_class)
- **min_samples_per_class**: 10 (minimum samples per class)

## DAG Construction

The DAG is constructed using **topological order**:

1. Assign random order to all nodes
2. Determine number of roots (3-15)
3. Calculate target edges based on `density`
4. Add edges only from nodes with lower order to higher order (guarantees acyclicity)
5. Ensure connectivity (each non-root has at least 1 parent)

This allows controlling the **density** of the graph:
- density=0: minimum graph (tree)
- density=1: maximally dense DAG

## Disconnected Subgraphs

The generator can create disconnected subgraphs for irrelevant features:
- Probability: 30%
- Each subgraph has minimum 3 nodes
- Subgraphs can have multiple roots
- Main subgraph retains at least 60% of nodes

## Sanity Checks

```bash
python sanity_checks.py
```

Verifies:
- ✅ Accuracy distribution (not trivial nor impossible)
- ✅ Variability of rankings between models
- ✅ Relevant vs irrelevant features
- ✅ Label permutation test (no data leakage)
- ✅ Learning curves (improves with more data)
- ✅ Invariance to permutations
- ✅ Correct discretization (prototypes, categories, entropy)
- ✅ Visualization of generated DAGs

## Feature Selection

The generator excludes from features only the **target node** (what we want to predict).

The target's parents **can be features** - they are valuable information for prediction. There is no data leakage because features are input values and the target is the result of the transformation.

## Custom Configuration

```python
from config import PriorConfig

prior = PriorConfig(
    n_samples_range=(100, 500),
    n_nodes_range=(30, 100),
    density_range=(0.1, 0.5),
    prob_classification=1.0,
    prob_missing_values=0.0,
    min_samples_per_class=20,
    activations=['identity', 'tanh', 'sigmoid', 'relu']
)

gen = SyntheticDatasetGenerator(prior=prior, seed=42)
```

## Tests

```bash
python tests.py
```

Includes tests for:
- DAG construction
- All transformations
- All activations
- Graph density
- Multiple roots
