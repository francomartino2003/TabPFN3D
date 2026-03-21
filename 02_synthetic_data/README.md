# 02 — Synthetic data generation

This module generates the synthetic time series classification datasets used to
fine-tune the adapted TabPFN model.  The approach extends the **DAG-based
synthetic prior** introduced in TabPFN (Hollmann et al., 2025) to natively
support temporal structure.

---

## Design philosophy

TabPFN's core insight is that a model trained on the right *distribution over
datasets* will generalise to unseen real tasks without any task-specific
training.  The original paper achieves this by sampling random datasets from a
Bayesian-inspired generative process: a random directed acyclic graph (DAG)
where every node computes a random function of its parents, ultimately
producing feature–label pairs that span a wide variety of tabular patterns
(Hollmann et al., 2025, Methods §"Synthetic data generation").

We extend this prior to **multivariate time series classification** by
introducing a new node type — the *series node* — that carries a full
temporal signal of length *T* instead of a single scalar.  Two design
decisions anchor how temporal structure enters the graph:

1. **GP-sampled roots (KernelSynth-style)**: each root series node is drawn
   from a Gaussian Process whose covariance is a random combination of base
   kernels, following the KernelSynth procedure of Ansari et al. (2024).
2. **Conv1D propagation (ROCKET-style)**: each internal series node applies a
   single 1-D convolution with random kernel length, dilation, and padding
   to its parent channels, inspired by the random convolutional kernel bank of
   Dempster et al. (2020).

The result is a family of synthetic classification tasks whose temporal
patterns — trend, periodicity, multi-scale local structure — span a prior that
is broad enough to cover the real UCR/UEA benchmark datasets.

---

## Module overview

| File | Role |
|------|------|
| `dag_structure.py` | Sample and build random DAG topologies |
| `generator.py` | Propagate observations through a DAG → `(X_train, X_test, y)` |
| `hyperparameters.py` | All sampling ranges, collected in frozen dataclasses |
| `augmentation.py` | Augmentation pipeline applied to training batches |
| `visualize_transforms.py` | Diagnostic visualisations |

---

## DAG topology (`dag_structure.py`)

Each synthetic dataset is defined by one random DAG.  The builder (`build_dag`)
proceeds in seven steps:

1. **Global parameters** — sample `root_d` (log-uniform ∈ [1, 6]), `n_layers`
   (log-uniform ∈ [1, 8]), connection drop probability ∈ [0.4, 0.8],
   and per-node probabilities for series / discrete types.
2. **Root layer** — create `root_d` nodes; each is independently assigned to
   type *series* (p = 0.5), *tabular*, or *discrete*.
3. **Hidden layers** — each layer has a log-uniform number of nodes (∈ [2, 18]);
   node types drawn per the sampled series/discrete probabilities.
4. **Guarantees** — at least one series root, one hidden series node, and one
   discrete node are enforced.  This prevents degenerate DAGs where temporal
   structure cannot propagate.
5. **Edges** — fully-connected between adjacent layers; each edge is
   independently dropped with the sampled probability (minimum 1 parent
   retained per node).
6. **Series-parent guarantee** — every series node must have at least one
   series parent; if the drop step broke this, the nearest upstream series
   node is re-connected.
7. **Role assignment** — a discrete node is chosen as the *target* (class
   label); series nodes are chosen as *features* (80 % univariate, otherwise
   log-uniform ∈ [1, 125]).  Layers beyond the deepest role node are pruned.

The three node types:

| Type | Role in the graph |
|------|------------------|
| **Series** | Carries a temporal signal of shape `(n, T)` |
| **Tabular** | Carries a scalar per observation |
| **Discrete** | Carries a class index; the target must be discrete |

---

## Generation pipeline (`generator.py`)

A `DatasetGenerator` is seeded once and produces one dataset of shape
`(n, m, T)`.  All structural choices (DAG, kernel parameters, conv weights,
activations, …) are fixed at construction time; only the random root values
vary per observation.

### Dataset-level parameters

Sampled once per dataset, before any observations are propagated:

| Parameter | Distribution | Range | Notes |
|-----------|-------------|-------|-------|
| `m` (channels) | DAG-determined; log-uniform if multivariate | [1, 125] | 80 % univariate |
| `T` (length) | Log-uniform | [6, 2100] | Capped so `m × T ≤ 2000` (PFN filter) |
| `n` (observations) | Log-uniform | [30, 1400] | Biased toward small — matches real data |

The PFN filter constraint `m × T ≤ 2000` is enforced at sampling time by
deriving `T_max(m) = min(2100, ⌊2000 / m⌋)` before drawing `T`.

### Root nodes

**Tabular roots** sample `n` i.i.d. scalars from `N(0, σ²)` or `U(-a, a)`.

**Discrete roots** sample a class index uniformly and map it to a fixed scalar
prototype value.

**Series roots** sample `n` functions from a Gaussian Process,
`f ∼ GP(0, K)`, where `K` is a composite kernel built by the
*KernelSynth* procedure (Ansari et al., 2024):

1. Draw `J ∼ log-uniform([1, 5])` base kernels from the bank below.
2. While `J > 1`: pick two kernels at random and combine them with `+` or `×`
   (each with probability 0.5).
3. The resulting `(T × T)` covariance matrix defines the shared prior over all
   series in this dataset.

**Kernel bank:**

| Kernel | Formula | Parameters |
|--------|---------|-----------|
| Linear | `k(t,t') = σ² (t−c)(t'−c)` | `σ ∈ [0.1, 1]`, `c ∈ [−0.5, 0.5]` |
| RBF | `k(t,t') = σ² exp(−(t−t')²/2ℓ²)` | `σ ∈ [0.1, 1]`, `ℓ ∈ [0.01, 0.5]` |
| Periodic | `k(t,t') = σ² exp(−2 sin²(π|t−t'|/p)/ℓ²)` | `σ ∈ [0.1, 1]`, `p ∈ [0.05, 1]`, `ℓ ∈ [0.1, 2]` |

The time grid is normalised to `[0, 1]` so that parameter ranges are
scale-invariant.  Samples are drawn via Cholesky decomposition of `K`.

### Internal series nodes (Conv1D propagation)

Each internal series node receives its parent channels
`(n, c_in, T)` and computes one output channel via a single causal or
centered `Conv1D`, inspired by the random convolutional kernels of ROCKET
(Dempster et al., 2020):

| Parameter | Distribution |
|-----------|-------------|
| Kernel length `K` | Uniform from `{1, 7, 9, 11}` |
| Dilation `D` | `D = ⌊2^x⌋`, `x ∼ U(0, log₂((T−1)/(K−1)))` — exponential spacing |
| Weights | `W ∼ N(0, 1)`, then zero-centered per filter (reduces DC bias) |
| Bias | `b ∼ U(−1, 1)` |
| Padding | Causal (left) or centered — sampled Bernoulli(0.5) **per dataset** |
| Activation | Sampled from a bank of 11 functions; *identity* is 5× upweighted for series nodes |
| Noise | Applied with log-uniform probability; `σ ∼ log-uniform([1e-5, 1])` |

The exponential dilation distribution gives the model receptive fields that
span multiple scales (local to near-global), matching the multi-scale
philosophy of ROCKET and its variants.

### Internal tabular and discrete nodes

Tabular nodes compute a weighted sum of their flattened inputs (series parents
are concatenated along the time axis) followed by a random activation and
optional noise.  Discrete nodes classify the same flattened input vector using
nearest-prototype assignment, mapping the result to a scalar.

### Dataset extraction

After propagation, the raw array `(n, m, T)` undergoes:

- **Predictive truncation** (p = 0.10): the observed window is shortened by a
  random fraction `[0.05, 0.50]` of `T`, so that some label information depends
  on unobserved future timesteps — creating harder, predictive tasks.
- **Kumaraswamy warping** (p = 0.10): a non-linear monotone warp is applied to
  one random channel to create asymmetric amplitude distributions.
- **Class filtering**: classes with fewer than 8 observations are dropped;
  datasets with fewer than 2 remaining classes are discarded.
- **Train / test split**: 80 % / 20 %, stratified per class.

### Variable-length series (p = 0.05)

With probability 0.05, per-observation lengths `T_i < T` are drawn from a
half-normal distribution.  Series are zero-padded (or edge-replicated) to
length `T` after the root GP sample, so all downstream Conv1D operations
see consistent arrays.  This produces NaN-free variable-length data at the
dataset level.

---

## Augmentation pipeline (`augmentation.py`)

Training batches contain **1/4 original datasets** and **3/4 augmented
copies**, generated on-the-fly.  Each augmented copy is independent.

The pipeline operates on train + test concatenated (`n_all` observations),
then splits back:

### Step 1 — Feature channel permutation
Random permutation of the `m` channels.  Teaches the model that feature
ordering carries no intrinsic meaning.

### Step 2 — Class label permutation
Random permutation of the `n_classes` labels.  Prevents the model from
learning any bias about class identity.

### Step 3 — Per-feature value transform
One transform is sampled independently per channel:

| Transform | Description | Frequency |
|-----------|-------------|-----------|
| `none` | Identity | ~50 % (5× upweighted) |
| `log` | `sign(x) · log(1 + |x|)` | ~10 % |
| `exp` | `sign(x) · (exp(|x|) − 1)`, clipped at `|x|=10` | ~10 % |
| `squash` | Robust scaling (median/IQR) + soft clip at `B=3` | ~10 % |
| `kdi` | Kernel Density Integral; `α ∼ LogNormal(0, 0.8²)` | ~10 % |
| `kuma` | Kumaraswamy CDF warp; `a, b ∼ LogNormal(0, 0.7²)` | ~10 % |

Statistics (median, IQR, KDE) are computed on the **full dataset** (train + test
pooled) to avoid data leakage and to match the behaviour seen at inference time.

### Step 4 — Temporal granularity transform
Applied at the **dataset level** (all channels at once):

| Mode | Probability | Description |
|------|------------|-------------|
| Identity | 50 % | No change |
| Pooling | 25 % | Sliding-window `K ∈ {2,3,4,5}`, stride `S ∈ [1,K]`; type = mean / max / min / global (min+max+mean triple-stacks channels) |
| Step-repeat | 25 % | Each timestep repeated `n` times (`n ∈ [2, 8]`); makes changes appear slower |

All modes enforce `m_new × T_new ≤ 2000` (PFN filter).

### Step 4b — Second channel permutation
Applied after the temporal transform so that when the *global* pool mode is
active (which produces `[min, max, mean]` channels in fixed order), the model
never sees a fixed ordering.

### Step 5 — Intentional missing values
Each channel independently receives NaN at a log-uniform rate in `[1%, 30%]`
with probability `p = 0.01` per channel.  This exposes the model to missing
data patterns it will encounter in real datasets.

### Step 6 — Group-size padding
The final `T` is right-zero-padded to the next multiple of `group_size = 8`,
matching the overlap tokenisation used by the model.

---

## Hyperparameters (`hyperparameters.py`)

All sampling ranges are collected in frozen `@dataclass` objects so that the
entire prior can be inspected and modified in one place without touching any
logic:

| Dataclass | Contents |
|-----------|---------|
| `DAGHyperparameters` | Root dimension, layer counts, node-type probabilities, edge drop probability |
| `RoleHyperparameters` | Univariate probability, feature count range |
| `GPKernelHyperparameters` | Kernel bank choices and per-kernel parameter ranges |
| `PropagationHyperparameters` | Conv kernel lengths, dilation cap, activation bank, noise ranges |
| `DatasetHyperparameters` | `n`, `T`, `m` ranges; PFN filter; warping; variable-length; predictive truncation |
| `AugmentationHyperparameters` | Transform weights, temporal mode probabilities, missing value rate |
| `GeneratorHyperparameters` | Top-level container combining all of the above |

---

## Usage

```bash
# Generate and visualise 5 datasets (saves PNGs to visualizations/)
python 02_synthetic_data/generator.py --n 5 --seed 0

# Visualise all augmentation transforms
python 02_synthetic_data/augmentation.py

# Visualise DAG structures only
python 02_synthetic_data/dag_structure.py --n 5 --seed 0
```

Datasets are consumed during training by `03_finetuning/worker_generator.py`,
which writes `.npz` batch files, and `03_finetuning/data_utils.py`, which
applies the augmentation pipeline on-the-fly.

---

## References

- **TabPFN v2.5 (synthetic prior):** Hollmann, N., Müller, S., Purucker, L.,
  Krishnakumar, A., Körfer, M., Hoo, B., Schirrmeister, R. T., & Hutter, F.
  *Accurate predictions on small data with a tabular foundation model.*
  Nature 637, 319–326 (2025). https://doi.org/10.1038/s41586-024-08328-6

- **KernelSynth (GP-based synthetic time series):** Ansari, A. F., Stella, L.,
  Turkmen, C., et al. *Chronos: Learning the language of time series.*
  arXiv:2403.07815 (2024). https://arxiv.org/abs/2403.07815

- **ROCKET (random convolutional kernels):** Dempster, A., Petitjean, F., &
  Webb, G. I. *ROCKET: Exceptionally fast and accurate time series
  classification using random convolutional kernels.* Data Min Knowl Disc 34,
  1454–1495 (2020). https://doi.org/10.1007/s10618-020-00701-z
