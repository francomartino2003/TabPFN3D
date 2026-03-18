"""
Hyperparameters for the kernel-DAG dataset generator and data augmentation pipeline.

Key differences from 11_final_generator:
  - Root nodes can be SERIES (GP-sampled with kernel bank) or TABULAR.
  - Internal series nodes use a single Conv1D (no pointwise → temporal split).
  - No time-index channels — series ancestry is guaranteed by DAG structure.
  - Kernel bank: Linear, RBF, Periodic (parameters sampled per root).
  - Conv1D: kernel length from {1,7,9,11}, centered N(0,1) weights, dilation
    sampled exponentially, padding causal or centered (Bernoulli 0.5).

PFN filter (matches 01_real_data/download.py):
  m*T <= 2000, labels <= 10.  No individual m or T limit.
  T is sampled log-uniform, constrained so that m * T <= 2000.
"""

from dataclasses import dataclass, field
from typing import Tuple


# ── DAG structure ──────────────────────────────────────────────────────────────

@dataclass
class DAGHyperparameters:
    """Hyperparameters that control the DAG topology."""

    # Root latent dimension d  (log-uniform int)
    # Upper bound raised to 8 to support up to ~5 series roots on average.
    root_d_range: Tuple[int, int] = (1, 6)

    # Number of hidden layers (log-uniform int — favors smaller)
    # Raised to 8 so large-m multivariate DAGs can have enough series nodes.
    n_layers_range: Tuple[int, int] = (1, 8)

    # Number of nodes per hidden layer (log-uniform int per layer — favors smaller)
    # Raised to 18: max theoretical series nodes ≈ 8×0.6 + 8×18×0.85 ≈ 127,
    # matching the n_features_range upper bound (2, 125).
    # Log-uniform sampling keeps most DAGs small (typical < 15 nodes/layer).
    nodes_per_layer_range: Tuple[int, int] = (2, 18)

    # Probability that a ROOT node is "series" (vs tabular/discrete)
    root_series_prob: float = 0.5

    # Probability that a HIDDEN node is "series" (vs tabular/discrete)
    series_node_prob_range: Tuple[float, float] = (0.3, 1.0)

    # Among non-series nodes, probability that a node is "discrete" (vs tabular)
    discrete_node_prob_range: Tuple[float, float] = (0.1, 1.0)

    # Probability of DROPPING a connection (uniform float, once per DAG)
    connection_drop_prob_range: Tuple[float, float] = (0.4, 0.8)

    # Minimum parents per non-root node after dropping
    min_parents: int = 1


# ── Role assignment ────────────────────────────────────────────────────────────

@dataclass
class RoleHyperparameters:
    """Hyperparameters that control feature / target assignment."""

    # Probability of univariate (1 feature channel); 86% of real PFN datasets are univariate
    univariate_prob: float = 0.80

    # If not univariate: log-uniform range for n_features (channels)
    # Upper bound matches real data range; further constrained at sampling time
    # by the PFN filter (m_eff * T_eff <= 4000).
    n_features_range: Tuple[int, int] = (1, 125)


# ── GP kernel bank (for series roots) ─────────────────────────────────────────

@dataclass
class GPKernelHyperparameters:
    """Hyperparameters for Gaussian Process kernel sampling at series roots."""

    # Number of base kernels J to combine (log-uniform int, favors 1)
    n_kernels_range: Tuple[int, int] = (1, 5)

    # Which kernels can be sampled (uniform choice)
    kernel_choices: Tuple[str, ...] = ('linear', 'rbf', 'periodic')

    # Linear kernel: k(t,t') = sigma^2 * (t - c) * (t' - c)
    linear_sigma_range: Tuple[float, float] = (0.1, 1)
    linear_c_range: Tuple[float, float] = (-0.5, 0.5)

    # RBF kernel: k(t,t') = sigma^2 * exp(-(t-t')^2 / (2 * ell^2))
    rbf_sigma_range: Tuple[float, float] = (0.1, 1)
    rbf_lengthscale_range: Tuple[float, float] = (0.01, 0.5)

    # Periodic kernel: k(t,t') = sigma^2 * exp(-2*sin^2(pi*|t-t'|/p) / ell^2)
    periodic_sigma_range: Tuple[float, float] = (0.1, 1)
    periodic_period_range: Tuple[float, float] = (0.05, 1.0)
    periodic_lengthscale_range: Tuple[float, float] = (0.1, 2.0)


# ── Propagation (operations per node) ─────────────────────────────────────────

@dataclass
class PropagationHyperparameters:
    """Hyperparameters for how node values are computed."""

    # Tabular root initialisation distribution
    root_init_choices: Tuple[str, ...] = ('normal',)

    # N(0, std):  std sampled uniform in range (per tabular root)
    root_normal_std_range: Tuple[float, float] = (0.5, 1)

    # U(-a, a):  a sampled uniform in range (per tabular root)
    root_uniform_a_range: Tuple[float, float] = (0.5, 1.5)

    # GP kernels for series roots
    gp_kernels: GPKernelHyperparameters = field(default_factory=GPKernelHyperparameters)

    # ── Internal series nodes: single Conv1D ──

    # Kernel length choices (uniform discrete)
    conv_kernel_length_choices: Tuple[int, ...] = (1, 7, 9, 11)

    # Max exponent A for dilation: d = floor(2^x), x ~ U(0, A)
    # A = log2((T-1)/(K-1)) computed at runtime; this is just a hard cap
    conv_max_dilation_exp: float = 10.0

    # Padding: 'left' (causal) or 'center' — sampled Bernoulli(0.5) per dataset
    conv_padding_causal_prob: float = 0.5

    # Per-node noise probability: sampled log-uniform (favors small/no noise)
    # Each internal node activates noise only with this probability.
    node_noise_prob_range: Tuple[float, float] = (0.5, 1.0)

    # Per-node output noise: std sampled log-uniform (favors small values)
    noise_std_range: Tuple[float, float] = (1e-5, 1.0)

    # Discrete nodes: number of classes k per node (log-uniform int)
    discrete_classes_range: Tuple[int, int] = (2, 10)

    # Activation bank (no plain relu — smooth_relu avoids dying neurons)
    activation_choices: Tuple[str, ...] = (
        'identity',     # f(x) = x
        'log',          # f(x) = sign(x) * log(1 + |x|)
        'sigmoid',      # f(x) = 1/(1+exp(-x))
        'abs',          # f(x) = |x|
        'sin',          # f(x) = sin(x)
        'tanh',         # f(x) = tanh(x)
        'square',       # f(x) = x^2
        'power',        # f(x) = sign(x) * |x|^0.5
        'softplus',     # f(x) = log(1 + exp(x))
        'smooth_relu',  # f(x) = x * sigmoid(x)  (SiLU / Swish)
        'modulo',       # f(x) = x mod 1
    )

    # Relative weight of 'identity' when sampling activations for series
    # (temporal) nodes.  Other activations each get weight 1.
    # e.g. 5 → identity is ~5x more likely than any other single activation.
    series_identity_weight: float = 5.0


# ── Dataset-level parameters ──────────────────────────────────────────────────

@dataclass
class DatasetHyperparameters:
    """Hyperparameters sampled once per dataset.

    Sampling order in DatasetGenerator._sample_dataset_params():
      1. m (channels) — from DAG structure; log-uniform when not univariate
      2. T — log-uniform in [t_range[0], t_max(m)], biased toward small values
             t_max(m) = min(t_range[1], max_feat_times_t // m)
      3. n — log-uniform in [min_samples, max_samples], biased toward small values

    PFN filter: m * T <= max_feat_times_t (2000), labels <= 10.
    """

    # Number of observations — log-uniform, strongly biased toward small
    # Real data: [27, 9414], median 482.  Generator biases toward small.
    min_samples: int = 30
    max_samples: int = 1400

    # Time-series length T — log-uniform, biased toward small
    # Real data: [8, 2000], median 315.  Capped at 2100 to keep
    # GP Cholesky (O(T^3)) tractable.
    t_range: Tuple[int, int] = (6, 2100)

    # PFN filter: m * T <= this (2000).
    # Used to derive t_max dynamically given m.
    max_feat_times_t: int = 2000

    # Hard cap on m (channels) regardless of PFN constraint.
    # Real data: max m=144 (UEA); 80% are univariate (m=1).
    max_m: int = 125

    # Train / test split ratio
    train_ratio: float = 0.8

    # Minimum total observations per class (classes below this are dropped)
    # Real data: min min-class = 8; set to match.
    min_samples_per_class: int = 8

    # Probability of applying Kumaraswamy warping to one random feature/series
    warping_prob: float = 0.1

    # Kumaraswamy CDF params F(x)=1-(1-x^a)^b (sampled per application)
    kumaraswamy_a_range: Tuple[float, float] = (1.5, 5.0)
    kumaraswamy_b_range: Tuple[float, float] = (1.5, 5.0)

    # ── Variable-length series ──
    variable_length_prob: float = 0.05
    variable_length_u_std_frac_range: Tuple[float, float] = (0.01, 0.3)

    # ── Convolution padding mode ──
    no_pre_padding_prob: float = 0.1
    edge_padding_prob: float = 0.1

    # ── Predictive truncation ──
    predictive_prob: float = 0.1
    predictive_u_frac_range: Tuple[float, float] = (0.05, 0.5)


# ── Augmentation pipeline ──────────────────────────────────────────────────────

@dataclass
class AugmentationHyperparameters:
    """Hyperparameters for the data augmentation pipeline (augmentation.py).

    Applied per synthetic dataset to create augmented copies for training.
    Pipeline: feature permutation → class permutation → per-feature value
    transform → temporal granularity → missing values → group-size padding.
    """

    # Per-feature value transforms and their sampling weights.
    # 'none' has higher weight so ~1/3 of features remain untouched.
    transform_choices: Tuple[str, ...] = ('none', 'log', 'exp', 'squash', 'kdi', 'kuma')
    transform_none_weight: float = 5.0   # weight for 'none'; others get weight 1.0

    # Dataset-level temporal granularity: [identity, pooling, interpolation]
    # Unnormalised — normalised to sum=1 at import time in augmentation.py.
    temporal_probs: Tuple[float, float, float] = (0.5, 0.25, 0.25)

    # Pooling type for temporal augmentation: [mean, max, min, global (min+max+mean)]
    # global is the most represented within the 25% pooling budget.
    pool_type_probs: Tuple[float, float, float, float] = (0.2, 0.2, 0.2, 0.4)

    # Probability that a given feature channel receives intentional missing values
    missing_feature_prob: float = 0.01

    # Limits enforced inside _temporal_granularity (augmentation, not generation)
    aug_max_T: int = 2000
    aug_max_m: int = 125
    aug_max_m_times_T: int = 2000


# ── Collected defaults ─────────────────────────────────────────────────────────

@dataclass
class GeneratorHyperparameters:
    """Top-level container for all generator hyperparameters."""

    dag: DAGHyperparameters = field(default_factory=DAGHyperparameters)
    roles: RoleHyperparameters = field(default_factory=RoleHyperparameters)
    propagation: PropagationHyperparameters = field(default_factory=PropagationHyperparameters)
    dataset: DatasetHyperparameters = field(default_factory=DatasetHyperparameters)
    augmentation: AugmentationHyperparameters = field(default_factory=AugmentationHyperparameters)
