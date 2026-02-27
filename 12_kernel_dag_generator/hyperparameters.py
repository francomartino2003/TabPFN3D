"""
Hyperparameters for the kernel-DAG dataset generator.

Key differences from 11_final_generator:
  - Root nodes can be SERIES (GP-sampled with kernel bank) or TABULAR.
  - Internal series nodes use a single Conv1D (no pointwise → temporal split).
  - No time-index channels — series ancestry is guaranteed by DAG structure.
  - Kernel bank: Linear, RBF, Periodic (parameters sampled per root).
  - Conv1D: kernel length from {1,7,9,11}, centered N(0,1) weights, dilation
    sampled exponentially, padding causal or centered (Bernoulli 0.5).
"""

from dataclasses import dataclass, field
from typing import Tuple


# ── DAG structure ──────────────────────────────────────────────────────────────

@dataclass
class DAGHyperparameters:
    """Hyperparameters that control the DAG topology."""

    # Root latent dimension d  (log-uniform int)
    root_d_range: Tuple[int, int] = (1, 4)

    # Number of hidden layers (log-uniform int — favors smaller)
    n_layers_range: Tuple[int, int] = (1, 5)

    # Number of nodes per hidden layer (log-uniform int per layer — favors smaller)
    nodes_per_layer_range: Tuple[int, int] = (2, 6)

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

    # Probability of univariate (1 feature)
    univariate_prob: float = 0.75

    # If not univariate: log-uniform range for n_features
    n_features_range: Tuple[int, int] = (1, 10)


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
    """Hyperparameters sampled once per dataset."""

    # Number of observations (uniform int)
    min_samples: int = 20
    max_samples: int = 1000

    # Time-series length T (uniform int, further constrained by feat*T)
    t_range: Tuple[int, int] = (20, 1024)

    # Constraint: n_features * T <= this
    max_feat_times_t: int = 1200

    # Train / test split ratio
    train_ratio: float = 0.8

    # Minimum total observations per class (classes below this are dropped)
    min_samples_per_class: int = 6

    # Probability of applying Kumaraswamy warping to one random feature/series
    warping_prob: float = 0.1

    # Kumaraswamy CDF params F(x)=1-(1-x^a)^b (sampled per application)
    kumaraswamy_a_range: Tuple[float, float] = (1.5, 5.0)
    kumaraswamy_b_range: Tuple[float, float] = (1.5, 5.0)

    # ── Variable-length series ──
    # With this probability, root series nodes are sampled with T_i = T - u_i
    # timesteps and zero-padded to T before propagation.
    # u_std is sampled once per dataset (log-uniform, favors small values).
    # u_i ~ |N(0, u_std)| per observation, clipped to [0, T-1].
    variable_length_prob: float = 0.05
    # u_std = u_std_frac * T;  frac sampled log-uniform from this range
    variable_length_u_std_frac_range: Tuple[float, float] = (0.01, 0.3)

    # ── Convolution padding mode ──
    # With this probability, series internal nodes use POST-padding (valid conv):
    #   apply Conv1D without padding (shorter output), then pad to restore T.
    #   Causal → pad left;  centered → pad symmetrically.
    # Otherwise (default): PRE-padding — pad input before conv, output is always T.
    no_pre_padding_prob: float = 0.1

    # With this probability the padding value is the EDGE of the signal
    # (replicate last/first valid sample) instead of zero.
    # Applies to: root variable-length padding, conv pre/post padding.
    edge_padding_prob: float = 0.1

    # ── Predictive truncation ──
    # With this probability, the last u timesteps are removed from X AFTER
    # propagation, yielding shape (n, m, T-u) instead of (n, m, T).
    # y is computed from the full T propagation, so labels can depend on the
    # unobserved future — simulating event prediction / forecasting tasks.
    # u = round(frac * T), frac sampled log-uniform (favors small truncations).
    predictive_prob: float = 0.15
    # Fraction of T to cut: log-uniform in [lo, hi]
    predictive_u_frac_range: Tuple[float, float] = (0.05, 0.5)


# ── Collected defaults ─────────────────────────────────────────────────────────

@dataclass
class GeneratorHyperparameters:
    """Top-level container for all generator hyperparameters."""

    dag: DAGHyperparameters = field(default_factory=DAGHyperparameters)
    roles: RoleHyperparameters = field(default_factory=RoleHyperparameters)
    propagation: PropagationHyperparameters = field(default_factory=PropagationHyperparameters)
    dataset: DatasetHyperparameters = field(default_factory=DatasetHyperparameters)
