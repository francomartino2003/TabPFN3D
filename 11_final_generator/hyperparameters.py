"""
Hyperparameters for the final dataset generator.

Every sampable quantity has:
  - A distribution type (uniform, log_uniform, geometric, choice, …)
  - A range or parameter set

This file is the SINGLE SOURCE OF TRUTH for all defaults.
New hyper-parameters are added here as the generator grows.
"""

from dataclasses import dataclass, field
from typing import Tuple


# ── DAG structure ──────────────────────────────────────────────────────────────

@dataclass
class DAGHyperparameters:
    """Hyperparameters that control the DAG topology."""

    # Root latent dimension d  (log-uniform int)
    root_d_range: Tuple[int, int] = (1, 16)

    # Number of hidden layers (log-uniform int — favors smaller)
    n_layers_range: Tuple[int, int] = (2, 8)

    # Number of nodes per hidden layer (log-uniform int per layer — favors smaller)
    nodes_per_layer_range: Tuple[int, int] = (2, 8)

    # Probability that a node is "series" (vs tabular/discrete)
    # Sampled uniform float once per DAG
    series_node_prob_range: Tuple[float, float] = (0.3, 1)

    # Among non-series nodes, probability that a node is "discrete" (vs continuous tabular)
    # Sampled uniform float once per DAG
    discrete_node_prob_range: Tuple[float, float] = (0.1, 1)

    # Probability of DROPPING a connection (uniform float, once per DAG)
    connection_drop_prob_range: Tuple[float, float] = (0.0, 0.8)

    # Minimum parents per non-root node after dropping
    min_parents: int = 1


# ── Role assignment ────────────────────────────────────────────────────────────

@dataclass
class RoleHyperparameters:
    """Hyperparameters that control feature / target assignment."""

    # Probability of univariate (1 feature)
    univariate_prob: float = 0.75

    # If not univariate: log-uniform range for n_features (can still be 1)
    n_features_range: Tuple[int, int] = (1, 10)


# ── Propagation (operations per node) ─────────────────────────────────────────

@dataclass
class PropagationHyperparameters:
    """Hyperparameters for how node values are computed."""

    # Root initialisation distribution
    root_init_choices: Tuple[str, ...] = ('normal', 'uniform')

    # N(0, std):  std sampled uniform in range
    root_normal_std_range: Tuple[float, float] = (0.5, 1.5)

    # U(-a, a):  a sampled uniform in range
    root_uniform_a_range: Tuple[float, float] = (0.5, 1.5)

    # Conv1 (pointwise, K=1): output channels (log-uniform int per series node)
    series_hidden_channels_range: Tuple[int, int] = (1, 32)

    # Conv2 (temporal): kernel size (log-uniform int per series node)
    kernel_size_range: Tuple[int, int] = (1, 16)

    # Conv2 (temporal): dilation factor (log-uniform int per series node)
    dilation_range: Tuple[int, int] = (1, 16)

    # Extra time-index channels for series nodes (log-uniform int per node — favors 0/1)
    # All series: add this many extra t-index channels.
    # Series with no series parents: +1 mandatory on top.
    n_extra_time_indices_range: Tuple[int, int] = (0, 4)

    # Per-node output noise: std sampled log-uniform (favors small values)
    noise_std_range: Tuple[float, float] = (1e-4, 1)

    # Discrete nodes: number of classes k per node (log-uniform int — favors fewer)
    discrete_classes_range: Tuple[int, int] = (2, 10)

    # Activation bank (same as folder 10)
    activation_choices: Tuple[str, ...] = (
        'identity',   # f(x) = x
        'log',        # f(x) = sign(x) * log(1 + |x|)
        'sigmoid',    # f(x) = 1/(1+exp(-x))
        'abs',        # f(x) = |x|
        'sin',        # f(x) = sin(x)
        'tanh',       # f(x) = tanh(x)
        'square',     # f(x) = x^2
        'power',      # f(x) = sign(x) * |x|^0.5
        'softplus',   # f(x) = log(1 + exp(x))
        'modulo',     # f(x) = x mod 1
    )


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

    # Padding policy for dilated conv in series nodes — sampled once per dataset,
    # all nodes share the same policy.
    #   'left'   → causal (zero-pad left, output[t] sees only past)
    #   'right'  → anti-causal (zero-pad right, output[t] sees only future)
    #   'center' → symmetric (zero-pad both sides, output[t] sees past + future)
    padding_policy_choices: Tuple[str, ...] = ('left', 'center', 'right')

    # Probability of applying Kumaraswamy warping to one random feature/series
    warping_prob: float = 0.1

    # Kumaraswamy CDF params F(x)=1-(1-x^a)^b (sampled per application)
    kumaraswamy_a_range: Tuple[float, float] = (1.5, 5.0)
    kumaraswamy_b_range: Tuple[float, float] = (1.5, 5.0)


# ── Collected defaults ─────────────────────────────────────────────────────────

@dataclass
class GeneratorHyperparameters:
    """Top-level container for all generator hyperparameters."""

    dag: DAGHyperparameters = field(default_factory=DAGHyperparameters)
    roles: RoleHyperparameters = field(default_factory=RoleHyperparameters)
    propagation: PropagationHyperparameters = field(default_factory=PropagationHyperparameters)
    dataset: DatasetHyperparameters = field(default_factory=DatasetHyperparameters)
