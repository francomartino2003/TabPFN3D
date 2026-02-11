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
    root_d_range: Tuple[int, int] = (2, 64)

    # Number of hidden layers (log-uniform int — favors smaller)
    n_layers_range: Tuple[int, int] = (1, 3)

    # Number of nodes per hidden layer (log-uniform int per layer — favors smaller)
    nodes_per_layer_range: Tuple[int, int] = (2, 4)

    # Probability that a node is "series" (vs tabular/discrete)
    # Sampled uniform float once per DAG
    series_node_prob_range: Tuple[float, float] = (0.3, 0.9)

    # Among non-series nodes, probability that a node is "discrete" (vs continuous tabular)
    # Sampled uniform float once per DAG
    discrete_node_prob_range: Tuple[float, float] = (0.1, 0.5)

    # Probability of DROPPING a connection (uniform float, once per DAG)
    connection_drop_prob_range: Tuple[float, float] = (0.0, 0.5)

    # Minimum parents per non-root node after dropping
    min_parents: int = 1


# ── Role assignment ────────────────────────────────────────────────────────────

@dataclass
class RoleHyperparameters:
    """Hyperparameters that control feature / target assignment."""

    # Number of feature nodes: geometric(p), capped at n_series_nodes and max_features
    n_features_geometric_p: float = 0.7

    # Maximum features
    max_features: int = 12


# ── Propagation (operations per node) ─────────────────────────────────────────

@dataclass
class PropagationHyperparameters:
    """Hyperparameters for how node values are computed."""

    # Root initialisation distribution
    root_init_choices: Tuple[str, ...] = ('normal', 'uniform')

    # N(0, std):  std sampled uniform in range
    root_normal_std_range: Tuple[float, float] = (1, 1)

    # U(-a, a):  a sampled uniform in range
    root_uniform_a_range: Tuple[float, float] = (1, 1)

    # Causal convolution kernel size per series node (uniform int)
    kernel_size_range: Tuple[int, int] = (3, 11)

    # Optional smoothing conv: a second causal conv with positive L1-normalised
    # kernel applied after the main conv and before the activation.
    # Probability (per-DAG) that each series node gets a smoothing conv.
    smoothing_conv_prob_range: Tuple[float, float] = (0.0, 0.8)
    # Kernel size for the smoothing conv (uniform int per node)
    smoothing_kernel_size_range: Tuple[int, int] = (3, 11)

    # Series nodes without series parents: minimum input dim before PE
    min_series_input_dim: int = 8

    # Discrete nodes: number of classes k per node (uniform int)
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
        'step',       # f(x) = 0 if x<0 else 1
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
    t_range: Tuple[int, int] = (20, 500)

    # Constraint: n_features * T <= this
    max_feat_times_t: int = 500

    # Train / test split ratio
    train_ratio: float = 0.8

    # Minimum total observations per class (classes below this are dropped)
    min_samples_per_class: int = 6


# ── Collected defaults ─────────────────────────────────────────────────────────

@dataclass
class GeneratorHyperparameters:
    """Top-level container for all generator hyperparameters."""

    dag: DAGHyperparameters = field(default_factory=DAGHyperparameters)
    roles: RoleHyperparameters = field(default_factory=RoleHyperparameters)
    propagation: PropagationHyperparameters = field(default_factory=PropagationHyperparameters)
    dataset: DatasetHyperparameters = field(default_factory=DatasetHyperparameters)
