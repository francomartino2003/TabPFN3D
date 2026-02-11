"""
Configuration for 3D Synthetic Dataset Generator v5.

Incorporates ideas from random_nn_generator:
- Memory and time transforms with rich variety
- Per-node noise configuration (log-uniform distributions)
- Per-node discretization (15% probability per node)
- Almost all NN with smooth activations (softplus, tanh, elu)
- Log-uniform sampling for most parameters (favors smaller values)

Key design decisions (v5):
- TIME inputs: linear (always) + sampled transforms (sin, cos, exp, log, etc.)
- MEMORY vector: log-uniform dimension, sampled once per sequence
- Per-node noise with varying distributions
- 15% probability per node for discretization
- No tree transformations
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass
class NodeNoiseConfig:
    """Configuration for per-node noise."""
    has_noise: bool = False
    distribution: str = 'normal'  # 'normal', 'uniform', 'laplace'
    scale: float = 0.0


@dataclass
class NodeDiscretizationConfig:
    """Configuration for per-node discretization."""
    is_discretization: bool = False
    n_classes: int = 2
    prototypes: Optional[np.ndarray] = None  # (n_classes, input_dim)
    class_values: Optional[np.ndarray] = None  # (n_classes,)


@dataclass
class PriorConfig3D_v5:
    """
    Prior distributions for sampling 3D temporal dataset configurations (v5).
    """
    
    # === Size constraints ===
    max_samples: int = 10000
    max_features: int = 15
    max_t_subseq: int = 1000
    max_T_total: int = 5000
    max_classes: int = 10
    max_complexity: int = 10_000_000
    
    # === Sample size ===
    n_samples_range: Tuple[int, int] = (100, 10000)
    
    # === Feature count ===
    prob_univariate: float = 0.70
    n_features_beta_a: float = 1.5
    n_features_beta_b: float = 4.0
    n_features_range: Tuple[int, int] = (2, 15)
    
    # === Temporal parameters ===
    T_total_range: Tuple[int, int] = (30, 2000)
    T_total_log_uniform: bool = True
    t_subseq_range: Tuple[int, int] = (10, 1000)
    t_subseq_log_uniform: bool = True
    
    # === Graph structure ===
    n_nodes_range: Tuple[int, int] = (8, 80)
    n_nodes_log_uniform: bool = True
    density_range: Tuple[float, float] = (0.2, 0.5)
    
    # Disconnected subgraphs for irrelevant features
    prob_disconnected_subgraph: float = 0.2
    n_disconnected_subgraphs_range: Tuple[int, int] = (1, 3)
    
    # === TIME inputs (v5) ===
    # Number of time transforms (log-uniform to favor smaller)
    n_time_transforms_range: Tuple[int, int] = (1, 32)
    n_time_transforms_log_uniform: bool = True
    
    # Extra copies of linear (to give it more weight)
    linear_extra_copies_p: float = 0.3  # Binomial(3, p)
    
    # === MEMORY inputs (v5) ===
    memory_dim_range: Tuple[int, int] = (1, 64)
    memory_dim_log_uniform: bool = True
    memory_init: str = 'uniform'  # 'uniform' or 'normal'
    
    # === Stochastic inputs (random at each timestep) ===
    stochastic_dim_range: Tuple[int, int] = (0, 8)
    
    # === NN Transformations (v5) ===
    # All NN with smooth activations (no dying neurons)
    nn_activations: Tuple[str, ...] = ('softplus', 'tanh', 'elu')
    prob_identity_activation: float = 0.3
    per_layer_activation: bool = True  # Different activation per layer/node
    
    # Weight initialization
    weight_init: str = 'xavier_normal'  # 'xavier_uniform' or 'xavier_normal'
    weight_scale_range: Tuple[float, float] = (0.8, 1.2)
    bias_std_range: Tuple[float, float] = (0.0, 0.1)
    
    # === Per-node discretization (v5) ===
    discretization_node_prob: float = 0.15
    n_categories_range: Tuple[int, int] = (2, 10)
    
    # === Per-node noise (v5) ===
    node_noise_prob_range: Tuple[float, float] = (0.01, 0.5)
    node_noise_prob_log_uniform: bool = True
    node_noise_std_range: Tuple[float, float] = (0.0001, 0.02)
    node_noise_std_log_uniform: bool = True
    noise_distributions: Tuple[str, ...] = ('normal', 'uniform', 'laplace')
    
    # === Sample generation mode ===
    prob_iid_mode: float = 0.55
    prob_sliding_window_mode: float = 0.30
    prob_mixed_mode: float = 0.15
    window_stride_range: Tuple[int, int] = (1, 10)
    
    # === Target configuration ===
    min_samples_per_class: int = 25
    max_target_offset: int = 20
    prob_future: float = 0.50
    offset_alpha: float = 1.5  # For offset distribution (favors small offsets)
    
    # === Distance preference ===
    distance_alpha: float = 1.5  # For feature distance (favors closer features)
    
    # === Post-processing ===
    prob_warping: float = 0.3
    warping_intensity_range: Tuple[float, float] = (0.1, 0.5)
    prob_quantization: float = 0.2
    n_quantization_bins_range: Tuple[int, int] = (5, 20)
    prob_missing_values: float = 0.1
    missing_rate_range: Tuple[float, float] = (0.01, 0.15)
    
    # === Train/test split ===
    train_ratio_range: Tuple[float, float] = (0.5, 0.8)


# Available time transforms with their parameter samplers
TIME_TRANSFORMS = [
    # (name, needs_params, param_sampler)
    ('linear', False, None),
    ('quadratic', False, None),
    ('cubic', False, None),
    ('sqrt', False, None),
    ('sin_k1', False, None),
    ('cos_k1', False, None),
    ('sin_k2', False, None),
    ('cos_k2', False, None),
    ('sin_k3', False, None),
    ('cos_k3', False, None),
    ('sin_k5', False, None),
    ('cos_k5', False, None),
    ('tanh_trend', True, lambda rng: {'beta': np.exp(rng.uniform(np.log(0.5), np.log(3.0)))}),
    ('exp_decay', True, lambda rng: {'gamma': np.exp(rng.uniform(np.log(0.5), np.log(5.0)))}),
    ('exp_growth', True, lambda rng: {'gamma': np.exp(rng.uniform(np.log(0.1), np.log(1.0)))}),
    ('log', False, None),
]


def apply_time_transform(transform: Dict, u: np.ndarray) -> np.ndarray:
    """Apply a time transform to normalized time u in [0, 1]."""
    name = transform['name']
    params = transform.get('params', {}) or {}
    
    if name == 'linear':
        return 2 * u - 1  # Scale to [-1, 1]
    elif name == 'quadratic':
        return 4 * u * (1 - u) - 0.5  # Parabola, scaled
    elif name == 'cubic':
        return 8 * (u - 0.5) ** 3
    elif name == 'sqrt':
        return 2 * np.sqrt(u) - 1
    elif name == 'sin_k1':
        return np.sin(2 * np.pi * u)
    elif name == 'cos_k1':
        return np.cos(2 * np.pi * u)
    elif name == 'sin_k2':
        return np.sin(4 * np.pi * u)
    elif name == 'cos_k2':
        return np.cos(4 * np.pi * u)
    elif name == 'sin_k3':
        return np.sin(6 * np.pi * u)
    elif name == 'cos_k3':
        return np.cos(6 * np.pi * u)
    elif name == 'sin_k5':
        return np.sin(10 * np.pi * u)
    elif name == 'cos_k5':
        return np.cos(10 * np.pi * u)
    elif name == 'tanh_trend':
        beta = params.get('beta', 2.0)
        return np.tanh(beta * (2 * u - 1))
    elif name == 'exp_decay':
        gamma = params.get('gamma', 2.0)
        return 2 * np.exp(-gamma * u) - 1
    elif name == 'exp_growth':
        gamma = params.get('gamma', 0.5)
        val = np.exp(gamma * u)
        return 2 * (val - 1) / (np.exp(gamma) - 1 + 1e-8) - 1  # Normalize to [-1, 1]
    elif name == 'log':
        return np.log1p(u) / np.log(2) * 2 - 1  # Normalize to [-1, 1]
    else:
        return 2 * u - 1  # Fallback to linear


@dataclass
class SampledConfig3D_v5:
    """
    Sampled configuration for a specific 3D dataset instance (v5).
    
    Contains all concrete values sampled from the prior.
    """
    # === Size ===
    n_samples: int
    n_features: int
    T_total: int
    t_subseq: int
    
    # === Graph structure ===
    n_nodes: int
    density: float
    n_disconnected_subgraphs: int
    
    # === TIME inputs ===
    time_transforms: List[Dict]  # List of {'name': str, 'params': dict}
    
    # === MEMORY inputs ===
    memory_dim: int
    memory_init: str  # 'uniform' or 'normal'
    
    # === Stochastic inputs ===
    stochastic_dim: int
    
    # === NN configuration ===
    activations: List[str]  # One per node (or layer)
    weight_init: str
    weight_scale: float
    bias_std: float
    
    # === Per-node configurations ===
    node_noise: List[NodeNoiseConfig]  # One per node
    node_discretization: List[NodeDiscretizationConfig]  # One per node
    
    # === Sample generation mode ===
    sample_mode: str  # 'iid', 'sliding_window', 'mixed'
    window_stride: int
    n_sequences: int
    
    # === Target configuration ===
    n_classes: int
    target_offset: int
    
    # === Distance preference ===
    spatial_distance_alpha: float
    
    # === Post-processing ===
    apply_warping: bool
    warping_intensity: float
    apply_quantization: bool
    n_quantization_bins: int
    apply_missing: bool
    missing_rate: float
    
    # === Train/test ===
    train_ratio: float
    
    # === Random seed ===
    seed: int = 0
    
    @classmethod
    def sample_from_prior(cls, prior: PriorConfig3D_v5, rng: np.random.Generator) -> 'SampledConfig3D_v5':
        """Sample a 3D dataset configuration from the prior (v5)."""
        
        def log_uniform_int(low: int, high: int) -> int:
            if low <= 0:
                low = 1
            return int(np.round(np.exp(rng.uniform(np.log(low), np.log(high)))))
        
        def log_uniform_float(low: float, high: float) -> float:
            if low <= 0:
                low = 1e-10
            return np.exp(rng.uniform(np.log(low), np.log(high)))
        
        # === Sample temporal parameters ===
        if prior.t_subseq_log_uniform:
            t_subseq = log_uniform_int(*prior.t_subseq_range)
        else:
            t_subseq = rng.integers(*prior.t_subseq_range)
        t_subseq = min(t_subseq, prior.max_t_subseq)
        t_subseq = max(t_subseq, 5)
        
        min_T = t_subseq + 10
        if prior.T_total_log_uniform:
            T_total = log_uniform_int(max(min_T, prior.T_total_range[0]), prior.T_total_range[1])
        else:
            T_total = rng.integers(max(min_T, prior.T_total_range[0]), prior.T_total_range[1])
        T_total = min(T_total, prior.max_T_total)
        
        # === Sample feature count ===
        if rng.random() < prior.prob_univariate:
            n_features = 1
        else:
            beta_sample = rng.beta(prior.n_features_beta_a, prior.n_features_beta_b)
            n_features = int(beta_sample * (prior.n_features_range[1] - prior.n_features_range[0]) 
                            + prior.n_features_range[0])
            n_features = max(2, min(n_features, prior.max_features))
        
        # === Enforce flattened feature limit (n_features * t_subseq < 500) ===
        max_t_for_features = 500 // n_features
        if t_subseq > max_t_for_features:
            t_subseq = max(50, max_t_for_features)
            T_total = max(t_subseq + 10, T_total)
        
        # === Sample sample count ===
        n_samples = rng.integers(*prior.n_samples_range)
        n_samples = min(n_samples, prior.max_samples)
        
        # === Sample TIME transforms (v5) ===
        if prior.n_time_transforms_log_uniform:
            n_transforms = log_uniform_int(*prior.n_time_transforms_range)
        else:
            n_transforms = rng.integers(*prior.n_time_transforms_range)
        
        # Always include linear
        time_transforms = [{'name': 'linear', 'params': None}]
        
        # Add extra copies of linear (to give it more weight)
        n_extra_linear = rng.binomial(3, prior.linear_extra_copies_p)
        for _ in range(n_extra_linear):
            time_transforms.append({'name': 'linear', 'params': None})
        
        # Add other transforms
        available = [t for t in TIME_TRANSFORMS if t[0] != 'linear']
        rng.shuffle(available)
        remaining = max(0, n_transforms - len(time_transforms))
        
        for i in range(min(remaining, len(available))):
            name, needs_params, param_sampler = available[i]
            params = param_sampler(rng) if needs_params else None
            time_transforms.append({'name': name, 'params': params})
        
        # === Sample MEMORY dimension (v5) ===
        if prior.memory_dim_log_uniform:
            memory_dim = log_uniform_int(*prior.memory_dim_range)
        else:
            memory_dim = rng.integers(*prior.memory_dim_range)
        
        # === Sample stochastic input dimension ===
        stochastic_dim = rng.integers(*prior.stochastic_dim_range)
        
        # Total roots = len(time_transforms) + memory_dim + stochastic_dim
        n_roots = len(time_transforms) + memory_dim + stochastic_dim
        
        # === Sample graph structure ===
        if prior.n_nodes_log_uniform:
            n_nodes = log_uniform_int(*prior.n_nodes_range)
        else:
            n_nodes = rng.integers(*prior.n_nodes_range)
        n_nodes = max(n_nodes, n_roots + n_features + 2)
        
        # === Enforce complexity limit ===
        for _ in range(10):
            complexity = n_samples * T_total * n_nodes
            if complexity <= prior.max_complexity:
                break
            scale = (prior.max_complexity / complexity) ** (1/3)
            n_samples = max(100, int(n_samples * scale))
            T_total = max(t_subseq + 10, int(T_total * scale))
            n_nodes = max(n_roots + n_features + 2, int(n_nodes * scale))
        
        density = rng.uniform(*prior.density_range)
        
        # Disconnected subgraphs
        n_disconnected = 0
        if n_nodes >= 10 and rng.random() < prior.prob_disconnected_subgraph:
            max_disconnected = min(prior.n_disconnected_subgraphs_range[1], (n_nodes - n_roots - 3) // 3)
            if max_disconnected >= 1:
                n_disconnected = rng.integers(1, max_disconnected + 1)
        
        # === Sample NN configuration ===
        weight_init = prior.weight_init
        weight_scale = rng.uniform(*prior.weight_scale_range)
        bias_std = rng.uniform(*prior.bias_std_range)
        
        # Sample activations (one per node)
        n_internal_nodes = n_nodes - n_roots  # Nodes that have transformations
        if prior.per_layer_activation:
            activations = [rng.choice(list(prior.nn_activations)) for _ in range(n_internal_nodes)]
        else:
            act = rng.choice(list(prior.nn_activations))
            activations = [act] * n_internal_nodes
        
        # Apply identity activation probability
        for i in range(len(activations)):
            if rng.random() < prior.prob_identity_activation:
                activations[i] = 'identity'
        
        # === Sample per-node noise (v5) ===
        if prior.node_noise_prob_log_uniform:
            node_noise_prob = log_uniform_float(*prior.node_noise_prob_range)
        else:
            node_noise_prob = rng.uniform(*prior.node_noise_prob_range)
        
        node_noise = []
        for _ in range(n_internal_nodes):
            has_noise = rng.random() < node_noise_prob
            if has_noise:
                dist = rng.choice(list(prior.noise_distributions))
                if prior.node_noise_std_log_uniform:
                    scale = log_uniform_float(*prior.node_noise_std_range)
                else:
                    scale = rng.uniform(*prior.node_noise_std_range)
                node_noise.append(NodeNoiseConfig(has_noise=True, distribution=dist, scale=scale))
            else:
                node_noise.append(NodeNoiseConfig(has_noise=False))
        
        # === Sample per-node discretization (v5) ===
        node_discretization = []
        for i in range(n_internal_nodes):
            is_disc = rng.random() < prior.discretization_node_prob
            if is_disc:
                n_classes = rng.integers(*prior.n_categories_range)
                # Prototypes will be generated during propagation when we know input dims
                node_discretization.append(NodeDiscretizationConfig(
                    is_discretization=True,
                    n_classes=n_classes,
                    prototypes=None,
                    class_values=None
                ))
            else:
                node_discretization.append(NodeDiscretizationConfig(is_discretization=False))
        
        # === Sample generation mode ===
        mode_probs = [prior.prob_iid_mode, prior.prob_sliding_window_mode, prior.prob_mixed_mode]
        mode_probs = np.array(mode_probs) / sum(mode_probs)
        sample_mode = rng.choice(['iid', 'sliding_window', 'mixed'], p=mode_probs)
        
        window_stride = rng.integers(*prior.window_stride_range) if sample_mode != 'iid' else 1
        n_sequences = rng.integers(2, 10) if sample_mode == 'mixed' else 1
        
        # === Target configuration ===
        max_classes_for_samples = max(2, n_samples // prior.min_samples_per_class)
        max_classes = min(prior.max_classes, max_classes_for_samples)
        n_classes = rng.integers(2, max(3, max_classes + 1))
        
        # Sample target offset (favors smaller offsets)
        max_offset = min(prior.max_target_offset, 15)
        possible_offsets = list(range(-min(5, t_subseq - 1), max_offset + 1))
        alpha = prior.offset_alpha
        
        offset_probs = []
        for k in possible_offsets:
            prob = 1.0 / (1.0 + abs(k) ** alpha)
            if k > 0:
                prob *= prior.prob_future
            elif k < 0:
                prob *= (1.0 - prior.prob_future)
            offset_probs.append(prob)
        
        offset_probs = np.array(offset_probs)
        offset_probs = offset_probs / offset_probs.sum()
        target_offset = int(rng.choice(possible_offsets, p=offset_probs))
        
        # === Optimize T_total ===
        if sample_mode == 'iid':
            burn_in = 5
            T_total = t_subseq + abs(target_offset) + burn_in
        elif sample_mode == 'sliding_window':
            estimated_windows = min(n_samples, 100)
            min_T_for_windows = t_subseq + estimated_windows * window_stride + abs(target_offset)
            max_T_for_ratio = int(t_subseq / 0.25)
            T_total = min(min_T_for_windows, max_T_for_ratio)
            T_total = max(T_total, t_subseq + abs(target_offset) + 10)
        else:
            burn_in = 10
            T_total = t_subseq + abs(target_offset) + burn_in + 20
        
        # === Post-processing ===
        apply_warping = rng.random() < prior.prob_warping
        warping_intensity = rng.uniform(*prior.warping_intensity_range)
        apply_quantization = rng.random() < prior.prob_quantization
        n_quantization_bins = rng.integers(*prior.n_quantization_bins_range)
        apply_missing = rng.random() < prior.prob_missing_values
        missing_rate = rng.uniform(*prior.missing_rate_range)
        train_ratio = rng.uniform(*prior.train_ratio_range)
        
        return cls(
            n_samples=n_samples,
            n_features=n_features,
            T_total=T_total,
            t_subseq=t_subseq,
            n_nodes=n_nodes,
            density=density,
            n_disconnected_subgraphs=n_disconnected,
            time_transforms=time_transforms,
            memory_dim=memory_dim,
            memory_init=prior.memory_init,
            stochastic_dim=stochastic_dim,
            activations=activations,
            weight_init=weight_init,
            weight_scale=weight_scale,
            bias_std=bias_std,
            node_noise=node_noise,
            node_discretization=node_discretization,
            sample_mode=sample_mode,
            window_stride=window_stride,
            n_sequences=n_sequences,
            n_classes=n_classes,
            target_offset=target_offset,
            spatial_distance_alpha=prior.distance_alpha,
            apply_warping=apply_warping,
            warping_intensity=warping_intensity,
            apply_quantization=apply_quantization,
            n_quantization_bins=n_quantization_bins,
            apply_missing=apply_missing,
            missing_rate=missing_rate,
            train_ratio=train_ratio,
            seed=int(rng.integers(0, 2**31))
        )
    
    def get_config_summary(self) -> str:
        """Get a summary of the configuration."""
        n_noisy = sum(1 for nc in self.node_noise if nc.has_noise)
        n_disc = sum(1 for dc in self.node_discretization if dc.is_discretization)
        
        transform_names = [t['name'] for t in self.time_transforms]
        
        lines = [
            f"n_samples: {self.n_samples}",
            f"n_features: {self.n_features}",
            f"T_total: {self.T_total}, t_subseq: {self.t_subseq}",
            f"n_nodes: {self.n_nodes}",
            f"Time transforms ({len(self.time_transforms)}): {transform_names}",
            f"Memory dim: {self.memory_dim} ({self.memory_init})",
            f"Stochastic dim: {self.stochastic_dim}",
            f"Activations: {set(self.activations)}",
            f"Weight init: {self.weight_init} (scale={self.weight_scale:.2f}), bias_std={self.bias_std:.3f}",
            f"Noisy nodes: {n_noisy}/{len(self.node_noise)}",
            f"Discretization nodes: {n_disc}/{len(self.node_discretization)}",
            f"Sample mode: {self.sample_mode}",
            f"Target: {self.n_classes} classes, offset={self.target_offset}",
        ]
        return '\n'.join(lines)
