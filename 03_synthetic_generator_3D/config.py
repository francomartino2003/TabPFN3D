"""
Configuration for 3D Synthetic Dataset Generator with Temporal Dependencies.

This module defines:
- PriorConfig3D: Distributions for sampling dataset configurations
- DatasetConfig3D: Concrete configuration for a single 3D dataset

Key design decisions (v4):
- Nodos raíz: 1 TIME (u = t/T normalizado) + 1 vector MEMORY (1-8 dims)
- MEMORY se samplea UNA VEZ por secuencia T (da variabilidad entre samples)
- Sin state inputs (no observan nodos en t-k)
- Ruido solo al final de cada transformación
- Target siempre de discretización (clasificación)
- Al menos 1 feature relevante y 1 continua
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass
class PriorConfig3D:
    """
    Prior distributions for sampling 3D temporal dataset configurations.
    
    v4 Design:
    - 1 TIME input: u = t/T (normalized, no activation)
    - 1 MEMORY vector: sampled once per sequence, provides sample variability
    - No state inputs
    """
    
    # === Size constraints ===
    max_samples: int = 10000
    max_features: int = 15
    max_t_subseq: int = 1000      # Max t in n×m×t (subsequence length)
    max_T_total: int = 5000       # Max T (total sequence before extraction)
    max_classes: int = 10
    max_complexity: int = 10_000_000  # Max n_samples × T_total × n_nodes
    
    # === Sample size ===
    n_samples_range: Tuple[int, int] = (100, 10000)
    
    # === Feature count ===
    # Real datasets: ~94% univariate, so we prioritize univariate
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
    # Larger DAGs for more stability (distributes impact across more nodes)
    n_nodes_range: Tuple[int, int] = (8, 20)
    n_nodes_log_uniform: bool = True
    density_range: Tuple[float, float] = (0.2, 0.5)
    
    # Disconnected subgraphs for irrelevant features
    prob_disconnected_subgraph: float = 0.2
    n_disconnected_subgraphs_range: Tuple[int, int] = (1, 3)
    
    # === Root inputs (v5 - inspired by random_nn_generator) ===
    # TIME inputs: 1 base linear (u = t/T) + extra transforms
    # Each transform can appear multiple times (linear gets more copies)
    # MEMORY vector: log-uniform dimension, sampled once per sequence
    
    # Number of time transforms (log-uniform to favor smaller)
    n_time_transforms_range: Tuple[int, int] = (1, 16)
    n_time_transforms_log_uniform: bool = True
    
    # Extra copies of linear (to give it more weight)
    linear_extra_copies_range: Tuple[int, int] = (0, 3)
    
    # Available time transforms (like random_nn_generator)
    time_transforms: List[str] = field(default_factory=lambda: [
        'linear',       # u (always included)
        'quadratic',    # u^2
        'cubic',        # u^3
        'sin_k1',       # sin(2πu)
        'cos_k1',       # cos(2πu)
        'sin_k2',       # sin(4πu)
        'cos_k2',       # cos(4πu)
        'sin_k3',       # sin(6πu)
        'cos_k3',       # cos(6πu)
        'sin_k5',       # sin(10πu)
        'cos_k5',       # cos(10πu)
        'tanh_trend',   # tanh(β(2u-1)), β ~ LogUniform(0.5, 3.0)
        'exp_decay',    # exp(-γu), γ ~ LogUniform(0.5, 5.0)
        'exp_growth',   # exp(γu), γ ~ LogUniform(0.1, 1.0)
        'log',          # log(1+u) normalized
        'sqrt',         # sqrt(u) normalized
    ])
    
    # MEMORY vector dimensions (log-uniform to favor smaller)
    memory_dim_range: Tuple[int, int] = (1, 64)
    memory_dim_log_uniform: bool = True
    
    # MEMORY initialization
    memory_init: str = 'uniform'  # 'uniform' or 'normal'
    
    # === Edge transformations (v5 - inspired by random_nn_generator) ===
    # Almost all NN with smooth activations, some discretization per-node
    
    # Per-node discretization probability (not global %)
    discretization_node_prob: float = 0.15
    
    # Discretization parameters
    n_categories_range: Tuple[int, int] = (2, 10)
    
    # NN activation - smooth activations only (no dying neurons)
    nn_activations: List[str] = field(default_factory=lambda: [
        'softplus',    # Smooth ReLU (primary)
        'tanh',        # Hyperbolic tangent
        'elu',         # ELU
    ])
    
    # Probability of identity activation (linear, no activation)
    prob_identity_activation: float = 0.3
    
    # === Per-node noise (v5 - inspired by random_nn_generator) ===
    # Each node can have noise with different probability and distribution
    node_noise_prob_range: Tuple[float, float] = (0.01, 0.5)  # Log-uniform
    node_noise_std_range: Tuple[float, float] = (0.0001, 0.02)  # Log-uniform
    noise_distributions: List[str] = field(default_factory=lambda: ['normal', 'uniform', 'laplace'])
    
    # === Sample generation mode ===
    prob_iid_mode: float = 0.55
    prob_sliding_window_mode: float = 0.30
    prob_mixed_mode: float = 0.15
    
    # Sliding window parameters
    window_stride_range: Tuple[int, int] = (1, 10)
    
    # === Target configuration (v4) ===
    # Target is ALWAYS classification (from discretization node)
    min_samples_per_class: int = 25
    
    # Target temporal offset - balanced probabilities
    # prob(offset=k) ∝ 1 / (1 + |k|^alpha)
    max_target_offset: int = 20
    # Balanced between future and past (50/50)
    prob_future: float = 0.50
    
    # === Distance preference ===
    distance_alpha: float = 1.5
    
    # === Post-processing ===
    prob_warping: float = 0.3
    warping_intensity_range: Tuple[float, float] = (0.1, 0.5)
    
    prob_quantization: float = 0.2
    n_quantization_bins_range: Tuple[int, int] = (5, 20)
    
    prob_missing_values: float = 0.1
    missing_rate_range: Tuple[float, float] = (0.01, 0.15)
    
    # === Train/test split ===
    train_ratio_range: Tuple[float, float] = (0.5, 0.8)


@dataclass  
class DatasetConfig3D:
    """
    Configuration for a specific 3D temporal dataset instance.
    
    v4 Design:
    - 1 TIME root (u = t/T)
    - memory_dim MEMORY roots (sampled once per sequence)
    - Target always from discretization
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
    
    # === Root configuration (v4) ===
    # TIME inputs
    n_extra_time_inputs: int      # Extra TIME inputs (0-7)
    time_input_activations: List[str]  # Activations for extra TIME inputs
    # MEMORY inputs
    memory_dim: int           # Number of MEMORY inputs (1-8)
    memory_noise_type: str    # 'normal' or 'uniform'
    memory_sigma: float       # For normal initialization
    memory_a: float           # For uniform initialization [-a, a]
    
    # === Edge transformations ===
    transform_probs: Dict[str, float]
    allowed_activations: List[str]
    prob_identity_activation: float
    n_categories: int
    tree_depth: int
    tree_max_features_fraction: float
    
    # === Noise (v4 - simplified) ===
    noise_scale: float  # Applied at end of each transformation
    
    # === Sample generation mode ===
    sample_mode: str  # 'iid', 'sliding_window', 'mixed'
    window_stride: int
    n_sequences: int
    
    # === Target configuration ===
    is_classification: bool
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
    seed: Optional[int] = None
    
    @classmethod
    def sample_from_prior(cls, prior: PriorConfig3D, rng: np.random.Generator) -> 'DatasetConfig3D':
        """Sample a 3D dataset configuration from the prior."""
        
        def log_uniform(low: float, high: float) -> float:
            return np.exp(rng.uniform(np.log(low), np.log(high)))
        
        # === Sample temporal parameters ===
        if prior.t_subseq_log_uniform:
            t_subseq = int(log_uniform(*prior.t_subseq_range))
        else:
            t_subseq = rng.integers(*prior.t_subseq_range)
        t_subseq = min(t_subseq, prior.max_t_subseq)
        t_subseq = max(t_subseq, 5)
        
        min_T = t_subseq + 10
        if prior.T_total_log_uniform:
            T_total = int(log_uniform(max(min_T, prior.T_total_range[0]), prior.T_total_range[1]))
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
        
        # === Sample extra TIME inputs (v4) ===
        n_extra_time_inputs = rng.integers(prior.n_extra_time_inputs_range[0], 
                                            prior.n_extra_time_inputs_range[1] + 1)
        
        # Sample activations for extra TIME inputs
        if n_extra_time_inputs > 0:
            time_input_activations = list(rng.choice(
                prior.time_activations, 
                size=n_extra_time_inputs, 
                replace=False
            ))
        else:
            time_input_activations = []
        
        # === Sample MEMORY dimension (v4) ===
        memory_dim = rng.integers(*prior.memory_dim_range)
        
        # Total roots = 1 (base TIME) + n_extra_time_inputs + memory_dim (MEMORY)
        n_roots = 1 + n_extra_time_inputs + memory_dim
        
        # === Sample graph structure ===
        if prior.n_nodes_log_uniform:
            n_nodes = int(log_uniform(*prior.n_nodes_range))
        else:
            n_nodes = rng.integers(*prior.n_nodes_range)
        # Ensure enough nodes for roots + features + target
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
        
        # === MEMORY initialization ===
        memory_noise_type = prior.memory_noise_type
        memory_sigma = rng.uniform(*prior.memory_sigma_range)
        memory_a = rng.uniform(*prior.memory_a_range)
        
        # === Sample edge transformation probabilities ===
        transform_probs = {
            'nn': prior.prob_nn_transform,
            'tree': prior.prob_tree_transform,
            'discretization': prior.prob_discretization,
        }
        total = sum(transform_probs.values())
        transform_probs = {k: v/total for k, v in transform_probs.items()}
        
        # Sample subset of activations
        n_activations = rng.integers(3, len(prior.activations) + 1)
        allowed_activations = list(rng.choice(prior.activations, size=n_activations, replace=False))
        if 'identity' not in allowed_activations:
            allowed_activations.append('identity')
        
        # Discretization
        n_categories = rng.integers(*prior.n_categories_range)
        max_cats_for_samples = max(2, n_samples // prior.min_samples_per_class)
        n_categories = min(n_categories, max_cats_for_samples)
        
        # Tree parameters
        tree_depth = rng.integers(*prior.tree_depth_range)
        tree_max_features_fraction = prior.tree_max_features_fraction
        
        # === Noise (v4 - simplified) ===
        if prior.noise_scale_log_uniform:
            noise_scale = log_uniform(*prior.noise_scale_range)
        else:
            noise_scale = rng.uniform(*prior.noise_scale_range)
        
        # === Sample generation mode ===
        mode_probs = [prior.prob_iid_mode, prior.prob_sliding_window_mode, prior.prob_mixed_mode]
        mode_probs = np.array(mode_probs) / sum(mode_probs)
        sample_mode = rng.choice(['iid', 'sliding_window', 'mixed'], p=mode_probs)
        
        window_stride = rng.integers(*prior.window_stride_range) if sample_mode != 'iid' else 1
        n_sequences = rng.integers(2, 10) if sample_mode == 'mixed' else 1
        
        # === Target configuration (v4: always classification) ===
        is_classification = True
        max_classes_for_samples = max(2, n_samples // prior.min_samples_per_class)
        max_classes = min(prior.max_classes, max_classes_for_samples)
        n_classes = rng.integers(2, max(3, max_classes + 1))
        
        # === Sample target offset (balanced future/past) ===
        max_offset = min(prior.max_target_offset, 15)
        possible_offsets = list(range(-min(5, t_subseq - 1), max_offset + 1))
        alpha = prior.distance_alpha
        
        offset_probs = []
        for k in possible_offsets:
            prob = 1.0 / (1.0 + abs(k) ** alpha)
            # Balanced future/past (50/50)
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
        else:  # mixed
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
            n_extra_time_inputs=n_extra_time_inputs,
            time_input_activations=time_input_activations,
            memory_dim=memory_dim,
            memory_noise_type=memory_noise_type,
            memory_sigma=memory_sigma,
            memory_a=memory_a,
            transform_probs=transform_probs,
            allowed_activations=allowed_activations,
            prob_identity_activation=prior.prob_identity_activation,
            n_categories=n_categories,
            tree_depth=tree_depth,
            tree_max_features_fraction=tree_max_features_fraction,
            noise_scale=noise_scale,
            sample_mode=sample_mode,
            window_stride=window_stride,
            n_sequences=n_sequences,
            is_classification=is_classification,
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
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'T_total': self.T_total,
            't_subseq': self.t_subseq,
            'n_nodes': self.n_nodes,
            'n_extra_time_inputs': self.n_extra_time_inputs,
            'time_input_activations': self.time_input_activations,
            'memory_dim': self.memory_dim,
            'sample_mode': self.sample_mode,
            'is_classification': self.is_classification,
            'n_classes': self.n_classes,
            'target_offset': self.target_offset,
            'spatial_distance_alpha': self.spatial_distance_alpha,
            'noise_scale': self.noise_scale,
            'train_ratio': self.train_ratio,
            'seed': self.seed
        }
