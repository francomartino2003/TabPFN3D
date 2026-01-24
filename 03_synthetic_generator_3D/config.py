"""
Configuration for 3D Synthetic Dataset Generator with Temporal Dependencies.

This module defines:
- PriorConfig3D: Distributions for sampling dataset configurations
- DatasetConfig3D: Concrete configuration for a single 3D dataset

Key design decisions (v3):
- Nodos raíz: solo temporales y estados (nodo X en t-k)
- Ruido solo como inicialización cuando t-k < 0
- Sin passthrough, más probabilidad de árboles
- DAG más pequeño
- Selección de features con preferencia por cercanía espacial (DAG)
- Selección de target_offset con preferencia por cercanía temporal
- distance_alpha controla ambas preferencias de forma unificada
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
import numpy as np


@dataclass
class StateInputConfig:
    """Configuration for a state input (node X at t-k)."""
    source_node: int  # Which node's past value to use
    lag: int          # How many timesteps back (k in t-k)


@dataclass
class PriorConfig3D:
    """
    Prior distributions for sampling 3D temporal dataset configurations.
    
    Limits:
    - max 10,000 samples (train + test)
    - max 15 features
    - max 1000 timesteps
    - max 10 classes
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
    # Probability of univariate series (1 feature only)
    prob_univariate: float = 0.4
    
    # Beta distribution for feature count when multivariate, scaled to [2, max_features]
    n_features_beta_a: float = 1.5
    n_features_beta_b: float = 4.0
    n_features_range: Tuple[int, int] = (2, 15)
    
    # === Temporal parameters ===
    # Total sequence length T (before extracting subsequences)
    T_total_range: Tuple[int, int] = (30, 2000)
    T_total_log_uniform: bool = True
    
    # Subsequence length for features (the 't' in n×m×t)
    t_subseq_range: Tuple[int, int] = (10, 1000)
    t_subseq_log_uniform: bool = True
    
    # === Graph structure - Distance preference controls complexity ===
    n_nodes_range: Tuple[int, int] = (8, 30)  # Larger DAGs, complexity controlled by distance
    n_nodes_log_uniform: bool = True
    
    # Edge density (0.0 = minimal tree, 1.0 = complete DAG)
    density_range: Tuple[float, float] = (0.1, 0.6)
    
    # Number of root/input nodes (nodes without parents)
    # Now only time + state inputs (no noise roots)
    # More roots = richer temporal dynamics
    n_roots_range: Tuple[int, int] = (4, 18)  # More roots allowed
    min_roots_fraction: float = 0.25
    max_roots_fraction: float = 0.60
    
    # Disconnected subgraphs for irrelevant features
    prob_disconnected_subgraph: float = 0.2  # Reduced (less relevant for univariate)
    n_disconnected_subgraphs_range: Tuple[int, int] = (1, 3)
    
    # === Root input type distribution ===
    # Only TIME and STATE inputs as roots (no noise roots)
    # Noise only used as initialization when t-k < 0
    min_time_inputs: int = 1    # At least 1 time input (linear always included)
    min_state_inputs: int = 1   # At least 1 state input
    
    # Distribution of roots between time and state
    # Reduced time fraction = more state inputs (state inputs correlate with better AUC)
    time_fraction_range: Tuple[float, float] = (0.2, 0.45)  # 20-45% time inputs → 55-80% state
    
    # === State input parameters ===
    # Lag range for state inputs (the k in t-k)
    state_lag_range: Tuple[int, int] = (1, 5)   # Reduced max lag (was 10)
    state_lag_distribution: str = 'geometric'   # 'uniform' or 'geometric' (favors smaller lags)
    state_lag_geometric_p: float = 0.6          # Higher p = smaller lags more probable (was 0.4)
    
    # Spatial preference for state source nodes
    # State inputs prefer to observe nodes closer to the target in the DAG
    state_source_distance_alpha: float = 2.0    # prob ∝ 1/(1 + distance^alpha)
    
    # α for tanh(α·s_{t-k}) to normalize state to [-1, 1]
    # Reduced range to preserve more information (high alpha → info loss)
    state_alpha_range: Tuple[float, float] = (0.3, 0.8)
    
    # === Time-dependent input activations ===
    # Available time activation functions (sampled with weights)
    time_activations: List[str] = field(default_factory=lambda: [
        'linear',        # u
        'quadratic',     # u^2
        'cubic',         # u^3
        'tanh',          # tanh(β(2u-1))
        'sin_1', 'sin_2', 'sin_3', 'sin_5',  # sin(2πku)
        'cos_1', 'cos_2', 'cos_3', 'cos_5',  # cos(2πku)
        'exp_decay',     # exp(-γu)
        'log'            # log(u + 0.1)
    ])
    
    # Weights for time activation sampling (all similar, slight preference for simple)
    time_activation_weights: Dict[str, float] = field(default_factory=lambda: {
        'linear': 1.5,      # Slight preference
        'quadratic': 1.2,
        'cubic': 1.0,
        'tanh': 1.0,
        'sin_1': 1.0, 'sin_2': 1.0, 'sin_3': 0.8, 'sin_5': 0.6,
        'cos_1': 1.0, 'cos_2': 1.0, 'cos_3': 0.8, 'cos_5': 0.6,
        'exp_decay': 1.0,
        'log': 1.0
    })
    
    # Parameters for time activations
    tanh_beta_range: Tuple[float, float] = (0.5, 3.0)
    exp_gamma_range: Tuple[float, float] = (0.5, 5.0)
    
    # === Noise for initialization (when t-k < 0) ===
    noise_types: List[str] = field(default_factory=lambda: ['normal', 'uniform', 'mixed'])
    init_sigma_range: Tuple[float, float] = (0.05, 0.5)  # Reduced further for cleaner signals
    init_a_range: Tuple[float, float] = (0.2, 0.6)       # Reduced further
    
    # === Edge transformations ===
    # Removed passthrough, higher tree probability for non-linear relationships
    prob_nn_transform: float = 0.40   # Neural network-like transformation
    prob_tree_transform: float = 0.45  # Decision tree - higher for complex patterns
    prob_discretization: float = 0.15  # Discretization (categorical)
    # No passthrough - removed
    
    # Probability of using identity activation in NN
    prob_identity_activation: float = 0.4
    
    # Available activation functions for NN transformation
    activations: List[str] = field(default_factory=lambda: [
        # === Bounded activations ===
        'identity',    # Linear (no activation) - unbounded but stable with Xavier init
        'tanh',        # Hyperbolic tangent: tanh(x) → [-1, 1]
        'sigmoid',     # Sigmoid: 1/(1+exp(-x)) → [0, 1]
        'sin',         # Sine: sin(x) → [-1, 1]
        'cos',         # Cosine: cos(x) → [-1, 1]
        'step',        # Step function: 1 if x>0 else 0 → {0, 1}
        'rank',        # Rank operation: percentile ranks → [0, 1]
        # === Unbounded activations ===
        'relu',        # ReLU: max(0, x) → [0, ∞)
        'leaky_relu',  # Leaky ReLU: x if x>0 else 0.01x → (-∞, ∞)
        'elu',         # ELU: x if x>0 else exp(x)-1 → (-1, ∞)
        'softplus',    # Smooth ReLU: log(1+exp(x)) → (0, ∞)
        'abs',         # Absolute value: |x| → [0, ∞)
        'square',      # Squaring: x^2 → [0, ∞)
        'log',         # Logarithm: log(|x| + eps) → (-∞, ∞)
        'power',       # Power function: sign(x)*|x|^p with random p
        'mod',         # Modulo operation: x mod 2 → [0, 2)
    ])
    
    # Discretization parameters
    n_categories_range: Tuple[int, int] = (2, 10)
    
    # Decision tree parameters
    tree_depth_range: Tuple[int, int] = (1, 5)
    tree_max_features_fraction: float = 0.7
    
    # === Edge noise - Variable per dataset (REDUCED) ===
    prob_edge_noise: float = 0.03      # Reduced (was 0.05)
    noise_scale_range: Tuple[float, float] = (0.001, 0.03)  # Reduced max (was 0.08)
    noise_scale_log_uniform: bool = True
    # More datasets are "clean" with minimal noise
    prob_low_noise_dataset: float = 0.5  # Increased (was 0.3)
    low_noise_scale_max: float = 0.002   # Reduced (was 0.005)
    
    # === Distance preference (unified for spatial and temporal) ===
    # prob(distance=d) ∝ 1 / (1 + d^alpha)
    # Higher alpha = stronger preference for closer distances
    distance_alpha: float = 1.5
    
    # === Sample generation mode ===
    # Increased IID probability (better AUC, no temporal leakage)
    prob_iid_mode: float = 0.55
    prob_sliding_window_mode: float = 0.30
    prob_mixed_mode: float = 0.15
    
    # Sliding window parameters
    window_stride_range: Tuple[int, int] = (1, 10)
    
    # === Target configuration ===
    prob_classification: float = 0.5
    force_classification: Optional[bool] = None
    min_samples_per_class: int = 25  # Increased for more robust learning
    
    # Target temporal offset - distance-based (uses distance_alpha)
    # prob(offset=k) ∝ 1 / (1 + |k|^distance_alpha)
    # This naturally favors offset=0, then ±1, ±2, etc.
    max_target_offset: int = 20       # Maximum temporal offset
    prob_future: float = 0.75         # Probability of future (vs past) when offset != 0
    
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
    
    Sampled from PriorConfig3D.
    """
    
    # === Size ===
    n_samples: int
    n_features: int
    T_total: int          # Total sequence length
    t_subseq: int         # Feature subsequence length
    
    # === Graph structure ===
    n_nodes: int
    density: float
    n_disconnected_subgraphs: int
    
    # === Root input configuration (NEW DESIGN) ===
    # No n_noise_inputs - noise only for initialization
    n_time_inputs: int
    n_state_inputs: int
    
    # State input configurations: list of (source_node_idx, lag) tuples
    # source_node_idx is the index in the non-root nodes (assigned after DAG build)
    # lag is the k in t-k
    state_configs: List[Tuple[int, int]]  # List of (source_node_relative_idx, lag)
    
    # Time input activations (one per time input)
    time_activations: List[str]
    time_activation_params: Dict[str, float]
    
    # State normalization parameter
    state_alpha: float
    
    # === Edge transformations ===
    transform_probs: Dict[str, float]
    allowed_activations: List[str]
    prob_identity_activation: float
    n_categories: int
    tree_depth: int
    tree_max_features_fraction: float
    
    # === Noise settings (for initialization only) ===
    noise_type: str
    noise_scale: float      # For edge noise (very small)
    edge_noise_prob: float
    init_sigma: float       # For state initialization
    init_a: float
    
    # === Sample generation mode ===
    sample_mode: str  # 'iid', 'sliding_window', 'mixed'
    window_stride: int
    n_sequences: int
    
    # === Target configuration ===
    is_classification: bool
    n_classes: int
    target_offset: int  # Temporal distance: 0=within, >0=future, <0=past
    
    # === Distance preference (for feature selection) ===
    spatial_distance_alpha: float  # Controls preference for spatially closer features
    
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
        
        # T_total must be larger than t_subseq
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
            t_subseq = max(50, max_t_for_features)  # Keep at least 50 timesteps
            T_total = max(t_subseq + 10, T_total)   # Ensure T_total > t_subseq
        
        # === Sample sample count ===
        n_samples = rng.integers(*prior.n_samples_range)
        n_samples = min(n_samples, prior.max_samples)
        
        # === Sample graph structure (REDUCED) ===
        if prior.n_nodes_log_uniform:
            n_nodes = int(log_uniform(*prior.n_nodes_range))
        else:
            n_nodes = rng.integers(*prior.n_nodes_range)
        # Ensure enough nodes for features + target + some latent
        n_nodes = max(n_nodes, n_features + 3)
        
        # === Enforce complexity limit ===
        for _ in range(10):
            complexity = n_samples * T_total * n_nodes
            if complexity <= prior.max_complexity:
                break
            scale = (prior.max_complexity / complexity) ** (1/3)
            n_samples = max(100, int(n_samples * scale))
            T_total = max(t_subseq + 10, int(T_total * scale))
            n_nodes = max(n_features + 3, int(n_nodes * scale))
        
        # Edge density
        density = rng.uniform(*prior.density_range)
        
        # Disconnected subgraphs (less likely for small graphs)
        n_disconnected = 0
        if n_nodes >= 8 and rng.random() < prior.prob_disconnected_subgraph:
            max_disconnected = min(prior.n_disconnected_subgraphs_range[1], (n_nodes - 5) // 3)
            if max_disconnected >= 1:
                n_disconnected = rng.integers(1, max_disconnected + 1)
        
        # === Sample root input types (ONLY TIME and STATE) ===
        # Calculate number of roots
        min_roots_by_fraction = max(2, int(n_nodes * prior.min_roots_fraction))
        max_roots_by_fraction = max(min_roots_by_fraction, int(n_nodes * prior.max_roots_fraction))
        
        low = max(prior.n_roots_range[0], min_roots_by_fraction)
        high = min(prior.n_roots_range[1], max_roots_by_fraction) + 1
        n_roots = rng.integers(low, high) if low < high else low
        
        # Distribute between time and state
        time_fraction = rng.uniform(*prior.time_fraction_range)
        n_time = max(prior.min_time_inputs, int(n_roots * time_fraction))
        n_state = max(prior.min_state_inputs, n_roots - n_time)
        
        # Adjust if we have too many
        while n_time + n_state > n_roots:
            if n_state > prior.min_state_inputs:
                n_state -= 1
            elif n_time > prior.min_time_inputs:
                n_time -= 1
            else:
                break
        
        # === Sample time activation functions (weighted, no forced linear) ===
        time_activations = []
        time_activation_params = {}
        
        # Build weighted probabilities
        available_acts = prior.time_activations.copy()
        weights = np.array([prior.time_activation_weights.get(a, 1.0) for a in available_acts])
        probs = weights / weights.sum()
        
        # Sample n_time activations WITHOUT replacement (all different)
        n_to_sample = min(n_time, len(available_acts))
        sampled_acts = list(rng.choice(available_acts, size=n_to_sample, replace=False, p=probs))
        time_activations = sampled_acts
        
        # Sample parameters for activations that need them
        for i, act in enumerate(time_activations):
            if act == 'tanh':
                time_activation_params[f'tanh_beta_{i}'] = log_uniform(*prior.tanh_beta_range)
            elif act == 'exp_decay':
                time_activation_params[f'exp_gamma_{i}'] = log_uniform(*prior.exp_gamma_range)
        
        # === Sample state configurations ===
        # Each state input references a non-root node at t-k
        # We'll assign source nodes later (after DAG is built)
        # For now, sample lags
        state_configs = []
        n_non_roots = n_nodes - n_roots
        
        for i in range(n_state):
            # Sample lag
            if prior.state_lag_distribution == 'geometric':
                # Geometric distribution favors smaller lags
                lag = 1 + int(rng.geometric(prior.state_lag_geometric_p))
                lag = min(lag, prior.state_lag_range[1])
            else:
                lag = rng.integers(*prior.state_lag_range)
            
            # Source node will be assigned later (use -1 as placeholder for "any non-root")
            # We'll properly assign after DAG is built
            source_idx = -1  # Placeholder
            state_configs.append((source_idx, lag))
        
        # === Sample state alpha ===
        state_alpha = log_uniform(*prior.state_alpha_range)
        
        # === Sample edge transformation probabilities (NO PASSTHROUGH) ===
        transform_probs = {
            'nn': prior.prob_nn_transform,
            'tree': prior.prob_tree_transform,
            'discretization': prior.prob_discretization,
        }
        # Normalize
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
        
        # === Noise (for initialization only) ===
        noise_type = rng.choice(prior.noise_types)
        
        # Some datasets are "clean" with minimal noise
        if rng.random() < prior.prob_low_noise_dataset:
            noise_scale = log_uniform(prior.noise_scale_range[0], prior.low_noise_scale_max)
        elif prior.noise_scale_log_uniform:
            noise_scale = log_uniform(*prior.noise_scale_range)
        else:
            noise_scale = rng.uniform(*prior.noise_scale_range)
        
        init_sigma = rng.uniform(*prior.init_sigma_range)
        init_a = rng.uniform(*prior.init_a_range)
        
        # === Sample generation mode ===
        mode_probs = [prior.prob_iid_mode, prior.prob_sliding_window_mode, prior.prob_mixed_mode]
        mode_probs = np.array(mode_probs) / sum(mode_probs)
        sample_mode = rng.choice(['iid', 'sliding_window', 'mixed'], p=mode_probs)
        
        # CRITICAL: For IID mode, we MUST have state inputs!
        # Time inputs are deterministic functions of t, so without state inputs,
        # all IID samples would have identical features/targets.
        # State inputs get initialized with noise when t-lag < 0, creating variability.
        if sample_mode == 'iid' and n_state == 0:
            # Force at least 1 state input by converting a time input
            if n_time > 1:
                n_time -= 1
                n_state = 1
            else:
                # If only 1 time input, add 1 state input (increase roots)
                n_state = 1
        
        window_stride = rng.integers(*prior.window_stride_range) if sample_mode != 'iid' else 1
        n_sequences = rng.integers(2, 10) if sample_mode == 'mixed' else 1
        
        # === Target configuration ===
        if prior.force_classification is not None:
            is_classification = prior.force_classification
        else:
            is_classification = rng.random() < prior.prob_classification
        
        if is_classification:
            max_classes_for_samples = max(2, n_samples // prior.min_samples_per_class)
            max_classes = min(prior.max_classes, max_classes_for_samples)
            n_classes = rng.integers(2, max(3, max_classes + 1))
        else:
            n_classes = 0
        
        # === Sample target offset FIRST (to optimize T_total) ===
        # Use a fixed reasonable max_offset, not dependent on T_total
        # This way we can then compute optimal T_total
        max_offset = min(prior.max_target_offset, 15)  # Cap at 15 for reasonable t_ratio
        
        # Generate possible offsets and their probabilities
        # prob(offset=k) ∝ 1 / (1 + |k|^alpha)
        possible_offsets = list(range(-min(5, t_subseq - 1), max_offset + 1))
        alpha = prior.distance_alpha
        
        offset_probs = []
        for k in possible_offsets:
            prob = 1.0 / (1.0 + abs(k) ** alpha)
            # Apply future vs past preference for non-zero offsets
            if k > 0:
                prob *= prior.prob_future
            elif k < 0:
                prob *= (1.0 - prior.prob_future)
            offset_probs.append(prob)
        
        offset_probs = np.array(offset_probs)
        offset_probs = offset_probs / offset_probs.sum()
        
        target_offset = int(rng.choice(possible_offsets, p=offset_probs))
        
        # === OPTIMIZE T_total to maximize t_ratio ===
        # For IID: we only need t_subseq + |offset| + minimal burn_in
        # For sliding_window: need more room for window extraction
        # For mixed: similar to sliding_window
        
        if sample_mode == 'iid':
            # Minimal T_total: just what we need
            # burn_in is for state initialization (states need some history)
            max_state_lag = max([sc[1] for sc in state_configs]) if state_configs else 1
            burn_in = max(5, max_state_lag + 2)  # Small burn-in
            T_total = t_subseq + abs(target_offset) + burn_in
        elif sample_mode == 'sliding_window':
            # Need more room for multiple windows, but still optimize
            # Estimate: need at least n_samples windows of size t_subseq
            # With stride, need: t_subseq + (n_windows - 1) * stride
            # But cap it to keep t_ratio reasonable
            estimated_windows = min(n_samples, 100)  # Cap window count estimation
            min_T_for_windows = t_subseq + estimated_windows * window_stride + abs(target_offset)
            # But cap T_total to keep t_ratio > 0.3 at least
            max_T_for_ratio = int(t_subseq / 0.25)  # t_ratio >= 0.25
            T_total = min(min_T_for_windows, max_T_for_ratio)
            T_total = max(T_total, t_subseq + abs(target_offset) + 10)
        else:  # mixed
            # Multiple independent sequences, each shorter
            max_state_lag = max([sc[1] for sc in state_configs]) if state_configs else 1
            burn_in = max(10, max_state_lag + 5)
            T_total = t_subseq + abs(target_offset) + burn_in + 20  # Small extra
        
        # === Post-processing ===
        apply_warping = rng.random() < prior.prob_warping
        warping_intensity = rng.uniform(*prior.warping_intensity_range)
        
        apply_quantization = rng.random() < prior.prob_quantization
        n_quantization_bins = rng.integers(*prior.n_quantization_bins_range)
        
        apply_missing = rng.random() < prior.prob_missing_values
        missing_rate = rng.uniform(*prior.missing_rate_range)
        
        # === Train ratio ===
        train_ratio = rng.uniform(*prior.train_ratio_range)
        
        return cls(
            n_samples=n_samples,
            n_features=n_features,
            T_total=T_total,
            t_subseq=t_subseq,
            n_nodes=n_nodes,
            density=density,
            n_disconnected_subgraphs=n_disconnected,
            n_time_inputs=n_time,
            n_state_inputs=n_state,
            state_configs=state_configs,
            time_activations=time_activations,
            time_activation_params=time_activation_params,
            state_alpha=state_alpha,
            transform_probs=transform_probs,
            allowed_activations=allowed_activations,
            prob_identity_activation=prior.prob_identity_activation,
            n_categories=n_categories,
            tree_depth=tree_depth,
            tree_max_features_fraction=tree_max_features_fraction,
            noise_type=noise_type,
            noise_scale=noise_scale,
            edge_noise_prob=prior.prob_edge_noise,
            init_sigma=init_sigma,
            init_a=init_a,
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
            'n_time_inputs': self.n_time_inputs,
            'n_state_inputs': self.n_state_inputs,
            'state_configs': self.state_configs,
            'time_activations': self.time_activations,
            'state_alpha': self.state_alpha,
            'sample_mode': self.sample_mode,
            'is_classification': self.is_classification,
            'n_classes': self.n_classes,
            'target_offset': self.target_offset,
            'spatial_distance_alpha': self.spatial_distance_alpha,
            'train_ratio': self.train_ratio,
            'seed': self.seed
        }
