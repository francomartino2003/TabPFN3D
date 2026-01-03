"""
Configuration for 3D Synthetic Dataset Generator with Temporal Dependencies.

This module defines:
- PriorConfig3D: Distributions for sampling dataset configurations
- DatasetConfig3D: Concrete configuration for a single 3D dataset

Key differences from 2D:
- Temporal dimension with T timesteps
- Three types of root inputs: noise, time-dependent, state (memory)
- Sample generation modes: IID, sliding window, mixed
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
import numpy as np


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
    # T_total > t_subseq to allow for target offsets and window extraction
    T_total_range: Tuple[int, int] = (30, 2000)
    T_total_log_uniform: bool = True
    
    # Subsequence length for features (the 't' in n×m×t)
    # This is the effective temporal dimension of the dataset
    t_subseq_range: Tuple[int, int] = (10, 1000)
    t_subseq_log_uniform: bool = True
    
    # === Graph structure (similar to 2D) ===
    n_nodes_range: Tuple[int, int] = (12, 80)
    n_nodes_log_uniform: bool = True
    
    # Edge density (0.0 = minimal tree, 1.0 = complete DAG)
    # Controls how many edges beyond the minimum (n-1) are added
    density_range: Tuple[float, float] = (0.01, 0.8)
    
    # Number of root/input nodes (nodes without parents)
    # For 3D, this is the total across noise + time + state inputs
    n_roots_range: Tuple[int, int] = (3, 30)
    max_roots_fraction: float = 0.50  # Root nodes can't exceed this fraction of total nodes
    
    # Disconnected subgraphs for irrelevant features
    prob_disconnected_subgraph: float = 0.3
    n_disconnected_subgraphs_range: Tuple[int, int] = (1, 5)
    
    # === Root input type distribution ===
    # Minimum counts for each type
    # - Time inputs: provide temporal structure
    # - State inputs: provide memory and autocorrelation
    # - Noise inputs: provide sample-to-sample variability (low count preferred)
    min_noise_inputs: int = 0   # Noise is optional (state provides variability via t0 init)
    min_time_inputs: int = 1    # At least 1 time input (linear always included)
    min_state_inputs: int = 2   # States provide memory and autocorrelation
    
    # Probability of adding noise inputs (keep low for temporal coherence)
    prob_noise_input: float = 0.3  # Low probability of adding noise roots
    max_noise_inputs: int = 2      # Cap on noise inputs
    
    # === Noise behavior ===
    # If true, noise only enters at t=0 and propagates as state
    # This increases autocorrelation in generated series
    noise_only_at_t0: bool = True
    prob_noise_only_at_t0: float = 0.7  # 70% of datasets use t0-only noise
    
    # === Time-dependent input activations ===
    # 'linear' is ALWAYS included as the first time input
    # Additional time inputs sample from other activations WITHOUT repetition
    time_activations: List[str] = field(default_factory=lambda: [
        'linear',        # u (ALWAYS first)
        'quadratic',     # u^2
        'cubic',         # u^3
        'tanh',          # tanh(β(2u-1))
        'sin_1', 'sin_2', 'sin_3', 'sin_5',  # sin(2πku)
        'cos_1', 'cos_2', 'cos_3', 'cos_5',  # cos(2πku)
        'exp_decay',     # exp(-γu)
        'log'            # log(u + 0.1)
    ])
    
    # Parameters for time activations
    tanh_beta_range: Tuple[float, float] = (0.5, 3.0)  # LogUniform
    exp_gamma_range: Tuple[float, float] = (0.5, 5.0)  # LogUniform
    
    # === State input parameters ===
    # α for tanh(α·s_{t-1}) to normalize state to [-1, 1]
    state_alpha_range: Tuple[float, float] = (0.5, 2.0)  # LogUniform
    
    # === Edge transformations (similar to 2D) ===
    # Probabilities for each node transformation type (applied per child node)
    prob_nn_transform: float = 0.40  # Neural network-like transformation
    prob_tree_transform: float = 0.10  # Decision tree-like transformation
    prob_discretization: float = 0.20  # Discretization (categorical)
    prob_passthrough: float = 0.30  # Pass-through: just copy one parent (preserves correlation)
    
    # Probability of using identity activation in NN (preserves temporal structure)
    # Higher than 2D because temporal series need more linear propagation
    prob_identity_activation: float = 0.5  # 50% of NN use identity
    
    # Available activation functions for NN transformation
    # Per paper: "identity, logarithm, sigmoid, absolute value, sine, 
    # hyperbolic tangent, rank operation, squaring, power functions, 
    # smooth ReLU, step function and modulo operation"
    activations: List[str] = field(default_factory=lambda: [
        'identity',    # Linear (no activation)
        'log',         # Logarithm: log(|x| + eps)
        'sigmoid',     # Sigmoid: 1/(1+exp(-x))
        'abs',         # Absolute value: |x|
        'sin',         # Sine: sin(x)
        'tanh',        # Hyperbolic tangent: tanh(x)
        'rank',        # Rank operation: convert to percentile ranks
        'square',      # Squaring: x^2
        'power',       # Power function: sign(x)*|x|^p with random p
        'softplus',    # Smooth ReLU: log(1+exp(x))
        'step',        # Step function: 1 if x>0 else 0
        'mod',         # Modulo operation: x mod period
    ])
    
    # Discretization parameters
    n_categories_range: Tuple[int, int] = (2, 10)
    
    # Decision tree parameters
    tree_depth_range: Tuple[int, int] = (1, 5)
    tree_max_features_fraction: float = 0.7  # Max fraction of parents to use as features
    
    # === Noise parameters ===
    noise_types: List[str] = field(default_factory=lambda: ['normal', 'uniform', 'mixed'])
    init_sigma_range: Tuple[float, float] = (0.1, 2.0)  # For Normal
    init_a_range: Tuple[float, float] = (0.5, 2.0)      # For Uniform
    
    # Edge noise - reduced to preserve temporal structure
    prob_edge_noise: float = 0.15  # Reduced from 0.3
    noise_scale_range: Tuple[float, float] = (0.005, 0.15)  # Reduced scale
    noise_scale_log_uniform: bool = True
    
    # === Sample generation mode ===
    # Probability of each mode
    # IID: each sample is independent, state resets (T_total ≈ t_subseq + burn-in)
    # Sliding window: extract windows from single long sequence (T_total can be long)
    # Mixed: multiple independent sequences
    prob_iid_mode: float = 0.35
    prob_sliding_window_mode: float = 0.45
    prob_mixed_mode: float = 0.20
    
    # Sliding window parameters
    window_stride_range: Tuple[int, int] = (1, 10)  # Stride between windows
    
    # === Target configuration ===
    prob_classification: float = 0.5
    # Force task type: None = sample, True = always classification, False = always regression
    force_classification: Optional[bool] = None
    # Minimum samples per class (to ensure balanced enough datasets)
    min_samples_per_class: int = 10
    
    # Target time offset distribution
    # offset = 0 means within subsequence (classification)
    # offset > 0 means future prediction
    # offset < 0 means past (rare)
    target_offset_probs: Dict[str, float] = field(default_factory=lambda: {
        'within': 0.4,      # Target within the feature subsequence
        'future_near': 0.4,  # 1-5 steps ahead
        'future_far': 0.15,   # 6-20 steps ahead
        'past': 0.05         # Before subsequence (rare)
    })
    
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
    density: float  # Edge density (0.0 = tree, 1.0 = complete DAG)
    n_disconnected_subgraphs: int
    
    # === Root input configuration ===
    n_noise_inputs: int
    n_time_inputs: int
    n_state_inputs: int
    
    # Time input activations (one per time input)
    time_activations: List[str]
    time_activation_params: Dict[str, float]  # β for tanh, γ for exp_decay
    
    # State normalization parameter
    state_alpha: float
    
    # === Edge transformations ===
    transform_probs: Dict[str, float]
    allowed_activations: List[str]
    prob_identity_activation: float  # Probability of using identity in NN
    n_categories: int
    tree_depth: int
    tree_max_features_fraction: float
    
    # === Noise settings ===
    noise_type: str
    noise_scale: float
    edge_noise_prob: float
    noise_only_at_t0: bool  # If True, noise only enters at t=0, then propagates as state
    init_sigma: float
    init_a: float
    
    # === Sample generation mode ===
    sample_mode: str  # 'iid', 'sliding_window', 'mixed'
    window_stride: int  # For sliding window mode
    n_sequences: int    # For mixed mode (how many T sequences)
    
    # === Target configuration ===
    is_classification: bool
    n_classes: int
    target_offset_type: str  # 'within', 'future_near', 'future_far', 'past'
    target_offset: int       # Actual offset value
    
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
        # First sample t_subseq (the 't' in n×m×t), then ensure T_total > t_subseq
        if prior.t_subseq_log_uniform:
            t_subseq = int(log_uniform(*prior.t_subseq_range))
        else:
            t_subseq = rng.integers(*prior.t_subseq_range)
        t_subseq = min(t_subseq, prior.max_t_subseq)
        t_subseq = max(t_subseq, 5)  # At least 5 timesteps
        
        # T_total must be larger than t_subseq (to allow target offsets)
        min_T = t_subseq + 10  # At least 10 extra for target offsets
        if prior.T_total_log_uniform:
            T_total = int(log_uniform(max(min_T, prior.T_total_range[0]), prior.T_total_range[1]))
        else:
            T_total = rng.integers(max(min_T, prior.T_total_range[0]), prior.T_total_range[1])
        T_total = min(T_total, prior.max_T_total)
        
        # === Sample feature count ===
        # Check if univariate
        if rng.random() < prior.prob_univariate:
            n_features = 1
        else:
            # Multivariate: use Beta distribution
            beta_sample = rng.beta(prior.n_features_beta_a, prior.n_features_beta_b)
            n_features = int(beta_sample * (prior.n_features_range[1] - prior.n_features_range[0]) 
                            + prior.n_features_range[0])
            n_features = max(2, min(n_features, prior.max_features))
        
        # === Sample sample count ===
        n_samples = rng.integers(*prior.n_samples_range)
        n_samples = min(n_samples, prior.max_samples)
        
        # === Sample graph structure ===
        if prior.n_nodes_log_uniform:
            n_nodes = int(log_uniform(*prior.n_nodes_range))
        else:
            n_nodes = rng.integers(*prior.n_nodes_range)
        # Ensure enough nodes for features + target + some latent
        n_nodes = max(n_nodes, n_features + 5)
        
        # === Enforce complexity limit ===
        # Iteratively reduce n_samples, T_total, n_nodes if complexity too high
        for _ in range(10):  # Max 10 iterations
            complexity = n_samples * T_total * n_nodes
            if complexity <= prior.max_complexity:
                break
            # Scale down proportionally
            scale = (prior.max_complexity / complexity) ** (1/3)
            n_samples = max(100, int(n_samples * scale))
            T_total = max(t_subseq + 10, int(T_total * scale))
            n_nodes = max(n_features + 5, int(n_nodes * scale))
        
        # Edge density (uniform in range)
        density = rng.uniform(*prior.density_range)
        
        # Disconnected subgraphs
        n_disconnected = 0
        if rng.random() < prior.prob_disconnected_subgraph:
            n_disconnected = rng.integers(*prior.n_disconnected_subgraphs_range)
        
        # === Sample root input types ===
        # Determine how many root nodes we'll have
        # Sample from n_roots_range but cap at max_roots_fraction of total nodes
        estimated_roots = rng.integers(*prior.n_roots_range)
        max_roots_by_fraction = max(3, int(n_nodes * prior.max_roots_fraction))
        estimated_roots = min(estimated_roots, max_roots_by_fraction)
        
        # === Distribute roots among time, state, and noise ===
        # Strategy: time and state are primary, noise is optional with low probability
        
        # Minimum slots for time and state
        min_time = prior.min_time_inputs
        min_state = prior.min_state_inputs
        
        # Remaining slots after minimums
        remaining = max(0, estimated_roots - min_time - min_state)
        
        # Distribute remaining between time and state
        # Sample time fraction from range (not fixed) for variability
        time_fraction = rng.uniform(0.15, 0.4)  # 15-40% time, rest state
        extra_time = int(remaining * time_fraction)
        extra_state = remaining - extra_time
        
        n_time = min_time + extra_time
        n_state = min_state + extra_state
        
        # Add noise inputs with low probability (capped)
        n_noise = 0
        if rng.random() < prior.prob_noise_input:
            n_noise = rng.integers(1, prior.max_noise_inputs + 1)
        
        # Ensure total doesn't exceed estimated_roots
        total_inputs = n_noise + n_time + n_state
        if total_inputs > estimated_roots:
            # Remove from noise first
            excess = total_inputs - estimated_roots
            if n_noise >= excess:
                n_noise -= excess
            else:
                excess -= n_noise
                n_noise = 0
                # Then from extra state
                n_state = max(min_state, n_state - excess)
        
        # === Decide if noise only at t=0 ===
        noise_only_at_t0 = (rng.random() < prior.prob_noise_only_at_t0)
        
        # === Sample time activation functions (NO REPETITION) ===
        time_activations = []
        time_activation_params = {}
        
        # First time input is ALWAYS 'linear'
        available_time_acts = [a for a in prior.time_activations if a != 'linear']
        time_activations.append('linear')
        
        # Additional time inputs sample from remaining activations WITHOUT repetition
        if n_time > 1:
            n_additional = min(n_time - 1, len(available_time_acts))
            additional_acts = list(rng.choice(available_time_acts, size=n_additional, replace=False))
            time_activations.extend(additional_acts)
        
        # Sample parameters for activations that need them
        for i, act in enumerate(time_activations):
            if act == 'tanh':
                time_activation_params[f'tanh_beta_{i}'] = log_uniform(*prior.tanh_beta_range)
            elif act == 'exp_decay':
                time_activation_params[f'exp_gamma_{i}'] = log_uniform(*prior.exp_gamma_range)
        
        # === Sample state alpha ===
        state_alpha = log_uniform(*prior.state_alpha_range)
        
        # === Sample edge transformation probabilities ===
        # Note: identity is now included as an activation in NN transform
        # passthrough = just copy one parent (preserves temporal correlation)
        transform_probs = {
            'nn': prior.prob_nn_transform,
            'tree': prior.prob_tree_transform,
            'discretization': prior.prob_discretization,
            'passthrough': prior.prob_passthrough
        }
        
        # Sample subset of activations to use for this dataset
        n_activations = rng.integers(3, len(prior.activations) + 1)
        allowed_activations = list(rng.choice(prior.activations, size=n_activations, replace=False))
        # Always include identity for potential linear transformations
        if 'identity' not in allowed_activations:
            allowed_activations.append('identity')
        
        # Discretization - number of categories
        n_categories = rng.integers(*prior.n_categories_range)
        # Limit by samples
        max_cats_for_samples = max(2, n_samples // prior.min_samples_per_class)
        n_categories = min(n_categories, max_cats_for_samples)
        
        # Tree parameters
        tree_depth = rng.integers(*prior.tree_depth_range)
        tree_max_features_fraction = prior.tree_max_features_fraction
        
        # === Noise ===
        noise_type = rng.choice(prior.noise_types)
        if prior.noise_scale_log_uniform:
            noise_scale = log_uniform(*prior.noise_scale_range)
        else:
            noise_scale = rng.uniform(*prior.noise_scale_range)
        
        init_sigma = rng.uniform(*prior.init_sigma_range)
        init_a = rng.uniform(*prior.init_a_range)
        
        # === Sample generation mode ===
        mode_probs = [prior.prob_iid_mode, prior.prob_sliding_window_mode, prior.prob_mixed_mode]
        mode_probs = np.array(mode_probs) / sum(mode_probs)
        sample_mode = rng.choice(['iid', 'sliding_window', 'mixed'], p=mode_probs)
        
        # For IID and mixed modes: T_total should be close to t_subseq
        # Because we reset state for each sample, we only need burn-in (10-50 steps)
        if sample_mode in ['iid', 'mixed']:
            burn_in = rng.integers(10, 50)
            T_total = t_subseq + burn_in
        
        window_stride = rng.integers(*prior.window_stride_range) if sample_mode != 'iid' else 1
        n_sequences = rng.integers(2, 10) if sample_mode == 'mixed' else 1
        
        # === Target configuration ===
        # Check if task type is forced
        if prior.force_classification is not None:
            is_classification = prior.force_classification
        else:
            is_classification = rng.random() < prior.prob_classification
        
        if is_classification:
            # Limit n_classes to ensure min_samples_per_class
            max_classes_for_samples = max(2, n_samples // prior.min_samples_per_class)
            max_classes = min(prior.max_classes, max_classes_for_samples)
            n_classes = rng.integers(2, max(3, max_classes + 1))
        else:
            n_classes = 0
        
        # Target offset
        offset_types = list(prior.target_offset_probs.keys())
        offset_probs = list(prior.target_offset_probs.values())
        offset_probs = np.array(offset_probs) / sum(offset_probs)
        target_offset_type = rng.choice(offset_types, p=offset_probs)
        
        # Sample actual offset based on type
        if target_offset_type == 'within':
            target_offset = 0
        elif target_offset_type == 'future_near':
            target_offset = rng.integers(1, min(6, T_total - t_subseq))
        elif target_offset_type == 'future_far':
            max_far = min(20, T_total - t_subseq)
            if max_far > 6:
                target_offset = rng.integers(6, max_far)
            else:
                target_offset = rng.integers(1, max(2, max_far))
        else:  # past
            target_offset = -rng.integers(1, min(5, t_subseq))
        
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
            n_noise_inputs=n_noise,
            n_time_inputs=n_time,
            n_state_inputs=n_state,
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
            noise_only_at_t0=noise_only_at_t0,
            init_sigma=init_sigma,
            init_a=init_a,
            sample_mode=sample_mode,
            window_stride=window_stride,
            n_sequences=n_sequences,
            is_classification=is_classification,
            n_classes=n_classes,
            target_offset_type=target_offset_type,
            target_offset=target_offset,
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
            'n_noise_inputs': self.n_noise_inputs,
            'n_time_inputs': self.n_time_inputs,
            'n_state_inputs': self.n_state_inputs,
            'time_activations': self.time_activations,
            'state_alpha': self.state_alpha,
            'sample_mode': self.sample_mode,
            'is_classification': self.is_classification,
            'n_classes': self.n_classes,
            'target_offset_type': self.target_offset_type,
            'target_offset': self.target_offset,
            'train_ratio': self.train_ratio,
            'seed': self.seed
        }

