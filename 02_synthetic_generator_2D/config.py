"""
Configuration and hyperparameters for the synthetic dataset generator.

This module defines the "prior" - all the hyperparameters that control
how datasets are generated. These should be varied aggressively during
training to cover a huge family of different generative processes.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


@dataclass
class PriorConfig:
    """
    Prior configuration for dataset generation.
    
    These are the hyperparameters of the generative process itself.
    They control the distribution from which individual datasets are sampled.
    
    All ranges and probabilities here should be experimented with during training.
    """
    
    # === Size parameters ===
    # Number of rows (samples) range - sampled uniformly up to 2048 per paper
    n_rows_range: Tuple[int, int] = (50, 10000)
    n_rows_log_uniform: bool = False  # Paper samples uniformly, not log-uniform
    
    # Number of features - Beta(0.95, 8.0) scaled to [1, 160] per paper
    # "We sample the number of features using a beta distribution (k=0.95, b=8.0) 
    # that we linearly scale to the range 1-160"
    n_features_beta_a: float = 0.95
    n_features_beta_b: float = 8.0
    n_features_range: Tuple[int, int] = (1, 160)
    
    # Maximum total cells per table (paper: 75,000)
    max_cells: int = 75000
    
    # === Train/Test Split parameters ===
    # Based on real dataset distributions (UCR/UEA archive analysis)
    # Most real datasets have train ratios between 0.2 and 0.8
    # with a peak around 0.5-0.7
    train_ratio_range: Tuple[float, float] = (0.2, 0.8)
    train_ratio_beta_a: float = 2.0  # Beta distribution shape param a
    train_ratio_beta_b: float = 2.0  # Beta distribution shape param b (symmetric around 0.5)
    
    # === Graph structure parameters ===
    # Number of nodes in the DAG (log-uniform as per paper)
    # Larger graphs = more layers of transformation = more complex relationships
    # More nodes = more latent variables = deeper causal structure
    n_nodes_range: Tuple[int, int] = (12, 250)
    n_nodes_log_uniform: bool = True
    
    # Edge density (0.0 = minimal tree, 1.0 = complete DAG)
    # Controls how many edges beyond the minimum (n-1) are added
    density_range: Tuple[float, float] = (0.01, 0.8)
    
    # Number of root/input nodes (nodes without parents)
    n_roots_range: Tuple[int, int] = (3, 15)
    
    # Probability of creating disconnected subgraphs (for irrelevant features)
    prob_disconnected_subgraph: float = 0.3
    n_disconnected_subgraphs_range: Tuple[int, int] = (1, 5)
    
    # === Transformation parameters ===
    # Probabilities for each node transformation type (applied per child node)
    prob_nn_transform: float = 0.5  # Neural network-like transformation
    prob_tree_transform: float = 0.2  # Decision tree-like transformation
    prob_discretization: float = 0.3  # Discretization (categorical)
    
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
    # Number of categories/prototypes
    n_categories_range: Tuple[int, int] = (2, 10)
    
    # Decision tree parameters
    tree_depth_range: Tuple[int, int] = (1, 5)
    tree_max_features_fraction: float = 0.7  # Max fraction of parents to use as features
    
    # === Noise parameters ===
    # All transformations add Gaussian noise N(0, noise_scale^2)
    noise_scale_range: Tuple[float, float] = (0.01, 1.0)
    noise_scale_log_uniform: bool = True
    
    # Initialization noise for root nodes
    # Normal: ε ~ N(0, σε²)
    init_sigma_range: Tuple[float, float] = (0.5, 2.0)
    # Uniform: ε ~ U(−a, a)
    init_a_range: Tuple[float, float] = (1.0, 3.0)
    
    # Noise types for root initialization
    noise_types: List[str] = field(default_factory=lambda: [
        'normal', 'uniform', 'mixed'
    ])
    
    # === Row dependency parameters ===
    # Some datasets have dependencies between rows (prototypes/clusters)
    prob_row_dependency: float = 0.3
    n_prototypes_range: Tuple[int, int] = (2, 20)
    prototype_noise_scale: float = 0.1
    
    # === Post-processing parameters ===
    # Warping (non-linear distortions)
    prob_warping: float = 0.5
    warping_intensity_range: Tuple[float, float] = (0.1, 2.0)
    
    # Quantization (binning continuous values)
    prob_quantization: float = 0.3
    n_quantization_bins_range: Tuple[int, int] = (2, 50)
    
    # Missing values
    prob_missing_values: float = 0.3
    missing_rate_range: Tuple[float, float] = (0.01, 0.3)
    
    # === Target parameters ===
    # Maximum number of classes for classification
    max_classes: int = 10
    
    # Minimum samples per class (to ensure balanced enough datasets)
    min_samples_per_class: int = 10
    
    # Probability of classification vs regression
    prob_classification: float = 0.5
    
    def sample_hyperparams(self, rng: Optional[np.random.Generator] = None) -> 'DatasetConfig':
        """
        Sample a specific dataset configuration from this prior.
        
        Args:
            rng: Random number generator (for reproducibility)
            
        Returns:
            DatasetConfig with specific values sampled from this prior
        """
        if rng is None:
            rng = np.random.default_rng()
            
        return DatasetConfig.sample_from_prior(self, rng)


@dataclass 
class DatasetConfig:
    """
    Configuration for a specific dataset instance.
    
    This is sampled from a PriorConfig and contains concrete values
    for generating one particular dataset.
    """
    
    # Size
    n_rows: int
    n_features: int
    
    # Graph structure
    n_nodes: int
    density: float  # Edge density (0.0 = tree, 1.0 = complete DAG)
    n_roots: int  # Number of input/root nodes
    n_disconnected_subgraphs: int
    
    # Transformation settings
    transform_probs: Dict[str, float]
    allowed_activations: List[str]
    n_categories: int
    tree_depth: int
    tree_max_features_fraction: float
    
    # Noise settings
    noise_type: str
    noise_scale: float
    init_sigma: float  # σε for initialization noise N(0, σε²)
    init_a: float      # a for initialization noise U(-a, a)
    
    # Row dependency
    has_row_dependency: bool
    n_prototypes: int
    prototype_noise_scale: float
    
    # Post-processing
    apply_warping: bool
    warping_intensity: float
    apply_quantization: bool
    n_quantization_bins: int
    apply_missing: bool
    missing_rate: float
    
    # Target
    is_classification: bool
    n_classes: int
    
    # Train/Test split
    train_ratio: float  # Fraction of data for training (0.2 - 0.8)
    
    # Random seed for reproducibility
    seed: Optional[int] = None
    
    @classmethod
    def sample_from_prior(cls, prior: PriorConfig, rng: np.random.Generator) -> 'DatasetConfig':
        """
        Sample a dataset configuration from a prior.
        
        Args:
            prior: The prior configuration to sample from
            rng: Random number generator
            
        Returns:
            A DatasetConfig instance with sampled values
        """
        
        def log_uniform(low: float, high: float) -> float:
            """Sample from log-uniform distribution."""
            return np.exp(rng.uniform(np.log(low), np.log(high)))
        
        # Sample number of features using Beta distribution (per paper)
        # Beta(0.95, 8.0) scaled to [1, 160]
        beta_sample = rng.beta(prior.n_features_beta_a, prior.n_features_beta_b)
        n_features = int(beta_sample * (prior.n_features_range[1] - prior.n_features_range[0]) 
                        + prior.n_features_range[0])
        n_features = max(1, n_features)
        
        # Sample number of rows uniformly (per paper)
        if prior.n_rows_log_uniform:
            n_rows = int(log_uniform(*prior.n_rows_range))
        else:
            n_rows = rng.integers(prior.n_rows_range[0], prior.n_rows_range[1] + 1)
        
        # Enforce max_cells constraint (paper: 75,000 cells)
        if n_rows * n_features > prior.max_cells:
            n_rows = prior.max_cells // n_features
            n_rows = max(prior.n_rows_range[0], n_rows)
        
        # Sample graph structure
        if prior.n_nodes_log_uniform:
            n_nodes = int(log_uniform(*prior.n_nodes_range))
        else:
            n_nodes = rng.integers(*prior.n_nodes_range)
        
        # Ensure we have at least n_features + 1 nodes (for features + target)
        n_nodes = max(n_nodes, n_features + 2)
        
        # Sample edge density (uniform in range)
        density = rng.uniform(*prior.density_range)
        
        # Sample number of root nodes (from fixed range)
        n_roots = rng.integers(*prior.n_roots_range)
        n_roots = min(n_roots, n_nodes // 2)  # At most half the nodes are roots
        n_roots = max(1, n_roots)  # At least one root
        
        # Disconnected subgraphs
        has_disconnected = rng.random() < prior.prob_disconnected_subgraph
        n_disconnected = rng.integers(*prior.n_disconnected_subgraphs_range) if has_disconnected else 0
        
        # Transformation probabilities (normalized)
        transform_probs = {
            'nn': prior.prob_nn_transform,
            'tree': prior.prob_tree_transform,
            'discretization': prior.prob_discretization
        }
        
        # Sample subset of activations to use for this dataset
        n_activations = rng.integers(3, len(prior.activations) + 1)
        allowed_activations = list(rng.choice(prior.activations, size=n_activations, replace=False))
        # Always include identity for potential linear transformations
        if 'identity' not in allowed_activations:
            allowed_activations.append('identity')
        
        # Discretization - number of categories
        n_categories = rng.integers(*prior.n_categories_range)
        # Ensure at least 10 samples per category
        max_categories_for_samples = max(2, n_rows // 10)
        n_categories = min(n_categories, max_categories_for_samples)
        
        # Tree parameters
        tree_depth = rng.integers(*prior.tree_depth_range)
        tree_max_features_fraction = prior.tree_max_features_fraction
        
        # Noise
        noise_type = rng.choice(prior.noise_types)
        if prior.noise_scale_log_uniform:
            noise_scale = log_uniform(*prior.noise_scale_range)
        else:
            noise_scale = rng.uniform(*prior.noise_scale_range)
        
        # Initialization noise hyperparameters
        init_sigma = rng.uniform(*prior.init_sigma_range)
        init_a = rng.uniform(*prior.init_a_range)
        
        # Row dependency
        has_row_dependency = rng.random() < prior.prob_row_dependency
        n_prototypes = rng.integers(*prior.n_prototypes_range) if has_row_dependency else 0
        
        # Post-processing
        apply_warping = rng.random() < prior.prob_warping
        warping_intensity = rng.uniform(*prior.warping_intensity_range)
        
        apply_quantization = rng.random() < prior.prob_quantization
        n_quantization_bins = rng.integers(*prior.n_quantization_bins_range)
        
        apply_missing = rng.random() < prior.prob_missing_values
        missing_rate = rng.uniform(*prior.missing_rate_range)
        
        # Target type
        is_classification = rng.random() < prior.prob_classification
        if is_classification:
            # Limit n_classes to ensure min_samples_per_class
            max_classes_for_samples = max(2, n_rows // prior.min_samples_per_class)
            max_classes = min(prior.max_classes, max_classes_for_samples)
            n_classes = rng.integers(2, max_classes + 1)
        else:
            n_classes = 0
        
        # Train/Test split ratio
        # Use Beta distribution scaled to train_ratio_range for realistic distribution
        beta_sample = rng.beta(prior.train_ratio_beta_a, prior.train_ratio_beta_b)
        train_ratio = prior.train_ratio_range[0] + beta_sample * (
            prior.train_ratio_range[1] - prior.train_ratio_range[0]
        )
        
        return cls(
            n_rows=n_rows,
            n_features=n_features,
            n_nodes=n_nodes,
            density=density,
            n_roots=n_roots,
            n_disconnected_subgraphs=n_disconnected,
            transform_probs=transform_probs,
            allowed_activations=allowed_activations,
            n_categories=n_categories,
            tree_depth=tree_depth,
            tree_max_features_fraction=tree_max_features_fraction,
            noise_type=noise_type,
            noise_scale=noise_scale,
            init_sigma=init_sigma,
            init_a=init_a,
            has_row_dependency=has_row_dependency,
            n_prototypes=n_prototypes,
            prototype_noise_scale=prior.prototype_noise_scale,
            apply_warping=apply_warping,
            warping_intensity=warping_intensity,
            apply_quantization=apply_quantization,
            n_quantization_bins=n_quantization_bins,
            apply_missing=apply_missing,
            missing_rate=missing_rate,
            is_classification=is_classification,
            n_classes=n_classes,
            train_ratio=train_ratio,
            seed=int(rng.integers(0, 2**31))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'n_rows': self.n_rows,
            'n_features': self.n_features,
            'n_nodes': self.n_nodes,
            'density': self.density,
            'n_roots': self.n_roots,
            'n_disconnected_subgraphs': self.n_disconnected_subgraphs,
            'transform_probs': self.transform_probs,
            'allowed_activations': self.allowed_activations,
            'n_categories': self.n_categories,
            'tree_depth': self.tree_depth,
            'tree_max_features_fraction': self.tree_max_features_fraction,
            'noise_type': self.noise_type,
            'noise_scale': self.noise_scale,
            'init_sigma': self.init_sigma,
            'init_a': self.init_a,
            'has_row_dependency': self.has_row_dependency,
            'n_prototypes': self.n_prototypes,
            'prototype_noise_scale': self.prototype_noise_scale,
            'apply_warping': self.apply_warping,
            'warping_intensity': self.warping_intensity,
            'apply_quantization': self.apply_quantization,
            'n_quantization_bins': self.n_quantization_bins,
            'apply_missing': self.apply_missing,
            'missing_rate': self.missing_rate,
            'is_classification': self.is_classification,
            'n_classes': self.n_classes,
            'train_ratio': self.train_ratio,
            'seed': self.seed
        }
