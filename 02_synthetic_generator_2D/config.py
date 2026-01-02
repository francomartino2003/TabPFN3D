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
    n_rows_range: Tuple[int, int] = (50, 2048)
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
    n_nodes_range: Tuple[int, int] = (50, 600)
    n_nodes_log_uniform: bool = True
    
    # Redirection probability P (Gamma distribution as per paper)
    # "The redirection probability P is sampled from a gamma distribution, P ~ Γ(α, β)"
    # Smaller P = denser graphs with more edges on average
    redirection_gamma_shape: float = 2.0  # α parameter
    redirection_gamma_rate: float = 5.0   # β parameter
    
    # Probability of creating disconnected subgraphs (for irrelevant features)
    prob_disconnected_subgraph: float = 0.3
    n_disconnected_subgraphs_range: Tuple[int, int] = (1, 5)
    
    # === Transformation parameters ===
    # Probabilities for each edge transformation type
    prob_nn_transform: float = 0.5  # Neural network-like transformation
    prob_tree_transform: float = 0.2  # Decision tree-like transformation
    prob_discretization: float = 0.2  # Discretization (categorical)
    prob_identity: float = 0.1  # Identity with noise
    
    # Neural network transformation parameters
    nn_hidden_range: Tuple[int, int] = (1, 5)  # Number of hidden "layers"
    nn_width_range: Tuple[int, int] = (1, 10)  # Width of hidden layers
    
    # Available activation functions (per paper):
    # "identity, logarithm, sigmoid, absolute value, sine, hyperbolic tangent, 
    # rank operation, squaring, power functions, smooth ReLU, step function 
    # and modulo operation"
    activations: List[str] = field(default_factory=lambda: [
        'identity', 'log', 'sigmoid', 'tanh', 'sin',
        'abs', 'square', 'power', 'softplus', 'step', 'mod', 'rank'
    ])
    
    # Discretization parameters
    # "We sample the number of categories K from a rounded gamma distribution 
    # with an offset of 2 to yield a minimum number of classes of 2"
    n_categories_gamma_shape: float = 2.0
    n_categories_gamma_scale: float = 2.0
    n_categories_min: int = 2
    n_categories_max: int = 10  # Paper limits to 10 classes
    
    # Decision tree parameters
    tree_depth_range: Tuple[int, int] = (1, 5)
    tree_n_splits_range: Tuple[int, int] = (1, 10)
    
    # === Noise parameters ===
    # Per paper: "1. Normal  2. Uniform  3. Mixed" (no laplace)
    # "Mixed: for each root node, we randomly select either normal or uniform"
    noise_types: List[str] = field(default_factory=lambda: [
        'normal', 'uniform', 'mixed'
    ])
    
    # Initialization noise parameters (per paper these are hyperparameters)
    # σε for Normal: ε ~ N(0, σε²)
    init_sigma_range: Tuple[float, float] = (0.5, 2.0)
    # a for Uniform: ε ~ U(−a, a)
    init_a_range: Tuple[float, float] = (1.0, 3.0)
    
    # Edge noise scale (Gaussian noise added at each edge)
    noise_scale_range: Tuple[float, float] = (0.01, 1.0)
    noise_scale_log_uniform: bool = True
    
    # Probability of adding noise to each edge
    prob_edge_noise: float = 0.8
    
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
    redirection_prob: float  # P in the paper - smaller = denser graphs
    n_disconnected_subgraphs: int
    
    # Transformation settings
    edge_transform_probs: Dict[str, float]
    nn_hidden: int
    nn_width: int
    allowed_activations: List[str]
    n_categories: int
    tree_depth: int
    tree_n_splits: int
    
    # Noise settings
    noise_type: str
    noise_scale: float
    edge_noise_prob: float
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
        
        # Redirection probability P ~ Gamma(α, β) where β is rate (not scale)
        # Using scale = 1/rate for numpy's gamma (which uses scale)
        redirection_prob = rng.gamma(prior.redirection_gamma_shape, 
                                      1.0 / prior.redirection_gamma_rate)
        # Clip to [0, 1] since it's a probability
        redirection_prob = min(1.0, redirection_prob)
        
        # Disconnected subgraphs
        has_disconnected = rng.random() < prior.prob_disconnected_subgraph
        n_disconnected = rng.integers(*prior.n_disconnected_subgraphs_range) if has_disconnected else 0
        
        # Transformation probabilities (could be varied per-dataset)
        edge_transform_probs = {
            'nn': prior.prob_nn_transform,
            'tree': prior.prob_tree_transform,
            'discretization': prior.prob_discretization,
            'identity': prior.prob_identity
        }
        
        # Sample subset of activations to use for this dataset
        n_activations = rng.integers(3, len(prior.activations) + 1)
        allowed_activations = list(rng.choice(prior.activations, size=n_activations, replace=False))
        
        # NN parameters
        nn_hidden = rng.integers(*prior.nn_hidden_range)
        nn_width = rng.integers(*prior.nn_width_range)
        
        # Discretization - Gamma distribution with offset of 2 (per paper)
        # But also limit based on n_rows to ensure at least 10 samples per category
        n_categories = int(rng.gamma(prior.n_categories_gamma_shape, 
                                      prior.n_categories_gamma_scale)) + prior.n_categories_min
        n_categories = min(n_categories, prior.n_categories_max)
        max_categories_for_samples = max(2, n_rows // 10)  # At least 10 samples per category
        n_categories = min(n_categories, max_categories_for_samples)
        
        # Tree parameters
        tree_depth = rng.integers(*prior.tree_depth_range)
        tree_n_splits = rng.integers(*prior.tree_n_splits_range)
        
        # Noise
        noise_type = rng.choice(prior.noise_types)
        if prior.noise_scale_log_uniform:
            noise_scale = log_uniform(*prior.noise_scale_range)
        else:
            noise_scale = rng.uniform(*prior.noise_scale_range)
        
        # Initialization noise hyperparameters (per paper)
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
            # Limit n_classes to ensure at least 10 samples per class
            max_classes_for_samples = max(2, n_rows // 10)
            max_classes = min(prior.max_classes, max_classes_for_samples)
            n_classes = rng.integers(2, max_classes + 1)
        else:
            n_classes = 0
        
        # Train/Test split ratio
        # Use Beta distribution scaled to train_ratio_range for realistic distribution
        # Beta(2,2) gives symmetric distribution peaked at 0.5
        beta_sample = rng.beta(prior.train_ratio_beta_a, prior.train_ratio_beta_b)
        train_ratio = prior.train_ratio_range[0] + beta_sample * (
            prior.train_ratio_range[1] - prior.train_ratio_range[0]
        )
        
        return cls(
            n_rows=n_rows,
            n_features=n_features,
            n_nodes=n_nodes,
            redirection_prob=redirection_prob,
            n_disconnected_subgraphs=n_disconnected,
            edge_transform_probs=edge_transform_probs,
            nn_hidden=nn_hidden,
            nn_width=nn_width,
            allowed_activations=allowed_activations,
            n_categories=n_categories,
            tree_depth=tree_depth,
            tree_n_splits=tree_n_splits,
            noise_type=noise_type,
            noise_scale=noise_scale,
            edge_noise_prob=prior.prob_edge_noise,
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
            'redirection_prob': self.redirection_prob,
            'n_disconnected_subgraphs': self.n_disconnected_subgraphs,
            'edge_transform_probs': self.edge_transform_probs,
            'nn_hidden': self.nn_hidden,
            'nn_width': self.nn_width,
            'allowed_activations': self.allowed_activations,
            'n_categories': self.n_categories,
            'tree_depth': self.tree_depth,
            'tree_n_splits': self.tree_n_splits,
            'noise_type': self.noise_type,
            'noise_scale': self.noise_scale,
            'edge_noise_prob': self.edge_noise_prob,
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

