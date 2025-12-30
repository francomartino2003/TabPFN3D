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
    # Number of rows (samples) range
    n_rows_range: Tuple[int, int] = (50, 10000)
    n_rows_log_uniform: bool = True  # If True, sample log-uniformly
    
    # Number of features (columns) range
    n_features_range: Tuple[int, int] = (2, 100)
    n_features_log_uniform: bool = True
    
    # === Train/Test Split parameters ===
    # Based on real dataset distributions (UCR/UEA archive analysis)
    # Most real datasets have train ratios between 0.2 and 0.8
    # with a peak around 0.5-0.7
    train_ratio_range: Tuple[float, float] = (0.2, 0.8)
    train_ratio_beta_a: float = 2.0  # Beta distribution shape param a
    train_ratio_beta_b: float = 2.0  # Beta distribution shape param b (symmetric around 0.5)
    
    # === Graph structure parameters ===
    # Number of nodes in the DAG (log-uniform as per paper)
    n_nodes_range: Tuple[int, int] = (5, 200)
    n_nodes_log_uniform: bool = True
    
    # Graph density parameter (Gamma distribution as per paper)
    # Higher values = denser graphs
    density_gamma_shape: float = 2.0
    density_gamma_scale: float = 1.0
    
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
    
    # Available activation functions (as per paper)
    activations: List[str] = field(default_factory=lambda: [
        'identity', 'log', 'sigmoid', 'tanh', 'sin', 'cos',
        'abs', 'square', 'cube', 'sqrt', 'relu', 'softplus',
        'step', 'mod', 'rank', 'exp_neg', 'gaussian'
    ])
    
    # Discretization parameters
    n_categories_range: Tuple[int, int] = (2, 10)
    
    # Decision tree parameters
    tree_depth_range: Tuple[int, int] = (1, 5)
    tree_n_splits_range: Tuple[int, int] = (1, 10)
    
    # === Noise parameters ===
    noise_types: List[str] = field(default_factory=lambda: [
        'normal', 'uniform', 'laplace', 'mixture'
    ])
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
    density: float
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
        
        # Sample size
        if prior.n_rows_log_uniform:
            n_rows = int(log_uniform(*prior.n_rows_range))
        else:
            n_rows = rng.integers(*prior.n_rows_range)
            
        if prior.n_features_log_uniform:
            n_features = int(log_uniform(*prior.n_features_range))
        else:
            n_features = rng.integers(*prior.n_features_range)
        
        # Sample graph structure
        if prior.n_nodes_log_uniform:
            n_nodes = int(log_uniform(*prior.n_nodes_range))
        else:
            n_nodes = rng.integers(*prior.n_nodes_range)
        
        # Ensure we have at least n_features + 1 nodes (for features + target)
        n_nodes = max(n_nodes, n_features + 2)
        
        density = rng.gamma(prior.density_gamma_shape, prior.density_gamma_scale)
        
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
        
        # Discretization
        n_categories = rng.integers(*prior.n_categories_range)
        
        # Tree parameters
        tree_depth = rng.integers(*prior.tree_depth_range)
        tree_n_splits = rng.integers(*prior.tree_n_splits_range)
        
        # Noise
        noise_type = rng.choice(prior.noise_types)
        if prior.noise_scale_log_uniform:
            noise_scale = log_uniform(*prior.noise_scale_range)
        else:
            noise_scale = rng.uniform(*prior.noise_scale_range)
        
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
        n_classes = rng.integers(2, prior.max_classes + 1) if is_classification else 0
        
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
            density=density,
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

