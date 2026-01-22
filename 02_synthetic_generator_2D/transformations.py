"""
Edge transformations for synthetic dataset generation.

This module implements the different types of transformations that can
be applied to compute child node values from parent node values:

1. NN Transformation: Linear combination with Xavier-initialized weights + bias + activation
2. Decision Tree: Select subset of parent features, apply threshold rules
3. Discretization: Distance to prototypes, output normalized category index

All transformations add Gaussian noise N(0, noise_scale²) at the end.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Callable, Dict, Any, Tuple
import numpy as np

try:
    from .config import DatasetConfig
except ImportError:
    from config import DatasetConfig


class Activation:
    """Collection of activation functions."""
    
    @staticmethod
    def identity(x: np.ndarray) -> np.ndarray:
        """Linear/identity activation."""
        return x
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def softplus(x: np.ndarray) -> np.ndarray:
        return np.log1p(np.exp(np.clip(x, -500, 500)))
    
    @staticmethod
    def sin(x: np.ndarray) -> np.ndarray:
        return np.sin(x)
    
    @staticmethod
    def cos(x: np.ndarray) -> np.ndarray:
        return np.cos(x)
    
    @staticmethod
    def abs(x: np.ndarray) -> np.ndarray:
        return np.abs(x)
    
    @staticmethod
    def square(x: np.ndarray) -> np.ndarray:
        return x ** 2
    
    @staticmethod
    def log(x: np.ndarray) -> np.ndarray:
        # Safe log: log(|x| + eps)
        return np.log(np.abs(x) + 1e-6)
    
    @staticmethod
    def step(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)
    
    @staticmethod
    def leaky_relu(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, 0.01 * x)
    
    @staticmethod
    def elu(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, np.exp(x) - 1)
    
    @staticmethod
    def rank(x: np.ndarray) -> np.ndarray:
        """Rank operation: convert to percentile ranks [0, 1]."""
        if x.ndim == 0:
            return np.array(0.5)
        if len(x) <= 1:
            return np.zeros_like(x) + 0.5
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(x))
        return ranks / (len(x) - 1 + 1e-10)
    
    @staticmethod
    def power(x: np.ndarray) -> np.ndarray:
        """Power function: sign(x) * |x|^p with random p in [0.5, 3]."""
        p = np.random.uniform(0.5, 3.0)
        return np.sign(x) * np.power(np.abs(x) + 1e-10, p)
    
    @staticmethod
    def mod(x: np.ndarray) -> np.ndarray:
        """Modulo operation with period ~2."""
        return np.mod(x, 2.0)
    
    @classmethod
    def get(cls, name: str) -> Callable[[np.ndarray], np.ndarray]:
        """Get activation function by name."""
        activations = {
            'identity': cls.identity,
            'relu': cls.relu,
            'tanh': cls.tanh,
            'sigmoid': cls.sigmoid,
            'softplus': cls.softplus,
            'sin': cls.sin,
            'cos': cls.cos,
            'abs': cls.abs,
            'square': cls.square,
            'log': cls.log,
            'step': cls.step,
            'leaky_relu': cls.leaky_relu,
            'elu': cls.elu,
            'rank': cls.rank,
            'power': cls.power,
            'mod': cls.mod,
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")
        return activations[name]


class EdgeTransformation(ABC):
    """Base class for edge transformations."""
    
    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Apply transformation to inputs.
        
        Args:
            inputs: Array of shape (n_samples, n_parents) containing values from parent nodes
                   
        Returns:
            Array of shape (n_samples,) containing output values
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get transformation parameters for serialization."""
        pass


@dataclass
class NNTransformation(EdgeTransformation):
    """
    Neural network-like transformation.
    
    For each parent, applies: weight * parent_value
    Then sums all weighted inputs, adds bias, and applies activation.
    Finally adds Gaussian noise.
    
    Structure:
    1. For each parent i: w_i * x_i (Xavier-initialized weights)
    2. Sum + bias: sum(w_i * x_i) + b
    3. Apply activation function (can be identity for linear)
    4. Add noise ~ N(0, noise_scale²)
    """
    
    weights: np.ndarray  # Shape (n_parents,) - one weight per parent
    bias: float
    activation: str  # Activation function name (including 'identity' for linear)
    noise_scale: float
    rng: np.random.Generator
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Apply NN transformation."""
        # Ensure 2D input: (n_samples, n_parents)
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        
        # Sanitize inputs - wider bounds to allow diversity
        # The temporal propagator handles cross-timestep clipping
        inputs = np.clip(inputs, -1e6, 1e6)
        inputs = np.nan_to_num(inputs, nan=0.0, posinf=0, neginf=0)
        
        n_samples = inputs.shape[0]
        n_parents = inputs.shape[1]
        
        # Handle dimension mismatch
        if len(self.weights) != n_parents:
            # Pad or truncate weights
            if len(self.weights) < n_parents:
                weights = np.zeros(n_parents)
                weights[:len(self.weights)] = self.weights
            else:
                weights = self.weights[:n_parents]
        else:
            weights = self.weights
        
        # Linear combination: sum(w_i * x_i) + bias
        # Shape: (n_samples,)
        output = inputs @ weights + self.bias
        
        # Apply activation
        act_fn = Activation.get(self.activation)
        output = act_fn(output)
        
        # Add Gaussian noise N(0, noise_scale²)
        if self.noise_scale > 0:
            output = output + self.rng.normal(0, self.noise_scale, size=output.shape)
        
        # Sanitize output - handle NaN/Inf from overflow
        output = np.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return output.squeeze()
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'type': 'nn',
            'n_parents': len(self.weights),
            'activation': self.activation,
            'noise_scale': self.noise_scale
        }


@dataclass
class TreeTransformation(EdgeTransformation):
    """
    Decision tree-like transformation.
    
    Selects a subset of parent features and applies threshold rules:
    - At each internal node: if parent_feature > threshold, go right, else left
    - Each leaf has an associated output value
    
    Parameters per split are randomly sampled:
    - feature_index: which parent to use (from subset)
    - threshold: random threshold value
    
    Finally adds Gaussian noise.
    """
    
    # Tree structure (binary tree stored as arrays)
    feature_indices: np.ndarray  # Which parent feature to split on at each internal node
    thresholds: np.ndarray       # Threshold at each internal node
    left_children: np.ndarray    # Index of left child (-1 for leaf)
    right_children: np.ndarray   # Index of right child (-1 for leaf)
    leaf_values: np.ndarray      # Output value at each node (used at leaves)
    noise_scale: float
    rng: np.random.Generator
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Apply tree transformation."""
        # Ensure 2D input
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        
        # Sanitize inputs
        inputs = np.nan_to_num(inputs, nan=0.0, posinf=1e6, neginf=-1e6)
        
        n_samples = inputs.shape[0]
        n_parents = inputs.shape[1]
        outputs = np.zeros(n_samples)
        
        for i in range(n_samples):
            node = 0  # Start at root
            while True:
                if self.left_children[node] == -1:  # Leaf node
                    outputs[i] = self.leaf_values[node]
                    break
                
                # Get feature index for this split
                feat_idx = self.feature_indices[node]
                # Ensure valid index
                feat_idx = feat_idx % n_parents if n_parents > 0 else 0
                
                # Get feature value
                feat_val = inputs[i, feat_idx]
                
                # Apply threshold rule
                if feat_val > self.thresholds[node]:
                    node = self.right_children[node]
                else:
                    node = self.left_children[node]
        
        # Add Gaussian noise N(0, noise_scale²)
        if self.noise_scale > 0:
            outputs = outputs + self.rng.normal(0, self.noise_scale, size=outputs.shape)
        
        # Sanitize output
        outputs = np.nan_to_num(outputs, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return outputs.squeeze()
    
    def get_params(self) -> Dict[str, Any]:
        n_internal = np.sum(self.left_children != -1)
        depth = int(np.ceil(np.log2(n_internal + 2))) if n_internal > 0 else 1
        return {
            'type': 'tree',
            'depth': depth,
            'n_splits': n_internal,
            'noise_scale': self.noise_scale
        }


@dataclass
class DiscretizationTransformation(EdgeTransformation):
    """
    Discretization transformation for categorical features.
    
    1. Receives all parent values as a vector
    2. Computes Euclidean distance to K randomly sampled prototype vectors
    3. Assigns to nearest prototype (category index)
    4. Normalizes output: category_index / n_categories (for further use in graph)
    5. Adds Gaussian noise
    """
    
    prototypes: np.ndarray  # Shape (n_categories, n_parents)
    n_categories: int
    noise_scale: float
    rng: np.random.Generator
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Apply discretization transformation."""
        # Ensure 2D input
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        
        # Sanitize inputs
        inputs = np.nan_to_num(inputs, nan=0.0, posinf=1e6, neginf=-1e6)
        
        n_samples = inputs.shape[0]
        n_parents = inputs.shape[1]
        
        # Handle dimension mismatch between prototypes and inputs
        proto_dim = self.prototypes.shape[1]
        if proto_dim != n_parents:
            # Adjust prototypes or inputs
            if proto_dim < n_parents:
                # Use only first proto_dim features
                inputs_adj = inputs[:, :proto_dim]
            else:
                # Pad inputs with zeros
                inputs_adj = np.zeros((n_samples, proto_dim))
                inputs_adj[:, :n_parents] = inputs
        else:
            inputs_adj = inputs
        
        # Compute distance to each prototype
        # Shape: (n_samples, n_categories)
        distances = np.zeros((n_samples, self.n_categories))
        for i, proto in enumerate(self.prototypes):
            diff = inputs_adj - proto
            distances[:, i] = np.linalg.norm(diff, axis=1)
        
        # Assign to nearest prototype (category index)
        category_indices = np.argmin(distances, axis=1)
        
        # Normalize: divide by n_categories for use in computational graph
        # Output is in range [0, 1)
        output = category_indices.astype(float) / self.n_categories
        
        # Add Gaussian noise N(0, noise_scale²)
        if self.noise_scale > 0:
            output = output + self.rng.normal(0, self.noise_scale, size=output.shape)
        
        return output.squeeze()
    
    def get_category_indices(self, inputs: np.ndarray) -> np.ndarray:
        """
        Get the raw categorical indices (for use as observed categorical features).
        Does NOT normalize or add noise.
        """
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        
        n_samples = inputs.shape[0]
        n_parents = inputs.shape[1]
        
        # Handle dimension mismatch
        proto_dim = self.prototypes.shape[1]
        if proto_dim != n_parents:
            if proto_dim < n_parents:
                inputs_adj = inputs[:, :proto_dim]
            else:
                inputs_adj = np.zeros((n_samples, proto_dim))
                inputs_adj[:, :n_parents] = inputs
        else:
            inputs_adj = inputs
        
        distances = np.zeros((n_samples, self.n_categories))
        for i, proto in enumerate(self.prototypes):
            diff = inputs_adj - proto
            distances[:, i] = np.linalg.norm(diff, axis=1)
        
        return np.argmin(distances, axis=1).squeeze()
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'type': 'discretization',
            'n_categories': self.n_categories,
            'noise_scale': self.noise_scale
        }


class PassthroughTransformation(EdgeTransformation):
    """
    Pass-through transformation: just copies one parent's value.
    
    Useful for preserving temporal correlation in 3D series.
    Selects one parent randomly and copies its value (with optional small noise).
    """
    
    def __init__(
        self,
        parent_index: int,
        noise_scale: float = 0.0,
        rng: Optional[np.random.Generator] = None
    ):
        super().__init__()
        self.parent_index = parent_index
        self.noise_scale = noise_scale
        self.rng = rng if rng is not None else np.random.default_rng()
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Copy one parent's value.
        
        Args:
            inputs: Array of shape (n_samples, n_parents) or (n_parents,)
        """
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        
        # Select the designated parent
        parent_idx = min(self.parent_index, inputs.shape[1] - 1)
        output = inputs[:, parent_idx].copy()
        
        # Add small noise if specified
        if self.noise_scale > 0:
            output += self.rng.normal(0, self.noise_scale, size=output.shape)
        
        # Sanitize
        output = np.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return output.squeeze()
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'type': 'passthrough',
            'parent_index': self.parent_index,
            'noise_scale': self.noise_scale
        }


class TransformationFactory:
    """
    Factory for creating random edge transformations.
    
    Samples transformation type and parameters according to the dataset config.
    """
    
    def __init__(self, config: DatasetConfig, rng: Optional[np.random.Generator] = None):
        """
        Initialize the factory.
        
        Args:
            config: Dataset configuration
            rng: Random number generator
        """
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng(config.seed)
    
    def create(self, n_parents: int) -> EdgeTransformation:
        """
        Create a random transformation for a node.
        
        Args:
            n_parents: Number of parent nodes (input dimension)
            
        Returns:
            EdgeTransformation instance
        """
        # Sample transformation type
        probs = self.config.transform_probs
        types = list(probs.keys())
        type_probs = np.array([probs[t] for t in types])
        type_probs = type_probs / type_probs.sum()
        
        transform_type = self.rng.choice(types, p=type_probs)
        
        # Get noise scale from config
        noise_scale = self.config.noise_scale
        
        if transform_type == 'nn':
            return self._create_nn(n_parents, noise_scale)
        elif transform_type == 'tree':
            return self._create_tree(n_parents, noise_scale)
        elif transform_type == 'passthrough':
            return self._create_passthrough(n_parents, noise_scale)
        else:  # discretization
            return self._create_discretization(n_parents, noise_scale)
    
    def _create_nn(self, n_parents: int, noise_scale: float) -> NNTransformation:
        """
        Create a neural network transformation.
        
        Uses Xavier initialization for weights: w ~ N(0, sqrt(2 / (n_in + n_out)))
        For our case n_out = 1, so: w ~ N(0, sqrt(2 / (n_parents + 1)))
        """
        input_dim = max(1, n_parents)
        
        # Xavier initialization
        # Variance = 2 / (fan_in + fan_out) = 2 / (n_parents + 1)
        xavier_std = np.sqrt(2.0 / (input_dim + 1))
        weights = self.rng.normal(0, xavier_std, size=(input_dim,))
        
        # Small bias
        bias = self.rng.normal(0, 0.1)
        
        # Sample activation with higher probability for identity (preserves temporal structure)
        prob_identity = getattr(self.config, 'prob_identity_activation', 0.3)
        if self.rng.random() < prob_identity and 'identity' in self.config.allowed_activations:
            activation = 'identity'
        else:
            # Sample from other activations
            other_activations = [a for a in self.config.allowed_activations if a != 'identity']
            if other_activations:
                activation = self.rng.choice(other_activations)
            else:
                activation = 'identity'
        
        return NNTransformation(
            weights=weights,
            bias=bias,
            activation=activation,
            noise_scale=noise_scale,
            rng=self.rng
        )
    
    def _create_tree(self, n_parents: int, noise_scale: float) -> TreeTransformation:
        """
        Create a decision tree transformation.
        
        Selects a random subset of parent features and creates a binary tree
        with randomly sampled (feature_index, threshold) pairs at each split.
        """
        depth = self.config.tree_depth
        n_internal = 2 ** depth - 1  # Number of internal nodes in complete binary tree
        n_leaves = 2 ** depth
        total_nodes = n_internal + n_leaves
        
        input_dim = max(1, n_parents)
        
        # Select subset of features to use
        max_features = max(1, int(input_dim * self.config.tree_max_features_fraction))
        n_features_to_use = self.rng.integers(1, max_features + 1)
        available_features = self.rng.choice(input_dim, size=min(n_features_to_use, input_dim), replace=False)
        
        # Initialize arrays
        feature_indices = np.zeros(total_nodes, dtype=int)
        thresholds = np.zeros(total_nodes)
        left_children = np.full(total_nodes, -1, dtype=int)
        right_children = np.full(total_nodes, -1, dtype=int)
        leaf_values = self.rng.normal(0, 1, size=(total_nodes,))
        
        # Build complete binary tree structure
        for node in range(n_internal):
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            
            if left_child < total_nodes:
                left_children[node] = left_child
            if right_child < total_nodes:
                right_children[node] = right_child
            
            # Randomly sample feature and threshold for this split
            feature_indices[node] = self.rng.choice(available_features)
            thresholds[node] = self.rng.normal(0, 1)  # Random threshold
        
        return TreeTransformation(
            feature_indices=feature_indices,
            thresholds=thresholds,
            left_children=left_children,
            right_children=right_children,
            leaf_values=leaf_values,
            noise_scale=noise_scale,
            rng=self.rng
        )
    
    def _create_discretization(self, n_parents: int, noise_scale: float) -> DiscretizationTransformation:
        """
        Create a discretization transformation.
        
        Creates K random prototype vectors. Each input is assigned to the nearest
        prototype, and the output is the normalized category index (index / K).
        """
        n_categories = self.config.n_categories
        input_dim = max(1, n_parents)
        
        # Sample prototypes (cluster centers) - random vectors
        prototypes = self.rng.normal(0, 1, size=(n_categories, input_dim))
        
        return DiscretizationTransformation(
            prototypes=prototypes,
            n_categories=n_categories,
            noise_scale=noise_scale,
            rng=self.rng
        )
    
    def _create_passthrough(self, n_parents: int, noise_scale: float) -> PassthroughTransformation:
        """
        Create a pass-through transformation.
        
        Just copies one parent's value. Useful for preserving temporal structure.
        """
        # Select random parent to copy
        parent_index = self.rng.integers(0, max(1, n_parents))
        
        # Very small noise (or none) to preserve correlation
        small_noise = noise_scale * 0.1  # 10% of normal noise
        
        return PassthroughTransformation(
            parent_index=parent_index,
            noise_scale=small_noise,
            rng=self.rng
        )


class RootNoiseGenerator:
    """
    Generates noise values for root/input nodes.
    
    Root nodes have no parents, so they receive injected noise
    that is then propagated through the DAG.
    
    Supports 3 initialization mechanisms:
    1. Normal: ε ~ N(0, σε²)
    2. Uniform: ε ~ U(−a, a)
    3. Mixed: randomly select normal or uniform per root
    """
    
    def __init__(self, config: DatasetConfig, rng: Optional[np.random.Generator] = None):
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng(config.seed)
        
        # Parameters from config
        self.sigma = config.init_sigma  # σε for normal
        self.a = config.init_a          # a for uniform
    
    def generate(self, n_samples: int) -> np.ndarray:
        """
        Generate noise values for a root node.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Array of shape (n_samples,) with noise values
        """
        noise_type = self.config.noise_type
        
        if noise_type == 'normal':
            return self.rng.normal(0, self.sigma, size=n_samples)
        elif noise_type == 'uniform':
            return self.rng.uniform(-self.a, self.a, size=n_samples)
        elif noise_type == 'mixed':
            # Randomly select normal or uniform for this root
            if self.rng.random() < 0.5:
                return self.rng.normal(0, self.sigma, size=n_samples)
            else:
                return self.rng.uniform(-self.a, self.a, size=n_samples)
        else:
            return self.rng.normal(0, self.sigma, size=n_samples)
