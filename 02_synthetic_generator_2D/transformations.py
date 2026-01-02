"""
Edge transformations for synthetic dataset generation.

This module implements the different types of transformations that can
be applied along edges of the causal DAG:

- Type A: Neural network-like transformations (linear + nonlinear activation)
- Type B: Discretization (for categorical features)
- Type C: Decision tree-like transformations (piecewise rules)
- Type D: Noise (added to all transformations)

Each transformation takes inputs from parent nodes and produces an output
for the child node.
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
        return x
    
    @staticmethod
    def log(x: np.ndarray) -> np.ndarray:
        # Safe log: log(|x| + eps)
        return np.log(np.abs(x) + 1e-6)
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
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
    def cube(x: np.ndarray) -> np.ndarray:
        return x ** 3
    
    @staticmethod
    def sqrt(x: np.ndarray) -> np.ndarray:
        # Safe sqrt: sqrt(|x|)
        return np.sqrt(np.abs(x))
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    @staticmethod
    def softplus(x: np.ndarray) -> np.ndarray:
        return np.log1p(np.exp(np.clip(x, -500, 500)))
    
    @staticmethod
    def step(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)
    
    @staticmethod
    def mod(x: np.ndarray) -> np.ndarray:
        # Modulo with period sampled between 1 and 5
        return np.mod(x, 2.0)
    
    @staticmethod
    def power(x: np.ndarray) -> np.ndarray:
        # Power function with random exponent (per paper: "power functions")
        # Using x^p where p is randomly chosen
        p = np.random.uniform(0.5, 3.0)
        return np.sign(x) * np.power(np.abs(x) + 1e-6, p)
    
    @staticmethod
    def rank(x: np.ndarray) -> np.ndarray:
        # Convert to ranks (normalized to [0, 1])
        if x.ndim == 1:
            order = x.argsort()
            ranks = np.empty_like(order)
            ranks[order] = np.arange(len(x))
            return ranks / (len(x) - 1 + 1e-6)
        else:
            # For batched input, rank within each batch
            result = np.zeros_like(x)
            for i in range(x.shape[0]):
                order = x[i].argsort()
                ranks = np.empty_like(order)
                ranks[order] = np.arange(len(x[i]))
                result[i] = ranks / (len(x[i]) - 1 + 1e-6)
            return result
    
    @staticmethod
    def exp_neg(x: np.ndarray) -> np.ndarray:
        # Exponential of negative: exp(-|x|)
        return np.exp(-np.abs(np.clip(x, -500, 500)))
    
    @staticmethod
    def gaussian(x: np.ndarray) -> np.ndarray:
        # Gaussian: exp(-x^2)
        return np.exp(-x ** 2)
    
    @classmethod
    def get(cls, name: str) -> Callable[[np.ndarray], np.ndarray]:
        """Get activation function by name."""
        activations = {
            'identity': cls.identity,
            'log': cls.log,
            'sigmoid': cls.sigmoid,
            'tanh': cls.tanh,
            'sin': cls.sin,
            'cos': cls.cos,
            'abs': cls.abs,
            'square': cls.square,
            'cube': cls.cube,
            'sqrt': cls.sqrt,
            'relu': cls.relu,
            'softplus': cls.softplus,
            'step': cls.step,
            'mod': cls.mod,
            'power': cls.power,
            'rank': cls.rank,
            'exp_neg': cls.exp_neg,
            'gaussian': cls.gaussian,
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
            inputs: Array of shape (n_samples, n_parents) or (n_parents,)
                   containing values from parent nodes
                   
        Returns:
            Array of shape (n_samples,) or scalar containing output values
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
    
    Combines inputs linearly, applies nonlinear activation(s),
    and optionally adds noise.
    
    Structure:
    1. Linear combination: sum(w_i * x_i) + bias
    2. Apply activation function
    3. Optionally repeat with more layers
    4. Add noise
    """
    
    weights: List[np.ndarray]  # Weight matrices for each layer
    biases: List[np.ndarray]   # Bias vectors for each layer
    activations: List[str]      # Activation function names
    noise_scale: float
    rng: np.random.Generator
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Apply NN transformation."""
        # Ensure 2D input
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        
        x = inputs
        
        # Apply each layer
        for i, (w, b, act_name) in enumerate(zip(self.weights, self.biases, self.activations)):
            # Linear transformation
            x = x @ w + b
            
            # Activation
            act_fn = Activation.get(act_name)
            x = act_fn(x)
        
        # Final linear to scalar output
        x = x.mean(axis=1)  # Reduce to scalar per sample
        
        # Add noise
        if self.noise_scale > 0:
            x = x + self.rng.normal(0, self.noise_scale, size=x.shape)
        
        return x.squeeze()
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'type': 'nn',
            'n_layers': len(self.weights),
            'activations': self.activations,
            'noise_scale': self.noise_scale
        }


@dataclass
class DiscretizationTransformation(EdgeTransformation):
    """
    Discretization transformation for categorical features.
    
    1. Computes distance to K prototypes
    2. Assigns to nearest prototype (categorical index)
    3. Maps category back to continuous value for further propagation
    """
    
    prototypes: np.ndarray       # Shape (n_categories, n_parents)
    category_embeddings: np.ndarray  # Shape (n_categories,)
    noise_scale: float
    rng: np.random.Generator
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Apply discretization transformation."""
        # Ensure 2D input
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        
        n_samples = inputs.shape[0]
        n_categories = len(self.prototypes)
        
        # Compute distance to each prototype
        distances = np.zeros((n_samples, n_categories))
        for i, proto in enumerate(self.prototypes):
            # Euclidean distance (or use first dimension if dimensions don't match)
            if len(proto) == inputs.shape[1]:
                diff = inputs - proto
            else:
                # Use only the matching dimensions
                min_dim = min(len(proto), inputs.shape[1])
                diff = inputs[:, :min_dim] - proto[:min_dim]
            distances[:, i] = np.linalg.norm(diff, axis=1)
        
        # Assign to nearest prototype
        category_indices = np.argmin(distances, axis=1)
        
        # Map to continuous embedding for propagation
        output = self.category_embeddings[category_indices]
        
        # Add noise
        if self.noise_scale > 0:
            output = output + self.rng.normal(0, self.noise_scale, size=output.shape)
        
        return output.squeeze()
    
    def get_category_indices(self, inputs: np.ndarray) -> np.ndarray:
        """Get the categorical indices (for use as observed features)."""
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        
        n_samples = inputs.shape[0]
        n_categories = len(self.prototypes)
        
        distances = np.zeros((n_samples, n_categories))
        for i, proto in enumerate(self.prototypes):
            if len(proto) == inputs.shape[1]:
                diff = inputs - proto
            else:
                min_dim = min(len(proto), inputs.shape[1])
                diff = inputs[:, :min_dim] - proto[:min_dim]
            distances[:, i] = np.linalg.norm(diff, axis=1)
        
        return np.argmin(distances, axis=1).squeeze()
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'type': 'discretization',
            'n_categories': len(self.prototypes),
            'noise_scale': self.noise_scale
        }


@dataclass
class TreeTransformation(EdgeTransformation):
    """
    Decision tree-like transformation.
    
    Applies piecewise rules based on thresholds:
    - If value > threshold: go right
    - Else: go left
    
    Each leaf has an associated output value.
    """
    
    # Tree structure (binary tree stored as arrays)
    thresholds: np.ndarray      # Threshold at each internal node
    feature_indices: np.ndarray  # Which input feature to split on
    left_children: np.ndarray   # Index of left child (-1 for leaf)
    right_children: np.ndarray  # Index of right child (-1 for leaf)
    leaf_values: np.ndarray     # Output value for each node (used at leaves)
    noise_scale: float
    rng: np.random.Generator
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Apply tree transformation."""
        # Ensure 2D input
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        
        n_samples = inputs.shape[0]
        outputs = np.zeros(n_samples)
        
        for i in range(n_samples):
            node = 0  # Start at root
            while True:
                if self.left_children[node] == -1:  # Leaf node
                    outputs[i] = self.leaf_values[node]
                    break
                
                # Get feature value for this sample
                feat_idx = self.feature_indices[node]
                if feat_idx < inputs.shape[1]:
                    feat_val = inputs[i, feat_idx]
                else:
                    feat_val = inputs[i, 0]  # Fallback to first feature
                
                # Decision
                if feat_val > self.thresholds[node]:
                    node = self.right_children[node]
                else:
                    node = self.left_children[node]
        
        # Add noise
        if self.noise_scale > 0:
            outputs = outputs + self.rng.normal(0, self.noise_scale, size=outputs.shape)
        
        return outputs.squeeze()
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'type': 'tree',
            'depth': int(np.log2(len(self.thresholds) + 1)),
            'noise_scale': self.noise_scale
        }


@dataclass
class IdentityTransformation(EdgeTransformation):
    """
    Identity transformation (with optional noise).
    
    Simply passes through the (weighted) sum of inputs.
    """
    
    weights: np.ndarray
    bias: float
    noise_scale: float
    rng: np.random.Generator
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Apply identity transformation."""
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        
        # Weighted sum
        if len(self.weights) == inputs.shape[1]:
            output = inputs @ self.weights + self.bias
        else:
            # Handle dimension mismatch
            min_dim = min(len(self.weights), inputs.shape[1])
            output = inputs[:, :min_dim] @ self.weights[:min_dim] + self.bias
        
        # Add noise
        if self.noise_scale > 0:
            output = output + self.rng.normal(0, self.noise_scale, size=output.shape)
        
        return output.squeeze()
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'type': 'identity',
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
        Create a random transformation for an edge.
        
        Args:
            n_parents: Number of parent nodes (input dimension)
            
        Returns:
            EdgeTransformation instance
        """
        # Sample transformation type
        probs = self.config.edge_transform_probs
        types = list(probs.keys())
        type_probs = np.array([probs[t] for t in types])
        type_probs = type_probs / type_probs.sum()
        
        transform_type = self.rng.choice(types, p=type_probs)
        
        # Determine noise scale
        add_noise = self.rng.random() < self.config.edge_noise_prob
        noise_scale = self.config.noise_scale if add_noise else 0.0
        
        if transform_type == 'nn':
            return self._create_nn(n_parents, noise_scale)
        elif transform_type == 'tree':
            return self._create_tree(n_parents, noise_scale)
        elif transform_type == 'discretization':
            return self._create_discretization(n_parents, noise_scale)
        else:  # identity
            return self._create_identity(n_parents, noise_scale)
    
    def _create_nn(self, n_parents: int, noise_scale: float) -> NNTransformation:
        """Create a neural network transformation."""
        n_layers = self.config.nn_hidden
        width = self.config.nn_width
        
        weights = []
        biases = []
        activations = []
        
        input_dim = max(1, n_parents)
        
        for i in range(n_layers):
            # Output dimension for this layer
            if i == n_layers - 1:
                out_dim = 1  # Final layer outputs scalar
            else:
                out_dim = width
            
            # Sample weights using Xavier-like initialization
            scale = np.sqrt(2.0 / (input_dim + out_dim))
            w = self.rng.normal(0, scale, size=(input_dim, out_dim))
            b = self.rng.normal(0, 0.1, size=(out_dim,))
            
            weights.append(w)
            biases.append(b)
            
            # Sample activation
            act = self.rng.choice(self.config.allowed_activations)
            activations.append(act)
            
            input_dim = out_dim
        
        return NNTransformation(
            weights=weights,
            biases=biases,
            activations=activations,
            noise_scale=noise_scale,
            rng=self.rng
        )
    
    def _create_discretization(self, n_parents: int, noise_scale: float) -> DiscretizationTransformation:
        """
        Create a discretization transformation.
        
        Per paper: "we map the vector to the index of the nearest neighbour in a 
        set of per node randomly sampled vectors {p1, …, pK}... We sample a second 
        set of embedding vectors {p'1,...,p'K} for each class"
        
        Both prototype vectors and embedding vectors are fully random.
        """
        n_categories = self.config.n_categories
        input_dim = max(1, n_parents)
        
        # Sample prototypes (cluster centers) - fully random
        prototypes = self.rng.normal(0, 1, size=(n_categories, input_dim))
        
        # Sample embeddings for each category - fully random (per paper)
        # "We sample a second set of embedding vectors {p'1,...,p'K} for each class"
        category_embeddings = self.rng.normal(0, 1, size=(n_categories,))
        
        return DiscretizationTransformation(
            prototypes=prototypes,
            category_embeddings=category_embeddings,
            noise_scale=noise_scale,
            rng=self.rng
        )
    
    def _create_tree(self, n_parents: int, noise_scale: float) -> TreeTransformation:
        """Create a decision tree transformation."""
        depth = self.config.tree_depth
        n_nodes = 2 ** depth - 1  # Number of internal nodes
        n_leaves = 2 ** depth     # Number of leaves
        total_nodes = n_nodes + n_leaves
        
        input_dim = max(1, n_parents)
        
        # Initialize arrays
        thresholds = np.zeros(total_nodes)
        feature_indices = np.zeros(total_nodes, dtype=int)
        left_children = np.full(total_nodes, -1, dtype=int)
        right_children = np.full(total_nodes, -1, dtype=int)
        leaf_values = self.rng.normal(0, 1, size=(total_nodes,))
        
        # Build tree structure
        for node in range(n_nodes):
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            
            if left_child < total_nodes:
                left_children[node] = left_child
            if right_child < total_nodes:
                right_children[node] = right_child
            
            # Sample threshold and feature
            thresholds[node] = self.rng.normal(0, 1)
            feature_indices[node] = self.rng.integers(0, input_dim)
        
        return TreeTransformation(
            thresholds=thresholds,
            feature_indices=feature_indices,
            left_children=left_children,
            right_children=right_children,
            leaf_values=leaf_values,
            noise_scale=noise_scale,
            rng=self.rng
        )
    
    def _create_identity(self, n_parents: int, noise_scale: float) -> IdentityTransformation:
        """Create an identity transformation."""
        input_dim = max(1, n_parents)
        
        # Sample weights
        weights = self.rng.normal(0, 1 / np.sqrt(input_dim), size=(input_dim,))
        bias = self.rng.normal(0, 0.1)
        
        return IdentityTransformation(
            weights=weights,
            bias=bias,
            noise_scale=noise_scale,
            rng=self.rng
        )


class RootNoiseGenerator:
    """
    Generates noise values for root nodes.
    
    Root nodes have no parents, so they receive injected noise
    that is then propagated through the DAG.
    
    Per paper, there are 3 initialization mechanisms:
    1. Normal: ε ~ N(0, σε²) where σε² is a hyperparameter
    2. Uniform: ε ~ U(−a, a) where a is a hyperparameter  
    3. Mixed: for each root node, randomly select normal or uniform
    """
    
    def __init__(self, config: DatasetConfig, rng: Optional[np.random.Generator] = None):
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng(config.seed)
        
        # Use σ and a from config (sampled as hyperparameters per paper)
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
            # ε ~ N(0, σε²) per paper
            return self.rng.normal(0, self.sigma, size=n_samples)
        elif noise_type == 'uniform':
            # ε ~ U(−a, a) per paper
            return self.rng.uniform(-self.a, self.a, size=n_samples)
        elif noise_type == 'mixed':
            # Per paper: "for each root node, we randomly select either 
            # a normal or uniform distribution"
            # This method is called once per root node, so we just pick one
            if self.rng.random() < 0.5:
                return self.rng.normal(0, self.sigma, size=n_samples)
            else:
                return self.rng.uniform(-self.a, self.a, size=n_samples)
        else:
            return self.rng.normal(0, self.sigma, size=n_samples)

