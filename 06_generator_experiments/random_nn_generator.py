"""
Random Neural Network Generator for Time Series.

Each network randomly samples:
- Number of memory dimensions
- Time input transformations (from a pool of options)
- Number of layers and nodes per layer
- Activation function
- Initialization parameters
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import os


@dataclass
class RandomNNConfig:
    """
    Configuration with ranges for random sampling.
    Each network will sample its own parameters from these ranges.
    """
    # Memory dimension range
    memory_dim_range: Tuple[int, int] = (3, 32)
    memory_init: str = 'uniform'  # 'uniform' or 'normal'
    
    # Stochastic input (random at each time step)
    stochastic_input_dim_range: Tuple[int, int] = (0, 5)
    
    # Time input options - will randomly select a subset
    # Each is a tuple of (name, function, probability of inclusion)
    # Available time transforms are defined in the generator
    n_time_transforms_range: Tuple[int, int] = (2, 8)
    
    # Network architecture ranges
    n_hidden_layers_range: Tuple[int, int] = (2, 8)
    n_nodes_per_layer_range: Tuple[int, int] = (10, 50)
    
    # Activation - will be randomly chosen
    activation_choices: Tuple[str, ...] = ('relu', 'tanh', 'leaky_relu')
    
    # Initialization
    weight_init_choices: Tuple[str, ...] = ('xavier_uniform', 'xavier_normal')
    weight_scale_range: Tuple[float, float] = (0.8, 1.2)
    bias_std_range: Tuple[float, float] = (0.0, 0.1)
    
    # Per-node noise configuration
    # Range for probability that a node has noise (sampled per network)
    node_noise_prob_range: Tuple[float, float] = (0.1, 0.5)
    # Range for noise std when a node has noise
    node_noise_std_range: Tuple[float, float] = (0.001, 0.02)
    # Available noise distributions
    noise_dist_choices: Tuple[str, ...] = ('normal', 'uniform', 'laplace')
    
    # Per-layer activation (if True, each layer gets a random activation)
    per_layer_activation: bool = True
    
    # Quantization nodes: some nodes act as classifiers
    # Probability that a node is a quantization node
    quantization_node_prob: float = 0.1
    # Range for number of classes/categories
    quantization_n_classes_range: Tuple[int, int] = (2, 5)
    
    # Sequence
    seq_length: int = 100


@dataclass
class NodeNoiseConfig:
    """Noise configuration for a single node."""
    has_noise: bool
    distribution: str = 'normal'  # 'normal', 'uniform', 'laplace'
    scale: float = 0.0  # std for normal/laplace, half-range for uniform


@dataclass
class NodeQuantizationConfig:
    """
    Quantization configuration for a single node.
    
    When is_quantization=True:
    - prototypes: (n_classes, input_dim) - class center vectors
    - class_values: (n_classes,) - output value for each class (sampled N(0,1))
    
    During propagation:
    1. Compute distance from input vector to each prototype
    2. Assign to nearest class
    3. Output the class_value for that class
    """
    is_quantization: bool
    n_classes: int = 0
    prototypes: np.ndarray = None  # (n_classes, input_dim)
    class_values: np.ndarray = None  # (n_classes,) - output for each class


@dataclass
class SampledConfig:
    """Configuration that was sampled for a specific network."""
    memory_dim: int
    memory_init: str
    stochastic_input_dim: int
    time_transforms: List[Dict[str, Any]]  # List of {name, func, params}
    n_hidden_layers: int
    nodes_per_layer: List[int]  # Can vary per layer
    activations: List[str]  # One activation per layer (for backwards compat)
    node_activations: List[List[str]] = None  # Per-node activations: List[List[str]]
    weight_init: str = 'xavier_normal'
    weight_scale: float = 1.0
    bias_std: float = 0.0
    # Per-node noise: List[List[NodeNoiseConfig]] - one list per layer, one config per node
    node_noise: List[List[NodeNoiseConfig]] = None
    node_noise_prob: float = 0.0  # The sampled noise probability for this network
    # Per-node quantization: List[List[NodeQuantizationConfig]]
    node_quantization: List[List[NodeQuantizationConfig]] = None
    seq_length: int = 100
    per_node_activation: bool = False  # If True, use node_activations instead of activations


class RandomNNGenerator:
    """
    Generate time series by propagating through a randomly configured neural network.
    """
    
    # Available time transforms
    TIME_TRANSFORMS = [
        # (name, needs_params, param_sampler or None)
        ('linear', False, None),  # u
        ('quadratic', False, None),  # u²
        ('cubic', False, None),  # u³
        ('tanh_trend', True, lambda rng: {'beta': rng.uniform(0.5, 3.0)}),  # tanh(β(2u-1))
        ('sin_k1', False, None),  # sin(2πu)
        ('cos_k1', False, None),  # cos(2πu)
        ('sin_k2', False, None),  # sin(4πu)
        ('cos_k2', False, None),  # cos(4πu)
        ('sin_k3', False, None),  # sin(6πu)
        ('cos_k3', False, None),  # cos(6πu)
        ('sin_k5', False, None),  # sin(10πu)
        ('cos_k5', False, None),  # cos(10πu)
        ('exp_decay', True, lambda rng: {'gamma': rng.uniform(0.5, 5.0)}),  # exp(-γu)
        ('exp_growth', True, lambda rng: {'gamma': rng.uniform(0.1, 1.0)}),  # exp(γu) - smaller gamma
        ('log', False, None),  # log(1 + u) - shifted to handle u=0
        ('sqrt', False, None),  # sqrt(u)
    ]
    
    def __init__(self, config: RandomNNConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        
        # Sample the network configuration
        self.sampled_config = self._sample_config()
        
        # Build the network
        self._build_network()
    
    def _log_uniform_int(self, low: int, high: int) -> int:
        """Sample integer from log-uniform distribution (favors smaller values)."""
        if low <= 0:
            low = 1  # Avoid log(0)
        log_low = np.log(low)
        log_high = np.log(high)
        log_sample = self.rng.uniform(log_low, log_high)
        return int(np.round(np.exp(log_sample)))
    
    def _log_uniform_float(self, low: float, high: float) -> float:
        """Sample float from log-uniform distribution (favors smaller values)."""
        if low <= 0:
            low = 1e-10  # Avoid log(0)
        log_low = np.log(low)
        log_high = np.log(high)
        log_sample = self.rng.uniform(log_low, log_high)
        return np.exp(log_sample)
    
    def _sample_config(self) -> SampledConfig:
        """Sample all configuration parameters for this network."""
        cfg = self.config
        
        # Sample memory dimension (log-uniform to favor smaller)
        memory_dim = self._log_uniform_int(max(1, cfg.memory_dim_range[0]), cfg.memory_dim_range[1])
        
        # Sample stochastic input dimension
        stochastic_dim = self.rng.integers(
            cfg.stochastic_input_dim_range[0], 
            cfg.stochastic_input_dim_range[1] + 1
        )
        
        # Sample time transforms (log-uniform to favor smaller)
        n_transforms = self._log_uniform_int(max(1, cfg.n_time_transforms_range[0]), cfg.n_time_transforms_range[1])
        
        # ONLY linear transforms (t) - no sin/cos/etc
        selected_transforms = []
        for _ in range(n_transforms):
            selected_transforms.append({'name': 'linear', 'params': None})
        
        # Sample network architecture (log-uniform to favor smaller)
        n_layers = self._log_uniform_int(max(1, cfg.n_hidden_layers_range[0]), cfg.n_hidden_layers_range[1])
        
        # Sample nodes per layer (log-uniform to favor smaller)
        nodes_per_layer = []
        for _ in range(n_layers):
            n_nodes = self._log_uniform_int(max(1, cfg.n_nodes_per_layer_range[0]), cfg.n_nodes_per_layer_range[1])
            nodes_per_layer.append(n_nodes)
        
        # Sample activations
        # Per-node activation mode: each node gets its own activation
        node_activations = []
        activations = []  # Keep for backwards compatibility
        for layer_idx in range(n_layers):
            n_nodes = nodes_per_layer[layer_idx]
            layer_acts = [self.rng.choice(list(cfg.activation_choices)) for _ in range(n_nodes)]
            node_activations.append(layer_acts)
            # For backwards compatibility, pick the most common or first
            activations.append(layer_acts[0] if layer_acts else 'identity')
        
        # Sample initialization
        weight_init = self.rng.choice(list(cfg.weight_init_choices))
        weight_scale = self.rng.uniform(cfg.weight_scale_range[0], cfg.weight_scale_range[1])
        bias_std = self.rng.uniform(cfg.bias_std_range[0], cfg.bias_std_range[1])
        
        # Sample noise probability for this network (log-uniform to favor smaller)
        node_noise_prob = self._log_uniform_float(max(0.001, cfg.node_noise_prob_range[0]), cfg.node_noise_prob_range[1])
        
        # Calculate input dimensions for each layer (needed for quantization prototypes)
        # Input to layer 0: total_input_dim = time_transforms + memory + stochastic
        n_time_inputs = len(selected_transforms)
        total_input_dim = n_time_inputs + memory_dim + stochastic_dim
        layer_input_dims = [total_input_dim] + nodes_per_layer[:-1]  # Input dim for each layer
        
        # Sample per-node noise configuration
        node_noise = []
        for layer_idx, n_nodes in enumerate(nodes_per_layer):
            layer_noise = []
            for node_idx in range(n_nodes):
                has_noise = self.rng.random() < node_noise_prob
                if has_noise:
                    dist = self.rng.choice(list(cfg.noise_dist_choices))
                    # Log-uniform for noise scale to favor smaller values
                    scale = self._log_uniform_float(max(1e-6, cfg.node_noise_std_range[0]), cfg.node_noise_std_range[1])
                    layer_noise.append(NodeNoiseConfig(has_noise=True, distribution=dist, scale=scale))
                else:
                    layer_noise.append(NodeNoiseConfig(has_noise=False))
            node_noise.append(layer_noise)
        
        # Sample per-node quantization configuration
        node_quantization = []
        for layer_idx, n_nodes in enumerate(nodes_per_layer):
            input_dim = layer_input_dims[layer_idx]
            layer_quant = []
            for node_idx in range(n_nodes):
                is_quant = self.rng.random() < cfg.quantization_node_prob
                if is_quant:
                    n_classes = self.rng.integers(
                        cfg.quantization_n_classes_range[0],
                        cfg.quantization_n_classes_range[1] + 1
                    )
                    # Sample prototype vectors (class centers)
                    prototypes = self.rng.normal(0, 1, (n_classes, input_dim))
                    # Sample output value for each class
                    class_values = self.rng.normal(0, 1, n_classes)
                    layer_quant.append(NodeQuantizationConfig(
                        is_quantization=True,
                        n_classes=n_classes,
                        prototypes=prototypes,
                        class_values=class_values
                    ))
                else:
                    layer_quant.append(NodeQuantizationConfig(is_quantization=False))
            node_quantization.append(layer_quant)
        
        return SampledConfig(
            memory_dim=memory_dim,
            memory_init=cfg.memory_init,
            stochastic_input_dim=stochastic_dim,
            time_transforms=selected_transforms,
            n_hidden_layers=n_layers,
            nodes_per_layer=nodes_per_layer,
            activations=activations,
            node_activations=node_activations,
            weight_init=weight_init,
            weight_scale=weight_scale,
            bias_std=bias_std,
            node_noise=node_noise,
            node_noise_prob=node_noise_prob,
            node_quantization=node_quantization,
            seq_length=cfg.seq_length,
            per_node_activation=True
        )
    
    def _compute_time_transform(self, name: str, u: np.ndarray, params: Optional[Dict] = None) -> np.ndarray:
        """
        Compute a time transform given normalized time u ∈ [0, 1].
        
        Args:
            name: Transform name
            u: Normalized time array, shape (T,)
            params: Optional parameters for the transform
        
        Returns:
            Transformed values, shape (T,)
        """
        if name == 'linear':
            return 2 * u - 1  # Map to [-1, 1]
        elif name == 'quadratic':
            return (2 * u - 1) ** 2
        elif name == 'cubic':
            return (2 * u - 1) ** 3
        elif name == 'tanh_trend':
            beta = params['beta']
            return np.tanh(beta * (2 * u - 1))
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
        elif name == 'exp_decay':
            gamma = params['gamma']
            # Normalize to [-1, 1] range approximately
            vals = np.exp(-gamma * u)
            return 2 * vals - 1
        elif name == 'exp_growth':
            gamma = params['gamma']
            # exp(γu) for small γ, normalize
            vals = np.exp(gamma * u)
            vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
            return 2 * vals - 1
        elif name == 'log':
            # log(1 + u) normalized to [-1, 1]
            vals = np.log(1 + u)
            vals = vals / (np.log(2) + 1e-8)  # Normalize since max is log(2)
            return 2 * vals - 1
        elif name == 'sqrt':
            # sqrt(u) normalized to [-1, 1]
            vals = np.sqrt(u)
            return 2 * vals - 1
        else:
            raise ValueError(f"Unknown transform: {name}")
    
    def _build_network(self):
        """Build random network weights and structure."""
        cfg = self.sampled_config
        
        # Calculate input dimension
        n_time_inputs = len(cfg.time_transforms)
        self.time_input_dim = n_time_inputs
        self.total_input_dim = n_time_inputs + cfg.memory_dim + cfg.stochastic_input_dim
        
        # Layer sizes: input -> hidden layers
        self.layer_sizes = [self.total_input_dim] + cfg.nodes_per_layer
        
        # Create weights for each layer
        self.weights = []
        self.biases = []
        
        for i in range(len(self.layer_sizes) - 1):
            in_dim = self.layer_sizes[i]
            out_dim = self.layer_sizes[i + 1]
            
            # Weight initialization
            if cfg.weight_init == 'xavier_uniform':
                a = cfg.weight_scale * np.sqrt(6.0 / (in_dim + out_dim))
                W = self.rng.uniform(-a, a, (out_dim, in_dim))
            else:  # xavier_normal
                std = cfg.weight_scale * np.sqrt(2.0 / in_dim)
                W = self.rng.normal(0, std, (out_dim, in_dim))
            
            # Bias initialization
            b = np.zeros(out_dim) if cfg.bias_std == 0 else self.rng.normal(0, cfg.bias_std, out_dim)
            
            self.weights.append(W)
            self.biases.append(b)
        
        self.n_layers = len(self.layer_sizes)
        self.total_nodes = sum(self.layer_sizes)
    
    def _activation(self, x: np.ndarray, act: str) -> np.ndarray:
        """Apply activation function.
        
        Available activations:
        - identity: f(x) = x
        - log: f(x) = sign(x) * log(1 + |x|)
        - sigmoid: f(x) = 1/(1+exp(-x))
        - abs: f(x) = |x|
        - sin: f(x) = sin(x)
        - tanh: f(x) = tanh(x)
        - rank: f(x) = normalized rank (percentile)
        - square: f(x) = x^2
        - power: f(x) = sign(x) * |x|^0.5
        - softplus: f(x) = log(1 + exp(x)) (smooth ReLU)
        - step: f(x) = 0 if x < 0 else 1
        - modulo: f(x) = x mod 1 (fractional part)
        - relu, leaky_relu, elu: classic activations
        """
        if act == 'identity':
            return x
        elif act == 'log':
            # Signed log: preserves sign, compresses magnitude
            return np.sign(x) * np.log1p(np.abs(x))
        elif act == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif act == 'abs':
            return np.abs(x)
        elif act == 'sin':
            return np.sin(x)
        elif act == 'tanh':
            return np.tanh(x)
        elif act == 'rank':
            # Rank-based: convert to percentile (0-1) along time axis
            # This creates step-like patterns based on relative values
            if x.ndim == 2:  # (n_samples, T)
                ranks = np.argsort(np.argsort(x, axis=1), axis=1)  # ranks along T
                return ranks / (x.shape[1] - 1) * 2 - 1  # Scale to [-1, 1]
            else:
                return x
        elif act == 'square':
            return x ** 2
        elif act == 'power':
            # Signed square root: preserves sign, compresses magnitude
            return np.sign(x) * np.sqrt(np.abs(x))
        elif act == 'softplus':
            # Smooth approximation to ReLU: log(1 + exp(x))
            return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -500, 20))))
        elif act == 'step':
            # Step function: binary output
            return np.where(x >= 0, 1.0, 0.0)
        elif act == 'modulo':
            # Modulo 1: creates sawtooth patterns
            return np.mod(x, 1.0)
        elif act == 'relu':
            return np.maximum(0, x)
        elif act == 'leaky_relu':
            return np.where(x > 0, x, 0.01 * x)
        elif act == 'elu':
            return np.where(x > 0, x, np.exp(np.clip(x, -500, 0)) - 1)
        else:
            raise ValueError(f"Unknown activation: {act}")
    
    def _generate_time_inputs(self, T: int) -> np.ndarray:
        """
        Generate time input features based on sampled transforms.
        
        Returns: (n_time_inputs, T) array
        """
        u = np.linspace(0, 1, T)  # Normalized time in [0, 1]
        
        inputs = []
        for transform in self.sampled_config.time_transforms:
            name = transform['name']
            params = transform['params']
            inputs.append(self._compute_time_transform(name, u, params))
        
        return np.array(inputs)  # (n_time_inputs, T)
    
    def _generate_memory(self, n_samples: int) -> np.ndarray:
        """Generate memory vectors for each sample."""
        cfg = self.sampled_config
        if cfg.memory_init == 'uniform':
            return self.rng.uniform(-1, 1, (n_samples, cfg.memory_dim))
        else:  # normal
            return np.clip(self.rng.normal(0, 0.5, (n_samples, cfg.memory_dim)), -1, 1)
    
    def _generate_stochastic_inputs(self, n_samples: int, T: int) -> Optional[np.ndarray]:
        """Generate random inputs that vary at each time step."""
        if self.sampled_config.stochastic_input_dim == 0:
            return None
        return self.rng.uniform(-1, 1, (n_samples, self.sampled_config.stochastic_input_dim, T))
    
    def propagate(self, n_samples: int = 5) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Propagate inputs through the network.
        
        Returns:
            output: Final layer output (n_samples, n_output_nodes, T)
            all_layers: List of all layer activations
        """
        cfg = self.sampled_config
        T = cfg.seq_length
        
        # Generate inputs
        time_inputs = self._generate_time_inputs(T)  # (time_dim, T)
        memory = self._generate_memory(n_samples)    # (n_samples, mem_dim)
        stochastic = self._generate_stochastic_inputs(n_samples, T)
        
        # Build input layer
        time_broadcast = np.broadcast_to(time_inputs[np.newaxis, :, :], (n_samples, self.time_input_dim, T))
        memory_broadcast = np.broadcast_to(memory[:, :, np.newaxis], (n_samples, cfg.memory_dim, T))
        
        if stochastic is not None:
            input_layer = np.concatenate([time_broadcast, memory_broadcast, stochastic], axis=1)
        else:
            input_layer = np.concatenate([time_broadcast, memory_broadcast], axis=1)
        
        all_layers = [input_layer]
        current = input_layer
        
        for layer_idx, (W, b) in enumerate(zip(self.weights, self.biases)):
            n_output_nodes = W.shape[0]
            out = np.zeros((n_samples, n_output_nodes, T))
            
            # Get quantization config for this layer
            layer_quant_config = cfg.node_quantization[layer_idx]
            
            # Process each node
            for node_idx in range(n_output_nodes):
                quant_cfg = layer_quant_config[node_idx]
                
                if quant_cfg.is_quantization:
                    # Quantization node: classify based on input and output class value
                    # current shape: (n_samples, input_dim, T)
                    # For each sample and time step, compute distance to prototypes
                    for t in range(T):
                        inputs = current[:, :, t]  # (n_samples, input_dim)
                        
                        # Handle dimension mismatch
                        proto_dim = quant_cfg.prototypes.shape[1]
                        if inputs.shape[1] != proto_dim:
                            if inputs.shape[1] > proto_dim:
                                inputs_adj = inputs[:, :proto_dim]
                            else:
                                inputs_adj = np.zeros((n_samples, proto_dim))
                                inputs_adj[:, :inputs.shape[1]] = inputs
                        else:
                            inputs_adj = inputs
                        
                        # Compute distance to each prototype: (n_samples, n_classes)
                        distances = np.zeros((n_samples, quant_cfg.n_classes))
                        for c in range(quant_cfg.n_classes):
                            diff = inputs_adj - quant_cfg.prototypes[c]
                            distances[:, c] = np.linalg.norm(diff, axis=1)
                        
                        # Assign to nearest class
                        class_assignments = np.argmin(distances, axis=1)  # (n_samples,)
                        
                        # Output the class value for each sample
                        out[:, node_idx, t] = quant_cfg.class_values[class_assignments]
                else:
                    # Normal node: linear + activation
                    w_node = W[node_idx, :]  # (input_dim,)
                    b_node = b[node_idx]
                    
                    # Linear: sum over input dimension
                    node_out = np.einsum('i,nit->nt', w_node, current) + b_node
                    
                    # Activation - per-node if available
                    if cfg.per_node_activation and cfg.node_activations is not None:
                        node_activation = cfg.node_activations[layer_idx][node_idx]
                    else:
                        node_activation = cfg.activations[layer_idx]
                    node_out = self._activation(node_out, node_activation)
                    
                    out[:, node_idx, :] = node_out
            
            # Add per-node noise
            layer_noise_config = cfg.node_noise[layer_idx]
            for node_idx, noise_cfg in enumerate(layer_noise_config):
                if noise_cfg.has_noise:
                    if noise_cfg.distribution == 'normal':
                        noise = self.rng.normal(0, noise_cfg.scale, (n_samples, T))
                    elif noise_cfg.distribution == 'uniform':
                        noise = self.rng.uniform(-noise_cfg.scale, noise_cfg.scale, (n_samples, T))
                    elif noise_cfg.distribution == 'laplace':
                        noise = self.rng.laplace(0, noise_cfg.scale, (n_samples, T))
                    else:
                        noise = 0
                    out[:, node_idx, :] = out[:, node_idx, :] + noise
            
            all_layers.append(out)
            current = out
        
        return current, all_layers
    
    def get_config_summary(self) -> str:
        """Get a summary of the sampled configuration."""
        cfg = self.sampled_config
        
        # Count noisy nodes per layer
        noisy_nodes_info = []
        total_noisy = 0
        total_nodes = 0
        for layer_idx, layer_noise in enumerate(cfg.node_noise):
            noisy_count = sum(1 for n in layer_noise if n.has_noise)
            total = len(layer_noise)
            total_noisy += noisy_count
            total_nodes += total
            if noisy_count > 0:
                noisy_nodes_info.append(f"L{layer_idx+1}:{noisy_count}")
        
        # Count quantization nodes per layer
        quant_nodes_info = []
        total_quant = 0
        for layer_idx, layer_quant in enumerate(cfg.node_quantization):
            quant_count = sum(1 for q in layer_quant if q.is_quantization)
            total_quant += quant_count
            if quant_count > 0:
                quant_nodes_info.append(f"L{layer_idx+1}:{quant_count}")
        
        # Format activations
        if cfg.per_node_activation and cfg.node_activations is not None:
            # Count activations across all nodes
            act_counts = {}
            for layer_acts in cfg.node_activations:
                for act in layer_acts:
                    act_counts[act] = act_counts.get(act, 0) + 1
            act_str = ', '.join([f"{act}:{cnt}" for act, cnt in sorted(act_counts.items())])
        else:
            act_str = ', '.join(cfg.activations)
        
        lines = [
            f"Memory: {cfg.memory_dim} dims ({cfg.memory_init})",
            f"Stochastic: {cfg.stochastic_input_dim} dims",
            f"Time transforms ({len(cfg.time_transforms)}): {[t['name'] for t in cfg.time_transforms]}",
            f"Layers: {cfg.nodes_per_layer}",
            f"Activations: [{act_str}]",
            f"Weight init: {cfg.weight_init} (scale={cfg.weight_scale:.2f})",
            f"Bias std: {cfg.bias_std:.3f}",
            f"Noisy nodes ({total_noisy}/{total_nodes}, p={cfg.node_noise_prob:.0%}): {', '.join(noisy_nodes_info) if noisy_nodes_info else 'none'}",
            f"Quantization nodes ({total_quant}/{total_nodes}): {', '.join(quant_nodes_info) if quant_nodes_info else 'none'}",
        ]
        return '\n'.join(lines)


def visualize_network(generator: RandomNNGenerator, n_samples: int = 5,
                     output_dir: str = "./output", network_id: int = 0,
                     skip_input_layer: bool = True, max_nodes_per_page: int = 6):
    """Visualize all layers of the network."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate time series
    output, all_layers = generator.propagate(n_samples)
    cfg = generator.sampled_config
    
    # Save config summary
    with open(f"{output_dir}/net{network_id:02d}_config.txt", 'w') as f:
        f.write(generator.get_config_summary())
    
    start_layer = 1 if skip_input_layer else 0
    
    for layer_idx in range(start_layer, len(all_layers)):
        layer_data = all_layers[layer_idx]
        n_nodes = layer_data.shape[1]
        
        n_pages = (n_nodes + max_nodes_per_page - 1) // max_nodes_per_page
        
        for page in range(n_pages):
            start_node = page * max_nodes_per_page
            end_node = min(start_node + max_nodes_per_page, n_nodes)
            nodes_this_page = end_node - start_node
            
            fig, axes = plt.subplots(n_samples, nodes_this_page, 
                                    figsize=(3 * nodes_this_page, 2.5 * n_samples))
            
            if n_samples == 1:
                axes = axes.reshape(1, -1)
            if nodes_this_page == 1:
                axes = axes.reshape(-1, 1)
            
            layer_size = generator.layer_sizes[layer_idx] if layer_idx < len(generator.layer_sizes) else n_nodes
            # Get activation for this layer (layer_idx 0 is input, so hidden layers start at 1)
            if layer_idx > 0 and layer_idx - 1 < len(cfg.activations):
                layer_act = cfg.activations[layer_idx - 1]
            else:
                layer_act = "input"
            title = f"Net {network_id} | Layer {layer_idx}: {layer_act} ({layer_size} nodes)"
            if n_pages > 1:
                title += f" [page {page+1}/{n_pages}]"
            fig.suptitle(title, fontsize=12)
            
            for sample_idx in range(n_samples):
                for node_offset, node_idx in enumerate(range(start_node, end_node)):
                    ax = axes[sample_idx, node_offset]
                    
                    ts = layer_data[sample_idx, node_idx, :]
                    color = plt.cm.tab10(node_offset % 10)
                    ax.plot(ts, color=color, linewidth=1)
                    
                    if sample_idx == 0:
                        ax.set_title(f"Node {node_idx}", fontsize=10)
                    if node_offset == 0:
                        ax.set_ylabel(f"S{sample_idx}", fontsize=10)
                    if sample_idx == n_samples - 1:
                        ax.set_xlabel("Time", fontsize=9)
            
            plt.tight_layout()
            
            if n_pages > 1:
                filename = f"{output_dir}/net{network_id:02d}_layer{layer_idx:02d}_page{page}.png"
            else:
                filename = f"{output_dir}/net{network_id:02d}_layer{layer_idx:02d}.png"
            
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()


def main():
    print("=" * 60)
    print("Random NN Generator Experiment")
    print("=" * 60)
    
    # Configuration with ranges
    # EXPERIMENT: Random inputs, random architecture, more noise
    config = RandomNNConfig(
        # Memory - RANDOM 1-8 dimensions
        memory_dim_range=(1, 8),
        memory_init='uniform',
        
        # Stochastic inputs - none
        stochastic_input_dim_range=(0, 0),
        
        # Time transforms - RANDOM 1-5 (all linear, no other activations)
        n_time_transforms_range=(1, 5),
        
        # Architecture - RANDOM layers (3-8) and nodes (4-16)
        n_hidden_layers_range=(3, 8),
        n_nodes_per_layer_range=(4, 16),
        
        # Activations - diverse per-node activations
        activation_choices=(
            'identity',   # f(x) = x
            'log',        # f(x) = sign(x) * log(1 + |x|)
            'sigmoid',    # f(x) = 1/(1+exp(-x))
            'abs',        # f(x) = |x|
            'sin',        # f(x) = sin(x)
            'tanh',       # f(x) = tanh(x)
            'rank',       # f(x) = normalized rank (percentile)
            'square',     # f(x) = x^2
            'power',      # f(x) = sign(x) * |x|^0.5
            'softplus',   # f(x) = log(1 + exp(x)) - smooth ReLU
            'step',       # f(x) = 0 if x < 0 else 1
            'modulo',     # f(x) = x mod 1
        ),
        
        # Initialization
        weight_init_choices=('xavier_normal',),
        weight_scale_range=(0.9, 1.1),
        bias_std_range=(0.0, 0.1),
        
        # Per-node noise - INCREASED (5-30% of nodes, higher std)
        node_noise_prob_range=(0.05, 0.30),
        node_noise_std_range=(0.01, 0.1),
        noise_dist_choices=('normal',),
        
        # Each layer gets its own random activation
        per_layer_activation=True,
        
        # Quantization nodes (discretization) - DISABLED
        quantization_node_prob=0.0,
        quantization_n_classes_range=(2, 5),
        
        # Sequence - 200 time steps
        seq_length=200,
    )
    
    output_dir = "/Users/franco/Documents/TabPFN3D/06_generator_experiments/random_all"
    
    n_networks = 8
    print(f"\nGenerating {n_networks} networks with random configurations...")
    
    for net_id in range(n_networks):
        generator = RandomNNGenerator(config, seed=42 + net_id)
        
        print(f"\n  Network {net_id}:")
        print(f"    {generator.get_config_summary().replace(chr(10), chr(10) + '    ')}")
        
        visualize_network(generator, n_samples=5, output_dir=output_dir,
                         network_id=net_id, skip_input_layer=True)
        print(f"    Done")
    
    print(f"\nDone! Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
