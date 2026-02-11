"""
DAG Generator with Preferential Attachment for Time Series.

Uses flexible DAG structure (not fully connected layers) with preferential attachment
via redirection probability. Each node connects to parents based on preferential attachment:
- Select a random parent from previous nodes
- With probability P, redirect to the parent's parent (if exists)
- This favors nodes with more children (preferential attachment)

N (number of nodes) ~ log-uniform
P (redirection probability) ~ gamma
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import os


@dataclass
class DAGPreferentialConfig:
    """
    Configuration with ranges for random sampling.
    Each network will sample its own parameters from these ranges.
    """
    # Memory dimensions: input -> trees -> output
    memory_input_dim_range: Tuple[int, int] = (2, 8)   # z_input dim
    memory_output_dim_range: Tuple[int, int] = (2, 8)  # z_output dim (used in DAG)
    memory_init: str = 'normal'  # 'uniform' or 'normal'
    memory_std_range: Tuple[float, float] = (0.01, 1.0)
    
    # Decision trees for memory mapping
    tree_depth_range: Tuple[int, int] = (2, 5)        # depth per tree
    tree_discrete_prob: float = 0.7                    # prob of discrete vs continuous tree
    tree_max_classes: int = 10                         # max leaves for discrete trees (= max classes)
    
    # Stochastic input (random at each time step)
    stochastic_input_dim_range: Tuple[int, int] = (0, 8)
    # Stochastic input std (log-uniform) - controls noise level at each time step
    stochastic_std_range: Tuple[float, float] = (0.001, 0.1)
    
    # Time input options
    n_time_transforms_range: Tuple[int, int] = (1, 8)
    
    # DAG structure parameters
    # N (number of nodes) ~ log-uniform between [a, b]
    n_nodes_range: Tuple[int, int] = (10, 100)
    
    # P (redirection probability) ~ gamma(alpha, beta)
    redirection_alpha: float = 2.0  # shape parameter
    redirection_beta: float = 5.0    # rate parameter
    
    # Number of root nodes (nodes with no parents in the DAG)
    # Note: All root nodes receive the same external input (time + memory + stochastic),
    # but each root node has its own weights and activations. This is independent of
    # the input dimensions (memory_dim, stochastic_dim, n_time_transforms).
    n_roots_range: Tuple[int, int] = (1, 5)
    
    # DAG density: average in-degree (edges per non-root node)
    # density = 1.0 -> each non-root has ~1 parent (chain/tree)
    # density = 2.0 -> each non-root has ~2 parents on average
    # density = 3.0+ -> dense DAG (many nodes with multiple parents)
    dag_density_range: Tuple[float, float] = (1.5, 3.0)
    
    # Activation - will be randomly chosen
    activation_choices: Tuple[str, ...] = (
        'identity', 'log', 'sigmoid', 'abs', 'sin', 'tanh', 'rank',
        'square', 'power', 'softplus', 'step', 'modulo', 'relu', 'leaky_relu', 'elu'
    )
    
    # Initialization
    weight_init_choices: Tuple[str, ...] = ('xavier_uniform', 'xavier_normal')
    weight_scale_range: Tuple[float, float] = (0.9, 1.1)
    bias_std_range: Tuple[float, float] = (0.0, 0.1)
    
    # Per-node noise configuration
    node_noise_prob_range: Tuple[float, float] = (0.05, 0.30)
    node_noise_std_range: Tuple[float, float] = (0.01, 0.1)
    noise_dist_choices: Tuple[str, ...] = ('normal',)
    
    # Quantization nodes
    quantization_node_prob: float = 0.0
    quantization_n_classes_range: Tuple[int, int] = (2, 5)
    
    # Sequence
    seq_length: int = 200


@dataclass
class TreeNode:
    """Node in a decision tree for memory mapping."""
    is_leaf: bool
    # For internal nodes:
    split_feature: int = -1      # which input dim to split on
    split_threshold: float = 0.0  # threshold for split
    left: Optional['TreeNode'] = None
    right: Optional['TreeNode'] = None
    # For leaf nodes:
    leaf_value: Optional[float] = None           # for discrete trees
    leaf_weights: Optional[np.ndarray] = None    # for continuous trees
    leaf_input_indices: Optional[List[int]] = None  # which inputs to combine


@dataclass
class MemoryTree:
    """Decision tree that maps memory_input to one memory_output dimension."""
    is_discrete: bool
    root: TreeNode = None
    depth: int = 0
    n_leaves: int = 0


@dataclass
class NodeNoiseConfig:
    """Noise configuration for a single node."""
    has_noise: bool
    distribution: str = 'normal'
    scale: float = 0.0


@dataclass
class NodeQuantizationConfig:
    """Quantization configuration for a single node."""
    is_quantization: bool
    n_classes: int = 0
    prototypes: np.ndarray = None
    class_values: np.ndarray = None


@dataclass
class DAGNode:
    """Represents a node in the DAG."""
    id: int
    parents: List[int]  # List of parent node IDs
    children: List[int]  # List of child node IDs
    is_root: bool
    topological_order: int
    activation: str
    weight: np.ndarray  # Weight vector for this node (one per parent)
    bias: float
    noise_config: NodeNoiseConfig
    quant_config: NodeQuantizationConfig


@dataclass
class SampledConfig:
    """Configuration that was sampled for a specific network."""
    memory_input_dim: int   # z_input dimension
    memory_output_dim: int  # z_output dimension (used in DAG)
    memory_init: str
    memory_std: float
    memory_trees: List[MemoryTree]  # one tree per output dim
    stochastic_input_dim: int
    stochastic_std: float
    time_transforms: List[Dict[str, Any]]
    n_nodes: int
    n_roots: int
    dag_density: float  # target average in-degree for non-root nodes
    redirection_prob: float
    node_activations: List[str]
    weight_init: str
    weight_scale: float
    bias_std: float
    node_noise: List[NodeNoiseConfig]
    node_noise_prob: float
    node_quantization: List[NodeQuantizationConfig]
    seq_length: int


class DAGPreferentialGenerator:
    """
    Generate time series by propagating through a DAG with preferential attachment.
    """
    
    def __init__(self, config: DAGPreferentialConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        
        # Sample the network configuration
        self.sampled_config = self._sample_config()
        
        # Build the DAG structure
        self._build_dag()
    
    def _log_uniform_int(self, low: int, high: int) -> int:
        """Sample integer from log-uniform distribution."""
        if low <= 0:
            low = 1
        log_low = np.log(low)
        log_high = np.log(high)
        log_sample = self.rng.uniform(log_low, log_high)
        return int(np.round(np.exp(log_sample)))
    
    def _log_uniform_float(self, low: float, high: float) -> float:
        """Sample float from log-uniform distribution."""
        if low <= 0:
            low = 1e-10
        log_low = np.log(low)
        log_high = np.log(high)
        log_sample = self.rng.uniform(log_low, log_high)
        return np.exp(log_sample)
    
    def _sample_value(self, init_type: str) -> float:
        """Sample a value in [-1, 1] using the specified distribution."""
        if init_type == 'uniform':
            return self.rng.uniform(-1, 1)
        else:  # 'normal'
            return np.clip(self.rng.normal(0, 0.5), -1, 1)
    
    def _build_tree_node(self, depth: int, max_depth: int, is_discrete: bool, 
                          input_dim: int, init_type: str) -> TreeNode:
        """Recursively build a decision tree node."""
        # Leaf node if max depth reached
        if depth >= max_depth:
            if is_discrete:
                # Discrete leaf: single value using init_type distribution
                return TreeNode(
                    is_leaf=True,
                    leaf_value=self._sample_value(init_type)
                )
            else:
                # Continuous leaf: linear combination of subset of inputs
                n_inputs_to_use = self.rng.integers(1, max(2, input_dim // 2 + 1))
                indices = self.rng.choice(input_dim, size=n_inputs_to_use, replace=False).tolist()
                # Weights using init_type, normalized by n_inputs to keep output in range
                weights = np.array([self._sample_value(init_type) for _ in range(n_inputs_to_use)]) / n_inputs_to_use
                return TreeNode(
                    is_leaf=True,
                    leaf_weights=weights,
                    leaf_input_indices=indices
                )
        
        # Internal node: split on random feature with threshold using init_type
        split_feature = self.rng.integers(0, input_dim)
        split_threshold = self._sample_value(init_type)
        
        return TreeNode(
            is_leaf=False,
            split_feature=split_feature,
            split_threshold=split_threshold,
            left=self._build_tree_node(depth + 1, max_depth, is_discrete, input_dim, init_type),
            right=self._build_tree_node(depth + 1, max_depth, is_discrete, input_dim, init_type)
        )
    
    def _build_memory_tree(self, input_dim: int, is_discrete: bool, max_depth: int, init_type: str) -> MemoryTree:
        """Build a single decision tree for one memory output dimension."""
        cfg = self.config
        
        # For discrete trees, limit depth to not exceed max_classes leaves
        if is_discrete:
            max_leaves = cfg.tree_max_classes
            max_depth = min(max_depth, int(np.floor(np.log2(max_leaves))))
            max_depth = max(1, max_depth)
        
        root = self._build_tree_node(0, max_depth, is_discrete, input_dim, init_type)
        n_leaves = 2 ** max_depth  # Full binary tree
        
        return MemoryTree(
            is_discrete=is_discrete,
            root=root,
            depth=max_depth,
            n_leaves=n_leaves
        )
    
    def _sample_config(self) -> SampledConfig:
        """Sample all configuration parameters for this network."""
        cfg = self.config
        
        # Sample memory input/output dimensions
        memory_input_dim = self._log_uniform_int(
            max(1, cfg.memory_input_dim_range[0]), cfg.memory_input_dim_range[1])
        memory_output_dim = self._log_uniform_int(
            max(1, cfg.memory_output_dim_range[0]), cfg.memory_output_dim_range[1])
        
        # Sample memory std
        memory_std = self._log_uniform_float(
            max(1e-6, cfg.memory_std_range[0]), cfg.memory_std_range[1])
        
        # Build decision trees for each memory output dimension
        memory_trees = []
        for _ in range(memory_output_dim):
            is_discrete = self.rng.random() < cfg.tree_discrete_prob
            depth = self._log_uniform_int(cfg.tree_depth_range[0], cfg.tree_depth_range[1])
            tree = self._build_memory_tree(memory_input_dim, is_discrete, depth, cfg.memory_init)
            memory_trees.append(tree)
        
        # Sample stochastic input dimension (log-uniform)
        stochastic_dim = self._log_uniform_int(
            max(1, cfg.stochastic_input_dim_range[0]),
            cfg.stochastic_input_dim_range[1]
        )
        
        # Sample stochastic input std
        stochastic_std = self._log_uniform_float(
            max(1e-6, cfg.stochastic_std_range[0]),
            cfg.stochastic_std_range[1]
        )
        
        # Sample time transforms
        n_transforms = self._log_uniform_int(max(1, cfg.n_time_transforms_range[0]), cfg.n_time_transforms_range[1])
        selected_transforms = []
        for _ in range(n_transforms):
            selected_transforms.append({'name': 'linear', 'params': None})
        
        # Sample N (number of nodes)
        n_nodes = self._log_uniform_int(max(1, cfg.n_nodes_range[0]), cfg.n_nodes_range[1])
        
        # Sample P (redirection probability)
        redirection_prob = np.clip(self.rng.gamma(cfg.redirection_alpha, 1.0 / cfg.redirection_beta), 0.0, 1.0)
        
        # Sample number of root nodes (log-uniform)
        n_roots = self._log_uniform_int(
            max(1, cfg.n_roots_range[0]),
            min(n_nodes, cfg.n_roots_range[1])
        )
        
        # Sample DAG density (average in-degree for non-root nodes)
        dag_density = self.rng.uniform(
            cfg.dag_density_range[0],
            cfg.dag_density_range[1]
        )
        
        # Sample activations for each node
        node_activations = [self.rng.choice(list(cfg.activation_choices)) for _ in range(n_nodes)]
        
        # Sample initialization
        weight_init = self.rng.choice(list(cfg.weight_init_choices))
        weight_scale = self.rng.uniform(cfg.weight_scale_range[0], cfg.weight_scale_range[1])
        bias_std = self.rng.uniform(cfg.bias_std_range[0], cfg.bias_std_range[1])
        
        # Sample noise probability
        node_noise_prob = self._log_uniform_float(max(0.001, cfg.node_noise_prob_range[0]), cfg.node_noise_prob_range[1])
        
        # Calculate input dimension for DAG (uses memory_output_dim)
        n_time_inputs = len(selected_transforms)
        total_input_dim = n_time_inputs + memory_output_dim + stochastic_dim
        
        # Sample per-node noise configuration
        node_noise = []
        for _ in range(n_nodes):
            has_noise = self.rng.random() < node_noise_prob
            if has_noise:
                dist = self.rng.choice(list(cfg.noise_dist_choices))
                scale = self._log_uniform_float(max(1e-6, cfg.node_noise_std_range[0]), cfg.node_noise_std_range[1])
                node_noise.append(NodeNoiseConfig(has_noise=True, distribution=dist, scale=scale))
            else:
                node_noise.append(NodeNoiseConfig(has_noise=False))
        
        # Sample per-node quantization configuration
        node_quantization = []
        for _ in range(n_nodes):
            is_quant = self.rng.random() < cfg.quantization_node_prob
            if is_quant:
                n_classes = self.rng.integers(
                    cfg.quantization_n_classes_range[0],
                    cfg.quantization_n_classes_range[1] + 1
                )
                prototypes = self.rng.normal(0, 1, (n_classes, total_input_dim))
                class_values = self.rng.normal(0, 1, n_classes)
                node_quantization.append(NodeQuantizationConfig(
                    is_quantization=True,
                    n_classes=n_classes,
                    prototypes=prototypes,
                    class_values=class_values
                ))
            else:
                node_quantization.append(NodeQuantizationConfig(is_quantization=False))
        
        return SampledConfig(
            memory_input_dim=memory_input_dim,
            memory_output_dim=memory_output_dim,
            memory_init=cfg.memory_init,
            memory_std=memory_std,
            memory_trees=memory_trees,
            stochastic_input_dim=stochastic_dim,
            stochastic_std=stochastic_std,
            time_transforms=selected_transforms,
            n_nodes=n_nodes,
            n_roots=n_roots,
            dag_density=dag_density,
            redirection_prob=redirection_prob,
            node_activations=node_activations,
            weight_init=weight_init,
            weight_scale=weight_scale,
            bias_std=bias_std,
            node_noise=node_noise,
            node_noise_prob=node_noise_prob,
            node_quantization=node_quantization,
            seq_length=cfg.seq_length
        )
    
    def _build_dag(self):
        """
        Build DAG structure using preferential attachment with redirection.
        
        Algorithm:
        1. Create N nodes with topological ordering
        2. First n_roots nodes are root nodes (no parents)
        3. For each non-root node:
           - Select a random parent from previous nodes
           - With probability P, redirect to parent's parent (if exists)
           - This creates preferential attachment (nodes with more children get more connections)
        4. Guarantees connectivity: each non-root node has at least one parent
        """
        cfg = self.sampled_config
        n_nodes = cfg.n_nodes
        n_roots = cfg.n_roots
        P = cfg.redirection_prob
        
        # Initialize nodes
        self.nodes: Dict[int, DAGNode] = {}
        self.topological_order: List[int] = []
        
        # Assign topological order (roots first, then shuffle non-roots)
        root_orders = list(range(n_roots))
        non_root_orders = list(range(n_roots, n_nodes))
        self.rng.shuffle(non_root_orders)
        topo_order = root_orders + non_root_orders
        
        # Track number of children per node (for preferential attachment)
        child_counts = np.zeros(n_nodes, dtype=int)
        
        # Create nodes and build edges
        for order in range(n_nodes):
            node_id = order
            is_root = order < n_roots
            
            # Initialize node structure
            parents = []
            children = []
            
            if not is_root:
                # Non-root node: must have at least one parent
                # Select initial parent from previous nodes
                available_parents = list(range(order))  # All previous nodes
                initial_parent = self.rng.choice(available_parents)
                
                # Apply redirection with probability P
                final_parent = initial_parent
                if self.rng.random() < P:
                    # Try to redirect to parent's parent
                    # Find parents of the initial parent
                    if initial_parent in self.nodes:
                        parent_parents = self.nodes[initial_parent].parents
                        if len(parent_parents) > 0:
                            # Redirect to a random parent of the initial parent
                            final_parent = self.rng.choice(parent_parents)
                
                parents.append(final_parent)
                child_counts[final_parent] += 1
                
                # Add additional parents with preferential attachment
                # Use dag_density to determine number of extra parents
                # density = 1.0 means ~1 parent, density = 2.5 means ~2.5 parents on average
                target_extra = cfg.dag_density - 1.0  # Extra parents beyond the primary one
                # Sample around target_extra using Poisson-like distribution
                n_additional = self.rng.poisson(max(0, target_extra))
                n_additional = min(n_additional, order - 1)  # Can't have more parents than previous nodes
                
                for _ in range(n_additional):
                    # Preferential attachment: probability proportional to (1 + child_count)
                    weights = 1.0 + child_counts[:order]
                    weights = weights / weights.sum()
                    additional_parent = self.rng.choice(order, p=weights)
                    
                    if additional_parent not in parents:
                        parents.append(additional_parent)
                        child_counts[additional_parent] += 1
                
                # Update children lists of parents
                for parent_id in parents:
                    if parent_id in self.nodes:
                        if node_id not in self.nodes[parent_id].children:
                            self.nodes[parent_id].children.append(node_id)
            
            # Calculate input dimension for this node
            if is_root:
                # Root nodes: input from external sources (uses memory_output_dim)
                n_time_inputs = len(cfg.time_transforms)
                input_dim = n_time_inputs + cfg.memory_output_dim + cfg.stochastic_input_dim
            else:
                # Non-root nodes: input from parents
                input_dim = len(parents)
            
            # Initialize weights and bias
            if cfg.weight_init == 'xavier_uniform':
                a = cfg.weight_scale * np.sqrt(6.0 / (input_dim + 1))
                weight = self.rng.uniform(-a, a, input_dim)
            else:  # xavier_normal
                std = cfg.weight_scale * np.sqrt(2.0 / input_dim)
                weight = self.rng.normal(0, std, input_dim)
            
            bias = 0.0 if cfg.bias_std == 0 else self.rng.normal(0, cfg.bias_std)
            
            # Create node
            self.nodes[node_id] = DAGNode(
                id=node_id,
                parents=parents,
                children=children,
                is_root=is_root,
                topological_order=topo_order[order],
                activation=cfg.node_activations[node_id],
                weight=weight,
                bias=bias,
                noise_config=cfg.node_noise[node_id],
                quant_config=cfg.node_quantization[node_id]
            )
            self.topological_order.append(node_id)
        
        # Update children lists (now that all nodes exist)
        for node_id, node in self.nodes.items():
            for parent_id in node.parents:
                if node_id not in self.nodes[parent_id].children:
                    self.nodes[parent_id].children.append(node_id)
        
        # Store input dimension (uses memory_output_dim for DAG)
        n_time_inputs = len(cfg.time_transforms)
        self.time_input_dim = n_time_inputs
        self.total_input_dim = n_time_inputs + cfg.memory_output_dim + cfg.stochastic_input_dim
    
    def _evaluate_tree_node(self, node: TreeNode, x: np.ndarray) -> np.ndarray:
        """
        Evaluate a tree node for a batch of inputs.
        x: (n_samples, input_dim)
        Returns: (n_samples,)
        """
        if node.is_leaf:
            if node.leaf_value is not None:
                # Discrete leaf: return constant value
                return np.full(x.shape[0], node.leaf_value)
            else:
                # Continuous leaf: linear combination of selected inputs
                selected = x[:, node.leaf_input_indices]  # (n_samples, n_selected)
                return np.dot(selected, node.leaf_weights)
        else:
            # Internal node: split based on feature and threshold
            left_mask = x[:, node.split_feature] <= node.split_threshold
            result = np.zeros(x.shape[0])
            if np.any(left_mask):
                result[left_mask] = self._evaluate_tree_node(node.left, x[left_mask])
            if np.any(~left_mask):
                result[~left_mask] = self._evaluate_tree_node(node.right, x[~left_mask])
            return result
    
    def _evaluate_memory_tree(self, tree: MemoryTree, x: np.ndarray) -> np.ndarray:
        """
        Evaluate a memory tree for a batch of inputs.
        x: (n_samples, memory_input_dim)
        Returns: (n_samples,)
        """
        return self._evaluate_tree_node(tree.root, x)
    
    def _compute_time_transform(self, name: str, u: np.ndarray, params: Optional[Dict] = None) -> np.ndarray:
        """Compute a time transform given normalized time u ∈ [0, 1]."""
        if name == 'linear':
            return 2 * u - 1
        else:
            raise ValueError(f"Unknown transform: {name}")
    
    def _activation(self, x: np.ndarray, act: str) -> np.ndarray:
        """Apply activation function."""
        if act == 'identity':
            return x
        elif act == 'log':
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
            if x.ndim == 2:
                ranks = np.argsort(np.argsort(x, axis=1), axis=1)
                return ranks / (x.shape[1] - 1) * 2 - 1
            else:
                return x
        elif act == 'square':
            return x ** 2
        elif act == 'power':
            return np.sign(x) * np.sqrt(np.abs(x))
        elif act == 'softplus':
            return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -500, 20))))
        elif act == 'step':
            return np.where(x >= 0, 1.0, 0.0)
        elif act == 'modulo':
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
        """Generate time input features."""
        u = np.linspace(0, 1, T)
        inputs = []
        for transform in self.sampled_config.time_transforms:
            name = transform['name']
            params = transform['params']
            inputs.append(self._compute_time_transform(name, u, params))
        return np.array(inputs)
    
    def _generate_memory(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate memory vectors by:
        1. Sample memory_input
        2. Apply decision trees to get memory_output
        
        Returns:
            memory_input: (n_samples, memory_input_dim)
            memory_output: (n_samples, memory_output_dim)
        """
        cfg = self.sampled_config
        
        # 1. Generate memory_input
        if cfg.memory_init == 'uniform':
            memory_input = self.rng.uniform(-1, 1, (n_samples, cfg.memory_input_dim))
        else:
            memory_input = np.clip(
                self.rng.normal(0, cfg.memory_std, (n_samples, cfg.memory_input_dim)), 
                -1, 1
            )
        
        # 2. Apply trees to get memory_output
        memory_output = np.zeros((n_samples, cfg.memory_output_dim))
        for i, tree in enumerate(cfg.memory_trees):
            memory_output[:, i] = self._evaluate_memory_tree(tree, memory_input)
        
        # Clip output to [-1, 1] for safety
        memory_output = np.clip(memory_output, -1, 1)
        
        return memory_input, memory_output
    
    def _generate_stochastic_inputs(self, n_samples: int, T: int) -> Optional[np.ndarray]:
        """Generate random inputs that vary at each time step."""
        if self.sampled_config.stochastic_input_dim == 0:
            return None
        # Use sampled stochastic_std to control noise level
        return self.rng.normal(0, self.sampled_config.stochastic_std, 
                              (n_samples, self.sampled_config.stochastic_input_dim, T))
    
    def propagate(self, n_samples: int = 5) -> Tuple[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]:
        """
        Propagate inputs through the DAG in topological order.
        
        Returns:
            output: Final node outputs (n_samples, n_nodes, T)
            all_nodes: List of all node activations
            memory_info: Dict with memory_input and memory_output
        """
        cfg = self.sampled_config
        T = cfg.seq_length
        
        # Generate inputs
        time_inputs = self._generate_time_inputs(T)  # (time_dim, T)
        memory_input, memory_output = self._generate_memory(n_samples)
        stochastic = self._generate_stochastic_inputs(n_samples, T)
        
        # Build input layer for root nodes (use memory_output)
        time_broadcast = np.broadcast_to(time_inputs[np.newaxis, :, :], (n_samples, self.time_input_dim, T))
        memory_broadcast = np.broadcast_to(memory_output[:, :, np.newaxis], (n_samples, cfg.memory_output_dim, T))
        
        if stochastic is not None:
            root_input = np.concatenate([time_broadcast, memory_broadcast, stochastic], axis=1)
        else:
            root_input = np.concatenate([time_broadcast, memory_broadcast], axis=1)
        
        # Store outputs for each node: (n_samples, n_nodes, T)
        node_outputs = np.zeros((n_samples, cfg.n_nodes, T))
        all_nodes = []
        
        # Propagate in topological order
        for node_id in self.topological_order:
            node = self.nodes[node_id]
            
            if node.is_root:
                # Root node: use external input
                # Aggregate all input dimensions
                node_input = root_input  # (n_samples, total_input_dim, T)
            else:
                # Non-root node: aggregate from parents
                parent_outputs = []
                for parent_id in node.parents:
                    parent_outputs.append(node_outputs[:, parent_id, :])  # (n_samples, T)
                
                if len(parent_outputs) == 1:
                    node_input = parent_outputs[0][:, np.newaxis, :]  # (n_samples, 1, T)
                else:
                    node_input = np.stack(parent_outputs, axis=1)  # (n_samples, n_parents, T)
            
            # Compute node output
            # node_input: (n_samples, input_dim, T)
            # node.weight: (input_dim,)
            # node.bias: scalar
            
            # Linear combination: sum over input dimension
            node_out = np.einsum('d,ndt->nt', node.weight, node_input) + node.bias
            
            # Apply activation
            node_out = self._activation(node_out, node.activation)
            
            # Add noise
            if node.noise_config.has_noise:
                if node.noise_config.distribution == 'normal':
                    noise = self.rng.normal(0, node.noise_config.scale, (n_samples, T))
                else:
                    noise = 0
                node_out = node_out + noise
            
            node_outputs[:, node_id, :] = node_out
            all_nodes.append(node_out)
        
        memory_info = {
            'memory_input': memory_input,
            'memory_output': memory_output
        }
        return node_outputs, all_nodes, memory_info
    
    def get_config_summary(self) -> str:
        """Get a summary of the sampled configuration."""
        cfg = self.sampled_config
        
        # Count edges
        total_edges = sum(len(node.parents) for node in self.nodes.values())
        
        # Count noisy nodes
        noisy_count = sum(1 for n in cfg.node_noise if n.has_noise)
        
        # Count tree types
        n_discrete = sum(1 for t in cfg.memory_trees if t.is_discrete)
        n_continuous = len(cfg.memory_trees) - n_discrete
        
        # Actual density = edges / (nodes - roots) for non-root nodes
        n_non_roots = cfg.n_nodes - cfg.n_roots
        actual_density = total_edges / n_non_roots if n_non_roots > 0 else 0
        
        lines = [
            f"Nodes: {cfg.n_nodes} (roots: {cfg.n_roots})",
            f"Edges: {total_edges} (actual density: {actual_density:.2f}, target: {cfg.dag_density:.2f})",
            f"Redirection prob: {cfg.redirection_prob:.3f}",
            f"Memory: {cfg.memory_input_dim} -> {cfg.memory_output_dim} (trees: {n_discrete}D/{n_continuous}C)",
            f"Stochastic: {cfg.stochastic_input_dim} dims",
            f"Time transforms: {len(cfg.time_transforms)}",
            f"Noisy nodes: {noisy_count}/{cfg.n_nodes}",
        ]
        return '\n'.join(lines)


def visualize_network(generator: DAGPreferentialGenerator, n_samples: int = 5,
                      output_dir: str = "./output", network_id: int = 0,
                      max_nodes_per_page: int = 6):
    """Visualize all nodes of the network."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate time series
    output, all_nodes, memory_info = generator.propagate(n_samples)
    cfg = generator.sampled_config
    
    # Save config summary
    with open(f"{output_dir}/net{network_id:02d}_config.txt", 'w') as f:
        f.write(generator.get_config_summary())
    
    n_nodes = cfg.n_nodes
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
        
        title = f"Net {network_id} | DAG Preferential (N={cfg.n_nodes}, P={cfg.redirection_prob:.3f})"
        if n_pages > 1:
            title += f" [page {page+1}/{n_pages}]"
        fig.suptitle(title, fontsize=12)
        
        for sample_idx in range(n_samples):
            for node_offset, node_idx in enumerate(range(start_node, end_node)):
                ax = axes[sample_idx, node_offset]
                
                ts = output[sample_idx, node_idx, :]
                color = plt.cm.tab10(node_offset % 10)
                ax.plot(ts, color=color, linewidth=1)
                
                node = generator.nodes[node_idx]
                node_type = "ROOT" if node.is_root else f"P{len(node.parents)}"
                in_edges = len(node.parents)
                out_edges = len(node.children)
                topo = node.topological_order
                node_info = f"topo={topo} in={in_edges} out={out_edges}"
                
                if sample_idx == 0:
                    ax.set_title(f"Node {node_idx} ({node_type}) | {node_info}", fontsize=9)
                if node_offset == 0:
                    ax.set_ylabel(f"S{sample_idx}", fontsize=10)
                if sample_idx == n_samples - 1:
                    ax.set_xlabel("Time", fontsize=9)
        
        plt.tight_layout()
        
        if n_pages > 1:
            filename = f"{output_dir}/net{network_id:02d}_page{page}.png"
        else:
            filename = f"{output_dir}/net{network_id:02d}.png"
        
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()


def check_memory_trees(generator: DAGPreferentialGenerator, n_samples: int = 100, output_dir: str = "."):
    """
    Check that memory trees work correctly.
    Visualize memory_input -> memory_output mapping.
    """
    cfg = generator.sampled_config
    
    print("\n" + "=" * 50)
    print("MEMORY TREES CHECK")
    print("=" * 50)
    print(f"Input dim: {cfg.memory_input_dim}, Output dim: {cfg.memory_output_dim}")
    
    # Generate memory
    memory_input, memory_output = generator._generate_memory(n_samples)
    
    print(f"memory_input shape: {memory_input.shape}, range: [{memory_input.min():.3f}, {memory_input.max():.3f}]")
    print(f"memory_output shape: {memory_output.shape}, range: [{memory_output.min():.3f}, {memory_output.max():.3f}]")
    
    # Check each tree
    for i, tree in enumerate(cfg.memory_trees):
        out_i = memory_output[:, i]
        unique_vals = len(np.unique(np.round(out_i, 4)))
        tree_type = "DISCRETE" if tree.is_discrete else "CONTINUOUS"
        print(f"  Tree {i}: {tree_type}, depth={tree.depth}, "
              f"out range=[{out_i.min():.3f}, {out_i.max():.3f}], unique≈{unique_vals}")
    
    # Plot if few dimensions
    if cfg.memory_output_dim <= 8:
        fig, axes = plt.subplots(2, min(4, cfg.memory_output_dim), 
                                 figsize=(3*min(4, cfg.memory_output_dim), 6))
        if cfg.memory_output_dim == 1:
            axes = np.array([[axes[0]], [axes[1]]])
        elif cfg.memory_output_dim <= 4:
            axes = axes.reshape(2, -1)
        
        for i in range(min(4, cfg.memory_output_dim)):
            # Histogram of output
            ax = axes[0, i]
            ax.hist(memory_output[:, i], bins=30, alpha=0.7)
            tree_type = "D" if cfg.memory_trees[i].is_discrete else "C"
            ax.set_title(f"Tree {i} ({tree_type})")
            ax.set_xlabel("value")
            
            # Scatter: input[0] vs output[i]
            ax = axes[1, i]
            ax.scatter(memory_input[:, 0], memory_output[:, i], alpha=0.5, s=10)
            ax.set_xlabel("input[0]")
            ax.set_ylabel(f"output[{i}]")
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, "memory_trees_check.png")
        plt.savefig(filepath, dpi=150)
        plt.close()
        print(f"\nSaved: {filepath}")
    
    return memory_input, memory_output


# =============================================================================
# DATASET GENERATION
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for dataset generation from DAG generator."""
    # Sample modes probabilities
    prob_iid: float = 1.0           # Each sample is independent sequence
    prob_sliding: float = 0.0       # Sliding window over single sequence
    prob_mixed: float = 0.0         # Multiple sequences, multiple windows each
    
    # Constraints
    max_samples: int = 1000
    max_features: int = 12
    max_timesteps: int = 1000       # max T (sequence length)
    max_m_times_t: int = 500        # max n_features * t_subseq
    
    # Subsequence length range (t_subseq)
    t_subseq_range: Tuple[int, int] = (20, 200)
    
    # Leftover margin range (log-uniform, prioritize small)
    t_margin_range: Tuple[int, int] = (1, 50)
    
    # Train/test split
    train_ratio: float = 0.8
    
    # Sliding window stride (fraction of t_subseq)
    stride_fraction: float = 0.25


@dataclass
class GeneratedDataset:
    """A generated dataset with train/test splits."""
    X_train: np.ndarray      # (n_train, n_features, t_subseq)
    y_train: np.ndarray      # (n_train,)
    X_test: np.ndarray       # (n_test, n_features, t_subseq)
    y_test: np.ndarray       # (n_test,)
    n_classes: int
    n_features: int
    t_subseq: int
    T: int                   # total sequence length used
    sample_mode: str
    feature_node_ids: List[int]
    label_tree_idx: int
    class_balance: Dict


def _assign_leaf_ids(node: TreeNode, leaf_id_map: Dict, current_id: List[int]):
    """
    Pre-order traversal to assign unique IDs to each leaf.
    leaf_id_map: dict mapping leaf node (by id) to leaf_id
    current_id: mutable [counter]
    """
    if node.is_leaf:
        leaf_id_map[id(node)] = current_id[0]
        current_id[0] += 1
    else:
        _assign_leaf_ids(node.left, leaf_id_map, current_id)
        _assign_leaf_ids(node.right, leaf_id_map, current_id)


def _get_leaf_index(node: TreeNode, x: np.ndarray, leaf_id_map: Dict) -> np.ndarray:
    """
    Traverse tree and return the leaf INDEX (class) for each sample.
    leaf_id_map: dict mapping leaf node id -> leaf class index
    x: (n_samples, input_dim)
    Returns: (n_samples,) with leaf indices
    """
    if node.is_leaf:
        leaf_id = leaf_id_map[id(node)]
        return np.full(x.shape[0], leaf_id, dtype=np.int32)
    else:
        left_mask = x[:, node.split_feature] <= node.split_threshold
        result = np.zeros(x.shape[0], dtype=np.int32)
        
        if np.any(left_mask):
            result[left_mask] = _get_leaf_index(node.left, x[left_mask], leaf_id_map)
        if np.any(~left_mask):
            result[~left_mask] = _get_leaf_index(node.right, x[~left_mask], leaf_id_map)
        return result


def get_discrete_tree_classes(tree: MemoryTree, x: np.ndarray) -> np.ndarray:
    """
    Get leaf indices (classes) for a discrete tree.
    x: (n_samples, input_dim)
    Returns: (n_samples,) with class indices 0, 1, 2, ...
    """
    # First pass: assign unique IDs to all leaves
    leaf_id_map = {}
    _assign_leaf_ids(tree.root, leaf_id_map, [0])
    
    # Second pass: get leaf index for each sample
    return _get_leaf_index(tree.root, x, leaf_id_map)


def _sample_n_features(rng: np.random.Generator, max_features: int) -> int:
    """Sample number of features with strong preference for few (univariate)."""
    # P(1)=50%, P(2)=25%, P(3)=12.5%, P(4+)=12.5%
    r = rng.random()
    if r < 0.50:
        return 1
    elif r < 0.75:
        return 2
    elif r < 0.875:
        return 3
    else:
        return min(rng.integers(4, max(5, max_features + 1)), max_features)


def _log_uniform_int(rng: np.random.Generator, low: int, high: int) -> int:
    """Sample integer from log-uniform distribution (favors smaller values)."""
    if low >= high:
        return low
    log_low = np.log(max(1, low))
    log_high = np.log(max(1, high))
    return int(np.exp(rng.uniform(log_low, log_high)))


def generate_dataset(
    dag_config: DAGPreferentialConfig,
    ds_config: DatasetConfig,
    seed: int = 42
) -> GeneratedDataset:
    """
    Generate a classification dataset using DAGPreferentialGenerator.
    
    Process:
    1. Sample sample_mode (iid, sliding_window, mixed)
    2. Sample n_features (prefer univariate)
    3. Sample t_subseq (subsequence length)
    4. Calculate required T based on mode
    5. Sample margin (leftover) with log-uniform (prefer small)
    6. Create generator with T = required + margin
    7. Extract features from nodes with in_degree > 1
    8. Labels from discrete tree leaf indices
    9. Split train/test without temporal leakage
    
    Returns:
        GeneratedDataset with train/test splits
    """
    rng = np.random.default_rng(seed)
    
    # 1. Sample mode
    mode_roll = rng.random()
    if mode_roll < ds_config.prob_iid:
        sample_mode = 'iid'
    elif mode_roll < ds_config.prob_iid + ds_config.prob_sliding:
        sample_mode = 'sliding_window'
    else:
        sample_mode = 'mixed'
    
    # 2. Sample n_features
    n_features = _sample_n_features(rng, ds_config.max_features)
    
    # 3. Sample t_subseq (respecting m*t constraint)
    max_t = min(ds_config.t_subseq_range[1], 
                ds_config.max_m_times_t // n_features,
                ds_config.max_timesteps)
    min_t = max(ds_config.t_subseq_range[0], 10)
    t_subseq = _log_uniform_int(rng, min_t, max_t)
    
    # 4. Sample n_samples
    n_samples = _log_uniform_int(rng, 50, ds_config.max_samples)
    
    # 5. Calculate required T based on mode
    if sample_mode == 'iid':
        # Each sample is independent: T = t_subseq + margin
        required_t = t_subseq
        n_sequences = n_samples
        
    elif sample_mode == 'sliding_window':
        # All samples from one sequence with sliding window
        stride = max(1, int(t_subseq * ds_config.stride_fraction))
        # T = t_subseq + (n_samples - 1) * stride
        required_t = t_subseq + (n_samples - 1) * stride
        n_sequences = 1
        
    else:  # mixed
        # Multiple sequences, multiple windows each
        n_sequences = max(2, n_samples // 10)
        samples_per_seq = n_samples // n_sequences
        stride = max(1, int(t_subseq * ds_config.stride_fraction))
        required_t = t_subseq + (samples_per_seq - 1) * stride
    
    # 6. Sample margin (log-uniform, prefer small)
    margin = _log_uniform_int(rng, ds_config.t_margin_range[0], ds_config.t_margin_range[1])
    T = min(required_t + margin, ds_config.max_timesteps)
    
    # Adjust if T is too small
    if T < t_subseq + 1:
        T = t_subseq + 1
    
    # 7. Create generator with this T
    dag_config_with_T = DAGPreferentialConfig(
        memory_input_dim_range=dag_config.memory_input_dim_range,
        memory_output_dim_range=dag_config.memory_output_dim_range,
        memory_init=dag_config.memory_init,
        memory_std_range=dag_config.memory_std_range,
        tree_depth_range=dag_config.tree_depth_range,
        tree_discrete_prob=dag_config.tree_discrete_prob,
        tree_max_classes=dag_config.tree_max_classes,
        stochastic_input_dim_range=dag_config.stochastic_input_dim_range,
        stochastic_std_range=dag_config.stochastic_std_range,
        n_time_transforms_range=dag_config.n_time_transforms_range,
        n_nodes_range=dag_config.n_nodes_range,
        redirection_alpha=dag_config.redirection_alpha,
        redirection_beta=dag_config.redirection_beta,
        n_roots_range=dag_config.n_roots_range,
        activation_choices=dag_config.activation_choices,
        weight_init_choices=dag_config.weight_init_choices,
        weight_scale_range=dag_config.weight_scale_range,
        bias_std_range=dag_config.bias_std_range,
        node_noise_prob_range=dag_config.node_noise_prob_range,
        node_noise_std_range=dag_config.node_noise_std_range,
        noise_dist_choices=dag_config.noise_dist_choices,
        quantization_node_prob=dag_config.quantization_node_prob,
        quantization_n_classes_range=dag_config.quantization_n_classes_range,
        seq_length=T,
    )
    
    generator = DAGPreferentialGenerator(dag_config_with_T, seed=seed)
    cfg = generator.sampled_config
    
    # 8. Check for discrete trees
    discrete_tree_indices = [i for i, tree in enumerate(cfg.memory_trees) if tree.is_discrete]
    if len(discrete_tree_indices) == 0:
        raise ValueError("No discrete trees found for labels. Increase tree_discrete_prob.")
    
    label_tree_idx = discrete_tree_indices[0]
    label_tree = cfg.memory_trees[label_tree_idx]
    
    # 9. Find feature nodes (STRICTLY in_degree > 1)
    feature_candidates = [node_id for node_id, node in generator.nodes.items() 
                         if len(node.parents) > 1]
    
    if len(feature_candidates) == 0:
        raise ValueError("No nodes with in_degree > 1 found. Need more complex DAG.")
    
    if len(feature_candidates) < n_features:
        raise ValueError(f"Only {len(feature_candidates)} nodes with in_degree>1, need {n_features}")
    
    # Select up to n_features (sorted by topological order)
    feature_candidates = sorted(feature_candidates, 
                                key=lambda nid: generator.nodes[nid].topological_order)
    
    # Random selection from candidates
    if len(feature_candidates) > n_features:
        selected_idx = rng.choice(len(feature_candidates), size=n_features, replace=False)
        feature_node_ids = [feature_candidates[i] for i in sorted(selected_idx)]
    else:
        feature_node_ids = feature_candidates
    
    # 10. Generate sequences and extract samples
    output, all_nodes, memory_info = generator.propagate(n_sequences)
    memory_input = memory_info['memory_input']
    
    X_list = []
    y_list = []
    sequence_ids = []  # Track which sequence each sample comes from
    
    if sample_mode == 'iid':
        # Each sample from different sequence, starting at t=0
        for seq_idx in range(min(n_sequences, n_samples)):
            start = 0
            end = start + t_subseq
            if end > T:
                continue
            
            # Extract features
            features = output[seq_idx, feature_node_ids, start:end]  # (n_features, t_subseq)
            X_list.append(features)
            
            # Get label
            y_val = get_discrete_tree_classes(label_tree, memory_input[seq_idx:seq_idx+1])[0]
            y_list.append(y_val)
            sequence_ids.append(seq_idx)
    
    elif sample_mode == 'sliding_window':
        seq_idx = 0
        stride = max(1, int(t_subseq * ds_config.stride_fraction))
        
        for sample_idx in range(n_samples):
            start = sample_idx * stride
            end = start + t_subseq
            if end > T:
                break
            
            features = output[seq_idx, feature_node_ids, start:end]
            X_list.append(features)
            
            y_val = get_discrete_tree_classes(label_tree, memory_input[seq_idx:seq_idx+1])[0]
            y_list.append(y_val)
            sequence_ids.append(seq_idx)
    
    else:  # mixed
        stride = max(1, int(t_subseq * ds_config.stride_fraction))
        samples_per_seq = n_samples // n_sequences
        
        for seq_idx in range(n_sequences):
            for win_idx in range(samples_per_seq):
                start = win_idx * stride
                end = start + t_subseq
                if end > T:
                    break
                
                features = output[seq_idx, feature_node_ids, start:end]
                X_list.append(features)
                
                y_val = get_discrete_tree_classes(label_tree, memory_input[seq_idx:seq_idx+1])[0]
                y_list.append(y_val)
                sequence_ids.append(seq_idx)
    
    if len(X_list) == 0:
        raise ValueError("No samples generated")
    
    X = np.stack(X_list, axis=0).astype(np.float32)  # (n_samples, n_features, t_subseq)
    y_raw = np.array(y_list, dtype=np.int32)
    
    # Remap classes to consecutive 0, 1, 2, ...
    unique_classes = np.unique(y_raw)
    if len(unique_classes) < 2:
        raise ValueError(f"Only {len(unique_classes)} class found. Need at least 2 classes.")
    
    class_map = {old: new for new, old in enumerate(unique_classes)}
    y = np.array([class_map[c] for c in y_raw], dtype=np.int32)
    
    # 11. Train/test split (without temporal leakage)
    n_total = len(X)
    
    if sample_mode == 'sliding_window':
        # Temporal split: first train_ratio% for train, rest for test
        split_idx = int(n_total * ds_config.train_ratio)
        train_idx = np.arange(split_idx)
        test_idx = np.arange(split_idx, n_total)
        
    elif sample_mode == 'mixed':
        # Sequence-level split to avoid leakage
        unique_seqs = list(set(sequence_ids))
        n_train_seqs = max(1, int(len(unique_seqs) * ds_config.train_ratio))
        rng.shuffle(unique_seqs)
        train_seqs = set(unique_seqs[:n_train_seqs])
        
        train_idx = np.array([i for i, sid in enumerate(sequence_ids) if sid in train_seqs])
        test_idx = np.array([i for i, sid in enumerate(sequence_ids) if sid not in train_seqs])
        
    else:  # iid - random split is fine
        perm = rng.permutation(n_total)
        split_idx = int(n_total * ds_config.train_ratio)
        train_idx = perm[:split_idx]
        test_idx = perm[split_idx:]
    
    # Handle edge cases
    if len(train_idx) == 0:
        train_idx = np.arange(max(1, n_total // 2))
        test_idx = np.arange(len(train_idx), n_total)
    if len(test_idx) == 0:
        test_idx = train_idx[-1:]
        train_idx = train_idx[:-1]
    
    # Compute class balance (with remapped classes)
    unique, counts = np.unique(y, return_counts=True)
    class_balance = {
        'n_classes': len(unique),
        'class_counts': dict(zip(unique.tolist(), counts.tolist())),
        'imbalance_ratio': float(counts.max() / counts.min()) if counts.min() > 0 else float('inf'),
    }
    
    return GeneratedDataset(
        X_train=X[train_idx],
        y_train=y[train_idx],
        X_test=X[test_idx],
        y_test=y[test_idx],
        n_classes=len(unique),
        n_features=len(feature_node_ids),
        t_subseq=t_subseq,
        T=T,
        sample_mode=sample_mode,
        feature_node_ids=feature_node_ids,
        label_tree_idx=label_tree_idx,
        class_balance=class_balance,
    )


def analyze_class_balance(y: np.ndarray) -> Dict:
    """Analyze class balance of labels."""
    unique, counts = np.unique(y, return_counts=True)
    n_classes = len(unique)
    total = len(y)
    
    balance_info = {
        'n_classes': n_classes,
        'class_counts': dict(zip(unique.tolist(), counts.tolist())),
        'class_proportions': dict(zip(unique.tolist(), (counts / total).tolist())),
        'min_class_size': int(counts.min()),
        'max_class_size': int(counts.max()),
        'imbalance_ratio': float(counts.max() / counts.min()) if counts.min() > 0 else float('inf'),
    }
    return balance_info


def visualize_generated_dataset(dataset: GeneratedDataset, output_dir: str = ".", 
                                 dataset_id: int = 0):
    """
    Visualize a GeneratedDataset.
    Shows samples by class, class balance, and summary info.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    X = dataset.X_train
    y = dataset.y_train
    n_samples = len(X)
    n_features = dataset.n_features
    t_subseq = dataset.t_subseq
    n_classes = dataset.n_classes
    balance = dataset.class_balance
    
    print(f"\n{'='*60}")
    print(f"DATASET {dataset_id}")
    print(f"{'='*60}")
    print(f"Shape: {len(dataset.X_train)+len(dataset.X_test)} total ({len(dataset.X_train)} train, {len(dataset.X_test)} test)")
    print(f"Features: {n_features} nodes | Timesteps: {t_subseq} | T: {dataset.T}")
    print(f"Mode: {dataset.sample_mode} | Label tree: {dataset.label_tree_idx}")
    print(f"Classes: {n_classes} | Balance: {balance['class_counts']}")
    print(f"Imbalance ratio: {balance['imbalance_ratio']:.2f}x")
    
    colors = plt.cm.tab10(np.arange(10))
    
    # Figure 1: One sample per class (all features overlaid)
    n_rows = min(n_classes, 8)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.5 * n_rows))
    if n_rows == 1:
        axes = [axes]
    
    total = len(dataset.X_train) + len(dataset.X_test)
    fig.suptitle(
        f"Dataset {dataset_id} | {total} × {n_features} × {t_subseq} | {n_classes} classes | "
        f"{dataset.sample_mode} | imbalance={balance['imbalance_ratio']:.1f}x",
        fontsize=11, fontweight='bold'
    )
    
    for class_idx in range(n_rows):
        ax = axes[class_idx]
        class_mask = y == class_idx
        if not np.any(class_mask):
            continue
        
        # Get first sample of this class
        sample_idx = np.where(class_mask)[0][0]
        sample = X[sample_idx]  # (n_features, t_subseq)
        
        for f in range(n_features):
            ax.plot(sample[f], label=f"F{f}" if n_features <= 6 else None, linewidth=1)
        
        count = balance['class_counts'].get(class_idx, 0)
        ax.set_ylabel(f"Class {class_idx}\n(n={count})", fontsize=9)
        if n_features <= 6:
            ax.legend(loc='upper right', fontsize=7, ncol=min(3, n_features))
        if class_idx == n_rows - 1:
            ax.set_xlabel("Time Step")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dataset{dataset_id:02d}_samples.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Class balance
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    classes = list(balance['class_counts'].keys())
    counts = list(balance['class_counts'].values())
    
    ax = axes[0]
    ax.bar(classes, counts, color=[colors[c % 10] for c in classes])
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(f"Class Distribution (n={total})")
    ax.axhline(y=total/n_classes, color='red', linestyle='--', label='balanced')
    ax.legend()
    
    ax = axes[1]
    ax.pie(counts, labels=[f"C{c}" for c in classes], autopct='%1.1f%%',
           colors=[colors[c % 10] for c in classes])
    ax.set_title("Class Proportions")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dataset{dataset_id:02d}_balance.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Multiple samples per class (grid view)
    min_class_count = min(balance['class_counts'].values())
    n_samples_per_class = max(1, min(5, min_class_count))
    n_classes_to_show = min(n_classes, 6)
    n_features_to_show = min(n_features, 4)
    
    fig, axes = plt.subplots(n_classes_to_show, n_samples_per_class, 
                             figsize=(3 * n_samples_per_class, 2.5 * n_classes_to_show))
    if n_classes_to_show == 1:
        axes = axes.reshape(1, -1)
    if n_samples_per_class == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(f"Dataset {dataset_id} | Samples by Class", fontsize=11)
    
    for class_idx in range(n_classes_to_show):
        class_mask = y == class_idx
        class_samples = X[class_mask][:n_samples_per_class]
        
        for sample_offset in range(n_samples_per_class):
            ax = axes[class_idx, sample_offset]
            
            if sample_offset < len(class_samples):
                sample = class_samples[sample_offset]
                for f in range(n_features_to_show):
                    ax.plot(sample[f], color=colors[f % 10], alpha=0.8, linewidth=1)
            else:
                ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)
            
            if class_idx == 0:
                ax.set_title(f"Sample {sample_offset}", fontsize=9)
            if sample_offset == 0:
                ax.set_ylabel(f"Class {class_idx}", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dataset{dataset_id:02d}_grid.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: dataset{dataset_id:02d}_samples.png, dataset{dataset_id:02d}_balance.png, dataset{dataset_id:02d}_grid.png")
    
    return dataset


def main():
    """Generate and visualize synthetic datasets."""
    print("=" * 60)
    print("DAG Preferential Dataset Generator")
    print("=" * 60)
    
    output_dir = os.path.join(os.path.dirname(__file__), "dag_datasets_v6")
    os.makedirs(output_dir, exist_ok=True)
    
    # --- DAG Config ---
    dag_config = DAGPreferentialConfig(
        # Memory trees
        memory_input_dim_range=(1, 5),        # log-uniform int
        memory_output_dim_range=(1, 1),       # log-uniform int
        memory_init='normal',                   # 'normal' o 'uniform' (también para hojas/th)
        memory_std_range=(0.1, 1.0),            # log-uniform float
        tree_depth_range=(2, 5),                # log-uniform int
        tree_discrete_prob=0.6,                 # fixed (prob árbol discreto) - alto para tener labels
        tree_max_classes=10,                    # fixed (max hojas discretas = max clases)
        
        # Stochastic input
        stochastic_input_dim_range=(2, 2),      # log-uniform int
        stochastic_std_range=(0.01, 1),    # log-uniform float
        
        # Time
        n_time_transforms_range=(2, 2),         # log-uniform int
        
        # DAG structure (preferential attachment)
        n_nodes_range=(30, 150),                # log-uniform int
        redirection_alpha=2.0,                  # gamma shape (P ~ gamma(α,β))
        redirection_beta=5.0,                   # gamma rate  
        n_roots_range=(3, 15),                  # log-uniform int
        dag_density_range=(2.5, 150),           # uniform float (in-degree promedio: 1=cadena, 2+=denso)
        
        # Activations
        activation_choices=(
            'identity', 'log', 'sigmoid', 'abs', 'sin', 'tanh', 'rank',
            'square', 'power', 'softplus', 'step', 'modulo'
        ),
        
        # Weights
        weight_init_choices=('xavier_normal',),
        weight_scale_range=(0.9, 1.1),
        bias_std_range=(0.0, 0.1),
        
        # Noise
        node_noise_prob_range=(0.005, 0.5),
        node_noise_std_range=(0.00001, 0.1),
        noise_dist_choices=('normal',),
        
        # Quantization (desactivado)
        quantization_node_prob=0.0,
        
        # Sequence (será sobrescrito por dataset generator)
        seq_length=200,
    )
    
    # --- Dataset Config ---
    ds_config = DatasetConfig(
        # Modes (solo IID por ahora)
        prob_iid=1.0,
        prob_sliding=0.0,
        prob_mixed=0.0,
        
        # Constraints
        max_samples=500,
        max_features=12,
        max_timesteps=500,
        max_m_times_t=500,
        
        # Subsequence
        t_subseq_range=(30, 150),
        t_margin_range=(1, 1),
        
        # Split
        train_ratio=0.8,
    )
    
    n_datasets = 12
    # --- fin config ---
    
    print(f"\nGenerating {n_datasets} datasets -> {output_dir}\n")
    
    success_count = 0
    for i in range(n_datasets):
        for attempt in range(5):
            try:
                seed = 42 + i * 100 + attempt
                dataset = generate_dataset(dag_config, ds_config, seed=seed)
                visualize_generated_dataset(dataset, output_dir=output_dir, dataset_id=i)
                success_count += 1
                break
            except ValueError as e:
                if attempt == 4:
                    print(f"Dataset {i}: Failed after 5 attempts - {e}")
                continue
    
    print(f"\nDone! Generated {success_count}/{n_datasets} datasets -> {output_dir}")


if __name__ == "__main__":
    main()
