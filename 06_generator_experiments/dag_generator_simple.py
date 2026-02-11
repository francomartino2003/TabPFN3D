"""
Simplified DAG Generator for Time Series Classification.

Key features:
- Input vector -> numeric output (dense) + categorical output (discretization)
- Categorical output = dataset label (balanced via k-means-like prototypes)
- DAG with preferential attachment and density control
- Stochastic input clipped to [-1, 1]
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import os


@dataclass
class DAGConfig:
    """Configuration for DAG generator."""
    # Input (z) configuration
    input_dim_range: Tuple[int, int] = (5, 100)      # uniform int
    input_std_range: Tuple[float, float] = (0.1, 1.0)  # log-uniform: std for normal init
    # Note: input_init is sampled 50/50 between 'normal' and 'uniform' per dataset
    
    # Categorical output (for labels) - always 1
    n_classes_range: Tuple[int, int] = (2, 10)      # uniform int
    
    # Numeric outputs from z (multiple dense layers)
    n_numeric_outputs_range: Tuple[int, int] = (1, 4)  # uniform int
    
    # Stochastic input (noise per timestep, clipped to [-1, 1])
    stochastic_dim_range: Tuple[int, int] = (0, 4)  # uniform int
    stochastic_std_range: Tuple[float, float] = (0.01, 0.5)  # log-uniform
    
    # Time input
    n_time_copies_range: Tuple[int, int] = (1, 4)   # uniform int (copies of t)
    # First time copy is always t (identity), extras get random activations
    time_activation_choices: Tuple[str, ...] = (
        'tanh', 'sin', 'abs', 'square', 'sigmoid', 'step'
    )
    
    # DAG structure
    n_nodes_range: Tuple[int, int] = (20, 100)      # log-uniform int
    n_roots_range: Tuple[int, int] = (2, 10)        # log-uniform int
    
    # Preferential attachment via redirection
    # P = redirection probability ~ Gamma(alpha, beta)
    # When adding edge: with prob P, redirect to parent's parent (creates preferential attachment)
    redirection_alpha: float = 2.0   # gamma shape
    redirection_beta: float = 5.0    # gamma rate
    
    # Density: controls extra parents beyond the primary one
    # 0 = each node has 1 parent, 1 = many extra parents
    dag_density_range: Tuple[float, float] = (0.0, 0.5)
    
    # Activations per node
    activation_choices: Tuple[str, ...] = (
        'identity', 'tanh', 'sin', 'abs', 'sigmoid', 'relu', 
        'leaky_relu', 'softplus', 'step', 'square'
    )
    
    # Weights
    weight_scale_range: Tuple[float, float] = (0.8, 1.2)  # uniform
    bias_std_range: Tuple[float, float] = (0.0, 0.2)      # uniform
    
    # Sequence
    seq_length: int = 200


@dataclass
class NodeNoiseConfig:
    """Noise config for a node."""
    has_noise: bool
    scale: float = 0.0


@dataclass
class DAGNode:
    """A node in the DAG."""
    id: int
    parents: List[int]
    children: List[int]
    is_root: bool
    topological_order: int
    activation: str
    weight: np.ndarray
    bias: float
    noise_config: NodeNoiseConfig


@dataclass
class SampledConfig:
    """Sampled configuration for a specific network."""
    input_dim: int
    input_init: str                  # 'normal' or 'uniform' (sampled 50/50)
    input_std: float
    n_classes: int
    prototypes: np.ndarray           # (n_classes, input_dim) for discretization
    class_values: np.ndarray         # (n_classes,) fixed value per class for propagation
    n_numeric_outputs: int           # number of numeric outputs from z
    numeric_weights: np.ndarray      # (n_numeric_outputs, input_dim) dense layers
    numeric_biases: np.ndarray       # (n_numeric_outputs,)
    numeric_activations: List[str]   # one per numeric output
    stochastic_dim: int
    stochastic_std: float
    n_time_copies: int
    time_activations: List[str]      # first is 'identity', rest are random
    n_nodes: int
    n_roots: int
    redirection_prob: float          # P for preferential attachment
    dag_density: float
    node_activations: List[str]
    weight_scale: float
    bias_std: float
    seq_length: int


class DAGGenerator:
    """
    Simplified DAG generator for time series classification.
    
    Input flow:
    1. Sample z_input ~ normal/uniform (input_dim)
    2. Compute numeric_output = activation(z_input @ weights + bias)
    3. Compute categorical_output = argmin distance to prototypes -> normalized to [-1, 1]
    4. categorical_output is the LABEL
    5. Combine [numeric_output, categorical_output, stochastic, time] -> root nodes
    6. Propagate through DAG with preferential attachment
    """
    
    def __init__(self, config: DAGConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        
        # Sample configuration
        self.sampled_config = self._sample_config()
        
        # Build DAG
        self._build_dag()
    
    def _log_uniform_int(self, low: int, high: int) -> int:
        """Sample int from log-uniform distribution."""
        if low >= high:
            return low
        log_low = np.log(max(1, low))
        log_high = np.log(max(1, high))
        return int(np.exp(self.rng.uniform(log_low, log_high)))
    
    def _log_uniform_float(self, low: float, high: float) -> float:
        """Sample float from log-uniform distribution."""
        if low >= high:
            return low
        log_low = np.log(max(1e-10, low))
        log_high = np.log(max(1e-10, high))
        return np.exp(self.rng.uniform(log_low, log_high))
    
    def _generate_balanced_prototypes(self, n_classes: int, input_dim: int) -> np.ndarray:
        """
        Generate prototypes that encourage balanced classes.
        Uses k-means-like initialization: spread prototypes evenly in space.
        """
        # Strategy: place prototypes at vertices of a simplex-like structure
        # For better balance, spread them maximally in the input space
        
        if n_classes == 2:
            # Two classes: opposite corners
            prototypes = np.zeros((2, input_dim))
            prototypes[0, :] = -0.5
            prototypes[1, :] = 0.5
            # Add some randomness
            prototypes += self.rng.normal(0, 0.2, prototypes.shape)
        else:
            # Use k-means++ like initialization for better spread
            prototypes = np.zeros((n_classes, input_dim))
            
            # First prototype: random
            prototypes[0] = self.rng.uniform(-0.8, 0.8, input_dim)
            
            # Subsequent prototypes: maximize distance to existing
            for i in range(1, n_classes):
                # Generate candidates
                n_candidates = 50
                candidates = self.rng.uniform(-0.8, 0.8, (n_candidates, input_dim))
                
                # Compute min distance to existing prototypes
                min_dists = np.full(n_candidates, np.inf)
                for j in range(i):
                    dists = np.sum((candidates - prototypes[j])**2, axis=1)
                    min_dists = np.minimum(min_dists, dists)
                
                # Select candidate with maximum min distance
                best_idx = np.argmax(min_dists)
                prototypes[i] = candidates[best_idx]
        
        return np.clip(prototypes, -1, 1)
    
    def _sample_config(self) -> SampledConfig:
        """Sample all configuration parameters."""
        cfg = self.config
        
        # Input - 50/50 normal or uniform
        input_dim = self.rng.integers(cfg.input_dim_range[0], cfg.input_dim_range[1] + 1)
        input_init = self.rng.choice(['normal', 'uniform'])
        input_std = self._log_uniform_float(cfg.input_std_range[0], cfg.input_std_range[1])
        
        # Classes and balanced prototypes (log-uniform to favor fewer classes)
        n_classes = self._log_uniform_int(cfg.n_classes_range[0], cfg.n_classes_range[1])
        prototypes = self._generate_balanced_prototypes(n_classes, input_dim)
        
        # Class values for propagation (same distribution as z input)
        if input_init == 'normal':
            class_values = self.rng.normal(0, input_std, n_classes)
        else:  # uniform
            class_values = self.rng.uniform(-1, 1, n_classes)
        
        # Multiple numeric outputs (dense layers from z)
        n_numeric_outputs = self.rng.integers(
            cfg.n_numeric_outputs_range[0], cfg.n_numeric_outputs_range[1] + 1
        )
        weight_scale = self.rng.uniform(cfg.weight_scale_range[0], cfg.weight_scale_range[1])
        std = weight_scale * np.sqrt(2.0 / input_dim)  # He init
        numeric_weights = self.rng.normal(0, std, (n_numeric_outputs, input_dim))
        bias_std = self.rng.uniform(cfg.bias_std_range[0], cfg.bias_std_range[1])
        numeric_biases = self.rng.normal(0, bias_std, n_numeric_outputs) if bias_std > 0 else np.zeros(n_numeric_outputs)
        numeric_activations = [self.rng.choice(list(cfg.activation_choices)) for _ in range(n_numeric_outputs)]
        
        # Stochastic
        stochastic_dim = self.rng.integers(cfg.stochastic_dim_range[0], cfg.stochastic_dim_range[1] + 1)
        stochastic_std = self._log_uniform_float(cfg.stochastic_std_range[0], cfg.stochastic_std_range[1])
        
        # Time - first is identity, rest have random activations
        n_time_copies = self.rng.integers(cfg.n_time_copies_range[0], cfg.n_time_copies_range[1] + 1)
        time_activations = ['identity']  # First is always raw t
        for _ in range(n_time_copies - 1):
            time_activations.append(self.rng.choice(list(cfg.time_activation_choices)))
        
        # DAG
        n_nodes = self._log_uniform_int(cfg.n_nodes_range[0], cfg.n_nodes_range[1])
        n_roots = self._log_uniform_int(cfg.n_roots_range[0], min(n_nodes // 2, cfg.n_roots_range[1]))
        n_roots = max(1, n_roots)
        
        # Redirection probability P ~ Gamma(alpha, beta), clipped to [0, 1]
        redirection_prob = np.clip(self.rng.gamma(cfg.redirection_alpha, 1.0 / cfg.redirection_beta), 0.0, 1.0)
        
        dag_density = self._log_uniform_float(cfg.dag_density_range[0], cfg.dag_density_range[1])
        
        # Node activations
        node_activations = [self.rng.choice(list(cfg.activation_choices)) for _ in range(n_nodes)]
        
        return SampledConfig(
            input_dim=input_dim,
            input_init=input_init,
            input_std=input_std,
            n_classes=n_classes,
            prototypes=prototypes,
            class_values=class_values,
            n_numeric_outputs=n_numeric_outputs,
            numeric_weights=numeric_weights,
            numeric_biases=numeric_biases,
            numeric_activations=numeric_activations,
            stochastic_dim=stochastic_dim,
            stochastic_std=stochastic_std,
            n_time_copies=n_time_copies,
            time_activations=time_activations,
            n_nodes=n_nodes,
            n_roots=n_roots,
            redirection_prob=redirection_prob,
            dag_density=dag_density,
            node_activations=node_activations,
            weight_scale=weight_scale,
            bias_std=bias_std,
            seq_length=cfg.seq_length,
        )
    
    def _build_dag(self):
        """
        Build DAG structure with unified preferential attachment via redirection.
        
        For each non-root node:
        - n_parents = 1 + round(density * (n_anteriores - 1))
        - Each parent selected via: pick random -> with prob P redirect to grandparent
        - Skip if parent already selected
        """
        cfg = self.sampled_config
        P = cfg.redirection_prob
        
        self.nodes: Dict[int, DAGNode] = {}
        self.topological_order: List[int] = []
        
        # Input dim for root nodes: numeric(n) + categorical(1) + stochastic + time
        root_input_dim = cfg.n_numeric_outputs + 1 + cfg.stochastic_dim + cfg.n_time_copies
        
        for node_id in range(cfg.n_nodes):
            is_root = node_id < cfg.n_roots
            
            if is_root:
                parents = []
                input_dim = root_input_dim
            else:
                parents = set()
                n_anteriores = node_id  # number of previous nodes
                
                # Target number of parents based on density
                # density=0 -> 1 parent, density=1 -> all previous nodes
                n_parents_target = 1 + int(round(cfg.dag_density * (n_anteriores - 1)))
                n_parents_target = min(n_parents_target, n_anteriores)
                
                # Try to add parents using preferential attachment with redirection
                max_attempts = n_parents_target * 3  # allow some retries for duplicates
                attempts = 0
                
                while len(parents) < n_parents_target and attempts < max_attempts:
                    attempts += 1
                    
                    # Pick random node from previous nodes
                    initial_parent = self.rng.integers(0, node_id)
                    
                    # Apply redirection with probability P
                    final_parent = initial_parent
                    if self.rng.random() < P:
                        # Try to redirect to grandparent
                        if initial_parent in self.nodes:
                            grandparents = self.nodes[initial_parent].parents
                            if len(grandparents) > 0:
                                final_parent = self.rng.choice(grandparents)
                    
                    # Add if not already a parent
                    parents.add(final_parent)
                
                parents = list(parents)
                input_dim = len(parents)
            
            # Initialize weights (He init)
            std = cfg.weight_scale * np.sqrt(2.0 / max(1, input_dim))
            weight = self.rng.normal(0, std, input_dim)
            bias = self.rng.normal(0, cfg.bias_std) if cfg.bias_std > 0 else 0.0
            
            self.nodes[node_id] = DAGNode(
                id=node_id,
                parents=parents,
                children=[],
                is_root=is_root,
                topological_order=node_id,
                activation=cfg.node_activations[node_id],
                weight=weight,
                bias=bias,
                noise_config=NodeNoiseConfig(has_noise=False),
            )
            self.topological_order.append(node_id)
        
        # Update children lists
        for node_id, node in self.nodes.items():
            for parent_id in node.parents:
                self.nodes[parent_id].children.append(node_id)
    
    def _activation(self, x: np.ndarray, act: str) -> np.ndarray:
        """Apply activation function."""
        if act == 'identity':
            return x
        elif act == 'tanh':
            return np.tanh(x)
        elif act == 'sin':
            return np.sin(x)
        elif act == 'abs':
            return np.abs(x)
        elif act == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif act == 'relu':
            return np.maximum(0, x)
        elif act == 'leaky_relu':
            return np.where(x > 0, x, 0.01 * x)
        elif act == 'softplus':
            return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -500, 20))))
        elif act == 'step':
            return np.where(x >= 0, 1.0, -1.0)
        elif act == 'square':
            return np.sign(x) * np.minimum(x**2, 10)
        else:
            return x
    
    def _generate_input(self, n_samples: int) -> np.ndarray:
        """Generate input vectors z."""
        cfg = self.sampled_config
        if cfg.input_init == 'uniform':
            return self.rng.uniform(-1, 1, (n_samples, cfg.input_dim))
        else:  # normal
            return np.clip(self.rng.normal(0, cfg.input_std, (n_samples, cfg.input_dim)), -1, 1)
    
    def _compute_numeric_output(self, z_input: np.ndarray) -> np.ndarray:
        """
        Compute multiple numeric outputs from input via dense layers.
        Returns: (n_samples, n_numeric_outputs)
        """
        cfg = self.sampled_config
        n_samples = z_input.shape[0]
        outputs = np.zeros((n_samples, cfg.n_numeric_outputs))
        
        for i in range(cfg.n_numeric_outputs):
            pre_act = z_input @ cfg.numeric_weights[i] + cfg.numeric_biases[i]
            outputs[:, i] = self._activation(pre_act, cfg.numeric_activations[i])
        
        return outputs
    
    def _compute_categorical_output(self, z_input: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute categorical output via discretization.
        Returns:
            labels: (n_samples,) int class indices
            class_out: (n_samples,) float class values for propagation
        """
        cfg = self.sampled_config
        
        # Compute distances to each prototype
        diff = z_input[:, np.newaxis, :] - cfg.prototypes[np.newaxis, :, :]
        distances = np.sum(diff**2, axis=2)
        
        # Assign to closest prototype
        labels = np.argmin(distances, axis=1)
        
        # Use fixed class values for propagation (sampled once per dataset)
        class_out = cfg.class_values[labels]
        
        return labels.astype(np.int32), class_out.astype(np.float32)
    
    def _generate_time_inputs(self, T: int) -> np.ndarray:
        """
        Generate time inputs with activations.
        First copy is always raw t in [-1, 1], rest have random activations.
        Returns: (n_time_copies, T)
        """
        cfg = self.sampled_config
        t = np.linspace(-1, 1, T)
        
        time_inputs = []
        for i, act in enumerate(cfg.time_activations):
            time_inputs.append(self._activation(t, act))
        
        return np.array(time_inputs)
    
    def _generate_stochastic_inputs(self, n_samples: int, T: int) -> Optional[np.ndarray]:
        """Generate stochastic inputs, clipped to [-1, 1]."""
        cfg = self.sampled_config
        if cfg.stochastic_dim == 0:
            return None
        noise = self.rng.normal(0, cfg.stochastic_std, (n_samples, cfg.stochastic_dim, T))
        return np.clip(noise, -1, 1)
    
    def propagate(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Propagate through the DAG.
        
        Returns:
            output: (n_samples, n_nodes, T) - all node outputs
            labels: (n_samples,) - class labels from categorical output
            info: dict with intermediate values
        """
        cfg = self.sampled_config
        T = cfg.seq_length
        
        # 1. Generate input
        z_input = self._generate_input(n_samples)
        
        # 2. Compute numeric and categorical outputs
        numeric_out = self._compute_numeric_output(z_input)
        labels, categorical_out = self._compute_categorical_output(z_input)
        
        # 3. Generate time and stochastic inputs
        time_inputs = self._generate_time_inputs(T)
        stochastic = self._generate_stochastic_inputs(n_samples, T)
        
        # 4. Build root input: [numeric(n), categorical(1), stochastic, time]
        # numeric_out: (n_samples, n_numeric_outputs) -> broadcast to (n_samples, n_numeric, T)
        numeric_broadcast = np.tile(numeric_out[:, :, np.newaxis], (1, 1, T))
        
        # categorical_out: (n_samples,) -> (n_samples, 1, T)
        categorical_broadcast = np.tile(categorical_out[:, np.newaxis, np.newaxis], (1, 1, T))
        
        root_input_parts = [numeric_broadcast, categorical_broadcast]
        if stochastic is not None:
            root_input_parts.append(stochastic)
        time_broadcast = np.tile(time_inputs[np.newaxis, :, :], (n_samples, 1, 1))
        root_input_parts.append(time_broadcast)
        
        root_input = np.concatenate(root_input_parts, axis=1)
        
        # 5. Propagate through DAG
        node_outputs = {}
        output = np.zeros((n_samples, cfg.n_nodes, T), dtype=np.float32)
        
        for node_id in self.topological_order:
            node = self.nodes[node_id]
            
            if node.is_root:
                pre_act = np.einsum('ndt,d->nt', root_input, node.weight) + node.bias
            else:
                parent_outputs = np.stack([node_outputs[p] for p in node.parents], axis=1)
                pre_act = np.einsum('npt,p->nt', parent_outputs, node.weight) + node.bias
            
            node_out = self._activation(pre_act, node.activation)
            node_outputs[node_id] = node_out
            output[:, node_id, :] = node_out
        
        info = {
            'z_input': z_input,
            'numeric_out': numeric_out,
            'categorical_out': categorical_out,
        }
        
        return output, labels, info
    
    def get_config_summary(self) -> str:
        """Get summary of sampled configuration."""
        cfg = self.sampled_config
        total_edges = sum(len(node.parents) for node in self.nodes.values())
        n_non_roots = cfg.n_nodes - cfg.n_roots
        actual_density = total_edges / n_non_roots if n_non_roots > 0 else 0
        n_multi_parent = sum(1 for n in self.nodes.values() if len(n.parents) > 1)
        
        time_acts = ', '.join(cfg.time_activations)
        lines = [
            f"Input: {cfg.input_dim}d ({cfg.input_init})",
            f"Classes: {cfg.n_classes}, Numeric outputs: {cfg.n_numeric_outputs}",
            f"Nodes: {cfg.n_nodes} (roots: {cfg.n_roots})",
            f"Edges: {total_edges} (avg in-degree: {actual_density:.2f})",
            f"Redirection P: {cfg.redirection_prob:.3f}",
            f"Nodes with >1 parent: {n_multi_parent}/{cfg.n_nodes}",
            f"Stochastic: {cfg.stochastic_dim}d, Time: [{time_acts}]",
        ]
        return '\n'.join(lines)


# =============================================================================
# DATASET GENERATION
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    max_samples: int = 500
    max_features: int = 12
    max_timesteps: int = 300
    max_m_times_t: int = 500
    t_subseq_range: Tuple[int, int] = (20, 150)
    train_ratio: float = 0.8


@dataclass
class GeneratedDataset:
    """Generated dataset."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    n_classes: int
    n_features: int
    t_subseq: int
    T: int
    feature_node_ids: List[int]
    class_balance: Dict
    generator: DAGGenerator


def _sample_n_features(rng: np.random.Generator, max_features: int) -> int:
    """Sample n_features with preference for few."""
    r = rng.random()
    if r < 0.50:
        return 1
    elif r < 0.75:
        return 2
    elif r < 0.90:
        return 3
    else:
        return min(rng.integers(4, max(5, max_features + 1)), max_features)


def generate_dataset(dag_config: DAGConfig, ds_config: DatasetConfig, seed: int = 42) -> GeneratedDataset:
    """Generate a classification dataset."""
    rng = np.random.default_rng(seed)
    
    # Sample parameters
    n_features = _sample_n_features(rng, ds_config.max_features)
    
    # Sample t_subseq
    max_t = min(ds_config.t_subseq_range[1], ds_config.max_m_times_t // n_features)
    t_subseq = rng.integers(ds_config.t_subseq_range[0], max(ds_config.t_subseq_range[0] + 1, max_t + 1))
    
    # Sample n_samples
    n_samples = rng.integers(100, ds_config.max_samples + 1)
    
    # T = t_subseq + small margin
    margin = rng.integers(1, 20)
    T = min(t_subseq + margin, ds_config.max_timesteps)
    
    # Create generator
    dag_config_with_T = DAGConfig(
        input_dim_range=dag_config.input_dim_range,
        input_std_range=dag_config.input_std_range,
        n_classes_range=dag_config.n_classes_range,
        n_numeric_outputs_range=dag_config.n_numeric_outputs_range,
        stochastic_dim_range=dag_config.stochastic_dim_range,
        stochastic_std_range=dag_config.stochastic_std_range,
        n_time_copies_range=dag_config.n_time_copies_range,
        time_activation_choices=dag_config.time_activation_choices,
        n_nodes_range=dag_config.n_nodes_range,
        n_roots_range=dag_config.n_roots_range,
        redirection_alpha=dag_config.redirection_alpha,
        redirection_beta=dag_config.redirection_beta,
        dag_density_range=dag_config.dag_density_range,
        activation_choices=dag_config.activation_choices,
        weight_scale_range=dag_config.weight_scale_range,
        bias_std_range=dag_config.bias_std_range,
        seq_length=T,
    )
    
    generator = DAGGenerator(dag_config_with_T, seed=seed)
    cfg = generator.sampled_config
    
    # Find good feature nodes (>1 parent, good temporal variance)
    output_check, _, _ = generator.propagate(20)
    
    feature_candidates = []
    for node_id, node in generator.nodes.items():
        if len(node.parents) > 1:
            node_out = output_check[:, node_id, :]
            var_time = np.var(node_out, axis=1).mean()
            if var_time > 0.01:
                feature_candidates.append((node_id, var_time))
    
    if len(feature_candidates) < n_features:
        raise ValueError(f"Only {len(feature_candidates)} good feature nodes, need {n_features}")
    
    # Sort by variance, take top n_features
    feature_candidates.sort(key=lambda x: -x[1])
    feature_node_ids = [nid for nid, _ in feature_candidates[:n_features]]
    
    # Generate data
    output, labels, info = generator.propagate(n_samples)
    
    # Extract features
    X = output[:, feature_node_ids, :t_subseq]
    y = labels
    
    # Remap classes to consecutive 0,1,2,...
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        raise ValueError("Only 1 class found")
    class_map = {old: new for new, old in enumerate(unique_classes)}
    y = np.array([class_map[c] for c in y], dtype=np.int32)
    
    # Split
    n_total = len(X)
    perm = rng.permutation(n_total)
    split_idx = int(n_total * ds_config.train_ratio)
    train_idx = perm[:split_idx]
    test_idx = perm[split_idx:]
    
    # Class balance
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
        feature_node_ids=feature_node_ids,
        class_balance=class_balance,
        generator=generator,
    )


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_dataset(dataset: GeneratedDataset, output_dir: str, dataset_id: int):
    """Visualize a generated dataset with class distribution and samples per class."""
    os.makedirs(output_dir, exist_ok=True)
    
    X_all = np.concatenate([dataset.X_train, dataset.X_test], axis=0)
    y_all = np.concatenate([dataset.y_train, dataset.y_test], axis=0)
    n_classes = dataset.n_classes
    balance = dataset.class_balance
    
    total = len(X_all)
    print(f"\nDataset {dataset_id}: {total} samples, {dataset.n_features} features, "
          f"{dataset.t_subseq} timesteps, {n_classes} classes, imbalance={balance['imbalance_ratio']:.2f}x")
    
    colors = plt.cm.tab10(np.arange(10))
    
    # Figure 1: Class distribution
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    classes = list(balance['class_counts'].keys())
    counts = list(balance['class_counts'].values())
    
    ax = axes[0]
    bars = ax.bar(classes, counts, color=[colors[c % 10] for c in classes])
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(f"Dataset {dataset_id}: Class Distribution (n={total})")
    ax.axhline(y=total/n_classes, color='red', linestyle='--', label='balanced', alpha=0.7)
    ax.legend()
    
    ax = axes[1]
    ax.pie(counts, labels=[f"C{c}" for c in classes], autopct='%1.1f%%',
           colors=[colors[c % 10] for c in classes])
    ax.set_title(f"Imbalance: {balance['imbalance_ratio']:.2f}x")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dataset{dataset_id:02d}_balance.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 2: 5 samples per class (grid)
    n_samples_per_class = 5
    n_rows = n_classes
    n_cols = n_samples_per_class
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(f"Dataset {dataset_id} | {total}×{dataset.n_features}×{dataset.t_subseq} | "
                 f"{n_classes} classes", fontsize=12)
    
    for class_idx in range(n_rows):
        mask = y_all == class_idx
        class_samples = X_all[mask]
        count = len(class_samples)
        
        for sample_idx in range(n_cols):
            ax = axes[class_idx, sample_idx]
            
            if sample_idx < len(class_samples):
                sample = class_samples[sample_idx]
                for f in range(dataset.n_features):
                    ax.plot(sample[f], color=colors[f % 10], linewidth=1, alpha=0.8)
            else:
                ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)
            
            if class_idx == 0:
                ax.set_title(f"Sample {sample_idx}", fontsize=9)
            if sample_idx == 0:
                ax.set_ylabel(f"Class {class_idx}\n(n={count})", fontsize=9)
            
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dataset{dataset_id:02d}_samples.png", dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Generate and visualize datasets."""
    print("=" * 60)
    print("Simplified DAG Generator with Preferential Attachment")
    print("=" * 60)
    
    # =========================================================================
    # CONFIGURATION - ALL PARAMETERS IN ONE PLACE
    # =========================================================================
    
    output_dir = "dag_simple_v6"                    # output folder name
    n_datasets = 24                                 # number of datasets to generate
    
    # --- INPUT (z) ---
    input_dim_range = (4, 100)                       # uniform int: dim of z vector
    input_std_range = (0.1, 1.0)                    # log-uniform: std for normal init
    # Note: input_init sampled 50/50 between 'normal' and 'uniform' per dataset
    
    # --- CLASSES (categorical output - always 1) ---
    n_classes_range = (2, 10)                        # log-uniform int: favors fewer classes
    
    # --- NUMERIC OUTPUTS (multiple dense layers from z) ---
    n_numeric_outputs_range = (1, 5)                # uniform int: how many numeric features from z
    
    # --- STOCHASTIC INPUT (noise per timestep) ---
    stochastic_dim_range = (1, 4)                   # uniform int: noise dimensions
    stochastic_std_range = (0.001, 1)              # log-uniform: noise std (clipped [-1,1])
    
    # --- TIME INPUT ---
    n_time_copies_range = (1, 4)                    # uniform int: copies of t
    # First is always raw t, rest have random activations from time_activation_choices
    time_activation_choices = ('identity', 'tanh', 'sin', 'sigmoid')
    
    # --- DAG STRUCTURE ---
    n_roots_range = (3, 50)                         # log-uniform int: root nodes (no parents)
    n_nodes_range = (10, 150)                       # log-uniform int: total DAG nodes
    
    
    # --- PREFERENTIAL ATTACHMENT (redirection) ---
    # P ~ Gamma(alpha, beta), with prob P redirect edge to parent's parent
    redirection_alpha = 2.0                         # gamma shape (higher = P closer to alpha/beta)
    redirection_beta = 5.0                          # gamma rate (higher = smaller P values)
    
    # --- DAG DENSITY ---
    # n_parents = 1 + round(density * (n_anteriores - 1))
    # 0 = chain (1 parent), 1 = fully connected to all previous
    dag_density_range = (0.01, 1.0)                 # log-uniform: favors sparse DAGs
    
    # --- NODE ACTIVATIONS ---
    activation_choices = (
        'identity', 'tanh', 'sin', 'abs', 'sigmoid', 
        'relu', 'leaky_relu', 'softplus', 'step', 'square'
    )
    
    # --- WEIGHTS ---
    weight_scale_range = (0.8, 1.2)                 # uniform: He init scale
    bias_std_range = (0.0, 0.2)                     # uniform: bias std
    
    # --- SEQUENCE ---
    seq_length = 200                                # max sequence length
    
    # --- DATASET ---
    max_samples = 400                               # max samples per dataset
    max_features = 8                                # max feature nodes (in_degree > 1)
    max_timesteps = 300                             # max T
    max_m_times_t = 500                             # constraint: features * timesteps
    t_subseq_range = (20, 150)                      # uniform int: subsequence length
    train_ratio = 0.8                               # train/test split
    
    # =========================================================================
    # BUILD CONFIGS FROM PARAMETERS
    # =========================================================================
    
    output_dir = os.path.join(os.path.dirname(__file__), output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    dag_config = DAGConfig(
        input_dim_range=input_dim_range,
        input_std_range=input_std_range,
        n_classes_range=n_classes_range,
        n_numeric_outputs_range=n_numeric_outputs_range,
        stochastic_dim_range=stochastic_dim_range,
        stochastic_std_range=stochastic_std_range,
        n_time_copies_range=n_time_copies_range,
        time_activation_choices=time_activation_choices,
        n_nodes_range=n_nodes_range,
        n_roots_range=n_roots_range,
        redirection_alpha=redirection_alpha,
        redirection_beta=redirection_beta,
        dag_density_range=dag_density_range,
        activation_choices=activation_choices,
        weight_scale_range=weight_scale_range,
        bias_std_range=bias_std_range,
        seq_length=seq_length,
    )
    
    ds_config = DatasetConfig(
        max_samples=max_samples,
        max_features=max_features,
        max_timesteps=max_timesteps,
        max_m_times_t=max_m_times_t,
        t_subseq_range=t_subseq_range,
        train_ratio=train_ratio,
    )
    
    print(f"\nGenerating {n_datasets} datasets -> {output_dir}\n")
    
    success = 0
    for i in range(n_datasets):
        for attempt in range(5):
            try:
                seed = 42 + i * 100 + attempt
                dataset = generate_dataset(dag_config, ds_config, seed=seed)
                visualize_dataset(dataset, output_dir, i)
                success += 1
                break
            except ValueError as e:
                if attempt == 4:
                    print(f"Dataset {i}: Failed - {e}")
    
    print(f"\nDone! Generated {success}/{n_datasets} -> {output_dir}")


if __name__ == "__main__":
    main()
