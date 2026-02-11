"""
Hybrid DAG Generator for Time Series (Simplified).

Architecture:
1. Sample z (memory vector, fixed per sample)
2. z → z_nn → z_features (with optional discretization)
3. Combination NN: [t, z_features, noise] → hidden layers → roots
4. roots → preferential attachment DAG

All activations are tanh. No separate t_nn - t goes directly to combination NN.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any


@dataclass
class NodeDiscretizationConfig:
    """Configuration for per-node discretization."""
    is_discretized: bool = False
    n_classes: int = 2
    prototypes: Optional[np.ndarray] = None  # (n_classes, input_dim) - reference vectors
    class_values: Optional[np.ndarray] = None  # (n_classes,) - output value per class


@dataclass
class HybridDAGConfig:
    """Configuration for the hybrid DAG generator."""
    
    # z (memory) parameters
    z_dim_range: Tuple[int, int] = (4, 12)
    z_distribution: str = 'normal'  # 'normal' or 'uniform'
    
    # NN over z: layers and nodes
    z_nn_layers_range: Tuple[int, int] = (2, 4)
    z_nn_nodes_range: Tuple[int, int] = (8, 24)  # Wider
    
    # Discretization in z_nn
    z_discretization_prob: float = 0.3
    z_discretization_n_classes_range: Tuple[int, int] = (2, 5)
    
    # Time input copies (like random_nn_generator)
    n_time_copies_range: Tuple[int, int] = (3, 8)  # Multiple copies of t
    
    # Activations for combo_nn (per-node random selection)
    combo_activations: Tuple[str, ...] = ('tanh', 'sin', 'identity', 'abs', 'step', 'sigmoid')
    
    # Combination NN: [t_copies, z_features, noise] -> hidden -> roots
    combo_nn_layers_range: Tuple[int, int] = (2, 4)
    combo_nn_nodes_range: Tuple[int, int] = (16, 32)  # Wide hidden layers
    
    # Noise dimension
    noise_std_range: Tuple[float, float] = (0.001, 0.1)
    
    # Root nodes (output of combo NN)
    n_roots_range: Tuple[int, int] = (8, 20)  # More roots
    
    # Preferential attachment DAG
    n_dag_nodes_range: Tuple[int, int] = (30, 100)
    redirection_alpha: float = 2.0
    redirection_beta: float = 5.0
    
    # Per-node noise in DAG
    dag_noise_prob_range: Tuple[float, float] = (0.01, 0.3)
    dag_noise_std_range: Tuple[float, float] = (0.001, 0.05)
    
    # Sequence length
    seq_length: int = 200


@dataclass
class SampledHybridConfig:
    """Sampled configuration for a specific network."""
    z_dim: int
    z_distribution: str
    z_nn_layers: int
    z_nn_nodes_per_layer: List[int]
    z_nn_discretization: List[NodeDiscretizationConfig]
    
    n_time_copies: int  # Number of copies of t in input
    
    combo_nn_layers: int
    combo_nn_nodes_per_layer: List[int]
    combo_node_activations: List[List[str]]  # Per-node activations for each layer
    
    noise_std: float
    
    n_roots: int
    
    n_dag_nodes: int
    redirection_prob: float
    dag_noise_prob: float
    dag_noise_std: float
    
    seq_length: int


class HybridDAGGenerator:
    """
    Generate time series using hybrid architecture:
    z_nn(z) + t + noise → combo_nn → roots → DAG
    """
    
    def __init__(self, config: HybridDAGConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        
        # Sample configuration
        self.sampled_config = self._sample_config()
        
        # Build networks
        self._build_z_nn()
        self._build_combo_nn()
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
    
    def _sample_config(self) -> SampledHybridConfig:
        """Sample all configuration parameters."""
        cfg = self.config
        
        # z (memory) parameters
        z_dim = self._log_uniform_int(cfg.z_dim_range[0], cfg.z_dim_range[1])
        
        # NN over z
        z_nn_layers = self.rng.integers(cfg.z_nn_layers_range[0], cfg.z_nn_layers_range[1] + 1)
        z_nn_nodes_per_layer = [
            self._log_uniform_int(cfg.z_nn_nodes_range[0], cfg.z_nn_nodes_range[1])
            for _ in range(z_nn_layers)
        ]
        
        # Sample discretization config for each node in z_nn
        z_nn_discretization = []
        for layer_idx in range(z_nn_layers):
            n_nodes = z_nn_nodes_per_layer[layer_idx]
            for node_idx in range(n_nodes):
                if self.rng.random() < cfg.z_discretization_prob:
                    n_classes = self.rng.integers(
                        cfg.z_discretization_n_classes_range[0],
                        cfg.z_discretization_n_classes_range[1] + 1
                    )
                    z_nn_discretization.append(NodeDiscretizationConfig(
                        is_discretized=True,
                        n_classes=n_classes,
                        prototypes=None,
                        class_values=None
                    ))
                else:
                    z_nn_discretization.append(NodeDiscretizationConfig(is_discretized=False))
        
        # Time copies
        n_time_copies = self.rng.integers(cfg.n_time_copies_range[0], cfg.n_time_copies_range[1] + 1)
        
        # Combination NN
        combo_nn_layers = self.rng.integers(cfg.combo_nn_layers_range[0], cfg.combo_nn_layers_range[1] + 1)
        combo_nn_nodes_per_layer = [
            self._log_uniform_int(cfg.combo_nn_nodes_range[0], cfg.combo_nn_nodes_range[1])
            for _ in range(combo_nn_layers)
        ]
        
        # Per-node activations for combo_nn (like random_nn_generator)
        combo_node_activations = []
        for layer_idx in range(combo_nn_layers):
            n_nodes = combo_nn_nodes_per_layer[layer_idx]
            layer_acts = [self.rng.choice(list(cfg.combo_activations)) for _ in range(n_nodes)]
            combo_node_activations.append(layer_acts)
        
        # Noise
        noise_std = self._log_uniform_float(cfg.noise_std_range[0], cfg.noise_std_range[1])
        
        # Roots
        n_roots = self.rng.integers(cfg.n_roots_range[0], cfg.n_roots_range[1] + 1)
        
        # DAG
        dag_low = max(n_roots + 5, cfg.n_dag_nodes_range[0])
        dag_high = max(dag_low, cfg.n_dag_nodes_range[1])
        n_dag_nodes = self._log_uniform_int(dag_low, dag_high)
        redirection_prob = np.clip(
            self.rng.gamma(cfg.redirection_alpha, 1.0 / cfg.redirection_beta), 
            0.0, 1.0
        )
        dag_noise_prob = self._log_uniform_float(cfg.dag_noise_prob_range[0], cfg.dag_noise_prob_range[1])
        dag_noise_std = self._log_uniform_float(cfg.dag_noise_std_range[0], cfg.dag_noise_std_range[1])
        
        return SampledHybridConfig(
            z_dim=z_dim,
            z_distribution=cfg.z_distribution,
            z_nn_layers=z_nn_layers,
            z_nn_nodes_per_layer=z_nn_nodes_per_layer,
            z_nn_discretization=z_nn_discretization,
            n_time_copies=n_time_copies,
            combo_nn_layers=combo_nn_layers,
            combo_nn_nodes_per_layer=combo_nn_nodes_per_layer,
            combo_node_activations=combo_node_activations,
            noise_std=noise_std,
            n_roots=n_roots,
            n_dag_nodes=n_dag_nodes,
            redirection_prob=redirection_prob,
            dag_noise_prob=dag_noise_prob,
            dag_noise_std=dag_noise_std,
            seq_length=cfg.seq_length
        )
    
    def _init_weights(self, in_dim: int, out_dim: int) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize weights with He normal (like random_nn_generator)."""
        std = np.sqrt(2.0 / in_dim)
        W = self.rng.normal(0, std, (out_dim, in_dim))
        b = self.rng.normal(0, 0.1, out_dim)
        return W, b
    
    def _build_z_nn(self):
        """Build NN over z (memory) with optional discretization nodes."""
        cfg = self.sampled_config
        self.z_nn_weights = []
        self.z_nn_biases = []
        
        in_dim = cfg.z_dim
        node_idx = 0
        
        for layer_idx in range(cfg.z_nn_layers):
            out_dim = cfg.z_nn_nodes_per_layer[layer_idx]
            W, b = self._init_weights(in_dim, out_dim)
            self.z_nn_weights.append(W)
            self.z_nn_biases.append(b)
            
            # Initialize discretization prototypes
            for local_node_idx in range(out_dim):
                disc_cfg = cfg.z_nn_discretization[node_idx]
                if disc_cfg.is_discretized:
                    # Prototypes in [-1, 1] (tanh output range)
                    disc_cfg.prototypes = self.rng.uniform(-1, 1, (disc_cfg.n_classes, in_dim))
                    disc_cfg.class_values = self.rng.uniform(-1, 1, disc_cfg.n_classes)
                node_idx += 1
            
            in_dim = out_dim
        
        self.z_features_dim = cfg.z_nn_nodes_per_layer[-1] if cfg.z_nn_layers > 0 else cfg.z_dim
    
    def _build_combo_nn(self):
        """Build combination NN: [t_copies, z_features, noise] -> hidden -> roots."""
        cfg = self.sampled_config
        self.combo_nn_weights = []
        self.combo_nn_biases = []
        
        # Input: t_copies + z_features + noise (1)
        self.time_dim = cfg.n_time_copies
        self.combo_input_dim = self.time_dim + self.z_features_dim + 1
        
        in_dim = self.combo_input_dim
        for layer_idx in range(cfg.combo_nn_layers):
            out_dim = cfg.combo_nn_nodes_per_layer[layer_idx]
            W, b = self._init_weights(in_dim, out_dim)
            self.combo_nn_weights.append(W)
            self.combo_nn_biases.append(b)
            in_dim = out_dim
        
        # Final layer: hidden -> roots
        W, b = self._init_weights(in_dim, cfg.n_roots)
        self.combo_nn_weights.append(W)
        self.combo_nn_biases.append(b)
    
    def _build_dag(self):
        """Build preferential attachment DAG from root nodes."""
        cfg = self.sampled_config
        n_roots = cfg.n_roots
        n_nodes = cfg.n_dag_nodes
        
        self.dag_parents = {i: [] for i in range(n_nodes)}
        self.dag_is_root = [i < n_roots for i in range(n_nodes)]
        
        child_counts = np.ones(n_nodes)
        
        for node_id in range(n_roots, n_nodes):
            probs = child_counts[:node_id].copy()
            probs = probs / probs.sum()
            
            primary_parent = self.rng.choice(node_id, p=probs)
            self.dag_parents[node_id].append(primary_parent)
            child_counts[primary_parent] += 1
            
            if self.rng.random() < cfg.redirection_prob and node_id > n_roots:
                n_additional = self.rng.integers(1, min(3, node_id))
                available = [i for i in range(node_id) if i != primary_parent]
                if available:
                    probs_add = child_counts[:node_id].copy()
                    probs_add[primary_parent] = 0
                    if probs_add.sum() > 0:
                        probs_add = probs_add / probs_add.sum()
                        additional_parents = self.rng.choice(
                            node_id, size=min(n_additional, len(available)), 
                            replace=False, p=probs_add
                        )
                        for additional_parent in additional_parents:
                            if additional_parent not in self.dag_parents[node_id]:
                                self.dag_parents[node_id].append(additional_parent)
                                child_counts[additional_parent] += 1
        
        self.dag_weights = []
        self.dag_biases = []
        self.dag_has_noise = []
        
        for node_id in range(n_nodes):
            if self.dag_is_root[node_id]:
                self.dag_weights.append(None)
                self.dag_biases.append(None)
            else:
                in_dim = len(self.dag_parents[node_id])
                W, b = self._init_weights(in_dim, 1)
                self.dag_weights.append(W.flatten())
                self.dag_biases.append(b[0])
            
            has_noise = self.rng.random() < cfg.dag_noise_prob
            self.dag_has_noise.append(has_noise)
    
    def _apply_discretization(self, node_value: np.ndarray, parent_values: np.ndarray,
                              disc_cfg: NodeDiscretizationConfig) -> np.ndarray:
        """Apply discretization to a node."""
        if not disc_cfg.is_discretized:
            return node_value
        
        n_samples = node_value.shape[0]
        parent_expanded = parent_values[:, np.newaxis, :]
        prototypes_expanded = disc_cfg.prototypes[np.newaxis, :, :]
        distances = np.sum((parent_expanded - prototypes_expanded) ** 2, axis=-1)
        closest_class = np.argmin(distances, axis=-1)
        return disc_cfg.class_values[closest_class]
    
    def _forward_z_nn(self, z: np.ndarray) -> np.ndarray:
        """Forward pass through z NN with discretization."""
        cfg = self.sampled_config
        x = z
        node_idx = 0
        
        for layer_idx in range(cfg.z_nn_layers):
            W = self.z_nn_weights[layer_idx]
            b = self.z_nn_biases[layer_idx]
            n_nodes = cfg.z_nn_nodes_per_layer[layer_idx]
            
            linear = x @ W.T + b
            out = np.tanh(linear)
            
            for local_node_idx in range(n_nodes):
                disc_cfg = cfg.z_nn_discretization[node_idx]
                if disc_cfg.is_discretized:
                    out[:, local_node_idx] = self._apply_discretization(
                        out[:, local_node_idx], x, disc_cfg
                    )
                node_idx += 1
            
            x = out
        
        return x
    
    def _forward_combo_nn(self, combo_input: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Forward pass through combo NN.
        
        combo_input: (n_samples, combo_input_dim, T)
        
        Returns:
            roots: (n_samples, n_roots, T)
            layer_outputs: list of (n_samples, layer_dim, T) for each layer
        """
        layer_outputs = []
        cfg = self.sampled_config
        x = combo_input  # (n_samples, in_dim, T)
        
        for layer_idx, (W, b) in enumerate(zip(self.combo_nn_weights, self.combo_nn_biases)):
            # W: (out_dim, in_dim), x: (n_samples, in_dim, T)
            pre_act = np.einsum('oi,nit->not', W, x) + b[:, np.newaxis]
            
            # Apply per-node activations (except last layer which goes to roots)
            if layer_idx < len(cfg.combo_node_activations):
                out = np.zeros_like(pre_act)
                layer_acts = cfg.combo_node_activations[layer_idx]
                for node_idx, act in enumerate(layer_acts):
                    out[:, node_idx, :] = self._activation(pre_act[:, node_idx, :], act)
            else:
                # Last layer (roots) - use tanh
                out = np.tanh(pre_act)
            
            layer_outputs.append(out)
            x = out
        
        return x, layer_outputs  # Last output is roots
    
    def _generate_time_inputs(self, T: int) -> np.ndarray:
        """
        Generate time input: multiple copies of t (like random_nn_generator).
        
        Returns: (n_time_copies, T)
        """
        cfg = self.sampled_config
        t = np.linspace(-1, 1, T)  # Normalized to [-1, 1]
        
        # Just repeat t multiple times (like random_nn_generator with only 'linear')
        return np.tile(t[np.newaxis, :], (cfg.n_time_copies, 1))  # (n_time_copies, T)
    
    def _activation(self, x: np.ndarray, act: str) -> np.ndarray:
        """Apply activation function (like random_nn_generator)."""
        if act == 'identity':
            return x
        elif act == 'tanh':
            return np.tanh(x)
        elif act == 'sin':
            return np.sin(x)
        elif act == 'abs':
            return np.abs(x)
        elif act == 'step':
            return np.where(x > 0, 1.0, 0.0)
        elif act == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        elif act == 'log':
            return np.sign(x) * np.log1p(np.abs(x))
        elif act == 'square':
            return np.clip(x ** 2, -10, 10)
        elif act == 'relu':
            return np.maximum(0, x)
        else:
            return np.tanh(x)  # default
    
    def propagate(self, n_samples: int = 5) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Propagate through the entire network.
        
        Returns:
            dag_outputs: (n_samples, n_dag_nodes, T)
            intermediates: dict with all intermediate values
        """
        cfg = self.sampled_config
        T = cfg.seq_length
        
        # 1. Sample z
        if cfg.z_distribution == 'normal':
            z = self.rng.normal(0, 1, (n_samples, cfg.z_dim))
        else:
            z = self.rng.uniform(-1, 1, (n_samples, cfg.z_dim))
        
        # 2. Forward z through z_nn
        z_features = self._forward_z_nn(z)  # (n_samples, z_features_dim)
        
        # 3. Generate time inputs (multiple copies of t)
        time_inputs = self._generate_time_inputs(T)  # (n_time_copies, T)
        
        # 4. Sample noise
        noise = self.rng.normal(0, cfg.noise_std, (n_samples, 1, T))
        
        # 5. Build combo input: [t_copies, z_features, noise]
        # Broadcast time inputs to all samples
        t_broadcast = np.broadcast_to(time_inputs[np.newaxis, :, :], (n_samples, self.time_dim, T))
        z_broadcast = np.broadcast_to(z_features[:, :, np.newaxis], (n_samples, self.z_features_dim, T))
        
        combo_input = np.concatenate([t_broadcast, z_broadcast, noise], axis=1)
        
        # 6. Forward through combo NN
        root_outputs, combo_layers = self._forward_combo_nn(combo_input)
        
        # 7. Propagate through DAG
        n_nodes = cfg.n_dag_nodes
        dag_outputs = np.zeros((n_samples, n_nodes, T))
        
        for root_id in range(cfg.n_roots):
            dag_outputs[:, root_id, :] = root_outputs[:, root_id, :]
        
        for node_id in range(cfg.n_roots, n_nodes):
            parents = self.dag_parents[node_id]
            parent_outputs = dag_outputs[:, parents, :]
            
            W = self.dag_weights[node_id]
            b = self.dag_biases[node_id]
            
            node_output = np.einsum('d,ndt->nt', W, parent_outputs) + b
            node_output = np.tanh(node_output)
            
            if self.dag_has_noise[node_id]:
                node_noise = self.rng.normal(0, cfg.dag_noise_std, (n_samples, T))
                node_output = node_output + node_noise
            
            dag_outputs[:, node_id, :] = node_output
        
        intermediates = {
            'z': z,
            'z_features': z_features,
            'time_inputs': time_inputs,
            'noise': noise,
            'combo_input': combo_input,
            'combo_layers': combo_layers,
            'root_outputs': root_outputs,
        }
        
        return dag_outputs, intermediates
    
    def get_discretization_info(self) -> Dict[str, Any]:
        """Get information about discretized nodes in z_nn."""
        cfg = self.sampled_config
        info = {
            'total_z_nodes': sum(cfg.z_nn_nodes_per_layer),
            'discretized_nodes': [],
        }
        
        node_idx = 0
        for layer_idx in range(cfg.z_nn_layers):
            n_nodes = cfg.z_nn_nodes_per_layer[layer_idx]
            for local_node_idx in range(n_nodes):
                disc_cfg = cfg.z_nn_discretization[node_idx]
                if disc_cfg.is_discretized:
                    info['discretized_nodes'].append({
                        'global_idx': node_idx,
                        'layer': layer_idx,
                        'local_idx': local_node_idx,
                        'n_classes': disc_cfg.n_classes,
                        'class_values': disc_cfg.class_values.tolist(),
                    })
                node_idx += 1
        
        info['n_discretized'] = len(info['discretized_nodes'])
        return info
    
    def summary(self) -> str:
        """Return a summary of the network architecture."""
        cfg = self.sampled_config
        disc_info = self.get_discretization_info()
        
        lines = [
            "=== Hybrid DAG Generator ===",
            "",
            f"=== z (memory) ===",
            f"  dim: {cfg.z_dim}",
            f"  distribution: {cfg.z_distribution}",
            "",
            f"=== z_nn ===",
            f"  layers: {cfg.z_nn_layers}",
            f"  nodes: {cfg.z_nn_nodes_per_layer}",
            f"  output dim: {self.z_features_dim}",
            f"  discretized: {disc_info['n_discretized']}/{disc_info['total_z_nodes']}",
            "",
            f"=== combo_nn ===",
            f"  input: {self.combo_input_dim} (t_copies:{self.time_dim} + z:{self.z_features_dim} + noise:1)",
            f"  layers: {cfg.combo_nn_layers}",
            f"  nodes: {cfg.combo_nn_nodes_per_layer}",
            f"  activations: {[list(set(acts)) for acts in cfg.combo_node_activations]}",
            f"  output: {cfg.n_roots} roots",
            "",
            f"=== DAG ===",
            f"  nodes: {cfg.n_dag_nodes}",
            f"  roots: {cfg.n_roots}",
        ]
        return '\n'.join(lines)


def visualize_combo_nn_layers(gen: HybridDAGGenerator, intermediates: Dict, 
                               output_dir: str, net_id: int):
    """Visualize each layer of the combo NN individually."""
    cfg = gen.sampled_config
    combo_layers = intermediates['combo_layers']
    T = cfg.seq_length
    t = np.arange(T)
    
    n_samples = combo_layers[0].shape[0]
    
    # Also visualize input
    combo_input = intermediates['combo_input']
    
    # Input layer visualization
    fig, axes = plt.subplots(n_samples, min(8, gen.combo_input_dim), 
                             figsize=(min(16, 2*gen.combo_input_dim), 2*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    n_show = min(8, gen.combo_input_dim)
    for node_idx in range(n_show):
        for sample_idx in range(n_samples):
            ax = axes[sample_idx, node_idx]
            ax.plot(t, combo_input[sample_idx, node_idx, :], linewidth=1)
            ax.set_ylim(-1.5, 1.5)
            if sample_idx == 0:
                if node_idx < gen.time_dim:
                    # All time copies are just 't'
                    ax.set_title(f't_{node_idx}')
                elif node_idx < gen.time_dim + gen.z_features_dim:
                    # z feature
                    ax.set_title(f'z_{node_idx - gen.time_dim}')
                else:
                    ax.set_title('noise')
            if node_idx == 0:
                ax.set_ylabel(f'S{sample_idx}')
    
    plt.suptitle(f'Net {net_id} - Combo NN Input ({gen.combo_input_dim} dims)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/net{net_id:02d}_combo_input.png', dpi=150)
    plt.close()
    
    # Each hidden layer
    for layer_idx, layer_out in enumerate(combo_layers[:-1]):  # Exclude final (roots)
        n_nodes = layer_out.shape[1]
        max_cols = 8
        n_pages = (n_nodes + max_cols - 1) // max_cols
        
        for page in range(n_pages):
            start_node = page * max_cols
            end_node = min(start_node + max_cols, n_nodes)
            nodes_this_page = end_node - start_node
            
            fig, axes = plt.subplots(n_samples, nodes_this_page,
                                    figsize=(2*nodes_this_page, 2*n_samples))
            if n_samples == 1 and nodes_this_page == 1:
                axes = np.array([[axes]])
            elif n_samples == 1:
                axes = axes.reshape(1, -1)
            elif nodes_this_page == 1:
                axes = axes.reshape(-1, 1)
            
            # Get activations for this layer
            layer_acts = cfg.combo_node_activations[layer_idx] if layer_idx < len(cfg.combo_node_activations) else None
            
            for col_idx, node_idx in enumerate(range(start_node, end_node)):
                for sample_idx in range(n_samples):
                    ax = axes[sample_idx, col_idx]
                    ax.plot(t, layer_out[sample_idx, node_idx, :], linewidth=1)
                    ax.set_ylim(-1.5, 1.5)  # Wider range for non-tanh activations
                    if sample_idx == 0:
                        act_name = layer_acts[node_idx] if layer_acts else '?'
                        ax.set_title(f'{act_name}', fontsize=8)
                    if col_idx == 0:
                        ax.set_ylabel(f'S{sample_idx}')
            
            title = f'Net {net_id} - Combo NN Layer {layer_idx} ({n_nodes} nodes)'
            if n_pages > 1:
                title += f' [page {page+1}/{n_pages}]'
            plt.suptitle(title, fontweight='bold')
            plt.tight_layout()
            
            suffix = f'_p{page}' if n_pages > 1 else ''
            plt.savefig(f'{output_dir}/net{net_id:02d}_combo_L{layer_idx}{suffix}.png', dpi=150)
            plt.close()
    
    # Roots (final layer)
    root_outputs = intermediates['root_outputs']
    n_roots = root_outputs.shape[1]
    max_cols = 8
    n_pages = (n_roots + max_cols - 1) // max_cols
    
    for page in range(n_pages):
        start_node = page * max_cols
        end_node = min(start_node + max_cols, n_roots)
        nodes_this_page = end_node - start_node
        
        fig, axes = plt.subplots(n_samples, nodes_this_page,
                                figsize=(2*nodes_this_page, 2*n_samples))
        if n_samples == 1 and nodes_this_page == 1:
            axes = np.array([[axes]])
        elif n_samples == 1:
            axes = axes.reshape(1, -1)
        elif nodes_this_page == 1:
            axes = axes.reshape(-1, 1)
        
        for col_idx, node_idx in enumerate(range(start_node, end_node)):
            for sample_idx in range(n_samples):
                ax = axes[sample_idx, col_idx]
                ax.plot(t, root_outputs[sample_idx, node_idx, :], linewidth=1)
                ax.set_ylim(-1.1, 1.1)
                if sample_idx == 0:
                    ax.set_title(f'Root {node_idx}')
                if col_idx == 0:
                    ax.set_ylabel(f'S{sample_idx}')
        
        title = f'Net {net_id} - Roots ({n_roots} nodes)'
        if n_pages > 1:
            title += f' [page {page+1}/{n_pages}]'
        plt.suptitle(title, fontweight='bold')
        plt.tight_layout()
        
        suffix = f'_p{page}' if n_pages > 1 else ''
        plt.savefig(f'{output_dir}/net{net_id:02d}_roots{suffix}.png', dpi=150)
        plt.close()


def visualize_summary(gen: HybridDAGGenerator, dag_outputs: np.ndarray, 
                      intermediates: Dict, output_dir: str, net_id: int):
    """Summary visualization."""
    cfg = gen.sampled_config
    disc_info = gen.get_discretization_info()
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 1. z_features
    ax = axes[0]
    z_features = intermediates['z_features']
    x = np.arange(gen.z_features_dim)
    width = 0.15
    n_samples = z_features.shape[0]
    for i in range(n_samples):
        ax.bar(x + i * width, z_features[i], width, label=f'S{i}')
    ax.set_title(f'z_features (dim={gen.z_features_dim}, disc={disc_info["n_discretized"]})')
    ax.legend()
    ax.set_ylim(-1.2, 1.2)
    
    # 2. Root outputs
    ax = axes[1]
    root_outputs = intermediates['root_outputs']
    for r in range(min(8, cfg.n_roots)):
        ax.plot(root_outputs[0, r, :], label=f'R{r}', alpha=0.8)
    ax.set_title(f'Root outputs (sample 0, {cfg.n_roots} roots)')
    ax.legend()
    ax.set_ylim(-1.1, 1.1)
    
    # 3. DAG outputs
    ax = axes[2]
    non_root_start = cfg.n_roots
    n_show = min(8, cfg.n_dag_nodes - cfg.n_roots)
    for i in range(n_show):
        node_id = non_root_start + i
        ax.plot(dag_outputs[0, node_id, :], label=f'N{node_id}', alpha=0.8)
    ax.set_title(f'DAG non-root outputs (sample 0)')
    ax.legend()
    ax.set_ylim(-1.5, 1.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/net{net_id:02d}_summary.png', dpi=150)
    plt.close()


def main():
    print("=" * 60)
    print("Hybrid DAG Generator - Combo NN Visualization")
    print("=" * 60)
    
    config = HybridDAGConfig(
        z_dim_range=(6, 12),
        z_distribution='normal',
        z_nn_layers_range=(2, 3),
        z_nn_nodes_range=(10, 20),
        z_discretization_prob=0.3,
        combo_nn_layers_range=(2, 4),
        combo_nn_nodes_range=(16, 32),
        n_roots_range=(10, 18),
        n_dag_nodes_range=(40, 80),
        seq_length=200,
    )
    
    output_dir = '06_generator_experiments/dag_hybrid_combo'
    os.makedirs(output_dir, exist_ok=True)
    
    n_nets = 3
    for net_id in range(n_nets):
        gen = HybridDAGGenerator(config, seed=2000 + net_id)
        cfg = gen.sampled_config
        
        print(f"\nNet {net_id}:")
        print(f"  z_dim={cfg.z_dim}, z_features={gen.z_features_dim}")
        print(f"  combo_nn: {cfg.combo_nn_nodes_per_layer} -> {cfg.n_roots} roots")
        print(f"  dag: {cfg.n_dag_nodes} nodes")
        
        dag_outputs, intermediates = gen.propagate(n_samples=5)
        
        # Detailed layer visualization
        visualize_combo_nn_layers(gen, intermediates, output_dir, net_id)
        
        # Summary
        visualize_summary(gen, dag_outputs, intermediates, output_dir, net_id)
        
        print(f"  Saved visualizations")
    
    print(f"\nDone! Saved to {output_dir}")


if __name__ == '__main__':
    main()
