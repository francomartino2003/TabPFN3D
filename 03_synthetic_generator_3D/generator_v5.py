"""
3D Synthetic Dataset Generator v5.

Simplified version that incorporates ideas from random_nn_generator:
- Dense NN-like propagation with smooth activations
- Per-node noise and discretization
- Rich time transforms and memory inputs
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from config_v5 import (
    PriorConfig3D_v5, SampledConfig3D_v5, 
    NodeNoiseConfig, NodeDiscretizationConfig,
    apply_time_transform
)


@dataclass
class PropagatedValues:
    """Container for propagated node values."""
    layer_values: List[np.ndarray]  # List of (n_samples, n_nodes_in_layer, T)
    layer_names: List[str]  # Names for each layer


class SimpleNNGenerator3D:
    """
    Simplified 3D generator using NN-like architecture.
    
    Structure:
    - Input layer: time transforms + memory + stochastic inputs
    - Hidden layers: NN with smooth activations, per-node noise/discretization
    - Output layer: values for feature/target selection
    """
    
    def __init__(self, config: SampledConfig3D_v5, seed: int = None):
        self.config = config
        self.rng = np.random.default_rng(seed or config.seed)
        
        # Build network structure
        self._build_network()
    
    def _build_network(self):
        """Build the network architecture."""
        cfg = self.config
        
        # Input dimensions
        n_time_inputs = len(cfg.time_transforms)
        n_memory_inputs = cfg.memory_dim
        n_stochastic_inputs = cfg.stochastic_dim
        self.input_dim = n_time_inputs + n_memory_inputs + n_stochastic_inputs
        
        # Compute number of hidden layers and nodes per layer
        # Use n_nodes to define total internal nodes, distribute across layers
        n_internal = cfg.n_nodes - self.input_dim
        n_internal = max(n_internal, cfg.n_features + 1)  # At least enough for features + target
        
        # Decide on number of layers (log-uniform would favor smaller)
        # For simplicity, we'll use 3-5 layers
        n_layers = min(5, max(3, int(np.sqrt(n_internal))))
        
        # Distribute nodes across layers
        nodes_per_layer = []
        remaining = n_internal
        for i in range(n_layers):
            if i == n_layers - 1:
                nodes = remaining
            else:
                # Roughly equal distribution with some variance
                avg = remaining // (n_layers - i)
                nodes = max(1, avg + self.rng.integers(-avg//3, avg//3 + 1))
            nodes_per_layer.append(nodes)
            remaining -= nodes
        
        self.nodes_per_layer = nodes_per_layer
        self.n_layers = n_layers
        
        # Initialize weights and biases for each layer
        self.weights = []
        self.biases = []
        
        in_dim = self.input_dim
        for layer_idx, out_dim in enumerate(nodes_per_layer):
            # Xavier initialization
            if cfg.weight_init == 'xavier_uniform':
                limit = np.sqrt(6 / (in_dim + out_dim)) * cfg.weight_scale
                W = self.rng.uniform(-limit, limit, (in_dim, out_dim))
            else:  # xavier_normal
                std = np.sqrt(2 / (in_dim + out_dim)) * cfg.weight_scale
                W = self.rng.normal(0, std, (in_dim, out_dim))
            
            b = self.rng.normal(0, cfg.bias_std, out_dim) if cfg.bias_std > 0 else np.zeros(out_dim)
            
            self.weights.append(W)
            self.biases.append(b)
            in_dim = out_dim
        
        # Assign activations, noise, and discretization configs to nodes
        # Flatten across layers for indexing
        node_idx = 0
        self.layer_configs = []  # (activation, noise_cfg, disc_cfg) for each node in each layer
        
        for layer_idx, n_nodes in enumerate(nodes_per_layer):
            layer_cfg = []
            for _ in range(n_nodes):
                if node_idx < len(cfg.activations):
                    act = cfg.activations[node_idx]
                else:
                    act = 'softplus'
                
                if node_idx < len(cfg.node_noise):
                    noise_cfg = cfg.node_noise[node_idx]
                else:
                    noise_cfg = NodeNoiseConfig(has_noise=False)
                
                if node_idx < len(cfg.node_discretization):
                    disc_cfg = cfg.node_discretization[node_idx]
                else:
                    disc_cfg = NodeDiscretizationConfig(is_discretization=False)
                
                layer_cfg.append((act, noise_cfg, disc_cfg))
                node_idx += 1
            
            self.layer_configs.append(layer_cfg)
        
        # Initialize discretization prototypes for nodes that need them
        self._init_discretization_prototypes()
    
    def _init_discretization_prototypes(self):
        """Initialize prototypes for discretization nodes."""
        in_dim = self.input_dim
        
        for layer_idx, layer_cfg in enumerate(self.layer_configs):
            for node_idx, (act, noise_cfg, disc_cfg) in enumerate(layer_cfg):
                if disc_cfg.is_discretization and disc_cfg.prototypes is None:
                    n_classes = disc_cfg.n_classes
                    # Sample prototype vectors (class centers in input space)
                    disc_cfg.prototypes = self.rng.normal(0, 1, (n_classes, in_dim))
                    # Sample output values for each class
                    disc_cfg.class_values = self.rng.normal(0, 1, n_classes)
            
            in_dim = self.nodes_per_layer[layer_idx]
    
    def _activation(self, x: np.ndarray, act: str) -> np.ndarray:
        """Apply activation function."""
        if act == 'identity':
            return x
        elif act == 'softplus':
            return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -500, 20))))
        elif act == 'tanh':
            return np.tanh(x)
        elif act == 'elu':
            return np.where(x > 0, x, np.exp(np.clip(x, -500, 0)) - 1)
        elif act == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        else:
            return x
    
    def _apply_noise(self, x: np.ndarray, noise_cfg: NodeNoiseConfig) -> np.ndarray:
        """Apply per-node noise."""
        if not noise_cfg.has_noise:
            return x
        
        shape = x.shape
        if noise_cfg.distribution == 'normal':
            noise = self.rng.normal(0, noise_cfg.scale, shape)
        elif noise_cfg.distribution == 'uniform':
            noise = self.rng.uniform(-noise_cfg.scale * np.sqrt(3), 
                                     noise_cfg.scale * np.sqrt(3), shape)
        elif noise_cfg.distribution == 'laplace':
            noise = self.rng.laplace(0, noise_cfg.scale / np.sqrt(2), shape)
        else:
            noise = 0
        
        return x + noise
    
    def _apply_discretization(self, node_value: np.ndarray, parent_values: np.ndarray, 
                               disc_cfg: NodeDiscretizationConfig) -> np.ndarray:
        """
        Apply discretization to a node.
        
        node_value: (n_samples, T) - current node value (not used, replaced)
        parent_values: (n_samples, n_parents, T) - parent values to determine class
        disc_cfg: discretization configuration
        
        Returns: (n_samples, T) - discretized output
        """
        if not disc_cfg.is_discretization:
            return node_value
        
        n_samples, T = node_value.shape
        n_classes, n_parents = disc_cfg.prototypes.shape
        
        # Reshape for distance computation
        # parent_values: (n_samples, n_parents, T) -> (n_samples, T, n_parents)
        parent_values = parent_values.transpose(0, 2, 1)  # (n_samples, T, n_parents)
        
        # Compute distance to each prototype
        # prototypes: (n_classes, n_parents)
        # parent_values: (n_samples, T, n_parents)
        
        # Expand for broadcasting
        prototypes = disc_cfg.prototypes[np.newaxis, np.newaxis, :, :]  # (1, 1, n_classes, n_parents)
        parent_expanded = parent_values[:, :, np.newaxis, :]  # (n_samples, T, 1, n_parents)
        
        # Compute squared distances
        distances = np.sum((parent_expanded - prototypes) ** 2, axis=-1)  # (n_samples, T, n_classes)
        
        # Find closest class
        closest_class = np.argmin(distances, axis=-1)  # (n_samples, T)
        
        # Map to class values
        output = disc_cfg.class_values[closest_class]  # (n_samples, T)
        
        return output
    
    def _generate_time_inputs(self, T: int) -> np.ndarray:
        """Generate time input vectors for T timesteps."""
        cfg = self.config
        u = np.linspace(0, 1, T)  # Normalized time
        
        time_inputs = []
        for transform in cfg.time_transforms:
            values = apply_time_transform(transform, u)
            time_inputs.append(values)
        
        return np.stack(time_inputs, axis=0)  # (n_time_inputs, T)
    
    def _generate_memory(self, n_samples: int) -> np.ndarray:
        """Generate memory vectors (one per sample, constant over time)."""
        cfg = self.config
        
        if cfg.memory_init == 'uniform':
            memory = self.rng.uniform(-1, 1, (n_samples, cfg.memory_dim))
        else:  # normal
            memory = np.clip(self.rng.normal(0, 0.5, (n_samples, cfg.memory_dim)), -1, 1)
        
        return memory  # (n_samples, memory_dim)
    
    def _generate_stochastic_inputs(self, n_samples: int, T: int) -> np.ndarray:
        """Generate stochastic inputs (random at each timestep)."""
        cfg = self.config
        if cfg.stochastic_dim == 0:
            return np.zeros((n_samples, 0, T))
        
        return self.rng.uniform(-1, 1, (n_samples, cfg.stochastic_dim, T))
    
    def propagate(self, n_samples: int, T: int) -> PropagatedValues:
        """
        Propagate inputs through the network.
        
        Returns: PropagatedValues with layer outputs
        """
        cfg = self.config
        
        # Generate inputs
        time_inputs = self._generate_time_inputs(T)  # (n_time_inputs, T)
        memory = self._generate_memory(n_samples)  # (n_samples, memory_dim)
        stochastic = self._generate_stochastic_inputs(n_samples, T)  # (n_samples, stochastic_dim, T)
        
        # Construct input layer
        # time_inputs: broadcast to (n_samples, n_time_inputs, T)
        time_broadcast = np.broadcast_to(time_inputs[np.newaxis, :, :], 
                                          (n_samples, time_inputs.shape[0], T))
        
        # memory: expand to (n_samples, memory_dim, T)
        memory_broadcast = np.broadcast_to(memory[:, :, np.newaxis], 
                                            (n_samples, cfg.memory_dim, T))
        
        # Combine inputs: (n_samples, input_dim, T)
        current = np.concatenate([time_broadcast, memory_broadcast, stochastic], axis=1)
        
        layer_values = [current.copy()]
        layer_names = ['Input']
        
        # Propagate through hidden layers
        for layer_idx in range(self.n_layers):
            W = self.weights[layer_idx]  # (in_dim, out_dim)
            b = self.biases[layer_idx]  # (out_dim,)
            layer_cfg = self.layer_configs[layer_idx]
            
            # current: (n_samples, in_dim, T)
            # We need to compute W @ current for each sample and timestep
            # current.transpose(0, 2, 1): (n_samples, T, in_dim)
            # @ W: (n_samples, T, out_dim)
            
            current_transposed = current.transpose(0, 2, 1)  # (n_samples, T, in_dim)
            out_transposed = np.einsum('nti,io->nto', current_transposed, W) + b  # (n_samples, T, out_dim)
            
            # out: (n_samples, out_dim, T)
            out = out_transposed.transpose(0, 2, 1)
            
            # Apply per-node activation, noise, and discretization
            for node_idx, (act, noise_cfg, disc_cfg) in enumerate(layer_cfg):
                node_values = out[:, node_idx, :]  # (n_samples, T)
                
                # Apply activation
                node_values = self._activation(node_values, act)
                
                # Apply noise
                node_values = self._apply_noise(node_values, noise_cfg)
                
                # Apply discretization (uses parent values)
                if disc_cfg.is_discretization:
                    # Parent values are the previous layer's output
                    node_values = self._apply_discretization(node_values, current, disc_cfg)
                
                out[:, node_idx, :] = node_values
            
            current = out
            layer_values.append(current.copy())
            layer_names.append(f'Layer {layer_idx + 1}')
        
        return PropagatedValues(layer_values=layer_values, layer_names=layer_names)
    
    def get_summary(self) -> str:
        """Get a summary of the network configuration."""
        cfg = self.config
        
        n_noisy = sum(1 for nc in cfg.node_noise if nc.has_noise)
        n_disc = sum(1 for dc in cfg.node_discretization if dc.is_discretization)
        
        lines = [
            f"Input dim: {self.input_dim} (time: {len(cfg.time_transforms)}, memory: {cfg.memory_dim}, stochastic: {cfg.stochastic_dim})",
            f"Layers: {self.nodes_per_layer}",
            f"Total nodes: {sum(self.nodes_per_layer)}",
            f"Activations: {set(cfg.activations)}",
            f"Noisy nodes: {n_noisy}",
            f"Discretization nodes: {n_disc}",
        ]
        return '\n'.join(lines)


@dataclass
class GeneratedDataset:
    """A complete generated dataset."""
    X: np.ndarray  # (n_samples, n_features, t_subseq)
    y: np.ndarray  # (n_samples,) - class labels
    feature_names: List[str]
    n_classes: int
    config_summary: str


def generate_dataset(generator: SimpleNNGenerator3D, config: SampledConfig3D_v5) -> GeneratedDataset:
    """
    Generate a complete dataset from the generator.
    
    Selects features and target from propagated values.
    """
    n_samples = config.n_samples
    T = config.T_total
    t_subseq = config.t_subseq
    n_features = config.n_features
    n_classes = config.n_classes
    
    # Propagate
    propagated = generator.propagate(n_samples, T)
    
    # Get last layer values (output layer)
    last_layer = propagated.layer_values[-1]  # (n_samples, n_nodes, T)
    n_output_nodes = last_layer.shape[1]
    
    # Select target node (preferably a discretization node)
    disc_nodes = []
    for i, dc in enumerate(config.node_discretization):
        if dc.is_discretization and i < n_output_nodes:
            disc_nodes.append(i)
    
    if disc_nodes:
        target_node_idx = generator.rng.choice(disc_nodes)
    else:
        target_node_idx = generator.rng.integers(0, n_output_nodes)
    
    # Select feature nodes (different from target, prioritize closer nodes)
    available_nodes = [i for i in range(n_output_nodes) if i != target_node_idx]
    
    if len(available_nodes) < n_features:
        # Use nodes from second-to-last layer if needed
        if len(propagated.layer_values) > 2:
            prev_layer = propagated.layer_values[-2]
            extra_nodes = prev_layer.shape[1]
            available_nodes.extend([n_output_nodes + i for i in range(extra_nodes)])
    
    # Prioritize nodes closer to target (by index distance for simplicity)
    distances = [abs(n % n_output_nodes - target_node_idx) for n in available_nodes]
    alpha = config.spatial_distance_alpha
    probs = np.array([1.0 / (1.0 + d ** alpha) for d in distances])
    probs = probs / probs.sum()
    
    n_to_select = min(n_features, len(available_nodes))
    feature_node_idxs = generator.rng.choice(
        available_nodes, size=n_to_select, replace=False, p=probs
    )
    
    # Extract subsequences
    # For IID mode, sample random start points
    # For simplicity, use IID sampling
    max_start = T - t_subseq - abs(config.target_offset)
    if max_start < 1:
        max_start = 1
    
    start_idxs = generator.rng.integers(0, max_start, size=n_samples)
    
    # Extract features: (n_samples, n_features, t_subseq)
    X = np.zeros((n_samples, n_to_select, t_subseq), dtype=np.float32)
    
    for sample_idx in range(n_samples):
        start = start_idxs[sample_idx]
        for feat_idx, node_idx in enumerate(feature_node_idxs):
            if node_idx < n_output_nodes:
                X[sample_idx, feat_idx, :] = last_layer[sample_idx, node_idx, start:start+t_subseq]
            else:
                # Node from previous layer
                prev_node_idx = node_idx - n_output_nodes
                prev_layer = propagated.layer_values[-2]
                X[sample_idx, feat_idx, :] = prev_layer[sample_idx, prev_node_idx, start:start+t_subseq]
    
    # Extract target values and discretize to classes
    target_start_offset = config.target_offset
    target_values = np.zeros(n_samples, dtype=np.float32)
    
    for sample_idx in range(n_samples):
        start = start_idxs[sample_idx]
        target_t = start + t_subseq + target_start_offset
        target_t = max(0, min(target_t, T - 1))
        target_values[sample_idx] = last_layer[sample_idx, target_node_idx, target_t]
    
    # Discretize target into n_classes using quantiles
    percentiles = np.linspace(0, 100, n_classes + 1)
    thresholds = np.percentile(target_values, percentiles[1:-1])
    y = np.digitize(target_values, thresholds)
    
    # Ensure all classes have at least some samples
    unique_classes = np.unique(y)
    if len(unique_classes) < n_classes:
        # Reassign to have more balanced classes
        sorted_idxs = np.argsort(target_values)
        samples_per_class = n_samples // n_classes
        y = np.zeros(n_samples, dtype=np.int32)
        for c in range(n_classes):
            start_idx = c * samples_per_class
            end_idx = start_idx + samples_per_class if c < n_classes - 1 else n_samples
            y[sorted_idxs[start_idx:end_idx]] = c
    
    feature_names = [f"Feature_{i}" for i in range(n_to_select)]
    
    return GeneratedDataset(
        X=X,
        y=y,
        feature_names=feature_names,
        n_classes=n_classes,
        config_summary=config.get_config_summary()
    )


def visualize_dataset(dataset: GeneratedDataset, output_path: str, dataset_name: str = "Synthetic"):
    """
    Visualize a generated dataset in the style of real dataset visualizations.
    
    Shows 5 samples from different classes with their time series.
    """
    import matplotlib.pyplot as plt
    
    X = dataset.X  # (n_samples, n_features, t_subseq)
    y = dataset.y
    n_samples, n_features, t_subseq = X.shape
    n_classes = dataset.n_classes
    
    # Select samples to visualize (one per class if possible, else random)
    unique_classes = np.unique(y)
    samples_to_show = []
    
    for cls in unique_classes[:5]:
        cls_samples = np.where(y == cls)[0]
        if len(cls_samples) > 0:
            samples_to_show.append(np.random.choice(cls_samples))
    
    # Fill up to 5 samples if needed
    while len(samples_to_show) < 5 and len(samples_to_show) < n_samples:
        remaining = [i for i in range(n_samples) if i not in samples_to_show]
        if remaining:
            samples_to_show.append(np.random.choice(remaining))
        else:
            break
    
    n_rows = len(samples_to_show)
    
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 2.5 * n_rows))
    if n_rows == 1:
        axes = [axes]
    
    # Title
    fig.suptitle(
        f"{dataset_name}\nShape: {n_samples} samples × {n_features} channels × {t_subseq} timesteps | {n_classes} classes",
        fontsize=14, fontweight='bold'
    )
    
    for row_idx, sample_idx in enumerate(samples_to_show):
        ax = axes[row_idx]
        sample_class = y[sample_idx]
        
        # Plot all features for this sample
        for feat_idx in range(n_features):
            ts = X[sample_idx, feat_idx, :]
            ax.plot(ts, label=f"Feature {feat_idx}" if n_features <= 5 else None, linewidth=1)
        
        ax.set_ylabel("Value")
        
        # Add sample/class label
        ax.text(0.01, 0.95, f"Sample {sample_idx} | Class: {sample_class + 1}", 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.legend(loc='upper right', fontsize=8)
        
        if row_idx == n_rows - 1:
            ax.set_xlabel("Time Step")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_generator(generator: SimpleNNGenerator3D, config: SampledConfig3D_v5,
                        output_dir: str = "./vis", dataset_id: int = 0):
    """Generate and visualize a dataset."""
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate dataset
    dataset = generate_dataset(generator, config)
    
    # Save config
    with open(f"{output_dir}/dataset{dataset_id:02d}_config.txt", 'w') as f:
        f.write(f"Dataset {dataset_id}\n")
        f.write("=" * 40 + "\n")
        f.write(dataset.config_summary)
        f.write("\n\n")
        f.write(generator.get_summary())
    
    # Visualize
    visualize_dataset(
        dataset, 
        f"{output_dir}/dataset{dataset_id:02d}.png",
        f"Synthetic Dataset {dataset_id}"
    )
    
    return dataset


def main():
    print("=" * 60)
    print("3D Generator v5 - Dataset Visualization")
    print("=" * 60)
    
    output_dir = "/Users/franco/Documents/TabPFN3D/05_flattened_benchmark/results/3d_generator_v5_vis"
    
    prior = PriorConfig3D_v5()
    n_datasets = 8
    
    print(f"\nGenerating {n_datasets} datasets...\n")
    
    for dataset_id in range(n_datasets):
        rng = np.random.default_rng(42 + dataset_id)
        config = SampledConfig3D_v5.sample_from_prior(prior, rng)
        
        generator = SimpleNNGenerator3D(config)
        dataset = visualize_generator(generator, config, output_dir, dataset_id)
        
        print(f"  Dataset {dataset_id}:")
        print(f"    Shape: {dataset.X.shape[0]} samples × {dataset.X.shape[1]} features × {dataset.X.shape[2]} timesteps")
        print(f"    Classes: {dataset.n_classes}, distribution: {np.bincount(dataset.y)}")
    
    print(f"\nDone! Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
