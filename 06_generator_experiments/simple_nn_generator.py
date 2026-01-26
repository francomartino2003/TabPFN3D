"""
Experimental neural network generator for time series.

Architecture:
- Memory vector (dim=6): Normal distribution ~N(0,1), fixed per sample
- Time inputs (7 total):
  - t_scaled: time normalized to [-1, 1]
  - sin(2*pi*k*t/T) for k=1,2,3
  - cos(2*pi*k*t/T) for k=1,2,3
- NN transformation with ReLU activation
- Small noise added after each activation
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence
import os


@dataclass
class NNConfig:
    """Configuration for the neural network generator."""
    # Input dimensions
    memory_dim: int = 6
    stochastic_input_dim: int = 0  # Random input that changes at each time step
    n_fourier_components: int = 3  # k=1,2,3 -> 6 time inputs (sin+cos) + 1 scaled time
    
    # Network architecture
    layer_sizes: Tuple[int, ...] = None  # Custom layer sizes (if None, uses n_nodes_per_layer)
    n_nodes_per_layer: int = 10  # Used if layer_sizes is None
    n_hidden_layers: int = 3
    activation: str = 'relu'  # 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'identity'
    
    # Weight initialization
    weight_init: str = 'xavier_uniform'  # 'xavier_uniform' or 'xavier_normal'
    weight_scale: float = 1.0  # Multiplier for Xavier init
    
    # Bias initialization: b ~ N(0, bias_std) or zeros if bias_std=0
    bias_std: float = 0.1
    
    # Memory initialization
    memory_init: str = 'uniform'  # 'uniform' (U[-1,1]) or 'normal' (N(0,1) clipped to ~[-1,1])
    
    # Noise and sequence
    noise_scale: float = 0.02
    seq_length: int = 100
    memory_skip_connections: bool = False  # If True, memory is fed to all layers


class SimpleNNGenerator:
    """
    Generate time series by propagating through a random neural network DAG.
    """
    
    def __init__(self, config: NNConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.default_rng(seed)
        
        # Calculate total input dimension
        # 1 (scaled time) + 2*n_fourier (sin+cos) + memory_dim + stochastic_input_dim
        self.time_input_dim = 1 + 2 * config.n_fourier_components
        self.total_input_dim = self.time_input_dim + config.memory_dim + config.stochastic_input_dim
        
        # Build network structure
        self._build_network()
    
    def _build_network(self):
        """Build random network weights and structure."""
        cfg = self.config
        
        # Network structure: input -> hidden layers
        self.layer_sizes = [self.total_input_dim]
        
        if cfg.layer_sizes is not None:
            # Use custom layer sizes
            for size in cfg.layer_sizes:
                self.layer_sizes.append(size)
        else:
            # Use uniform layer sizes
            for _ in range(cfg.n_hidden_layers):
                self.layer_sizes.append(cfg.n_nodes_per_layer)
        
        # Create weights for each layer
        self.weights = []
        self.biases = []
        
        for i in range(len(self.layer_sizes) - 1):
            if i == 0:
                # First layer: input is time + memory
                in_dim = self.layer_sizes[i]
            elif cfg.memory_skip_connections:
                # Subsequent layers with skip: input is previous output + memory
                in_dim = self.layer_sizes[i] + cfg.memory_dim
            else:
                # No skip connections: just previous layer output
                in_dim = self.layer_sizes[i]
            
            out_dim = self.layer_sizes[i + 1]
            
            # Weight initialization
            if cfg.weight_init == 'xavier_uniform':
                # Xavier uniform: W ~ U[-a, a], a = sqrt(6 / (fan_in + fan_out))
                a = cfg.weight_scale * np.sqrt(6.0 / (in_dim + out_dim))
                W = self.rng.uniform(-a, a, (out_dim, in_dim))
            else:  # xavier_normal
                # Xavier normal: W ~ N(0, std), std = sqrt(2 / (fan_in + fan_out))
                std = cfg.weight_scale * np.sqrt(2.0 / (in_dim))
                W = self.rng.normal(0, std, (out_dim, in_dim))
            
            # Bias initialization: zeros or N(0, bias_std)
            b = np.zeros(out_dim) if cfg.bias_std == 0 else self.rng.normal(0, cfg.bias_std, out_dim)
            
            self.weights.append(W)
            self.biases.append(b)
        
        # Store layer info for visualization
        self.n_layers = len(self.layer_sizes)
        self.total_nodes = sum(self.layer_sizes)
    
    def _activation(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.config.activation == 'relu':
            return np.maximum(0, x)
        elif self.config.activation == 'tanh':
            return np.tanh(x)
        elif self.config.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif self.config.activation == 'leaky_relu':
            return np.where(x > 0, x, 0.1 * x)
        elif self.config.activation == 'identity':
            return x
        else:
            return np.maximum(0, x)  # default to relu
    
    def _generate_time_inputs(self, T: int) -> np.ndarray:
        """
        Generate time-dependent inputs.
        
        Returns: (time_input_dim, T) array
        """
        t = np.arange(T)
        inputs = []
        
        # Scaled time: [-1, 1]
        t_scaled = 2 * t / (T - 1) - 1 if T > 1 else np.zeros(T)
        inputs.append(t_scaled)
        
        # Fourier components: sin and cos
        for k in range(1, self.config.n_fourier_components + 1):
            phase = 2 * np.pi * k * t / T
            inputs.append(np.sin(phase))
            inputs.append(np.cos(phase))
        
        return np.array(inputs)  # (time_input_dim, T)
    
    def _generate_memory(self, n_samples: int) -> np.ndarray:
        """
        Generate memory vectors for each sample.
        
        Returns: (n_samples, memory_dim) array with values roughly in [-1, 1]
        """
        if self.config.memory_init == 'uniform':
            # Uniform in [-1, 1]
            return self.rng.uniform(-1, 1, (n_samples, self.config.memory_dim))
        else:  # normal
            # Normal with std=0.5, so ~95% of values in [-1, 1]
            return np.clip(self.rng.normal(0, 0.5, (n_samples, self.config.memory_dim)), -1, 1)
    
    def _generate_stochastic_inputs(self, n_samples: int, T: int) -> np.ndarray:
        """
        Generate random inputs that vary at each time step.
        
        Returns: (n_samples, stochastic_input_dim, T) array or None if dim=0
        """
        if self.config.stochastic_input_dim == 0:
            return None
        return self.rng.uniform(-1, 1, (n_samples, self.config.stochastic_input_dim, T))
    
    def propagate(self, n_samples: int = 5) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Propagate through the network to generate time series.
        Memory is fed as input to ALL layers (skip connections).
        
        Returns:
            - all_node_values: List of arrays, one per layer
              Each array has shape (n_samples, layer_size, T)
        """
        T = self.config.seq_length
        
        # Generate inputs
        time_inputs = self._generate_time_inputs(T)  # (time_dim, T)
        memory = self._generate_memory(n_samples)    # (n_samples, mem_dim)
        stochastic = self._generate_stochastic_inputs(n_samples, T)  # (n_samples, stoch_dim, T) or None
        
        # Build input layer: combine time, memory, and stochastic inputs
        # time_inputs: (time_dim, T) -> broadcast to (n_samples, time_dim, T)
        # memory: (n_samples, mem_dim) -> expand to (n_samples, mem_dim, T)
        
        time_broadcast = np.broadcast_to(time_inputs, (n_samples, self.time_input_dim, T))
        memory_broadcast = np.repeat(memory[:, :, np.newaxis], T, axis=2)  # (n_samples, mem_dim, T)
        
        # Input layer: (n_samples, total_input_dim, T)
        if stochastic is not None:
            input_layer = np.concatenate([time_broadcast, memory_broadcast, stochastic], axis=1)
        else:
            input_layer = np.concatenate([time_broadcast, memory_broadcast], axis=1)
        
        # Store all layer activations
        all_layers = [input_layer]
        
        current = input_layer  # (n_samples, in_dim, T)
        
        for layer_idx, (W, b) in enumerate(zip(self.weights, self.biases)):
            # For layers after the first, concatenate memory (skip connection) if enabled
            if layer_idx > 0 and self.config.memory_skip_connections:
                current = np.concatenate([current, memory_broadcast], axis=1)
            
            # W: (out_dim, in_dim), b: (out_dim,)
            # current: (n_samples, in_dim, T)
            
            # Linear transformation: for each sample and time step
            # Result: (n_samples, out_dim, T)
            out = np.einsum('oi,nit->not', W, current) + b[:, np.newaxis]
            
            # Activation
            out = self._activation(out)
            
            # Add noise
            noise = self.rng.normal(0, self.config.noise_scale, out.shape)
            out = out + noise
            
            all_layers.append(out)
            current = out
        
        return all_layers
    
    def get_layer_info(self) -> List[str]:
        """Get descriptions of each layer."""
        info = []
        info.append(f"Input ({self.layer_sizes[0]}): time({self.time_input_dim}) + memory({self.config.memory_dim})")
        for i, size in enumerate(self.layer_sizes[1:], 1):
            if i == 1 or not self.config.memory_skip_connections:
                info.append(f"Hidden {i} ({size}): {self.config.activation}")
            else:
                info.append(f"Hidden {i} ({size}): {self.config.activation} +mem")
        return info


def visualize_network(generator: SimpleNNGenerator, n_samples: int = 5, 
                      output_dir: str = "visualizations", network_id: int = 0,
                      skip_input_layer: bool = True):
    """
    Visualize all nodes across all layers for multiple samples.
    Each sample gets its own row, nodes as columns.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate data
    all_layers = generator.propagate(n_samples)
    layer_info = generator.get_layer_info()
    
    T = generator.config.seq_length
    time = np.arange(T)
    
    # Color palette for nodes
    node_colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    # Skip input layer if requested
    start_layer = 1 if skip_input_layer else 0
    
    # Create one figure per layer
    for layer_idx in range(start_layer, len(all_layers)):
        layer_data = all_layers[layer_idx]
        info = layer_info[layer_idx]
        
        # layer_data: (n_samples, n_nodes, T)
        n_nodes = layer_data.shape[1]
        
        # Layout: rows = samples, cols = nodes (max 6 per page)
        max_cols = 6
        n_pages = (n_nodes + max_cols - 1) // max_cols
        
        for page in range(n_pages):
            start_node = page * max_cols
            end_node = min(start_node + max_cols, n_nodes)
            nodes_this_page = end_node - start_node
            
            fig, axes = plt.subplots(n_samples, nodes_this_page, 
                                    figsize=(3 * nodes_this_page, 2.5 * n_samples))
            
            if n_samples == 1 and nodes_this_page == 1:
                axes = np.array([[axes]])
            elif n_samples == 1:
                axes = axes.reshape(1, -1)
            elif nodes_this_page == 1:
                axes = axes.reshape(-1, 1)
            
            page_title = f"Network {network_id} | Layer {layer_idx}: {info}"
            if n_pages > 1:
                page_title += f" (nodes {start_node}-{end_node-1})"
            fig.suptitle(page_title, fontsize=14, fontweight='bold')
            
            for col_idx, node_idx in enumerate(range(start_node, end_node)):
                color = node_colors[node_idx % 20]
                
                for sample_idx in range(n_samples):
                    ax = axes[sample_idx, col_idx]
                    series = layer_data[sample_idx, node_idx, :]
                    
                    ax.plot(time, series, color=color, linewidth=1.2)
                    ax.grid(True, alpha=0.3)
                    
                    # Labels
                    if sample_idx == 0:
                        ax.set_title(f"Node {node_idx}", fontsize=10)
                    if col_idx == 0:
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
    print("Simple NN Generator Experiment")
    print("=" * 60)
    
    # Configuration
    config = NNConfig(
        # Inputs
        memory_dim=6,
        memory_init='normal',  # 'uniform' or 'normal'
        stochastic_input_dim=0,  # Random input at each time step
        n_fourier_components=0,  # -> 7 time inputs total
        
        # Architecture
        layer_sizes=None,
        n_nodes_per_layer=24,
        n_hidden_layers=10,
        activation='relu',
        memory_skip_connections=False,
        
        # Initialization
        weight_init='xavier_normal',  # 'xavier_uniform' or 'xavier_normal'
        weight_scale=1.0,
        bias_std=0.01,  # Bias ~ N(0, bias_std), use 0 for no bias
        
        # Noise and sequence
        noise_scale=0.00000,
        seq_length=100,
    )
    
    print(f"\nConfiguration:")
    print(f"  Memory dim: {config.memory_dim} ({config.memory_init})")
    print(f"  Stochastic input dim: {config.stochastic_input_dim}")
    print(f"  Time inputs: 1 + 2*{config.n_fourier_components} = {1 + 2*config.n_fourier_components}")
    total_in = config.memory_dim + config.stochastic_input_dim + 1 + 2*config.n_fourier_components
    print(f"  Total input dim: {total_in}")
    if config.layer_sizes:
        print(f"  Layer sizes: {config.layer_sizes}")
    else:
        print(f"  Hidden layers: {config.n_hidden_layers} x {config.n_nodes_per_layer} nodes")
    print(f"  Activation: {config.activation}")
    print(f"  Weight init: {config.weight_init} (scale={config.weight_scale})")
    print(f"  Bias std: {config.bias_std}")
    print(f"  Noise scale: {config.noise_scale}")
    print(f"  Memory skip connections: {config.memory_skip_connections}")
    print(f"  Sequence length: {config.seq_length}")
    
    output_dir = "/Users/franco/Documents/TabPFN3D/06_generator_experiments/tanh_experiment"
    
    # Generate 8 different networks
    n_networks = 8
    print(f"\nGenerating {n_networks} networks...")
    
    for net_id in range(n_networks):
        # Create generator with different seed for each network
        generator = SimpleNNGenerator(config, seed=42 + net_id)
        
        print(f"\n  Network {net_id}:")
        visualize_network(generator, n_samples=5, output_dir=output_dir, 
                         network_id=net_id, skip_input_layer=True)
        print(f"    Done")
    
    print(f"\nDone! Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
