"""
Dataset Generator - Uses RandomNNGenerator to create classification datasets.

Only adds:
- Sample mode logic (iid, sliding_window, mixed)
- Target selection (discretization node)
- Feature selection (prioritize close to target)
- Dataset constraints
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
from typing import List, Tuple

from random_nn_generator import RandomNNGenerator, RandomNNConfig


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    # Sample modes - IID ONLY now
    prob_iid: float = 1.0
    prob_sliding: float = 0.0
    prob_mixed: float = 0.0
    
    # Constraints
    max_samples: int = 1000
    max_features: int = 8
    max_timesteps: int = 1000
    max_m_times_t: int = 500
    
    # Feature distribution (strongly prefer univariate)
    # P(1)=60%, P(2)=20%, P(3)=10%, P(4+)=10%
    
    # Target offset range
    max_target_offset: int = 15
    
    # Train/test split
    train_ratio: float = 0.8


@dataclass  
class GeneratedDataset:
    """A generated dataset."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    n_classes: int
    sample_mode: str
    target_offset: int


def sample_n_features(rng: np.random.Generator) -> int:
    """Sample number of features with strong preference for univariate."""
    r = rng.random()
    if r < 0.60:
        return 1
    elif r < 0.80:
        return 2
    elif r < 0.90:
        return 3
    else:
        return rng.integers(4, 9)


def generate_dataset(
    nn_config: RandomNNConfig,
    ds_config: DatasetConfig,
    seed: int = 42
) -> GeneratedDataset:
    """
    Generate a classification dataset using RandomNNGenerator.
    """
    rng = np.random.default_rng(seed)
    
    # Sample dataset parameters
    n_features = sample_n_features(rng)
    
    # Sample timesteps (enforce m*t <= 500)
    max_t = min(ds_config.max_timesteps, ds_config.max_m_times_t // n_features)
    t_subseq = rng.integers(20, max(21, max_t + 1))
    
    # Sample mode
    mode_roll = rng.random()
    if mode_roll < ds_config.prob_iid:
        sample_mode = 'iid'
    elif mode_roll < ds_config.prob_iid + ds_config.prob_sliding:
        sample_mode = 'sliding_window'
    else:
        sample_mode = 'mixed'
    
    # Sample number of samples
    n_samples = rng.integers(50, ds_config.max_samples + 1)
    
    # Sample target offset (favor small)
    offsets = list(range(-5, ds_config.max_target_offset + 1))
    offset_probs = np.array([1.0 / (1.0 + abs(k) ** 1.5) for k in offsets])
    offset_probs /= offset_probs.sum()
    target_offset = int(rng.choice(offsets, p=offset_probs))
    
    # Create generator with sequence length = t_subseq + small margin
    # Target is at t_subseq + target_offset (can be negative)
    T = t_subseq + abs(target_offset) + 5
    nn_config_copy = RandomNNConfig(
        memory_dim_range=nn_config.memory_dim_range,
        memory_init=nn_config.memory_init,
        stochastic_input_dim_range=nn_config.stochastic_input_dim_range,
        n_time_transforms_range=nn_config.n_time_transforms_range,
        n_hidden_layers_range=nn_config.n_hidden_layers_range,
        n_nodes_per_layer_range=nn_config.n_nodes_per_layer_range,
        activation_choices=nn_config.activation_choices,
        weight_init_choices=nn_config.weight_init_choices,
        weight_scale_range=nn_config.weight_scale_range,
        bias_std_range=nn_config.bias_std_range,
        node_noise_prob_range=nn_config.node_noise_prob_range,
        node_noise_std_range=nn_config.node_noise_std_range,
        noise_dist_choices=nn_config.noise_dist_choices,
        per_layer_activation=nn_config.per_layer_activation,
        quantization_node_prob=nn_config.quantization_node_prob,
        quantization_n_classes_range=nn_config.quantization_n_classes_range,
        seq_length=T,
    )
    
    # Generate sequences
    if sample_mode == 'iid':
        n_sequences = n_samples
    elif sample_mode == 'sliding_window':
        n_sequences = 1
        stride = max(1, t_subseq // 4)
        T = t_subseq + stride * n_samples + abs(target_offset) + 10
        T = min(T, 5000)
        nn_config_copy = RandomNNConfig(
            memory_dim_range=nn_config.memory_dim_range,
            memory_init=nn_config.memory_init,
            stochastic_input_dim_range=nn_config.stochastic_input_dim_range,
            n_time_transforms_range=nn_config.n_time_transforms_range,
            n_hidden_layers_range=nn_config.n_hidden_layers_range,
            n_nodes_per_layer_range=nn_config.n_nodes_per_layer_range,
            activation_choices=nn_config.activation_choices,
            weight_init_choices=nn_config.weight_init_choices,
            weight_scale_range=nn_config.weight_scale_range,
            bias_std_range=nn_config.bias_std_range,
            node_noise_prob_range=nn_config.node_noise_prob_range,
            node_noise_std_range=nn_config.node_noise_std_range,
            noise_dist_choices=nn_config.noise_dist_choices,
            per_layer_activation=nn_config.per_layer_activation,
            quantization_node_prob=nn_config.quantization_node_prob,
            quantization_n_classes_range=nn_config.quantization_n_classes_range,
            seq_length=T,
        )
    else:  # mixed
        n_sequences = max(2, n_samples // 10)
    
    generator = RandomNNGenerator(nn_config_copy, seed=seed)
    cfg = generator.sampled_config
    
    # Propagate
    _, all_layers = generator.propagate(n_sequences)
    
    # Select target node - ANY node, apply quantization later
    # No restriction on layers
    n_layers = len(all_layers) - 1  # exclude input layer
    target_candidates = []
    
    for layer_idx in range(n_layers):
        layer_data = all_layers[layer_idx + 1]  # skip input
        n_nodes = layer_data.shape[1]
        
        for node_idx in range(n_nodes):
            # Add all nodes as candidates
            target_candidates.append((layer_idx, node_idx))
    
    # Pick random target (uniform across all nodes)
    idx = rng.integers(0, len(target_candidates))
    target_layer, target_node = target_candidates[idx]
    n_classes = rng.integers(2, 8)  # Apply quantization after
    use_quantization = False  # Always discretize afterward
    
    # Select feature nodes (exclude target, prefer close layers, avoid dead neurons)
    feature_candidates = []
    feature_variances = []
    
    for layer_idx in range(n_layers):
        layer_data = all_layers[layer_idx + 1]  # (n_seq, n_nodes, T)
        n_nodes = layer_data.shape[1]
        
        for node_idx in range(n_nodes):
            if layer_idx == target_layer and node_idx == target_node:
                continue
            
            # Compute variance across samples AND time
            node_values = layer_data[:, node_idx, :]  # (n_seq, T)
            
            # Variance across samples (for each timestep, then mean)
            var_across_samples = np.mean(np.var(node_values, axis=0))
            # Variance across time (for each sample, then mean)  
            var_across_time = np.mean(np.var(node_values, axis=1))
            # Combined variance score
            total_var = var_across_samples + var_across_time
            
            feature_candidates.append((layer_idx, node_idx))
            feature_variances.append(total_var)
    
    feature_variances = np.array(feature_variances)
    
    # Filter out dead neurons (very low variance)
    # Use both percentile AND absolute threshold
    percentile_threshold = np.percentile(feature_variances, 50)  # Bottom 50% are suspect
    absolute_threshold = 0.01  # Minimum absolute variance
    alive_mask = feature_variances > max(percentile_threshold, absolute_threshold)
    
    if alive_mask.sum() >= n_features:
        # Use only alive neurons, weight by variance only
        alive_candidates = [c for c, alive in zip(feature_candidates, alive_mask) if alive]
        alive_variances = feature_variances[alive_mask]
        
        # Simple: just weight by variance (no layer preference)
        feat_probs = alive_variances / (alive_variances.sum() + 1e-8)
        
        n_features = min(n_features, len(alive_candidates))
        selected_idxs = rng.choice(len(alive_candidates), size=n_features, replace=False, p=feat_probs)
        selected_features = [alive_candidates[i] for i in selected_idxs]
    else:
        # Fallback: uniform random
        n_features = min(n_features, len(feature_candidates))
        selected_idxs = rng.choice(len(feature_candidates), size=n_features, replace=False)
        selected_features = [feature_candidates[i] for i in selected_idxs]
    
    # Extract samples
    X_list = []
    y_list = []
    
    if sample_mode == 'iid':
        for seq_idx in range(min(n_sequences, n_samples)):
            # Always start from 0 to preserve the full temporal structure
            # This matches how random_all shows sequences (from t=0)
            start = 0
            
            features = []
            for layer_idx, node_idx in selected_features:
                layer_data = all_layers[layer_idx + 1]
                values = layer_data[seq_idx, node_idx, start:start + t_subseq]
                features.append(values)
            X_list.append(np.stack(features, axis=0))
            
            target_t = start + t_subseq + target_offset
            target_t = max(0, min(target_t, T - 1))
            target_val = all_layers[target_layer + 1][seq_idx, target_node, target_t]
            y_list.append(target_val)
    
    elif sample_mode == 'sliding_window':
        stride = max(1, t_subseq // 4)
        seq_idx = 0
        
        for sample_idx in range(n_samples):
            start = sample_idx * stride
            if start + t_subseq + abs(target_offset) >= T:
                break
            
            features = []
            for layer_idx, node_idx in selected_features:
                layer_data = all_layers[layer_idx + 1]
                values = layer_data[seq_idx, node_idx, start:start + t_subseq]
                features.append(values)
            X_list.append(np.stack(features, axis=0))
            
            target_t = start + t_subseq + target_offset
            target_t = max(0, min(target_t, T - 1))
            target_val = all_layers[target_layer + 1][seq_idx, target_node, target_t]
            y_list.append(target_val)
    
    else:  # mixed
        samples_per_seq = n_samples // n_sequences
        sequence_ids = []  # Track which sequence each sample comes from
        for seq_idx in range(n_sequences):
            for _ in range(samples_per_seq):
                max_start = T - t_subseq - abs(target_offset) - 1
                start = rng.integers(0, max(1, max_start))
                
                features = []
                for layer_idx, node_idx in selected_features:
                    layer_data = all_layers[layer_idx + 1]
                    values = layer_data[seq_idx, node_idx, start:start + t_subseq]
                    features.append(values)
                X_list.append(np.stack(features, axis=0))
                
                target_t = start + t_subseq + target_offset
                target_t = max(0, min(target_t, T - 1))
                target_val = all_layers[target_layer + 1][seq_idx, target_node, target_t]
                y_list.append(target_val)
                sequence_ids.append(seq_idx)
    
    if len(X_list) == 0:
        raise ValueError("No samples generated")
    
    X = np.stack(X_list, axis=0).astype(np.float32)
    y_raw = np.array(y_list)
    
    # Discretize target into classes
    n_total = len(y_raw)
    sorted_idx = np.argsort(y_raw)
    y = np.zeros(n_total, dtype=np.int32)
    samples_per_class = n_total // n_classes
    for c in range(n_classes):
        start = c * samples_per_class
        end = start + samples_per_class if c < n_classes - 1 else n_total
        y[sorted_idx[start:end]] = c
    
    # Train/test split
    if sample_mode == 'sliding_window':
        # Temporal split for sliding window
        split_idx = int(n_total * ds_config.train_ratio)
        train_idx = np.arange(split_idx)
        test_idx = np.arange(split_idx, n_total)
    elif sample_mode == 'mixed':
        # SEQUENCE-LEVEL split for mixed mode (avoid data leakage!)
        # Split sequences, not samples
        unique_seqs = list(range(n_sequences))
        n_train_seqs = max(1, int(n_sequences * ds_config.train_ratio))
        rng.shuffle(unique_seqs)
        train_seqs = set(unique_seqs[:n_train_seqs])
        test_seqs = set(unique_seqs[n_train_seqs:])
        
        train_idx = np.array([i for i, sid in enumerate(sequence_ids) if sid in train_seqs])
        test_idx = np.array([i for i, sid in enumerate(sequence_ids) if sid in test_seqs])
    else:  # iid - each sample is independent
        perm = rng.permutation(n_total)
        split_idx = int(n_total * ds_config.train_ratio)
        train_idx = perm[:split_idx]
        test_idx = perm[split_idx:]
    
    # Check if dataset is "boring" (features have low variance)
    feature_temporal_vars = []
    for f in range(X.shape[1]):
        # Variance of each feature across time (averaged over samples)
        var_across_time = np.mean(np.var(X[:, f, :], axis=1))
        feature_temporal_vars.append(var_across_time)
    
    avg_temporal_var = np.mean(feature_temporal_vars)
    if avg_temporal_var < 0.01:
        raise ValueError(f"Dataset too boring: avg temporal variance = {avg_temporal_var:.4f}")
    
    return GeneratedDataset(
        X_train=X[train_idx],
        y_train=y[train_idx],
        X_test=X[test_idx],
        y_test=y[test_idx],
        n_classes=n_classes,
        sample_mode=sample_mode,
        target_offset=target_offset
    )


def visualize_dataset(dataset: GeneratedDataset, output_path: str, name: str):
    """Visualize dataset like real dataset visualizations."""
    X = dataset.X_train
    y = dataset.y_train
    n_samples, n_features, t = X.shape
    
    # One sample per class
    samples_to_show = []
    for c in range(dataset.n_classes):
        idxs = np.where(y == c)[0]
        if len(idxs) > 0:
            samples_to_show.append(idxs[0])
    samples_to_show = samples_to_show[:5]
    
    n_rows = len(samples_to_show)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 2.5 * n_rows))
    if n_rows == 1:
        axes = [axes]
    
    total = len(dataset.X_train) + len(dataset.X_test)
    fig.suptitle(
        f"{name}\n"
        f"Shape: {total} samples × {n_features} channels × {t} timesteps | {dataset.n_classes} classes\n"
        f"Mode: {dataset.sample_mode} | Target offset: {dataset.target_offset}",
        fontsize=12, fontweight='bold'
    )
    
    for row, idx in enumerate(samples_to_show):
        ax = axes[row]
        for f in range(n_features):
            ax.plot(X[idx, f, :], label=f"F{f}" if n_features <= 5 else None, linewidth=1)
        ax.set_ylabel("Value")
        ax.text(0.01, 0.95, f"Sample {idx} | Class: {y[idx]+1}",
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        if n_features <= 5:
            ax.legend(loc='upper right', fontsize=8)
        if row == n_rows - 1:
            ax.set_xlabel("Time Step")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 60)
    print("Dataset Generator (based on RandomNNGenerator)")
    print("=" * 60)
    
    output_dir = "/Users/franco/Documents/TabPFN3D/06_generator_experiments/generated_datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    # Use same config as random_all experiment
    # Random inputs, diverse activations, noise, NO discretization
    nn_config = RandomNNConfig(
        # Memory - RANDOM 1-8 dimensions
        memory_dim_range=(1, 8),
        memory_init='uniform',
        
        # Stochastic inputs - none
        stochastic_input_dim_range=(0, 0),
        
        # Time transforms - RANDOM 1-5 (all linear)
        n_time_transforms_range=(1, 5),
        
        # Architecture - shallow networks (2-5 layers) to keep patterns clear
        n_hidden_layers_range=(2, 5),
        n_nodes_per_layer_range=(4, 16),
        
        # Activations - diverse per-node activations (12 types)
        activation_choices=(
            'identity', 'log', 'sigmoid', 'abs', 'sin', 'tanh',
            'rank', 'square', 'power', 'softplus', 'step', 'modulo',
        ),
        
        # Initialization
        weight_init_choices=('xavier_normal',),
        weight_scale_range=(0.9, 1.1),
        bias_std_range=(0.0, 0.1),
        
        # Per-node noise - 5-30% of nodes, higher std
        node_noise_prob_range=(0.05, 0.30),
        node_noise_std_range=(0.01, 0.1),
        noise_dist_choices=('normal',),
        
        per_layer_activation=True,
        
        # NO quantization nodes - will apply quantization to target later
        quantization_node_prob=0.0,
        quantization_n_classes_range=(2, 8),
        
        seq_length=200,  # Will be overwritten
    )
    
    ds_config = DatasetConfig()
    
    n_datasets = 16
    print(f"\nGenerating {n_datasets} datasets...\n")
    
    for i in range(n_datasets):
        # Try multiple seeds until we get an interesting dataset
        for attempt in range(10):
            try:
                seed = 42 + i * 100 + attempt
                dataset = generate_dataset(nn_config, ds_config, seed=seed)
                
                n_total = len(dataset.X_train) + len(dataset.X_test)
                n_feat = dataset.X_train.shape[1]
                t = dataset.X_train.shape[2]
                
                print(f"Dataset {i}: {n_total} × {n_feat} × {t} | {dataset.n_classes} classes | {dataset.sample_mode}")
                
                visualize_dataset(dataset, f"{output_dir}/dataset{i:02d}.png", f"Synthetic Dataset {i}")
                break  # Success, move to next dataset
                
            except ValueError as e:
                if "boring" in str(e).lower():
                    continue  # Try again with different seed
                else:
                    print(f"Dataset {i}: Error - {e}")
                    break
            except Exception as e:
                print(f"Dataset {i}: Error - {e}")
                break
        else:
            print(f"Dataset {i}: Failed after 10 attempts")
    
    print(f"\nDone! Saved to: {output_dir}")


if __name__ == "__main__":
    main()
