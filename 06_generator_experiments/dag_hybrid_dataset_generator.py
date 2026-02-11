"""
Dataset Generator using Hybrid DAG Generator.

Features: nodes from DAG in a subsequence window
Target: a value from z_features (fixed per sample -> classification label)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
from typing import List, Tuple

from dag_hybrid_generator import HybridDAGGenerator, HybridDAGConfig


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    # Sample modes
    prob_iid: float = 1.0
    prob_sliding: float = 0.0
    prob_mixed: float = 0.0
    
    # Constraints
    max_samples: int = 1000
    max_features: int = 12
    max_timesteps: int = 1000
    max_m_times_t: int = 500
    
    # Train/test split
    train_ratio: float = 0.8
    
    # Number of classes
    n_classes_range: Tuple[int, int] = (2, 8)


@dataclass  
class GeneratedDataset:
    """A generated dataset."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    n_classes: int
    sample_mode: str
    target_z_feature_idx: int  # Which z_feature is the target


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
    dag_config: HybridDAGConfig,
    ds_config: DatasetConfig,
    seed: int = 42
) -> GeneratedDataset:
    """
    Generate a classification dataset using HybridDAGGenerator.
    
    Features: DAG nodes in a time window
    Target: discretized z_feature (fixed per sample)
    """
    rng = np.random.default_rng(seed)
    
    # Sample dataset parameters
    n_features = sample_n_features(rng)
    
    # Sample timesteps (enforce m*t <= max_m_times_t)
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
    
    # Sample number of classes
    n_classes = rng.integers(ds_config.n_classes_range[0], ds_config.n_classes_range[1] + 1)
    
    # Create generator with appropriate sequence length
    T = t_subseq + 10  # Small margin
    
    dag_config_copy = HybridDAGConfig(
        z_dim_range=dag_config.z_dim_range,
        z_std_range=dag_config.z_std_range,
        z_nn_layers_range=dag_config.z_nn_layers_range,
        z_nn_nodes_range=dag_config.z_nn_nodes_range,
        t_nn_layers_range=dag_config.t_nn_layers_range,
        t_nn_nodes_range=dag_config.t_nn_nodes_range,
        noise_std_range=dag_config.noise_std_range,
        n_roots_range=dag_config.n_roots_range,
        n_dag_nodes_range=dag_config.n_dag_nodes_range,
        redirection_alpha=dag_config.redirection_alpha,
        redirection_beta=dag_config.redirection_beta,
        activation_choices=dag_config.activation_choices,
        weight_scale_range=dag_config.weight_scale_range,
        bias_std_range=dag_config.bias_std_range,
        dag_noise_prob_range=dag_config.dag_noise_prob_range,
        dag_noise_std_range=dag_config.dag_noise_std_range,
        seq_length=T,
    )
    
    # Generate sequences
    if sample_mode == 'iid':
        n_sequences = n_samples
    elif sample_mode == 'sliding_window':
        n_sequences = 1
        stride = max(1, t_subseq // 4)
        T = t_subseq + stride * n_samples + 10
        T = min(T, 5000)
        dag_config_copy.seq_length = T
    else:  # mixed
        n_sequences = max(2, n_samples // 10)
    
    generator = HybridDAGGenerator(dag_config_copy, seed=seed)
    cfg = generator.sampled_config
    
    # Propagate
    dag_outputs, intermediates = generator.propagate(n_sequences)
    # dag_outputs: (n_sequences, n_dag_nodes, T)
    # intermediates['z_features']: (n_sequences, z_features_dim)
    
    z_features = intermediates['z_features']  # (n_sequences, z_features_dim)
    
    # Select target: one dimension of z_features
    target_z_idx = rng.integers(0, generator.z_features_dim)
    
    # Select feature nodes from DAG (exclude roots, prefer high variance)
    n_dag_nodes = cfg.n_dag_nodes
    feature_candidates = []
    feature_variances = []
    
    for node_id in range(n_dag_nodes):
        # Skip root nodes (they don't fully depend on the DAG dynamics)
        if generator.dag_is_root[node_id]:
            continue
        
        node_values = dag_outputs[:, node_id, :]  # (n_sequences, T)
        
        # Variance across samples and time
        var_across_samples = np.mean(np.var(node_values, axis=0))
        var_across_time = np.mean(np.var(node_values, axis=1))
        total_var = var_across_samples + var_across_time
        
        feature_candidates.append(node_id)
        feature_variances.append(total_var)
    
    feature_variances = np.array(feature_variances)
    
    # Filter out dead neurons
    percentile_threshold = np.percentile(feature_variances, 50)
    absolute_threshold = 0.01
    alive_mask = feature_variances > max(percentile_threshold, absolute_threshold)
    
    if alive_mask.sum() >= n_features:
        alive_candidates = [c for c, alive in zip(feature_candidates, alive_mask) if alive]
        alive_variances = feature_variances[alive_mask]
        
        feat_probs = alive_variances / (alive_variances.sum() + 1e-8)
        feat_probs = feat_probs / feat_probs.sum()
        
        n_features = min(n_features, len(alive_candidates))
        selected_idxs = rng.choice(len(alive_candidates), size=n_features, replace=False, p=feat_probs)
        selected_features = [alive_candidates[i] for i in selected_idxs]
    else:
        n_features = min(n_features, len(feature_candidates))
        selected_idxs = rng.choice(len(feature_candidates), size=n_features, replace=False)
        selected_features = [feature_candidates[i] for i in selected_idxs]
    
    # Extract samples
    X_list = []
    y_raw_list = []
    
    if sample_mode == 'iid':
        for seq_idx in range(min(n_sequences, n_samples)):
            start = 0
            
            features = []
            for node_id in selected_features:
                values = dag_outputs[seq_idx, node_id, start:start + t_subseq]
                features.append(values)
            X_list.append(np.stack(features, axis=0))
            
            # Target: z_feature value (fixed per sample)
            y_raw_list.append(z_features[seq_idx, target_z_idx])
    
    elif sample_mode == 'sliding_window':
        stride = max(1, t_subseq // 4)
        seq_idx = 0
        
        for sample_idx in range(n_samples):
            start = sample_idx * stride
            if start + t_subseq >= T:
                break
            
            features = []
            for node_id in selected_features:
                values = dag_outputs[seq_idx, node_id, start:start + t_subseq]
                features.append(values)
            X_list.append(np.stack(features, axis=0))
            
            # Same z_features for all windows from same sequence
            y_raw_list.append(z_features[seq_idx, target_z_idx])
    
    else:  # mixed
        samples_per_seq = n_samples // n_sequences
        sequence_ids = []
        
        for seq_idx in range(n_sequences):
            for _ in range(samples_per_seq):
                max_start = T - t_subseq - 1
                start = rng.integers(0, max(1, max_start))
                
                features = []
                for node_id in selected_features:
                    values = dag_outputs[seq_idx, node_id, start:start + t_subseq]
                    features.append(values)
                X_list.append(np.stack(features, axis=0))
                
                y_raw_list.append(z_features[seq_idx, target_z_idx])
                sequence_ids.append(seq_idx)
    
    if len(X_list) == 0:
        raise ValueError("No samples generated")
    
    X = np.stack(X_list, axis=0).astype(np.float32)
    y_raw = np.array(y_raw_list)
    
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
        split_idx = int(n_total * ds_config.train_ratio)
        train_idx = np.arange(split_idx)
        test_idx = np.arange(split_idx, n_total)
    elif sample_mode == 'mixed':
        unique_seqs = list(range(n_sequences))
        n_train_seqs = max(1, int(n_sequences * ds_config.train_ratio))
        rng.shuffle(unique_seqs)
        train_seqs = set(unique_seqs[:n_train_seqs])
        test_seqs = set(unique_seqs[n_train_seqs:])
        
        train_idx = np.array([i for i, sid in enumerate(sequence_ids) if sid in train_seqs])
        test_idx = np.array([i for i, sid in enumerate(sequence_ids) if sid in test_seqs])
    else:  # iid
        perm = rng.permutation(n_total)
        split_idx = int(n_total * ds_config.train_ratio)
        train_idx = perm[:split_idx]
        test_idx = perm[split_idx:]
    
    # Check if dataset is boring
    feature_temporal_vars = []
    for f in range(X.shape[1]):
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
        target_z_feature_idx=target_z_idx
    )


def visualize_dataset(dataset: GeneratedDataset, output_path: str, name: str):
    """Visualize dataset."""
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
        f"Mode: {dataset.sample_mode} | Target: z_feature[{dataset.target_z_feature_idx}]",
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
    print("Dataset Generator (based on HybridDAGGenerator)")
    print("=" * 60)
    
    output_dir = "/Users/franco/Documents/TabPFN3D/06_generator_experiments/dag_hybrid_datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    dag_config = HybridDAGConfig(
        # z (memory)
        z_dim_range=(2, 8),
        z_std_range=(0.01, 1.0),
        z_nn_layers_range=(1, 3),
        z_nn_nodes_range=(4, 12),
        
        # t (time)
        t_nn_layers_range=(1, 3),
        t_nn_nodes_range=(4, 12),
        
        # noise
        noise_std_range=(0.001, 0.1),
        
        # roots and DAG
        n_roots_range=(3, 10),
        n_dag_nodes_range=(20, 80),
        redirection_alpha=2.0,
        redirection_beta=5.0,
        
        # activations
        activation_choices=(
            'identity', 'relu', 'tanh', 'sigmoid', 'sin',
            'leaky_relu', 'elu', 'softplus', 'abs', 'square'
        ),
        
        # weights
        weight_scale_range=(0.8, 1.2),
        bias_std_range=(0.0, 0.1),
        
        # DAG noise
        dag_noise_prob_range=(0.01, 0.3),
        dag_noise_std_range=(0.001, 0.05),
        
        # sequence
        seq_length=200,
    )
    
    ds_config = DatasetConfig()
    
    n_datasets = 16
    print(f"\nGenerating {n_datasets} datasets...\n")
    
    for i in range(n_datasets):
        for attempt in range(10):
            try:
                seed = 42 + i * 100 + attempt
                dataset = generate_dataset(dag_config, ds_config, seed=seed)
                
                n_total = len(dataset.X_train) + len(dataset.X_test)
                n_feat = dataset.X_train.shape[1]
                t = dataset.X_train.shape[2]
                
                print(f"Dataset {i}: {n_total} × {n_feat} × {t} | {dataset.n_classes} classes | {dataset.sample_mode} | target=z[{dataset.target_z_feature_idx}]")
                
                visualize_dataset(dataset, f"{output_dir}/dataset{i:02d}.png", f"Hybrid DAG Dataset {i}")
                break
                
            except ValueError as e:
                if "boring" in str(e).lower():
                    continue
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
