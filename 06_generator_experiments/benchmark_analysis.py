"""
Deep Analysis Benchmark - Save ALL metadata and run sensitivity experiments.
"""

import numpy as np
import os
import json
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics import accuracy_score, roc_auc_score
from tabpfn import TabPFNClassifier
import matplotlib.pyplot as plt

from random_nn_generator import RandomNNGenerator, RandomNNConfig


@dataclass
class FullMetadata:
    """Complete metadata for a generated dataset."""
    # Seed info
    master_seed: int = 0
    network_seed: int = 0  # Seed for network structure + weights
    feature_selection_seed: int = 0  # Seed for feature selection
    
    # Network config (sampled)
    memory_dim: int = 0
    memory_init: str = ''
    n_time_transforms: int = 0
    time_transform_types: List[str] = field(default_factory=list)
    n_hidden_layers: int = 0
    nodes_per_layer: List[int] = field(default_factory=list)
    activations_per_layer: List[str] = field(default_factory=list)
    weight_init: str = ''
    weight_scale: float = 0.0
    bias_std: float = 0.0
    
    # Noise config
    node_noise_prob: float = 0.0
    node_noise_std: float = 0.0
    noise_distribution: str = ''
    n_noisy_nodes_per_layer: List[int] = field(default_factory=list)
    
    # Quantization config
    quantization_node_prob: float = 0.0
    n_quantization_nodes_per_layer: List[int] = field(default_factory=list)
    quantization_classes_per_node: Dict[str, int] = field(default_factory=dict)
    
    # Target selection
    target_layer: int = 0
    target_node: int = 0
    target_is_quantization: bool = False
    target_n_classes: int = 0
    target_offset: int = 0
    
    # Feature selection
    n_features: int = 0
    feature_positions: List[Tuple[int, int]] = field(default_factory=list)  # (layer, node) pairs
    feature_variances: List[float] = field(default_factory=list)
    feature_distances_to_target: List[int] = field(default_factory=list)
    
    # Dataset config
    sample_mode: str = ''
    T_total: int = 0
    t_subsequence: int = 0
    n_sequences_generated: int = 0
    n_samples_total: int = 0
    n_train: int = 0
    n_test: int = 0
    
    # Variance stats
    avg_temporal_variance: float = 0.0
    avg_sample_variance: float = 0.0
    per_feature_temporal_variance: List[float] = field(default_factory=list)
    
    # Results
    accuracy: float = 0.0
    auc: Optional[float] = None


def generate_with_metadata(
    nn_config: RandomNNConfig,
    master_seed: int,
    network_seed: int = None,
    feature_seed: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, FullMetadata]:
    """
    Generate dataset and return full metadata.
    """
    if network_seed is None:
        network_seed = master_seed
    if feature_seed is None:
        feature_seed = master_seed + 1000
    
    rng_master = np.random.default_rng(master_seed)
    rng_feature = np.random.default_rng(feature_seed)
    
    meta = FullMetadata(
        master_seed=master_seed,
        network_seed=network_seed,
        feature_selection_seed=feature_seed,
    )
    
    # Sample dataset parameters
    r = rng_master.random()
    if r < 0.60:
        n_features = 1
    elif r < 0.80:
        n_features = 2
    elif r < 0.90:
        n_features = 3
    else:
        n_features = rng_master.integers(4, 9)
    
    # Sample timesteps
    max_t = min(1000, 500 // n_features)
    t_subseq = int(rng_master.integers(20, max(21, max_t)))
    
    # Sample mode
    mode_roll = rng_master.random()
    if mode_roll < 0.60:
        sample_mode = 'iid'
    elif mode_roll < 0.85:
        sample_mode = 'sliding_window'
    else:
        sample_mode = 'mixed'
    
    n_samples = int(rng_master.integers(50, 1001))
    
    # Target offset
    offsets = list(range(-5, 16))
    offset_probs = np.array([1.0 / (1.0 + abs(k) ** 1.5) for k in offsets])
    offset_probs /= offset_probs.sum()
    target_offset = int(rng_master.choice(offsets, p=offset_probs))
    
    # Compute T
    T = t_subseq + abs(target_offset) + 20
    
    if sample_mode == 'sliding_window':
        stride = max(1, t_subseq // 4)
        T = t_subseq + stride * n_samples + abs(target_offset) + 10
        T = min(T, 5000)
        n_sequences = 1
    elif sample_mode == 'mixed':
        n_sequences = max(2, n_samples // 10)
    else:
        n_sequences = n_samples
    
    meta.sample_mode = sample_mode
    meta.T_total = T
    meta.t_subsequence = t_subseq
    meta.n_sequences_generated = n_sequences
    meta.target_offset = target_offset
    
    # Create generator with specific seed for network
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
    
    generator = RandomNNGenerator(nn_config_copy, seed=network_seed)
    cfg = generator.sampled_config
    
    # Extract network metadata
    meta.memory_dim = cfg.memory_dim
    meta.memory_init = nn_config.memory_init
    meta.n_time_transforms = len(cfg.time_transforms)
    meta.time_transform_types = [t['name'] for t in cfg.time_transforms]
    meta.n_hidden_layers = cfg.n_hidden_layers
    meta.nodes_per_layer = list(cfg.nodes_per_layer)
    meta.activations_per_layer = list(cfg.activations)
    meta.weight_init = cfg.weight_init
    meta.weight_scale = float(cfg.weight_scale)
    meta.bias_std = float(cfg.bias_std)
    meta.node_noise_prob = float(cfg.node_noise_prob)
    meta.quantization_node_prob = nn_config.quantization_node_prob
    
    # Count noisy and quantization nodes per layer, extract noise stats
    noisy_per_layer = []
    quant_per_layer = []
    quant_classes = {}
    noise_scales = []
    noise_dists = []
    
    for layer_idx, layer_noise in enumerate(cfg.node_noise):
        noisy_count = 0
        for n in layer_noise:
            if n.has_noise:
                noisy_count += 1
                noise_scales.append(n.scale)
                noise_dists.append(n.distribution)
        noisy_per_layer.append(noisy_count)
    
    for layer_idx, layer_quant in enumerate(cfg.node_quantization):
        quant_count = 0
        for node_idx, q in enumerate(layer_quant):
            if q.is_quantization:
                quant_count += 1
                quant_classes[f"L{layer_idx}_N{node_idx}"] = q.n_classes
        quant_per_layer.append(quant_count)
    
    meta.n_noisy_nodes_per_layer = noisy_per_layer
    meta.n_quantization_nodes_per_layer = quant_per_layer
    meta.quantization_classes_per_node = quant_classes
    meta.node_noise_std = float(np.mean(noise_scales)) if noise_scales else 0.0
    meta.noise_distribution = noise_dists[0] if noise_dists else 'none'
    
    # Propagate
    _, all_layers = generator.propagate(n_sequences)
    
    # Select target (prefer quantization in later layers)
    n_layers = len(all_layers) - 1
    target_candidates = []
    
    for layer_idx in range(n_layers):
        layer_data = all_layers[layer_idx + 1]
        n_nodes = layer_data.shape[1]
        
        for node_idx in range(n_nodes):
            quant_cfg = cfg.node_quantization[layer_idx][node_idx]
            if quant_cfg.is_quantization:
                target_candidates.append((layer_idx, node_idx, quant_cfg.n_classes))
    
    if not target_candidates:
        last_layer_idx = n_layers - 1
        n_nodes = all_layers[last_layer_idx + 1].shape[1]
        target_layer = last_layer_idx
        target_node = int(rng_feature.integers(0, n_nodes))
        n_classes = int(rng_feature.integers(2, 6))
        target_is_quant = False
    else:
        weights = np.array([layer + 1 for layer, _, _ in target_candidates], dtype=float)
        weights /= weights.sum()
        idx = rng_feature.choice(len(target_candidates), p=weights)
        target_layer, target_node, n_classes = target_candidates[idx]
        target_is_quant = True
    
    meta.target_layer = target_layer
    meta.target_node = target_node
    meta.target_is_quantization = target_is_quant
    meta.target_n_classes = n_classes
    
    # Select features (using feature_seed)
    feature_candidates = []
    feature_variances = []
    
    for layer_idx in range(n_layers):
        layer_data = all_layers[layer_idx + 1]
        n_nodes = layer_data.shape[1]
        
        for node_idx in range(n_nodes):
            if layer_idx == target_layer and node_idx == target_node:
                continue
            
            node_values = layer_data[:, node_idx, :]
            var_across_samples = np.mean(np.var(node_values, axis=0))
            var_across_time = np.mean(np.var(node_values, axis=1))
            total_var = var_across_samples + var_across_time
            
            feature_candidates.append((layer_idx, node_idx))
            feature_variances.append(total_var)
    
    feature_variances = np.array(feature_variances)
    
    # Filter dead neurons
    percentile_threshold = np.percentile(feature_variances, 50)
    absolute_threshold = 0.01
    alive_mask = feature_variances > max(percentile_threshold, absolute_threshold)
    
    if alive_mask.sum() >= n_features:
        alive_candidates = [c for c, alive in zip(feature_candidates, alive_mask) if alive]
        alive_variances = feature_variances[alive_mask]
        
        distances = np.array([abs(layer - target_layer) for layer, _ in alive_candidates])
        dist_weights = 1.0 / (1.0 + distances ** 1.5)
        var_weights = alive_variances / (alive_variances.max() + 1e-8)
        
        feat_probs = dist_weights * (0.2 + 0.8 * var_weights ** 0.5)
        feat_probs /= feat_probs.sum()
        
        n_features = min(n_features, len(alive_candidates))
        selected_idxs = rng_feature.choice(len(alive_candidates), size=n_features, replace=False, p=feat_probs)
        selected_features = [alive_candidates[i] for i in selected_idxs]
        selected_variances = [float(alive_variances[i]) for i in selected_idxs]
    else:
        distances = np.array([abs(layer - target_layer) for layer, _ in feature_candidates])
        feat_probs = 1.0 / (1.0 + distances ** 1.5)
        feat_probs /= feat_probs.sum()
        
        n_features = min(n_features, len(feature_candidates))
        selected_idxs = rng_feature.choice(len(feature_candidates), size=n_features, replace=False, p=feat_probs)
        selected_features = [feature_candidates[i] for i in selected_idxs]
        selected_variances = [float(feature_variances[i]) for i in selected_idxs]
    
    meta.n_features = n_features
    meta.feature_positions = [(int(l), int(n)) for l, n in selected_features]
    meta.feature_variances = selected_variances
    meta.feature_distances_to_target = [abs(l - target_layer) for l, _ in selected_features]
    
    # Extract samples
    X_list = []
    y_list = []
    
    if sample_mode == 'iid':
        for seq_idx in range(min(n_sequences, n_samples)):
            max_start = T - t_subseq - abs(target_offset) - 1
            start = int(rng_master.integers(0, max(1, max_start)))
            
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
                start = int(rng_master.integers(0, max(1, max_start)))
                
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
    
    # Discretize target
    n_total = len(y_raw)
    sorted_idx = np.argsort(y_raw)
    y = np.zeros(n_total, dtype=np.int32)
    samples_per_class = n_total // n_classes
    for c in range(n_classes):
        start_c = c * samples_per_class
        end_c = start_c + samples_per_class if c < n_classes - 1 else n_total
        y[sorted_idx[start_c:end_c]] = c
    
    # Variance stats
    temporal_vars = []
    sample_vars = []
    for f in range(X.shape[1]):
        tv = float(np.mean(np.var(X[:, f, :], axis=1)))
        sv = float(np.mean(np.var(X[:, f, :], axis=0)))
        temporal_vars.append(tv)
        sample_vars.append(sv)
    
    meta.avg_temporal_variance = float(np.mean(temporal_vars))
    meta.avg_sample_variance = float(np.mean(sample_vars))
    meta.per_feature_temporal_variance = temporal_vars
    
    if meta.avg_temporal_variance < 0.01:
        raise ValueError(f"Dataset too boring: avg temporal variance = {meta.avg_temporal_variance:.4f}")
    
    # Train/test split
    if sample_mode == 'sliding_window':
        # Temporal split for sliding window
        split_idx = int(n_total * 0.7)
        train_idx = np.arange(split_idx)
        test_idx = np.arange(split_idx, n_total)
    elif sample_mode == 'mixed':
        # SEQUENCE-LEVEL split for mixed mode (avoid data leakage!)
        # Split sequences, not samples
        unique_seqs = list(range(n_sequences))
        n_train_seqs = max(1, int(n_sequences * 0.7))
        rng_master.shuffle(unique_seqs)
        train_seqs = set(unique_seqs[:n_train_seqs])
        test_seqs = set(unique_seqs[n_train_seqs:])
        
        train_idx = [i for i, sid in enumerate(sequence_ids) if sid in train_seqs]
        test_idx = [i for i, sid in enumerate(sequence_ids) if sid in test_seqs]
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
    else:  # iid - each sample is independent
        perm = rng_master.permutation(n_total)
        split_idx = int(n_total * 0.7)
        train_idx = perm[:split_idx]
        test_idx = perm[split_idx:]
    
    meta.n_samples_total = n_total
    meta.n_train = len(train_idx)
    meta.n_test = len(test_idx)
    
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    
    return X_train, y_train, X_test, y_test, meta


def evaluate_with_tabpfn(X_train, y_train, X_test, y_test, n_classes):
    """Evaluate with TabPFN and return accuracy and AUC."""
    # Flatten
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    if X_train_flat.shape[1] > 500:
        return None, None
    
    tabpfn = TabPFNClassifier(n_estimators=4, ignore_pretraining_limits=True)
    tabpfn.fit(X_train_flat, y_train)
    
    y_pred = tabpfn.predict(X_test_flat)
    y_proba = tabpfn.predict_proba(X_test_flat)
    
    acc = accuracy_score(y_test, y_pred)
    
    if n_classes == 2:
        auc = roc_auc_score(y_test, y_proba[:, 1])
    else:
        try:
            auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
        except:
            auc = None
    
    return acc, auc


def visualize_with_metadata(X_train, y_train, X_test, y_test, meta: FullMetadata, filepath: str):
    """Visualize dataset with full metadata."""
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    
    n_samples, n_features, T = X.shape
    n_classes = meta.target_n_classes
    
    # One sample per class
    samples_to_show = []
    for c in range(n_classes):
        idxs = np.where(y == c)[0]
        if len(idxs) > 0:
            samples_to_show.append(idxs[0])
    samples_to_show = samples_to_show[:5]
    
    n_rows = len(samples_to_show)
    fig = plt.figure(figsize=(16, 3 * n_rows + 4))
    
    # Title
    auc_str = f"{meta.auc:.3f}" if meta.auc else "N/A"
    fig.suptitle(
        f"Dataset | Acc: {meta.accuracy:.3f} | AUC: {auc_str}\n"
        f"Shape: {meta.n_samples_total} × {meta.n_features} × {meta.t_subsequence} | "
        f"{meta.target_n_classes} classes | Mode: {meta.sample_mode}",
        fontsize=12, fontweight='bold'
    )
    
    # Main plots
    for row, idx in enumerate(samples_to_show):
        ax = fig.add_subplot(n_rows + 1, 1, row + 1)
        for f in range(n_features):
            ax.plot(X[idx, f, :], label=f"F{f} (L{meta.feature_positions[f][0]},N{meta.feature_positions[f][1]})", linewidth=1)
        ax.set_ylabel("Value")
        ax.text(0.01, 0.95, f"Sample {idx} | Class: {y[idx]}",
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.legend(loc='upper right', fontsize=7)
        if row == n_rows - 1:
            ax.set_xlabel("Time Step")
    
    # Metadata text box
    ax_text = fig.add_subplot(n_rows + 1, 1, n_rows + 1)
    ax_text.axis('off')
    
    meta_text = f"""NETWORK CONFIG:
  Memory: dim={meta.memory_dim}, init={meta.memory_init}
  Time transforms: n={meta.n_time_transforms}, types={meta.time_transform_types[:5]}{'...' if len(meta.time_transform_types) > 5 else ''}
  Layers: {meta.n_hidden_layers}, nodes={meta.nodes_per_layer}
  Activations: {meta.activations_per_layer}
  Weights: init={meta.weight_init}, scale={meta.weight_scale:.3f}, bias_std={meta.bias_std:.4f}
  
NOISE CONFIG:
  prob={meta.node_noise_prob:.3f}, std={meta.node_noise_std:.4f}, dist={meta.noise_distribution}
  noisy nodes/layer: {meta.n_noisy_nodes_per_layer}

QUANTIZATION:
  prob={meta.quantization_node_prob:.2f}
  quant nodes/layer: {meta.n_quantization_nodes_per_layer}
  
TARGET: layer={meta.target_layer}, node={meta.target_node}, is_quant={meta.target_is_quantization}, offset={meta.target_offset}
FEATURES: positions={meta.feature_positions}, variances={[f'{v:.4f}' for v in meta.feature_variances]}
VARIANCE: temporal={meta.avg_temporal_variance:.4f}, sample={meta.avg_sample_variance:.4f}
SEEDS: master={meta.master_seed}, network={meta.network_seed}, feature={meta.feature_selection_seed}"""
    
    ax_text.text(0.02, 0.95, meta_text, transform=ax_text.transAxes, fontsize=8,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=120, bbox_inches='tight')
    plt.close()


def run_full_analysis(n_datasets: int = 64, output_dir: str = None):
    """Run full analysis with metadata."""
    if output_dir is None:
        output_dir = "/Users/franco/Documents/TabPFN3D/06_generator_experiments/analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    nn_config = RandomNNConfig(
        memory_dim_range=(1, 64),
        memory_init='uniform',
        stochastic_input_dim_range=(0, 0),
        n_time_transforms_range=(0, 64),
        n_hidden_layers_range=(2, 12),
        n_nodes_per_layer_range=(4, 96),
        activation_choices=('softplus', 'tanh', 'elu'),
        weight_init_choices=('xavier_normal',),
        weight_scale_range=(0.9, 1.1),
        bias_std_range=(0.0, 0.05),
        node_noise_prob_range=(0.01, 0.9),
        node_noise_std_range=(0.0001, 0.02),
        noise_dist_choices=('normal',),
        per_layer_activation=True,
        quantization_node_prob=0.1,
        quantization_n_classes_range=(2, 10),
        seq_length=100,
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{output_dir}/results_{timestamp}.json"
    
    print("=" * 70)
    print("Deep Analysis Benchmark - Full Metadata")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print(f"Results: {results_file}\n")
    
    all_results = []
    high_auc_configs = []  # Track configs with AUC >= 0.9
    
    for i in range(n_datasets):
        for attempt in range(10):
            try:
                seed = 42 + i * 100 + attempt
                
                X_train, y_train, X_test, y_test, meta = generate_with_metadata(
                    nn_config, master_seed=seed, network_seed=seed, feature_seed=seed + 1000
                )
                
                acc, auc = evaluate_with_tabpfn(X_train, y_train, X_test, y_test, meta.target_n_classes)
                
                if acc is None:
                    continue
                
                meta.accuracy = float(acc)
                meta.auc = float(auc) if auc else None
                
                # Save visualization
                vis_path = f"{output_dir}/dataset{i:02d}.png"
                visualize_with_metadata(X_train, y_train, X_test, y_test, meta, vis_path)
                
                # Create result dict manually - CONVERT ALL TO NATIVE PYTHON TYPES
                result = {
                    'dataset_id': int(i),
                    'seed': int(seed),
                    'accuracy': float(acc),
                    'auc': float(auc) if auc else None,
                    'sample_mode': str(meta.sample_mode),
                    'n_train': int(meta.n_train),
                    'n_test': int(meta.n_test),
                    'n_features': int(meta.n_features),
                    't_subsequence': int(meta.t_subsequence),
                    'T_total': int(meta.T_total),
                    'target_n_classes': int(meta.target_n_classes),
                    'target_layer': int(meta.target_layer),
                    'target_node': int(meta.target_node),
                    'target_offset': int(meta.target_offset),
                    'target_is_quantization': bool(meta.target_is_quantization),
                    'memory_dim': int(meta.memory_dim),
                    'n_time_transforms': int(meta.n_time_transforms),
                    'time_transform_types': [str(t) for t in meta.time_transform_types],
                    'n_hidden_layers': int(meta.n_hidden_layers),
                    'nodes_per_layer': [int(n) for n in meta.nodes_per_layer],
                    'activations_per_layer': [str(a) for a in meta.activations_per_layer],
                    'weight_init': str(meta.weight_init),
                    'weight_scale': float(meta.weight_scale),
                    'bias_std': float(meta.bias_std),
                    'node_noise_prob': float(meta.node_noise_prob),
                    'node_noise_std': float(meta.node_noise_std),
                    'noise_distribution': str(meta.noise_distribution),
                    'n_noisy_nodes_per_layer': [int(n) for n in meta.n_noisy_nodes_per_layer],
                    'n_quantization_nodes_per_layer': [int(n) for n in meta.n_quantization_nodes_per_layer],
                    'feature_positions': [[int(l), int(n)] for l, n in meta.feature_positions],
                    'feature_variances': [float(v) for v in meta.feature_variances],
                    'avg_temporal_variance': float(meta.avg_temporal_variance),
                    'avg_sample_variance': float(meta.avg_sample_variance),
                }
                all_results.append(result)
                
                # Save incrementally
                with open(results_file, 'w') as f:
                    json.dump(all_results, f, indent=2)
                
                auc_str = f"{auc:.3f}" if auc else "N/A"
                marker = " ★★★" if auc and auc >= 0.9 else ""
                print(f"[{i+1:2d}/{n_datasets}] {meta.n_train}×{meta.n_features}×{meta.t_subsequence} | "
                      f"{meta.target_n_classes} cls | {meta.sample_mode:15s} | "
                      f"Acc: {acc:.3f} | AUC: {auc_str}{marker}", flush=True)
                
                # Track high AUC configs
                if auc and auc >= 0.9:
                    high_auc_configs.append({
                        'dataset_id': i,
                        'seed': seed,
                        'meta': meta,
                        'X_train': X_train,
                        'y_train': y_train,
                        'X_test': X_test,
                        'y_test': y_test,
                    })
                
                break
                
            except ValueError as e:
                if "boring" in str(e).lower():
                    continue
                else:
                    print(f"[{i+1:2d}/{n_datasets}] Error: {e}")
                    break
            except Exception as e:
                print(f"[{i+1:2d}/{n_datasets}] Error: {e}")
                break
        else:
            print(f"[{i+1:2d}/{n_datasets}] Failed after 10 attempts")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    accs = [r['accuracy'] for r in all_results]
    aucs = [r['auc'] for r in all_results if r['auc'] is not None]
    
    print(f"Successful: {len(all_results)}/{n_datasets}")
    print(f"Accuracy: mean={np.mean(accs):.3f}, std={np.std(accs):.3f}")
    if aucs:
        print(f"AUC:      mean={np.mean(aucs):.3f}, std={np.std(aucs):.3f}")
    
    print(f"\nHigh AUC datasets (>=0.9): {len(high_auc_configs)}")
    
    # Run sensitivity experiments on high AUC configs
    if high_auc_configs:
        print("\n" + "=" * 70)
        print("SENSITIVITY EXPERIMENTS on High AUC Configs")
        print("=" * 70)
        
        sensitivity_results = []
        
        for cfg in high_auc_configs[:5]:  # Max 5 configs to test
            dataset_id = cfg['dataset_id']
            original_seed = cfg['seed']
            original_meta = cfg['meta']
            
            print(f"\n--- Dataset {dataset_id} (Original AUC: {original_meta.auc:.3f}) ---")
            
            # Experiment 1: Same network, different feature selection
            print("\nExp 1: Same network, different feature selection seeds:")
            exp1_results = []
            for feat_seed_offset in range(5):
                feat_seed = original_seed + 2000 + feat_seed_offset * 100
                try:
                    X_tr, y_tr, X_te, y_te, meta = generate_with_metadata(
                        nn_config, 
                        master_seed=original_seed,
                        network_seed=original_seed,  # Same network
                        feature_seed=feat_seed       # Different feature selection
                    )
                    acc, auc = evaluate_with_tabpfn(X_tr, y_tr, X_te, y_te, meta.target_n_classes)
                    if acc:
                        exp1_results.append({
                            'feat_seed': feat_seed,
                            'acc': acc,
                            'auc': auc,
                            'features': meta.feature_positions
                        })
                        auc_str = f"{auc:.3f}" if auc else "N/A"
                        print(f"  feat_seed={feat_seed}: Acc={acc:.3f}, AUC={auc_str}, features={meta.feature_positions}")
                except Exception as e:
                    print(f"  feat_seed={feat_seed}: Error - {e}")
            
            # Experiment 2: Same structure, different weights/biases
            print("\nExp 2: Different weight initializations (same structure):")
            exp2_results = []
            for weight_seed_offset in range(5):
                weight_seed = original_seed + 5000 + weight_seed_offset * 100
                try:
                    X_tr, y_tr, X_te, y_te, meta = generate_with_metadata(
                        nn_config,
                        master_seed=original_seed,
                        network_seed=weight_seed,    # Different network weights
                        feature_seed=original_seed + 1000  # Same feature selection logic
                    )
                    acc, auc = evaluate_with_tabpfn(X_tr, y_tr, X_te, y_te, meta.target_n_classes)
                    if acc:
                        exp2_results.append({
                            'weight_seed': weight_seed,
                            'acc': acc,
                            'auc': auc
                        })
                        auc_str = f"{auc:.3f}" if auc else "N/A"
                        print(f"  weight_seed={weight_seed}: Acc={acc:.3f}, AUC={auc_str}")
                except Exception as e:
                    print(f"  weight_seed={weight_seed}: Error - {e}")
            
            sensitivity_results.append({
                'dataset_id': dataset_id,
                'original_seed': original_seed,
                'original_auc': original_meta.auc,
                'exp1_feature_selection': exp1_results,
                'exp2_weight_init': exp2_results
            })
        
        # Save sensitivity results
        sensitivity_file = f"{output_dir}/sensitivity_{timestamp}.json"
        with open(sensitivity_file, 'w') as f:
            json.dump(sensitivity_results, f, indent=2)
        print(f"\nSensitivity results saved to: {sensitivity_file}")
    
    print(f"\nAll results saved to: {results_file}")
    return all_results


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    run_full_analysis(n_datasets=n)
