"""
Dataset Generator V2 - Uses RandomNNGenerator V2 to create classification datasets.

Label is a discrete memory dimension (fixed for all t).
Features are nodes from the network.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
from typing import List, Tuple

from random_nn_generator_v2 import RandomNNGenerator, RandomNNConfig, DiscreteMemoryDimConfig


@dataclass  
class GeneratedDataset:
    """A generated dataset."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    n_classes: int
    n_features: int
    t_length: int


def sample_n_features_geometric(rng: np.random.Generator, max_features: int, p: float = 0.5) -> int:
    """
    Sample number of features from truncated geometric distribution.
    p = probability of "success" (stopping). Higher p = more likely to get 1.
    Returns value in [1, max_features].
    """
    # Geometric: P(X=k) = (1-p)^(k-1) * p
    for k in range(1, max_features + 1):
        if rng.random() < p:
            return k
    return max_features


def _causal_conv1d_same(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    1D causal convolution that preserves length.

    y[t] = sum_{k=0..K-1} kernel[k] * x[t-k]  (with x[t<0] = 0)

    Args:
        x: shape (T,)
        kernel: shape (K,)
    Returns:
        y: shape (T,)
    """
    x = np.asarray(x)
    kernel = np.asarray(kernel)
    if x.ndim != 1 or kernel.ndim != 1:
        raise ValueError(f"_causal_conv1d_same expects 1D arrays, got x.ndim={x.ndim}, kernel.ndim={kernel.ndim}")
    if kernel.shape[0] < 1:
        return x.copy()

    K = int(kernel.shape[0])
    # Pad on the left for causality
    x_pad = np.pad(x, (K - 1, 0), mode="constant", constant_values=0.0)
    # Convolution, keep only valid window -> length T
    y = np.convolve(x_pad, kernel, mode="valid")
    return y


def apply_random_causal_convs(
    X: np.ndarray,
    rng: np.random.Generator,
    prob: float = 0.0,
    n_convs_range: Tuple[int, int] = (1, 1),
    kernel_size_range: Tuple[int, int] = (3, 11),
    per_feature_kernel: bool = True,
    normalize_l1: bool = True,
    kernel_mode: str = "positive",
) -> Tuple[np.ndarray, dict]:
    """
    Optional post-processing: apply random causal convolutions to some samples.

    - Preserves shape (n_samples, n_features, T)
    - Applies a random number of conv layers sequentially
    - For each conv: sample kernel length, sample kernel weights, apply causal conv

    Args:
        X: (n_samples, n_features, T)
        prob: probability of applying the transform to a given sample
        n_convs_range: (min, max) number of sequential convs per transformed sample
        kernel_size_range: (min, max) kernel size (will be clamped to >=1)
        per_feature_kernel: if True, sample a different kernel per feature; else one shared kernel
        normalize_l1: if True, normalize kernel by L1 norm (stabilizes scale)
        kernel_mode: 'positive' (default) or 'signed'
            - 'positive': kernel weights are non-negative (acts like a smoothing/filtering kernel)
            - 'signed': kernel weights can be negative (can create sign flips/cancellations)
    Returns:
        X_out: transformed array
        info: dict with metadata (n_transformed, etc.)
    """
    if prob <= 0.0:
        return X, {"enabled": False, "n_transformed": 0}
    if X.ndim != 3:
        raise ValueError(f"apply_random_causal_convs expects X.ndim==3, got {X.ndim}")

    n_samples, n_features, T = X.shape
    if n_samples == 0:
        return X, {"enabled": True, "n_transformed": 0}

    # Sample which samples to transform
    mask = rng.random(n_samples) < prob
    idxs = np.where(mask)[0]
    if idxs.size == 0:
        return X, {"enabled": True, "n_transformed": 0}

    X_out = X.copy()

    min_convs, max_convs = n_convs_range
    min_k, max_k = kernel_size_range
    min_convs = max(1, int(min_convs))
    max_convs = max(min_convs, int(max_convs))
    min_k = max(1, int(min_k))
    max_k = max(min_k, int(max_k))

    # Apply per selected sample
    for i in idxs:
        n_convs = int(rng.integers(min_convs, max_convs + 1))
        for _ in range(n_convs):
            K = int(rng.integers(min_k, max_k + 1))
            # Keep K reasonable wrt T
            K = min(K, max(1, T))

            if per_feature_kernel:
                kernels = rng.normal(0.0, 1.0, size=(n_features, K))
                if kernel_mode == "positive":
                    kernels = np.abs(kernels)
                if normalize_l1:
                    denom = np.sum(np.abs(kernels), axis=1, keepdims=True) + 1e-8
                    kernels = kernels / denom
                for f in range(n_features):
                    X_out[i, f, :] = _causal_conv1d_same(X_out[i, f, :], kernels[f])
            else:
                kernel = rng.normal(0.0, 1.0, size=(K,))
                if kernel_mode == "positive":
                    kernel = np.abs(kernel)
                if normalize_l1:
                    kernel = kernel / (np.sum(np.abs(kernel)) + 1e-8)
                for f in range(n_features):
                    X_out[i, f, :] = _causal_conv1d_same(X_out[i, f, :], kernel)

    return X_out.astype(np.float32, copy=False), {
        "enabled": True,
        "n_transformed": int(idxs.size),
        "transformed_indices": idxs.tolist(),
        "prob": float(prob),
        "n_convs_range": tuple(n_convs_range),
        "kernel_size_range": tuple(kernel_size_range),
        "per_feature_kernel": bool(per_feature_kernel),
        "normalize_l1": bool(normalize_l1),
        "kernel_mode": str(kernel_mode),
    }


def visualize_before_after(
    X_before: np.ndarray,
    X_after: np.ndarray,
    y: np.ndarray,
    output_path: str,
    name: str,
    samples_per_class: int = 2,
    only_indices: List[int] | None = None,
):
    """Visualize before vs after for selected samples in a class-wise grid."""
    if only_indices is not None:
        only_indices = np.array(list(only_indices), dtype=int)
        Xb = X_before[only_indices]
        Xa = X_after[only_indices]
        yy = y[only_indices]
    else:
        Xb, Xa, yy = X_before, X_after, y

    classes = sorted(np.unique(yy))
    n_classes = len(classes)
    if n_classes == 0:
        return

    n_cols = min(samples_per_class, 5) * 2  # before+after per sample
    fig, axes = plt.subplots(n_classes, n_cols, figsize=(3 * n_cols, 2.6 * n_classes), squeeze=False)
    fig.suptitle(
        f"{name} | Before vs After (causal conv) | n_feat={Xb.shape[1]} T={Xb.shape[2]}",
        fontsize=11,
        fontweight="bold",
    )

    colors = plt.cm.tab10(np.linspace(0, 1, max(Xb.shape[1], 1)))

    for row, cls in enumerate(classes):
        cls_idxs = np.where(yy == cls)[0]
        n_available = len(cls_idxs)
        n_show = min(samples_per_class, n_available)

        for s in range(n_show):
            idx = cls_idxs[s]
            # two panels: before, after
            for panel in range(2):
                ax = axes[row, 2 * s + panel]
                Xcur = Xb[idx] if panel == 0 else Xa[idx]
                title = "Before" if panel == 0 else "After"
                for f in range(Xcur.shape[0]):
                    ax.plot(Xcur[f, :], color=colors[f % 10], linewidth=0.8, alpha=0.85)
                ax.set_title(f"C{cls} {title}", fontsize=9)
                ax.tick_params(labelsize=7)

                # Match y-lims for before/after comparison on this sample
                x_min = min(Xb[idx].min(), Xa[idx].min())
                x_max = max(Xb[idx].max(), Xa[idx].max())
                if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
                    pad = 0.05 * (x_max - x_min)
                    ax.set_ylim(x_min - pad, x_max + pad)

        # Hide unused columns
        for c in range(2 * n_show, n_cols):
            axes[row, c].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_dataset(
    nn_config: RandomNNConfig,
    max_samples: int,
    max_features: int,
    feature_geometric_p: float,
    t_subseq_range: Tuple[int, int],
    t_excess_range: Tuple[int, int],
    max_m_times_t: int,
    train_ratio: float,
    causal_conv_prob: float = 0.0,
    causal_conv_n_convs_range: Tuple[int, int] = (1, 1),
    causal_conv_kernel_size_range: Tuple[int, int] = (3, 11),
    causal_conv_per_feature_kernel: bool = True,
    causal_conv_normalize_l1: bool = True,
    causal_conv_kernel_mode: str = "positive",
    causal_conv_visualize_path: str | None = None,
    causal_conv_visualize_samples_per_class: int = 2,
    causal_conv_visualize_only_transformed: bool = True,
    seed: int = 42
) -> GeneratedDataset:
    """
    Generate a classification dataset using RandomNNGenerator V2.
    
    Label is a discrete memory dimension (fixed for all t).
    Features are nodes from the network.
    """
    rng = np.random.default_rng(seed)
    
    # Sample number of features (geometric distribution favoring small)
    n_features = sample_n_features_geometric(rng, max_features, feature_geometric_p)
    
    # Sample timesteps (enforce m*t <= max_m_times_t)
    max_t = min(t_subseq_range[1], max_m_times_t // n_features)
    t_subseq = rng.integers(t_subseq_range[0], max(t_subseq_range[0] + 1, max_t + 1))
    
    # Sample excess and total T
    t_excess = rng.integers(t_excess_range[0], t_excess_range[1] + 1)
    T = t_subseq + t_excess
    
    # Sample number of samples
    n_samples = rng.integers(50, max_samples + 1)
    
    # Create generator with this T
    nn_config_copy = RandomNNConfig(
        memory_dim_range=nn_config.memory_dim_range,
        memory_init=nn_config.memory_init,
        memory_discrete_prob=nn_config.memory_discrete_prob,
        memory_discrete_classes_range=nn_config.memory_discrete_classes_range,
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
    
    generator = RandomNNGenerator(nn_config_copy, seed=seed)
    cfg = generator.sampled_config
    
    # Check for discrete memory dims - need at least one for label
    if not cfg.memory_discrete_configs or len(cfg.memory_discrete_configs) == 0:
        raise ValueError("No discrete memory dimensions - need at least one for label")
    
    # Select one discrete dim as label (random choice)
    label_disc_cfg = cfg.memory_discrete_configs[rng.integers(0, len(cfg.memory_discrete_configs))]
    label_dim_idx = label_disc_cfg.dim_idx
    n_classes = label_disc_cfg.n_classes
    class_values = label_disc_cfg.class_values  # The sampled propagation values for each class
    
    # Generate memory and get labels BEFORE propagation
    memory = generator._generate_memory(n_samples)
    
    # Extract labels from the discrete dimension
    label_values = memory[:, label_dim_idx]
    
    # Map the sampled class values to class indices (0, 1, 2, ...)
    # Each unique class_value corresponds to a class index
    y = np.zeros(n_samples, dtype=np.int32)
    for class_idx, class_val in enumerate(class_values):
        mask = np.isclose(label_values, class_val)
        y[mask] = class_idx
    
    # Propagate with this memory (pass memory to ensure consistency!)
    _, all_layers = generator.propagate(n_samples, memory=memory)
    
    # Select feature nodes (avoid dead neurons)
    n_layers = len(all_layers) - 1
    feature_candidates = []
    feature_variances = []
    
    for layer_idx in range(n_layers):
        layer_data = all_layers[layer_idx + 1]
        n_nodes = layer_data.shape[1]
        
        for node_idx in range(n_nodes):
            node_values = layer_data[:, node_idx, :]
            var_across_samples = np.mean(np.var(node_values, axis=0))
            var_across_time = np.mean(np.var(node_values, axis=1))
            total_var = var_across_samples + var_across_time
            
            feature_candidates.append((layer_idx, node_idx))
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
        
        n_features = min(n_features, len(alive_candidates))
        selected_idxs = rng.choice(len(alive_candidates), size=n_features, replace=False, p=feat_probs)
        selected_features = [alive_candidates[i] for i in selected_idxs]
    else:
        n_features = min(n_features, len(feature_candidates))
        selected_idxs = rng.choice(len(feature_candidates), size=n_features, replace=False)
        selected_features = [feature_candidates[i] for i in selected_idxs]
    
    # Extract samples - FIXED start position for all samples (sampled once per dataset)
    start = rng.integers(0, t_excess + 1)
    
    X_list = []
    for seq_idx in range(n_samples):
        features = []
        for layer_idx, node_idx in selected_features:
            layer_data = all_layers[layer_idx + 1]
            values = layer_data[seq_idx, node_idx, start:start + t_subseq]
            features.append(values)
        X_list.append(np.stack(features, axis=0))
    
    X = np.stack(X_list, axis=0).astype(np.float32)
    X_before_conv = X.copy() if (causal_conv_visualize_path is not None and causal_conv_prob > 0) else None

    # Optional post-processing: random causal convolutions on some samples
    X, conv_info = apply_random_causal_convs(
        X,
        rng=rng,
        prob=causal_conv_prob,
        n_convs_range=causal_conv_n_convs_range,
        kernel_size_range=causal_conv_kernel_size_range,
        per_feature_kernel=causal_conv_per_feature_kernel,
        normalize_l1=causal_conv_normalize_l1,
        kernel_mode=causal_conv_kernel_mode,
    )

    # Optional: visualize before vs after
    if causal_conv_visualize_path is not None and X_before_conv is not None and conv_info.get("enabled", False):
        only_idxs = None
        if causal_conv_visualize_only_transformed:
            only_idxs = conv_info.get("transformed_indices", None)
            if only_idxs is not None and len(only_idxs) == 0:
                only_idxs = None
        visualize_before_after(
            X_before=X_before_conv,
            X_after=X,
            y=y,
            output_path=causal_conv_visualize_path,
            name=os.path.basename(causal_conv_visualize_path),
            samples_per_class=causal_conv_visualize_samples_per_class,
            only_indices=only_idxs,
        )
    
    # Train/test split - STRATIFIED to ensure all classes in both sets
    n_total = len(y)
    unique_classes, class_counts = np.unique(y, return_counts=True)
    
    # Check if stratified split is possible (need at least 2 samples per class)
    min_samples_per_class = class_counts.min()
    if min_samples_per_class < 2:
        raise ValueError(f"Class with only {min_samples_per_class} sample(s) - need at least 2 for stratified split")
    
    # Stratified split: each class gets train_ratio in train, rest in test
    train_idx = []
    test_idx = []
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)
        n_train_cls = max(1, int(len(cls_indices) * train_ratio))
        train_idx.extend(cls_indices[:n_train_cls])
        test_idx.extend(cls_indices[n_train_cls:])
    
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    
    # Shuffle within train and test
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    
    # Verify all classes are in both sets
    train_classes = set(np.unique(y[train_idx]))
    test_classes = set(np.unique(y[test_idx]))
    if not test_classes.issubset(train_classes):
        missing = test_classes - train_classes
        raise ValueError(f"Test has classes {missing} not in train - stratified split failed")
    
    # Check if dataset is "boring"
    feature_temporal_vars = []
    for f in range(X.shape[1]):
        var_across_time = np.mean(np.var(X[:, f, :], axis=1))
        feature_temporal_vars.append(var_across_time)
    
    avg_temporal_var = np.mean(feature_temporal_vars)
    if avg_temporal_var < 0.01:
        raise ValueError(f"Dataset too boring: avg temporal variance = {avg_temporal_var:.4f}")
    
    # Check class balance
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        raise ValueError(f"Only {len(unique_classes)} class found - need at least 2")
    
    return GeneratedDataset(
        X_train=X[train_idx],
        y_train=y[train_idx],
        X_test=X[test_idx],
        y_test=y[test_idx],
        n_classes=n_classes,
        n_features=n_features,
        t_length=t_subseq
    )


def visualize_dataset(dataset: GeneratedDataset, output_path: str, name: str, samples_per_class: int = 5):
    """Visualize dataset with multiple observations per class in a grid."""
    X = np.concatenate([dataset.X_train, dataset.X_test], axis=0)
    y = np.concatenate([dataset.y_train, dataset.y_test], axis=0)
    n_samples, n_features, t = X.shape
    
    # Collect samples per class
    classes = sorted(np.unique(y))
    n_classes = len(classes)
    
    # Grid: rows = classes, cols = samples_per_class
    n_cols = min(samples_per_class, 5)
    n_rows = n_classes
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows), squeeze=False)
    
    # Class distribution info
    class_counts = [np.sum(y == c) for c in classes]
    balance = min(class_counts) / max(class_counts) if max(class_counts) > 0 else 1.0
    
    fig.suptitle(
        f"{name}\n"
        f"{n_samples} samples × {n_features} feat × {t} t | {n_classes} classes | balance={balance:.2f}",
        fontsize=11, fontweight='bold'
    )
    
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_features, 1)))
    
    for row, cls in enumerate(classes):
        cls_idxs = np.where(y == cls)[0]
        n_available = len(cls_idxs)
        
        for col in range(n_cols):
            ax = axes[row, col]
            
            if col < n_available:
                idx = cls_idxs[col]
                for f in range(n_features):
                    ax.plot(X[idx, f, :], color=colors[f % 10], linewidth=0.8, alpha=0.8)
                ax.set_title(f"Class {cls} (#{col+1})", fontsize=9)
            else:
                ax.set_visible(False)
            
            ax.tick_params(labelsize=7)
            if col == 0:
                ax.set_ylabel(f"C{cls} (n={n_available})", fontsize=8)
            if row == n_rows - 1:
                ax.set_xlabel("t", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def get_default_nn_config(seq_length: int = 200) -> RandomNNConfig:
    """
    Get the default RandomNNConfig with all parameters.
    This is the single source of truth for config parameters.
    Can be imported by other scripts (e.g., compare_distributions.py).
    """
    return RandomNNConfig(
        # Memory
        memory_dim_range=(1, 4),                      # uniform int
        memory_init='normal',                          # 'uniform' or 'normal'
        memory_normal_std_range=(0.1, 1.0),           # log-uniform: std for normal init
        memory_discrete_prob=0.3,                      # prob each dim is discrete
        memory_discrete_classes_range=(2, 10),         # log-uniform: classes per discrete dim
        
        # Stochastic input
        stochastic_input_dim_range=(0, 4),            # uniform int: noise dims
        stochastic_input_std_range=(0.0001, 1),      # log-uniform: std for stochastic noise
        
        # Time transforms
        n_time_transforms_range=(1, 4),               # uniform int
        
        # Architecture
        n_hidden_layers_range=(2, 8),                  # log-uniform
        n_nodes_per_layer_range=(4, 30),               # log-uniform
        
        # Activations (per-node)
        activation_choices=(
            'identity', 'log', 'sigmoid', 'abs', 'sin', 'tanh',
            'rank', 'square', 'power', 'softplus', 'step', 'modulo',
        ),
        
        # Weights
        weight_init_choices=('xavier_normal',),
        weight_scale_range=(0.9, 1.1),                 # uniform
        bias_std_range=(0.0, 0.1),                     # uniform
        
        # Per-node noise
        node_noise_prob_range=(0.001, 1),            # log-uniform (prob per node has noise)
        node_noise_std_range=(0.0001, 1),              # log-uniform (noise scale)
        noise_dist_choices=('normal',),
        
        # Quantization (not used - label from memory)
        per_layer_activation=True,
        quantization_node_prob=0.0,
        quantization_n_classes_range=(2, 8),
        
        seq_length=seq_length,
    )


def get_default_dataset_params() -> dict:
    """
    Get default dataset generation parameters.
    Can be imported by other scripts.
    
    feature_geometric_p controls univariate probability:
      p=0.5: P(1 feat)=50%, P(2)=25%, P(3)=12.5%
      p=0.7: P(1 feat)=70%, P(2)=21%, P(3)=6.3%
    """
    return {
        'max_samples': 1000,
        'max_features': 12,
        'feature_geometric_p': 0.7,  # Increased: ~70% univariate
        't_subseq_range': (20, 500),
        't_excess_range': (5, 50),
        'max_m_times_t': 500,
        'train_ratio': 0.8,

        # Optional post-processing experiment: random causal convolutions
        # Applies to a subset of samples, preserving length.
        'causal_conv_prob': 0.0,              # set >0 (e.g. 0.3) to enable
        'causal_conv_n_convs_range': (1, 2),  # sequential conv layers
        'causal_conv_kernel_size_range': (3, 11),
        'causal_conv_per_feature_kernel': True,
        'causal_conv_normalize_l1': True,
        'causal_conv_kernel_mode': 'positive',  # 'positive' or 'signed'
        'causal_conv_visualize': False,
        'causal_conv_visualize_samples_per_class': 2,
        'causal_conv_visualize_only_transformed': True,
    }


def main():
    print("=" * 60)
    print("Dataset Generator V2 (based on RandomNNGenerator V2)")
    print("=" * 60)
    
    # =========================================================================
    # CONFIGURATION - edit get_default_nn_config() and get_default_dataset_params()
    # to change parameters (shared with compare_distributions.py)
    # =========================================================================
    
    output_dir = "generated_datasets_v2"
    n_datasets = 16
    
    # Get configs from single source of truth
    nn_config = get_default_nn_config(seq_length=200)
    dataset_params = get_default_dataset_params()
    
    max_samples = dataset_params['max_samples']
    max_features = dataset_params['max_features']
    feature_geometric_p = dataset_params['feature_geometric_p']
    t_subseq_range = dataset_params['t_subseq_range']
    t_excess_range = dataset_params['t_excess_range']
    max_m_times_t = dataset_params['max_m_times_t']
    train_ratio = dataset_params['train_ratio']
    
    # =========================================================================
    # BUILD OUTPUT DIR
    # =========================================================================
    
    output_dir = os.path.join(os.path.dirname(__file__), output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating {n_datasets} datasets -> {output_dir}\n")
    
    for i in range(n_datasets):
        for attempt in range(10):
            try:
                seed = 42 + i * 100 + attempt
                dataset = generate_dataset(
                    nn_config=nn_config,
                    max_samples=max_samples,
                    max_features=max_features,
                    feature_geometric_p=feature_geometric_p,
                    t_subseq_range=t_subseq_range,
                    t_excess_range=t_excess_range,
                    max_m_times_t=max_m_times_t,
                    train_ratio=train_ratio,
                    causal_conv_prob=dataset_params.get('causal_conv_prob', 0.0),
                    causal_conv_n_convs_range=dataset_params.get('causal_conv_n_convs_range', (1, 1)),
                    causal_conv_kernel_size_range=dataset_params.get('causal_conv_kernel_size_range', (3, 11)),
                    causal_conv_per_feature_kernel=dataset_params.get('causal_conv_per_feature_kernel', True),
                    causal_conv_normalize_l1=dataset_params.get('causal_conv_normalize_l1', True),
                    causal_conv_kernel_mode=dataset_params.get('causal_conv_kernel_mode', 'positive'),
                    causal_conv_visualize_path=(
                        f"{output_dir}/dataset{i:02d}_before_after.png"
                        if dataset_params.get('causal_conv_visualize', False) and dataset_params.get('causal_conv_prob', 0.0) > 0
                        else None
                    ),
                    causal_conv_visualize_samples_per_class=dataset_params.get('causal_conv_visualize_samples_per_class', 2),
                    causal_conv_visualize_only_transformed=dataset_params.get('causal_conv_visualize_only_transformed', True),
                    seed=seed
                )
                
                n_total = len(dataset.X_train) + len(dataset.X_test)
                print(f"Dataset {i}: {n_total} × {dataset.n_features} × {dataset.t_length} | {dataset.n_classes} classes")
                
                visualize_dataset(dataset, f"{output_dir}/dataset{i:02d}.png", f"Synthetic Dataset {i}")
                break
                
            except ValueError as e:
                if attempt == 9:
                    print(f"Dataset {i}: Failed - {e}")
            except Exception as e:
                print(f"Dataset {i}: Error - {e}")
                break
    
    print(f"\nDone! Saved to: {output_dir}")


if __name__ == "__main__":
    main()
