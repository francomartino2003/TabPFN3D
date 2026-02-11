#!/usr/bin/env python3
"""
Dataset generator based on random DAGs with positional encoding and causal convolutions.

Pipeline:
  1. Sample hyperparameters (log-uniform favoring simpler configs).
  2. Build a random DAG: V nodes, M root nodes, max 3 parents per non-root node.
  3. Build network parameters: per non-root node, weights for linear combination
     of parents, a causal 1D conv kernel, and an activation function.
  4. For each sample (observation):
     a. Sample root nodes: each root is (T, d) = m_i + PE(t), m_i ~ N(0,std) or U(-a,a).
     b. Propagate through DAG: linear comb of parents → causal conv → activation.
  5. Assign feature nodes and target node (non-root, at random).
  6. Pool features: alpha*mean + (1-alpha)*max over d → (T,) per node.
  7. Pool target: same d-pooling → (T,), then pool over T → scalar.
  8. Discretize target scalars into classes via K-means.
  9. Train/test split.

Self-contained: no imports from 08_latent_pe_experiments.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

# ============================================================
# ACTIVATION FUNCTIONS
# ============================================================

ACTIVATION_CHOICES = (
    "identity",
    "log",
    "sigmoid",
    "abs",
    "sin",
    "tanh",
    "power",
    "softplus",
    "step",
    "modulo",
)


def apply_activation(x: np.ndarray, act: str) -> np.ndarray:
    """Apply activation element-wise."""
    if act == "identity":
        return x
    if act == "log":
        return np.sign(x) * np.log1p(np.abs(x))
    if act == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    if act == "abs":
        return np.abs(x)
    if act == "sin":
        return np.sin(x)
    if act == "tanh":
        return np.tanh(x)
    if act == "power":
        return np.sign(x) * np.sqrt(np.abs(x))
    if act == "softplus":
        return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -500, 20))))
    if act == "step":
        return np.where(x >= 0, 1.0, 0.0)
    if act == "modulo":
        return np.mod(x, 1.0)
    raise ValueError(f"Unknown activation: {act!r}")


# ============================================================
# POSITIONAL ENCODING
# ============================================================


def positional_encoding(t: int, d: int) -> np.ndarray:
    """Sinusoidal PE of dimension d at time step t. Returns (d,)."""
    pe = np.zeros(d, dtype=np.float64)
    for i in range(d):
        if i % 2 == 0:
            pe[i] = np.sin(t / (10000 ** (i / d)))
        else:
            pe[i] = np.cos(t / ((10000 ** ((i - 1) / d))))
    return pe


def positional_encoding_matrix(T: int, d: int) -> np.ndarray:
    """Precompute PE for all t in [0, T). Returns (T, d)."""
    pe = np.zeros((T, d), dtype=np.float64)
    pos = np.arange(T, dtype=np.float64)[:, None]  # (T, 1)
    i = np.arange(d, dtype=np.float64)[None, :]      # (1, d)
    # even indices: sin, odd indices: cos
    freq = 1.0 / (10000 ** (np.floor(i / 2) * 2 / d))
    pe[:, 0::2] = np.sin(pos * freq[:, 0::2])
    pe[:, 1::2] = np.cos(pos * freq[:, 1::2])
    return pe


# ============================================================
# CAUSAL 1D CONVOLUTION (depthwise)
# ============================================================


def causal_conv1d_depthwise(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Causal 1D convolution (depthwise): mixes features across time per channel.
    x: (T, d), kernel: (K, d).
    output[t, c] = sum_{k=0}^{min(K-1,t)} kernel[k, c] * x[t-k, c].
    """
    T, d = x.shape
    K = kernel.shape[0]
    out = np.zeros_like(x)
    padded = np.zeros((T + K - 1, d), dtype=x.dtype)
    padded[K - 1:] = x
    for c in range(d):
        rev_k = kernel[::-1, c]
        out[:, c] = np.convolve(padded[:, c], rev_k, mode="valid")
    return out


# ============================================================
# HYPERPARAMETER CONFIGURATION
# ============================================================
# All distributions used to sample dataset hyperparameters.
# Edit ranges, distribution types, and comments here.
# LogUniform(a, b) means: exp(Uniform(log(a), log(b))), favors smaller values.
# ============================================================

HYPER_CONFIG: Dict[str, Any] = {
    # --- DAG structure ---
    # V: total number of nodes in the DAG.
    # LogUniform(4, 30). Favors smaller (simpler) graphs.
    "V": {"dist": "log_uniform", "low": 4, "high": 30},

    # M: number of root nodes. LogUniform(1, 5). Always M < V.
    "M": {"dist": "log_uniform", "low": 1, "high": 5},

    # max_parents: each non-root node has between 1 and min(max_parents, available).
    # Fixed at 3.
    "max_parents": 3,

    # --- Dimensions ---
    # d: latent dimension per node. LogUniform(4, 512). Favors smaller d.
    "d": {"dist": "log_uniform", "low": 4, "high": 512},

    # T: temporal length. LogUniform(20, 1000).
    "T": {"dist": "log_uniform", "low": 20, "high": 1000},

    # --- Hard constraints ---
    "max_samples": 1000,
    "max_features": 12,
    "max_timesteps": 1000,
    "max_feat_times_t": 500,  # n_features * T_out <= 500 (after warmup)

    # --- Root initialization ---
    # init_type: "normal" or "uniform", chosen at random (50/50).
    # root_std: std for N(0, std). LogUniform(0.05, 1.5).
    "root_std": {"dist": "log_uniform", "low": 0.05, "high": 1.5},
    # root_a: half-width for U(-a, a). Uniform(0.1, 1.0). Max 1.
    "root_a": {"dist": "log_uniform", "low": 0.1, "high": 1.0},

    # --- Network (intermediate layers) ---
    # kernel_size: causal conv kernel size. LogUniform(2, 15).
    "kernel_size": {"dist": "uniform", "low": 2, "high": 15},
    # gain: Xavier gain for weight initialization. Uniform(0.5, 2.0).
    "gain": {"dist": "uniform", "low": 0.5, "high": 2.0},

    # --- Global pooling ---
    # alpha per node: alpha*mean + (1-alpha)*max over d.
    # Uniform(0, 1) per node.
    "alpha_pool": {"dist": "uniform", "low": 0.0, "high": 1.0},

    # --- Dataset ---
    # n_features: number of feature nodes.
    # Geometric(p=0.7) starting at 1, capped at max_features.
    # P(1)=0.70, P(2)=0.21, P(3)=0.063, P(4+)=0.027  — heavily univariate.
    "n_features": {"dist": "geometric", "p": 0.7},
    # n_classes: number of target classes. LogUniform(2, 10).
    "n_classes": {"dist": "log_uniform", "low": 2, "high": 10},
    # n_samples: total samples (train + test). LogUniform(50, 1000).
    "n_samples": {"dist": "uniform", "low": 50, "high": 1000},
}


# ============================================================
# SAMPLING HELPERS
# ============================================================


def log_uniform_int(rng: np.random.Generator, low: int, high: int) -> int:
    """Sample integer from log-uniform distribution on [low, high]."""
    v = np.exp(rng.uniform(np.log(low), np.log(high)))
    return int(np.clip(np.round(v), low, high))


def log_uniform_float(rng: np.random.Generator, low: float, high: float) -> float:
    """Sample float from log-uniform distribution on [low, high]."""
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def xavier_std(in_dim: int, out_dim: int, gain: float = 1.0) -> float:
    return gain * np.sqrt(2.0 / (in_dim + out_dim))


# ============================================================
# DATA STRUCTURES
# ============================================================


@dataclass
class HyperParams:
    """Sampled hyperparameters for one dataset."""
    V: int
    M: int
    d: int
    T: int
    kernel_size: int
    gain: float
    init_type: str       # "normal" or "uniform"
    root_std: float
    root_a: float
    n_features: int
    n_classes: int
    n_samples: int


@dataclass
class DAG:
    """DAG structure. parents[i] = list of parent indices for node i (empty for roots)."""
    V: int
    M: int
    roots: List[int]             # indices of root nodes
    parents: List[List[int]]     # parents[i] = [parent_idx, ...]
    topo_order: List[int]        # topological order (roots first)


@dataclass
class NodeParams:
    """Parameters for one non-root node."""
    w_comb: np.ndarray     # (n_parents,) — weights for linear combination
    b_comb: float          # bias for linear combination
    kernel: np.ndarray     # (K, d) — causal conv kernel
    activation: str        # activation function name


@dataclass
class NetworkParams:
    """Network parameters for all non-root nodes."""
    node_params: Dict[int, NodeParams]  # node_idx -> NodeParams
    alpha_pool: Dict[int, float]        # node_idx -> alpha for d-pooling (all nodes incl. roots)
    alpha_pool_T: float                 # alpha for T-pooling of target


@dataclass
class GeneratedDataset:
    """Output dataset."""
    X_train: np.ndarray    # (n_train, n_features, T)
    X_test: np.ndarray     # (n_test, n_features, T)
    y_train: np.ndarray    # (n_train,) int
    y_test: np.ndarray     # (n_test,) int
    n_classes: int
    hyper: HyperParams
    dag: DAG
    feature_nodes: List[int]
    target_node: int


# ============================================================
# 1. SAMPLE HYPERPARAMETERS
# ============================================================


def sample_hyperparams(rng: np.random.Generator) -> HyperParams:
    """Sample all hyperparameters from HYPER_CONFIG distributions."""
    cfg = HYPER_CONFIG

    V = log_uniform_int(rng, cfg["V"]["low"], cfg["V"]["high"])
    M = log_uniform_int(rng, cfg["M"]["low"], cfg["M"]["high"])
    M = min(M, V - 2)  # ensure at least 2 non-root nodes (1 feature + 1 target)
    M = max(M, 1)

    d = log_uniform_int(rng, cfg["d"]["low"], cfg["d"]["high"])
    T = log_uniform_int(rng, cfg["T"]["low"], cfg["T"]["high"])

    kernel_size = log_uniform_int(rng, cfg["kernel_size"]["low"], cfg["kernel_size"]["high"])
    gain = float(rng.uniform(cfg["gain"]["low"], cfg["gain"]["high"]))

    init_type = rng.choice(["normal", "uniform"]).item()
    root_std = log_uniform_float(rng, cfg["root_std"]["low"], cfg["root_std"]["high"])
    root_a = float(rng.uniform(cfg["root_a"]["low"], cfg["root_a"]["high"]))

    n_features = int(rng.geometric(cfg["n_features"]["p"]))
    n_classes = log_uniform_int(rng, cfg["n_classes"]["low"], cfg["n_classes"]["high"])
    n_samples = log_uniform_int(rng, cfg["n_samples"]["low"], cfg["n_samples"]["high"])
    n_samples = min(n_samples, cfg["max_samples"])
    n_features = min(n_features, cfg["max_features"])

    # Enforce constraints
    T = min(T, cfg["max_timesteps"])
    kernel_size = min(kernel_size, T)  # kernel can't be bigger than T
    warmup = kernel_size - 1
    T_out = T - warmup

    # Enforce n_features * T_out <= max_feat_times_t
    max_ft = cfg["max_feat_times_t"]
    if n_features * T_out > max_ft:
        # Try reducing T first
        T_out_needed = max(1, max_ft // n_features)
        T = T_out_needed + warmup
        T_out = T - warmup
    if n_features * T_out > max_ft:
        # Still too big, reduce features
        n_features = max(1, max_ft // T_out)
    T = max(warmup + 2, T)  # ensure T_out >= 2

    # Ensure we have enough non-root nodes for features + target
    non_root = V - M
    needed = n_features + 1  # features + 1 target
    if non_root < needed:
        V = M + needed
    n_features = min(n_features, V - M - 1)

    return HyperParams(
        V=V, M=M, d=d, T=T,
        kernel_size=kernel_size, gain=gain,
        init_type=init_type, root_std=root_std, root_a=root_a,
        n_features=n_features, n_classes=n_classes, n_samples=n_samples,
    )


# ============================================================
# 2. BUILD DAG
# ============================================================


def build_dag(V: int, M: int, rng: np.random.Generator, max_parents: int = 3) -> DAG:
    """
    Build a random DAG with V nodes, M root nodes (indices 0..M-1).
    Non-root nodes are indices M..V-1.
    Each non-root node gets 1..min(max_parents, idx) parents from nodes with smaller index.
    Nodes are already in topological order by index.
    """
    roots = list(range(M))
    parents: List[List[int]] = [[] for _ in range(V)]

    for i in range(M, V):
        # Available parents: all nodes with index < i
        available = list(range(i))
        n_parents = rng.integers(1, min(max_parents, len(available)) + 1)
        chosen = rng.choice(available, size=n_parents, replace=False).tolist()
        parents[i] = sorted(chosen)

    topo_order = list(range(V))  # indices are already topologically sorted
    return DAG(V=V, M=M, roots=roots, parents=parents, topo_order=topo_order)


# ============================================================
# 3. BUILD NETWORK PARAMETERS
# ============================================================


def build_network(
    dag: DAG,
    d: int,
    kernel_size: int,
    gain: float,
    rng: np.random.Generator,
) -> NetworkParams:
    """Sample weights, kernels, and activations for all non-root nodes."""
    node_params: Dict[int, NodeParams] = {}
    K = kernel_size

    for i in dag.topo_order:
        if i in dag.roots:
            continue
        n_parents = len(dag.parents[i])

        # Linear combination weights: one weight per parent, one bias
        std_lin = xavier_std(n_parents, 1, gain)
        w_comb = rng.normal(0, std_lin, size=n_parents).astype(np.float64)
        b_comb = float(rng.normal(0, std_lin))

        # Causal conv kernel: (K, d)
        std_conv = xavier_std(K, 1, gain)
        kernel = rng.normal(0, std_conv, size=(K, d)).astype(np.float64)

        # Activation
        act = rng.choice(ACTIVATION_CHOICES).item()

        node_params[i] = NodeParams(
            w_comb=w_comb, b_comb=b_comb, kernel=kernel, activation=act,
        )

    # Pooling alphas: one per node (including roots), and one for T-pooling of target
    alpha_pool = {}
    cfg_alpha = HYPER_CONFIG["alpha_pool"]
    for i in range(dag.V):
        alpha_pool[i] = float(rng.uniform(cfg_alpha["low"], cfg_alpha["high"]))
    alpha_pool_T = float(rng.uniform(cfg_alpha["low"], cfg_alpha["high"]))

    return NetworkParams(node_params=node_params, alpha_pool=alpha_pool, alpha_pool_T=alpha_pool_T)


# ============================================================
# 4. PROPAGATE DAG (single sample)
# ============================================================


def sample_roots(
    M: int, T: int, d: int,
    init_type: str, root_std: float, root_a: float,
    pe_matrix: np.ndarray,
    rng: np.random.Generator,
) -> Dict[int, np.ndarray]:
    """Sample M root nodes, each (T, d). root[t] = m_i + PE(t)."""
    roots = {}
    for i in range(M):
        if init_type == "normal":
            m_i = rng.normal(0, root_std, size=d).astype(np.float64)
        else:
            m_i = rng.uniform(-root_a, root_a, size=d).astype(np.float64)
        roots[i] = m_i[None, :] + pe_matrix  # (T, d) broadcast
    return roots


def propagate_dag(
    root_values: Dict[int, np.ndarray],
    dag: DAG,
    net: NetworkParams,
) -> Dict[int, np.ndarray]:
    """
    Propagate through DAG. Returns dict node_idx -> (T, d) for all nodes.
    """
    values: Dict[int, np.ndarray] = dict(root_values)

    for i in dag.topo_order:
        if i in values:
            continue
        params = net.node_params[i]
        parent_indices = dag.parents[i]

        # Linear combination of parents: sum_p w_p * parent_p[t,:] + b
        T, d = values[parent_indices[0]].shape
        combined = np.full((T, d), params.b_comb, dtype=np.float64)
        for p_idx, w in zip(parent_indices, params.w_comb):
            combined += w * values[p_idx]

        # Causal conv 1D
        conv_out = causal_conv1d_depthwise(combined, params.kernel)

        # Activation
        values[i] = apply_activation(conv_out, params.activation)

    return values


# ============================================================
# 5. POOL AND EXTRACT FEATURES / TARGET
# ============================================================


def pool_over_d(node_val: np.ndarray, alpha: float) -> np.ndarray:
    """alpha*mean + (1-alpha)*max over d dimension. node_val: (T, d) -> (T,)."""
    return alpha * node_val.mean(axis=1) + (1.0 - alpha) * node_val.max(axis=1)


def pool_over_T(series: np.ndarray, alpha: float) -> float:
    """alpha*mean + (1-alpha)*max over T dimension. series: (T,) -> scalar."""
    return float(alpha * series.mean() + (1.0 - alpha) * series.max())


def _connected_component(node: int, dag: DAG) -> set:
    """Return the connected component of `node` treating the DAG as undirected."""
    # Build undirected adjacency
    adj: Dict[int, set] = {i: set() for i in range(dag.V)}
    for i in range(dag.V):
        for p in dag.parents[i]:
            adj[i].add(p)
            adj[p].add(i)
    # BFS
    visited = set()
    queue = [node]
    while queue:
        n = queue.pop()
        if n in visited:
            continue
        visited.add(n)
        for nb in adj[n]:
            if nb not in visited:
                queue.append(nb)
    return visited


def assign_roles(
    dag: DAG, n_features: int, rng: np.random.Generator,
) -> Tuple[List[int], int]:
    """
    Assign feature nodes and target node from non-root nodes.
    Features MUST be in the same connected component as the target
    (treating the DAG as an undirected graph) so they share root ancestors.
    Returns (feature_node_indices, target_node_index).
    """
    non_root = [i for i in range(dag.V) if i not in dag.roots]
    rng.shuffle(non_root)

    # Try each candidate as target until we find one whose connected
    # component has enough other non-root nodes for features.
    for candidate in non_root:
        comp = _connected_component(candidate, dag)
        # Other non-root nodes in the same component
        same_comp = [n for n in non_root if n != candidate and n in comp]
        if len(same_comp) >= min(n_features, 1):
            target_node = candidate
            rng.shuffle(same_comp)
            feature_nodes = sorted(same_comp[:n_features])
            return feature_nodes, target_node

    # Fallback (should not happen with a well-formed DAG):
    # just pick whatever is available
    target_node = non_root[0]
    feature_nodes = sorted(non_root[1 : 1 + n_features])
    return feature_nodes, target_node


# ============================================================
# 6. DISCRETIZE TARGET (K-means)
# ============================================================


def discretize_kmeans(
    y: np.ndarray, n_classes: int, rng: np.random.Generator,
    return_centroids: bool = False,
):
    """
    Simple 1D K-means to discretize scalar values into n_classes.

    Returns:
        labels (n_samples,) int32
        centroids (n_classes,) float64  — only if return_centroids=True
    """
    y = y.astype(np.float64)
    n = len(y)
    if n_classes >= n:
        labels = np.arange(n, dtype=np.int32) % n_classes
        centroids = np.array([y[labels == k].mean() if (labels == k).any() else 0.0
                              for k in range(n_classes)])
        return (labels, centroids) if return_centroids else labels

    # Initialize centroids via quantiles (more stable than random)
    quantiles = np.linspace(0, 100, n_classes + 2)[1:-1]
    centroids = np.percentile(y, quantiles)

    for _ in range(50):
        # Assign
        dists = np.abs(y[:, None] - centroids[None, :])  # (n, K)
        labels = dists.argmin(axis=1)
        # Update
        new_centroids = np.array([y[labels == k].mean() if (labels == k).any() else centroids[k]
                                  for k in range(n_classes)])
        if np.allclose(centroids, new_centroids, atol=1e-10):
            break
        centroids = new_centroids

    # Final assignment
    dists = np.abs(y[:, None] - centroids[None, :])
    labels = dists.argmin(axis=1).astype(np.int32)
    return (labels, centroids) if return_centroids else labels


# ============================================================
# 7. GENERATE ONE DATASET
# ============================================================


def generate_dataset(rng: np.random.Generator, hyper: HyperParams | None = None) -> GeneratedDataset:
    """
    Full pipeline: sample hyperparams → build DAG → build network → generate samples
    → assign roles → pool → discretize → split.
    """
    if hyper is None:
        hyper = sample_hyperparams(rng)

    dag = build_dag(hyper.V, hyper.M, rng, max_parents=HYPER_CONFIG["max_parents"])
    net = build_network(dag, hyper.d, hyper.kernel_size, hyper.gain, rng)
    feature_nodes, target_node = assign_roles(dag, hyper.n_features, rng)

    # Precompute PE matrix
    pe_matrix = positional_encoding_matrix(hyper.T, hyper.d)

    # Warmup: skip first kernel_size - 1 timesteps in output
    warmup = hyper.kernel_size - 1
    T_out = hyper.T - warmup

    N = hyper.n_samples
    n_feat = len(feature_nodes)

    X_all = np.zeros((N, n_feat, T_out), dtype=np.float64)
    y_scalars = np.zeros(N, dtype=np.float64)

    for s in range(N):
        root_values = sample_roots(
            hyper.M, hyper.T, hyper.d,
            hyper.init_type, hyper.root_std, hyper.root_a,
            pe_matrix, rng,
        )
        all_values = propagate_dag(root_values, dag, net)

        # Extract features: pool over d, then trim warmup
        for f_idx, node_id in enumerate(feature_nodes):
            series = pool_over_d(all_values[node_id], net.alpha_pool[node_id])
            X_all[s, f_idx, :] = series[warmup:]

        # Extract target: pool over d, trim warmup, then pool over T
        target_series = pool_over_d(all_values[target_node], net.alpha_pool[target_node])
        target_series = target_series[warmup:]
        y_scalars[s] = pool_over_T(target_series, net.alpha_pool_T)

    # Train/test split (80/20) — split BEFORE discretising so that
    # K-means centroids are fitted on train only (no leakage).
    n_train = max(1, int(0.8 * N))
    indices = np.arange(N)
    rng.shuffle(indices)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    # Discretize target: fit K-means on train, assign test to nearest centroid
    y_train_labels, centroids = discretize_kmeans(
        y_scalars[train_idx], hyper.n_classes, rng, return_centroids=True,
    )
    # Assign test labels to nearest train centroid
    dists_test = np.abs(y_scalars[test_idx, None] - centroids[None, :])
    y_test_labels = dists_test.argmin(axis=1).astype(np.int32)

    return GeneratedDataset(
        X_train=X_all[train_idx],
        X_test=X_all[test_idx],
        y_train=y_train_labels,
        y_test=y_test_labels,
        n_classes=hyper.n_classes,
        hyper=hyper,
        dag=dag,
        feature_nodes=feature_nodes,
        target_node=target_node,
    )


# ============================================================
# CLI
# ============================================================


def main():
    import argparse

    p = argparse.ArgumentParser(description="Generate DAG-based classification datasets")
    p.add_argument("--n-datasets", type=int, default=5, help="Number of datasets to generate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print-info", action="store_true", help="Print dataset info")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    for i in range(args.n_datasets):
        ds = generate_dataset(rng)
        if args.print_info:
            h = ds.hyper
            print(
                f"Dataset {i}: V={h.V} M={h.M} d={h.d} T={h.T} K={h.kernel_size} "
                f"init={h.init_type} n_feat={h.n_features} n_cls={h.n_classes} "
                f"n_samp={h.n_samples} | X_train={ds.X_train.shape} X_test={ds.X_test.shape} "
                f"y_train classes={np.unique(ds.y_train).tolist()}"
            )


if __name__ == "__main__":
    main()
