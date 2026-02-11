#!/usr/bin/env python3
"""
DAG + PE experiment: big-picture DAG (1 root -> 3 nodes -> 2 nodes), each node dim d.

- m_init is fixed (sampled once). At each t we set root = m_t = m_init + PE(t).
- Each internal node: input = concat of parent vectors (d * num_parents), output = d.
  One W (d, d*parents), one b (d), activation applied to each of the d components. No encoder in the DAG.
- Encoder (d -> 1, no activation) is used only after propagation, to get one scalar per node for visualization.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple

from generate_pe_series import (
    positional_encoding,
    sample_latent,
    apply_activation,
    ACTIVATION_CHOICES,
)

# DAG: layer sizes (fully connected between consecutive layers)
DAG_LAYERS = [1, 3, 2]  # 1 root, 3 middle, 2 bottom -> 6 nodes total


def xavier_std(in_dim: int, out_dim: int, gain: float = 1.0) -> float:
    return gain * np.sqrt(2.0 / (in_dim + out_dim))


def build_dag(
    d: int,
    layers: List[int],
    rng: np.random.Generator,
    gain: float = 1.0,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, str]], np.ndarray, float]:
    """
    Build DAG weights per node (except root).
    Each node: W (d, parent_dim), b (d), act_name.
    Returns: list of (W, b, act) for each non-root node in order; then w_enc (d,), b_enc.
    """
    node_params = []
    for layer_idx in range(1, len(layers)):
        n_parents = layers[layer_idx - 1]
        parent_dim = n_parents * d
        n_this = layers[layer_idx]
        for _ in range(n_this):
            std = xavier_std(parent_dim, d, gain)
            W = rng.normal(0, std, size=(d, parent_dim)).astype(np.float64)
            b = rng.normal(0, std, size=d).astype(np.float64)
            act = rng.choice(ACTIVATION_CHOICES).item()
            node_params.append((W, b, act))

    # Encoder: d -> 1, no activation
    std_enc = xavier_std(d, 1, gain)
    w_enc = rng.normal(0, std_enc, size=d).astype(np.float64)
    b_enc = float(rng.normal(0, std_enc))
    return node_params, w_enc, b_enc


def propagate_dag(
    m_t: np.ndarray,
    node_params: List[Tuple[np.ndarray, np.ndarray, str]],
    layers: List[int],
    d: int,
) -> List[np.ndarray]:
    """
    One forward pass: each node receives d*parents, applies W and b, then activation to get d outputs.
    No encoder here. Returns list of node vectors (each d,) in order: root, L1_0, L1_1, L1_2, L2_0, L2_1.
    """
    values_by_layer = [[m_t.copy()]]  # layer 0: root
    for layer_idx in range(1, len(layers)):
        n_this = layers[layer_idx]
        parent_values = values_by_layer[layer_idx - 1]
        parent_dim = len(parent_values) * d
        parent_concat = np.concatenate(parent_values, axis=0)
        layer_vals = []
        param_idx = sum(layers[1:layer_idx])  # number of nodes in previous layers (excluding root)
        for j in range(n_this):
            W, b, act = node_params[param_idx + j]
            out = apply_activation(W @ parent_concat + b, act)
            layer_vals.append(out)
        values_by_layer.append(layer_vals)

    return [v for layer in values_by_layer for v in layer]


def encode_node_random(value_d: np.ndarray, w_enc: np.ndarray, b_enc: float) -> float:
    """Encode d-dim vector to scalar with learned weights (no activation)."""
    return float(np.dot(w_enc, value_d) + b_enc)


def encode_node_sum(value_d: np.ndarray) -> float:
    """Encode d-dim vector to scalar as simple sum of dimensions."""
    return float(np.sum(value_d))


def run_experiment(
    d: int,
    T: int,
    rng: np.random.Generator,
    *,
    latent_init: str = "normal",
    std: float = 0.5,
    a: float = 1.0,
    gain: float = 1.0,
    layers: List[int] | None = None,
    encoder: str = "random",
) -> Tuple[np.ndarray, List[str]]:
    """
    Run DAG+PE experiment. Returns (series, node_labels).
    series shape (T, n_nodes). encoder: "random" (w,b) or "sum" (sum of d dimensions).
    """
    layers = layers or DAG_LAYERS
    n_nodes = sum(layers)
    m_init = sample_latent(d, latent_init, rng, std=std, a=a)  # fixed for all t
    node_params, w_enc, b_enc = build_dag(d, layers, rng, gain=gain)

    use_sum = encoder == "sum"

    series = np.zeros((T, n_nodes))
    for t in range(T):
        pe_t = positional_encoding(t, d)
        m_t = m_init + pe_t  # root at t (m_init fixed)
        node_values = propagate_dag(m_t, node_params, layers, d)  # DAG only: W, b, activation per node
        for i, v in enumerate(node_values):
            series[t, i] = encode_node_sum(v) if use_sum else encode_node_random(v, w_enc, b_enc)

    labels = ["root"]
    for layer_idx in range(1, len(layers)):
        for j in range(layers[layer_idx]):
            labels.append(f"L{layer_idx}_{j}")
    return series, labels


def main():
    p = argparse.ArgumentParser(description="DAG+PE: one series per node, encoder d->1")
    p.add_argument("--d", type=int, default=8, help="Node dimension")
    p.add_argument("--T", type=int, default=200, help="Time length")
    p.add_argument("--init", choices=("normal", "uniform"), default="normal")
    p.add_argument("--std", type=float, default=0.5)
    p.add_argument("--a", type=float, default=1.0)
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="results/dag_pe.png")
    p.add_argument("--encoder", choices=("random", "sum"), default="random",
                    help="Encoder: random (w,b) or sum (sum of d dimensions)")
    p.add_argument("--d-values", type=int, nargs="+", default=None, help="One PNG per d (T fixed 200)")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out).parent
    out_stem = Path(args.out).stem
    T = args.T if args.d_values is None else 200

    if args.d_values is not None:
        for d in args.d_values:
            series, labels = run_experiment(
                d=d, T=T, rng=rng,
                latent_init=args.init, std=args.std, a=args.a, gain=args.gain,
                encoder=args.encoder,
            )
            path = out_dir / f"{out_stem}_d{d}.png"
            _plot(series, labels, d, T, path, args.encoder)
        return

    series, labels = run_experiment(
        d=args.d, T=T, rng=rng,
        latent_init=args.init, std=args.std, a=args.a, gain=args.gain,
        encoder=args.encoder,
    )
    path = Path(args.out)
    _plot(series, labels, args.d, T, path, args.encoder)


def _plot(series: np.ndarray, labels: List[str], d: int, T: int, path: Path, encoder: str = "random") -> None:
    n_nodes = series.shape[1]
    t = np.arange(T)
    fig, axes = plt.subplots(n_nodes, 1, figsize=(8, 1.5 * n_nodes), sharex=True)
    if n_nodes == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(t, series[:, i], color="tab:blue", linewidth=0.9)
        ax.set_ylabel(labels[i], fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("t")
    enc_label = "encoder=sum" if encoder == "sum" else "encoder=random"
    fig.suptitle(f"DAG+PE (d={d}, T={T}, {enc_label}) — one series per node", fontsize=11, fontweight="bold")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", path)


if __name__ == "__main__":
    main()
