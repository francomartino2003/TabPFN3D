#!/usr/bin/env python3
"""
DAG + PE with scalar internal nodes. Roots: 3 different m (same p_t), each (T, d).
L1: each node j uses root j only; combines min(m_t) and mean(m_t) -> scalar, then conv 1D causal -> activation.
L2, L3: linear combination of parents (per t) -> conv 1D causal -> activation.
No encoder. Visualize one series per internal node (root not plotted). One PNG per d.
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

DAG_LAYERS = [3, 3, 2, 2]  # 3 roots (each dim d, different m same p_t), L1: 3 nodes, L2: 2, L3: 2
DEFAULT_KERNEL_SIZE = 5


def xavier_std(in_dim: int, out_dim: int, gain: float = 1.0) -> float:
    return gain * np.sqrt(2.0 / (in_dim + out_dim))


def causal_conv1d_scalar(series: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Causal 1D convolution on a single series.
    series: (T,), kernel: (K,). output[t] = sum_{k=0}^{K-1} kernel[k] * series[t-k] (series[t-k]=0 if t-k<0).
    Returns (T,).
    """
    T = series.size
    K = kernel.size
    padded = np.zeros(T + K - 1, dtype=series.dtype)
    padded[K - 1 :] = series
    out = np.convolve(padded, kernel[::-1], mode="valid")
    assert out.shape[0] == T
    return out


# L1: (w_min, w_mean, b, kernel, act). L2/L3: (W, b, kernel, act).
Layer1Param = Tuple[float, float, float, np.ndarray, str]
LinearParam = Tuple[np.ndarray, float, np.ndarray, str]


def build_dag_scalar(
    d: int,
    layers: List[int],
    rng: np.random.Generator,
    gain: float = 1.0,
    kernel_size: int = DEFAULT_KERNEL_SIZE,
) -> Tuple[List[Layer1Param], List[LinearParam]]:
    """
    L1: each node combines min(m_t) and mean(m_t) of its root -> (w_min, w_mean, b, kernel, act).
    L2/L3: linear comb of parents -> (W, b, kernel, act).
    Returns (layer1_params, rest_params).
    """
    layer1_params: List[Layer1Param] = []
    rest_params: List[LinearParam] = []

    K = max(1, kernel_size)
    conv_std = xavier_std(K, 1, gain)
    std_2in = gain * np.sqrt(2.0 / (2 + 1))  # 2 inputs (min, mean) -> 1 output

    n_l1 = layers[1]
    for j in range(n_l1):
        w_min = float(rng.normal(0, std_2in))
        w_mean = float(rng.normal(0, std_2in))
        b = float(rng.normal(0, std_2in))
        kernel = rng.normal(0, conv_std, size=K).astype(np.float64)
        act = rng.choice(ACTIVATION_CHOICES).item()
        layer1_params.append((w_min, w_mean, b, kernel, act))

    for layer_idx in range(2, len(layers)):
        n_parents = layers[layer_idx - 1]
        n_this = layers[layer_idx]
        std = xavier_std(n_parents, 1, gain)
        for _ in range(n_this):
            W = rng.normal(0, std, size=(1, n_parents)).astype(np.float64)
            b = float(rng.normal(0, std))
            kernel = rng.normal(0, conv_std, size=K).astype(np.float64)
            act = rng.choice(ACTIVATION_CHOICES).item()
            rest_params.append((W, b, kernel, act))

    return layer1_params, rest_params


def propagate_dag_scalar(
    roots: np.ndarray,
    layer1_params: List[Layer1Param],
    rest_params: List[LinearParam],
    layers: List[int],
) -> np.ndarray:
    """
    roots: (3, T, d) — three different m + same p_t.
    L1: each node j uses roots[j], combines min and mean at each t -> conv -> activation.
    L2/L3: linear comb of parent series -> conv -> activation.
    Returns (T, n_internal).
    """
    T = roots.shape[1]
    all_layer_series = []

    # Layer 1: min/mean per root
    n_l1 = layers[1]
    layer_series = np.zeros((T, n_l1), dtype=np.float64)
    for j in range(n_l1):
        w_min, w_mean, b, kernel, act = layer1_params[j]
        # At each t: min(roots[j,t,:]), mean(roots[j,t,:])
        m_t = roots[j, :, :]  # (T, d)
        z = w_min * m_t.min(axis=1) + w_mean * m_t.mean(axis=1) + b
        z = causal_conv1d_scalar(z, kernel)
        layer_series[:, j] = apply_activation(z, act)
    all_layer_series.append(layer_series)
    parent_series = layer_series

    # Layers 2, 3: linear -> conv -> activation
    rest_idx = 0
    for layer_idx in range(2, len(layers)):
        n_this = layers[layer_idx]
        layer_series = np.zeros((T, n_this), dtype=np.float64)
        for j in range(n_this):
            W, b, kernel, act = rest_params[rest_idx + j]
            z = (parent_series @ W.T).ravel() + b
            z = causal_conv1d_scalar(z, kernel)
            layer_series[:, j] = apply_activation(z, act)
        rest_idx += n_this
        all_layer_series.append(layer_series)
        parent_series = layer_series

    return np.hstack(all_layer_series)


def run_experiment(
    d: int,
    T: int,
    rng: np.random.Generator,
    *,
    latent_init: str = "normal",
    std: float = 0.5,
    a: float = 1.0,
    gain: float = 1.0,
    kernel_size: int = DEFAULT_KERNEL_SIZE,
    layers: List[int] | None = None,
) -> Tuple[np.ndarray, List[str]]:
    """Returns series (T, n_internal). Three roots (different m, same p_t); L1 = min/mean comb."""
    layers = layers or DAG_LAYERS
    n_internal = sum(layers[1:])
    n_roots = layers[0]  # 3 roots (one per L1 node)
    m_inits = [sample_latent(d, latent_init, rng, std=std, a=a) for _ in range(n_roots)]
    layer1_params, rest_params = build_dag_scalar(
        d, layers, rng, gain=gain, kernel_size=kernel_size
    )

    roots = np.zeros((n_roots, T, d), dtype=np.float64)
    for t in range(T):
        pe_t = positional_encoding(t, d)
        for j in range(n_roots):
            roots[j, t, :] = m_inits[j] + pe_t
    series = propagate_dag_scalar(roots, layer1_params, rest_params, layers)

    labels = []
    for layer_idx in range(1, len(layers)):
        for j in range(layers[layer_idx]):
            labels.append(f"L{layer_idx}_{j}")
    return series, labels


def run_experiment_multi_m(
    d: int,
    T: int,
    n_latents: int,
    rng: np.random.Generator,
    *,
    latent_init: str = "normal",
    std: float = 0.5,
    a: float = 1.0,
    gain: float = 1.0,
    kernel_size: int = DEFAULT_KERNEL_SIZE,
    layers: List[int] | None = None,
) -> Tuple[np.ndarray, List[str]]:
    """Same DAG; n_latents different triplets of m (each col = 3 new m, same p_t). Returns (n_latents, T, n_internal), labels."""
    layers = layers or DAG_LAYERS
    n_internal = sum(layers[1:])
    n_roots = layers[0]
    layer1_params, rest_params = build_dag_scalar(
        d, layers, rng, gain=gain, kernel_size=kernel_size
    )

    out = np.zeros((n_latents, T, n_internal))
    for k in range(n_latents):
        m_inits = [sample_latent(d, latent_init, rng, std=std, a=a) for _ in range(n_roots)]
        roots = np.zeros((n_roots, T, d), dtype=np.float64)
        for t in range(T):
            pe_t = positional_encoding(t, d)
            for j in range(n_roots):
                roots[j, t, :] = m_inits[j] + pe_t
        out[k, :, :] = propagate_dag_scalar(roots, layer1_params, rest_params, layers)

    labels = []
    for layer_idx in range(1, len(layers)):
        for j in range(layers[layer_idx]):
            labels.append(f"L{layer_idx}_{j}")
    return out, labels


def main():
    p = argparse.ArgumentParser(description="DAG+PE scalar nodes (root dim d, internal dim 1), no encoder")
    p.add_argument("--d", type=int, default=8, help="Root latent dimension")
    p.add_argument("--T", type=int, default=200, help="Time length")
    p.add_argument("--init", choices=("normal", "uniform"), default="normal")
    p.add_argument("--std", type=float, default=0.5)
    p.add_argument("--a", type=float, default=1.0)
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="results/dag_pe_scalar.png")
    p.add_argument("--d-values", type=int, nargs="+", default=None, help="One PNG per d (use --T for length)")
    p.add_argument("--n-latents", type=int, default=6, help="When using --d-values: columns = different m (one PNG per d)")
    p.add_argument("--kernel-size", type=int, default=DEFAULT_KERNEL_SIZE, help="Tamaño del kernel de la conv 1D causal por nodo")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out).parent
    out_stem = Path(args.out).stem
    T = args.T
    n_latents = args.n_latents

    if args.d_values is not None:
        # Resultados del mismo experimento dentro de una carpeta (out_dir/out_stem/)
        experiment_dir = out_dir / out_stem
        experiment_dir.mkdir(parents=True, exist_ok=True)
        for d in args.d_values:
            series_3d, labels = run_experiment_multi_m(
                d=d, T=T, n_latents=n_latents, rng=rng,
                latent_init=args.init, std=args.std, a=args.a, gain=args.gain,
                kernel_size=args.kernel_size,
            )
            path = experiment_dir / f"{out_stem}_d{d}.png"
            _plot_multi_m(series_3d, labels, d, T, n_latents, path)
        return

    series, labels = run_experiment(
        d=args.d, T=T, rng=rng,
        latent_init=args.init, std=args.std, a=args.a, gain=args.gain,
        kernel_size=args.kernel_size,
    )
    _plot(series, labels, args.d, T, Path(args.out))


def _plot(series: np.ndarray, labels: List[str], d: int, T: int, path: Path) -> None:
    n_nodes = series.shape[1]
    t = np.arange(T)
    fig, axes = plt.subplots(n_nodes, 1, figsize=(8, 1.4 * n_nodes), sharex=True)
    if n_nodes == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(t, series[:, i], color="tab:blue", linewidth=0.9)
        ax.set_ylabel(labels[i], fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("t")
    fig.suptitle(
        f"DAG+PE scalar (d={d}, T={T}): 3 roots (distinct m, same p_t) | L1=min/mean→conv→act, L2/L3=lin→conv→act",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", path)


def _plot_multi_m(
    series_3d: np.ndarray,
    labels: List[str],
    d: int,
    T: int,
    n_latents: int,
    path: Path,
) -> None:
    """series_3d: (n_latents, T, n_internal). Rows = nodes, cols = different m."""
    n_latents_, _T, n_nodes = series_3d.shape
    assert n_latents_ == n_latents and _T == T
    t = np.arange(T)
    # Figura más ancha y alargada: más pulgadas por columna y por fila
    fig, axes = plt.subplots(
        n_nodes, n_latents, figsize=(5.8 * n_latents, 2.5 * n_nodes), sharex=True, sharey=False
    )
    if n_nodes == 1:
        axes = axes.reshape(1, -1)
    if n_latents == 1:
        axes = axes.reshape(-1, 1)
    for i in range(n_nodes):
        for j in range(n_latents):
            ax = axes[i, j]
            ax.plot(t, series_3d[j, :, i], color="tab:blue", linewidth=0.8)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(f"m_{j+1}", fontsize=9)
            if j == 0:
                ax.set_ylabel(labels[i], fontsize=9)
            if i == n_nodes - 1:
                ax.set_xlabel("t", fontsize=8)
    fig.suptitle(
        f"DAG+PE scalar — 3 roots (distinct m), L1=min/mean→conv→act — d={d}, T={T} | cols = {n_latents} triplets m",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", path)


if __name__ == "__main__":
    main()
