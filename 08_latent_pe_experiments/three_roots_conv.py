#!/usr/bin/env python3
"""
Experimento: 3 nodos raíz (R^{T×d}) + capas L1, L2, L3 (cada una 3 nodos R^{T×d}).

- Raíces: para cada raíz i, m_i ~ sample(d); root_i[t] = m_i + PE(t). Shape (3, T, d).
- L1: padres = 3 raíces. Por nodo: comb lineal por t → conv causal 1D → activación.
- L2: padres = 3 L1. Mismo procedimiento.
- L3: padres = 3 L2. Mismo procedimiento.
- Visualización: combinación lineal al azar entre mean y max pooling → 12 series (3 raíz + 3 L1 + 3 L2 + 3 L3).
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

N_ROOTS = 3
N_PER_LAYER = 3  # L1, L2, L3 tienen 3 nodos cada una
N_LAYERS = 3  # L1, L2, L3


def xavier_std(in_dim: int, out_dim: int, gain: float = 1.0) -> float:
    return gain * np.sqrt(2.0 / (in_dim + out_dim))


def causal_conv1d_depthwise(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Causal 1D convolution (depthwise): mezcla en tiempo por canal.
    x: (T, d), kernel: (K, d).
    output[t, c] = sum_{k=0}^{min(K-1,t)} kernel[k, c] * x[t-k, c].
    """
    T, d = x.shape
    K = kernel.shape[0]
    out = np.zeros_like(x)
    # Pad izquierdo para poder indexar x[t-k] con t-k >= 0
    padded = np.zeros((T + K - 1, d), dtype=x.dtype)
    padded[K - 1 :] = x
    for c in range(d):
        # Correlación causal: out[t,c] = sum_k kernel[k,c] * x[t-k,c]
        rev_k = kernel[::-1, c]  # kernel[K-1], kernel[K-2], ..., kernel[0]
        out[:, c] = np.convolve(padded[:, c], rev_k, mode="valid")
    return out


def _compute_layer(
    parents: np.ndarray,
    w_comb: np.ndarray,
    kernels: np.ndarray,
    activations: List[str],
    T: int,
    d: int,
) -> np.ndarray:
    """parents: (n_parents, T, d). w_comb: (n_nodes, n_parents). kernels: (n_nodes, K, d)."""
    n_nodes = w_comb.shape[0]
    n_parents = parents.shape[0]
    out = np.zeros((n_nodes, T, d), dtype=np.float64)
    for j in range(n_nodes):
        combined = np.zeros((T, d), dtype=np.float64)
        for t in range(T):
            for p in range(n_parents):
                combined[t, :] += w_comb[j, p] * parents[p, t, :]
        conv_out = causal_conv1d_depthwise(combined, kernels[j])
        out[j, :, :] = apply_activation(conv_out, activations[j])
    return out


def _forward_to_series(
    roots: np.ndarray,
    w_comb_1: np.ndarray,
    kernels_1: np.ndarray,
    acts_1: List[str],
    w_comb_2: np.ndarray,
    kernels_2: np.ndarray,
    acts_2: List[str],
    w_comb_3: np.ndarray,
    kernels_3: np.ndarray,
    acts_3: List[str],
    alpha: np.ndarray,
    T: int,
    d: int,
) -> np.ndarray:
    """Dado roots (3, T, d) y parámetros de la red, devuelve series (T, 12)."""
    n_total_nodes = N_ROOTS + N_LAYERS * N_PER_LAYER
    layer1 = _compute_layer(roots, w_comb_1, kernels_1, acts_1, T, d)
    layer2 = _compute_layer(layer1, w_comb_2, kernels_2, acts_2, T, d)
    layer3 = _compute_layer(layer2, w_comb_3, kernels_3, acts_3, T, d)
    series = np.zeros((T, n_total_nodes), dtype=np.float64)
    idx = 0
    for i in range(N_ROOTS):
        m, x = roots[i].mean(axis=1), roots[i].max(axis=1)
        series[:, idx] = alpha[idx] * m + (1 - alpha[idx]) * x
        idx += 1
    for j in range(N_PER_LAYER):
        m, x = layer1[j].mean(axis=1), layer1[j].max(axis=1)
        series[:, idx] = alpha[idx] * m + (1 - alpha[idx]) * x
        idx += 1
    for j in range(N_PER_LAYER):
        m, x = layer2[j].mean(axis=1), layer2[j].max(axis=1)
        series[:, idx] = alpha[idx] * m + (1 - alpha[idx]) * x
        idx += 1
    for j in range(N_PER_LAYER):
        m, x = layer3[j].mean(axis=1), layer3[j].max(axis=1)
        series[:, idx] = alpha[idx] * m + (1 - alpha[idx]) * x
        idx += 1
    return series


def run_experiment(
    d: int,
    T: int,
    rng: np.random.Generator,
    *,
    latent_init: str = "normal",
    root_std: float = 0.5,
    a: float = 1.0,
    kernel_size: int = 5,
    gain: float = 1.0,
) -> Tuple[np.ndarray, List[str]]:
    """
    Devuelve series escalares (T, 12) y labels para 3 raíz + 3 L1 + 3 L2 + 3 L3.
    Cada serie es alpha*mean + (1-alpha)*max sobre d, con alpha muestreado al azar por nodo.
    """
    K = kernel_size
    std_lin = xavier_std(N_PER_LAYER, 1, gain)
    std_conv = xavier_std(K * d, d, gain)
    n_total_nodes = N_ROOTS + N_LAYERS * N_PER_LAYER  # 3 + 9 = 12

    # --- Raíces: (3, T, d) ---
    roots = np.zeros((N_ROOTS, T, d), dtype=np.float64)
    for i in range(N_ROOTS):
        m_i = sample_latent(d, latent_init, rng, std=root_std, a=a)
        for t in range(T):
            roots[i, t, :] = m_i + positional_encoding(t, d)

    # --- Red (L1, L2, L3) y alpha ---
    w_comb_1 = rng.normal(0, std_lin, size=(N_PER_LAYER, N_ROOTS)).astype(np.float64)
    kernels_1 = rng.normal(0, std_conv, size=(N_PER_LAYER, K, d)).astype(np.float64)
    acts_1 = [rng.choice(ACTIVATION_CHOICES).item() for _ in range(N_PER_LAYER)]
    w_comb_2 = rng.normal(0, std_lin, size=(N_PER_LAYER, N_PER_LAYER)).astype(np.float64)
    kernels_2 = rng.normal(0, std_conv, size=(N_PER_LAYER, K, d)).astype(np.float64)
    acts_2 = [rng.choice(ACTIVATION_CHOICES).item() for _ in range(N_PER_LAYER)]
    w_comb_3 = rng.normal(0, std_lin, size=(N_PER_LAYER, N_PER_LAYER)).astype(np.float64)
    kernels_3 = rng.normal(0, std_conv, size=(N_PER_LAYER, K, d)).astype(np.float64)
    acts_3 = [rng.choice(ACTIVATION_CHOICES).item() for _ in range(N_PER_LAYER)]
    alpha = rng.uniform(0, 1, size=n_total_nodes).astype(np.float64)

    series = _forward_to_series(
        roots, w_comb_1, kernels_1, acts_1, w_comb_2, kernels_2, acts_2,
        w_comb_3, kernels_3, acts_3, alpha, T, d,
    )
    labels = (
        [f"root_{i}" for i in range(N_ROOTS)]
        + [f"L1_{j}" for j in range(N_PER_LAYER)]
        + [f"L2_{j}" for j in range(N_PER_LAYER)]
        + [f"L3_{j}" for j in range(N_PER_LAYER)]
    )
    return series, labels


def run_experiment_3cols(
    d: int,
    T: int,
    n_columns: int,
    rng: np.random.Generator,
    *,
    latent_init: str = "normal",
    root_std: float = 0.5,
    a: float = 1.0,
    kernel_size: int = 5,
    gain: float = 1.0,
) -> Tuple[np.ndarray, List[str]]:
    """
    Misma red (L1, L2, L3 fijos); en cada columna se reinicializan solo las raíces.
    Devuelve (n_columns, T, 12) y labels.
    """
    K = kernel_size
    std_lin = xavier_std(N_PER_LAYER, 1, gain)
    std_conv = xavier_std(K * d, d, gain)
    n_total_nodes = N_ROOTS + N_LAYERS * N_PER_LAYER

    # --- Red fija (una sola vez) ---
    w_comb_1 = rng.normal(0, std_lin, size=(N_PER_LAYER, N_ROOTS)).astype(np.float64)
    kernels_1 = rng.normal(0, std_conv, size=(N_PER_LAYER, K, d)).astype(np.float64)
    acts_1 = [rng.choice(ACTIVATION_CHOICES).item() for _ in range(N_PER_LAYER)]
    w_comb_2 = rng.normal(0, std_lin, size=(N_PER_LAYER, N_PER_LAYER)).astype(np.float64)
    kernels_2 = rng.normal(0, std_conv, size=(N_PER_LAYER, K, d)).astype(np.float64)
    acts_2 = [rng.choice(ACTIVATION_CHOICES).item() for _ in range(N_PER_LAYER)]
    w_comb_3 = rng.normal(0, std_lin, size=(N_PER_LAYER, N_PER_LAYER)).astype(np.float64)
    kernels_3 = rng.normal(0, std_conv, size=(N_PER_LAYER, K, d)).astype(np.float64)
    acts_3 = [rng.choice(ACTIVATION_CHOICES).item() for _ in range(N_PER_LAYER)]
    alpha = rng.uniform(0, 1, size=n_total_nodes).astype(np.float64)

    out = np.zeros((n_columns, T, n_total_nodes), dtype=np.float64)
    for c in range(n_columns):
        roots = np.zeros((N_ROOTS, T, d), dtype=np.float64)
        for i in range(N_ROOTS):
            m_i = sample_latent(d, latent_init, rng, std=root_std, a=a)
            for t in range(T):
                roots[i, t, :] = m_i + positional_encoding(t, d)
        out[c, :, :] = _forward_to_series(
            roots, w_comb_1, kernels_1, acts_1, w_comb_2, kernels_2, acts_2,
            w_comb_3, kernels_3, acts_3, alpha, T, d,
        )
    labels = (
        [f"root_{i}" for i in range(N_ROOTS)]
        + [f"L1_{j}" for j in range(N_PER_LAYER)]
        + [f"L2_{j}" for j in range(N_PER_LAYER)]
        + [f"L3_{j}" for j in range(N_PER_LAYER)]
    )
    return out, labels


def main():
    p = argparse.ArgumentParser(
        description="3 raíces (T×d) + L1/L2/L3 (comb lineal + conv causal + act), viz: comb lineal al azar mean+max"
    )
    p.add_argument("--d", type=int, default=8, help="Dimensión d de raíces y capa 1")
    p.add_argument("--T", type=int, default=100, help="Longitud temporal")
    p.add_argument("--init", choices=("normal", "uniform"), default="normal")
    p.add_argument(
        "--root-std",
        type=float,
        default=0.5,
        help="Desviación estándar de N(0, root_std) para samplear los vectores iniciales m_i de las raíces (init=normal)",
    )
    p.add_argument("--a", type=float, default=1.0, help="Semiancho [-a,a] para init=uniform de las raíces")
    p.add_argument("--kernel-size", type=int, default=5, help="Tamaño kernel conv causal 1D")
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="results/three_roots_conv.png")
    p.add_argument(
        "--d-values",
        type=int,
        nargs="+",
        default=None,
        help="Un PNG por d (guarda en out_dir/out_stem/)",
    )
    p.add_argument(
        "--n-columns",
        type=int,
        default=1,
        help="Si >1: misma red, cada columna con raíces reinicializadas (mismo ancho total de figura)",
    )
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out).parent
    out_stem = Path(args.out).stem
    T = args.T

    n_cols = args.n_columns

    if args.d_values is not None:
        experiment_dir = out_dir / out_stem
        experiment_dir.mkdir(parents=True, exist_ok=True)
        for d in args.d_values:
            if n_cols > 1:
                series_3d, labels = run_experiment_3cols(
                    d=d, T=T, n_columns=n_cols, rng=rng,
                    latent_init=args.init, root_std=args.root_std, a=args.a,
                    kernel_size=args.kernel_size, gain=args.gain,
                )
                path = experiment_dir / f"{out_stem}_d{d}.png"
                _plot_3cols(series_3d, labels, d, T, n_cols, path, args.kernel_size)
            else:
                series, labels = run_experiment(
                    d=d, T=T, rng=rng,
                    latent_init=args.init, root_std=args.root_std, a=args.a,
                    kernel_size=args.kernel_size, gain=args.gain,
                )
                path = experiment_dir / f"{out_stem}_d{d}.png"
                _plot(series, labels, d, T, path, args.kernel_size)
        return

    if n_cols > 1:
        series_3d, labels = run_experiment_3cols(
            d=args.d, T=T, n_columns=n_cols, rng=rng,
            latent_init=args.init, root_std=args.root_std, a=args.a,
            kernel_size=args.kernel_size, gain=args.gain,
        )
        _plot_3cols(series_3d, labels, args.d, T, n_cols, Path(args.out), args.kernel_size)
    else:
        series, labels = run_experiment(
            d=args.d, T=T, rng=rng,
            latent_init=args.init, root_std=args.root_std, a=args.a,
            kernel_size=args.kernel_size, gain=args.gain,
        )
        _plot(series, labels, args.d, T, Path(args.out), args.kernel_size)


def _plot(
    series: np.ndarray,
    labels: List[str],
    d: int,
    T: int,
    path: Path,
    kernel_size: int = 5,
) -> None:
    warmup = kernel_size - 1  # recortar transitorio inicial de la conv causal
    series = series[warmup:, :]
    t = np.arange(warmup, T)
    n_nodes = series.shape[1]
    fig, axes = plt.subplots(n_nodes, 1, figsize=(10, 1.2 * n_nodes), sharex=True)
    if n_nodes == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(t, series[:, i], color="tab:blue", linewidth=0.9)
        ax.set_ylabel(labels[i], fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("t")
    fig.suptitle(
        f"3 raíces + L1/L2/L3 (comb lin + conv causal + act), d={d}, T={T} — α·mean+(1-α)·max (sin warmup)",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", path)


def _plot_3cols(
    series_3d: np.ndarray,
    labels: List[str],
    d: int,
    T: int,
    n_columns: int,
    path: Path,
    kernel_size: int = 5,
) -> None:
    """series_3d: (n_columns, T, n_nodes). 12 filas × n_columns columnas; mismo ancho total 10."""
    warmup = kernel_size - 1  # recortar transitorio inicial de la conv causal
    series_3d = series_3d[:, warmup:, :]
    t = np.arange(warmup, T)
    n_cols, _T, n_nodes = series_3d.shape
    assert n_cols == n_columns and _T == T - warmup
    fig, axes = plt.subplots(
        n_nodes,
        n_columns,
        figsize=(10, 1.2 * n_nodes),
        sharex=True,
        sharey=False,
    )
    if n_nodes == 1:
        axes = axes.reshape(1, -1)
    if n_columns == 1:
        axes = axes.reshape(-1, 1)
    for i in range(n_nodes):
        for c in range(n_columns):
            ax = axes[i, c]
            ax.plot(t, series_3d[c, :, i], color="tab:blue", linewidth=0.8)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(f"init_{c+1}", fontsize=9)
            if c == 0:
                ax.set_ylabel(labels[i], fontsize=9)
            if i == n_nodes - 1:
                ax.set_xlabel("t", fontsize=8)
    fig.suptitle(
        f"3 raíces + L1/L2/L3, d={d}, T={T} — {n_columns} cols (raíces reinic.), sin warmup",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", path)


if __name__ == "__main__":
    main()
