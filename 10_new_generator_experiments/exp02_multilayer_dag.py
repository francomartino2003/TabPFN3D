"""
Experiment 02: Multi-layer DAG of series  [3, 2, 3].

Layer 0 (roots): 3 series, each built like exp01:
    sample m ~ N(0,1)^d  →  replicate T+3(K-1) times  →  add PE
    → causal conv (d→1) + activation  →  output length T+2(K-1)
Layer 1: 2 series, fully connected to layer 0.
    Each node receives 3 parent series stacked → (3, T+2(K-1))
    → causal conv (3→1) + activation → output length T+(K-1)
Layer 2: 3 series, fully connected to layer 1.
    Each node receives 2 parent series stacked → (2, T+(K-1))
    → causal conv (2→1) + activation → output length T

All convolutions and activations are sampled once (fixed by seed).
Only the root m vectors are re-sampled per propagation.

Intermediate nodes receive their parents' output (length T), but we need
length T+K-1 as input to their conv.  To achieve this WITHOUT padding,
every series is produced at length T + total_extra, where total_extra
accounts for the K-1 shrinkage at every downstream layer.  Roots produce
T + 2*(K-1) steps; layer 1 nodes produce T + (K-1) steps; layer 2 produces T.

Visualisation: one subplot per node (3+2+3 = 8), each showing n_prop
overlaid propagations (different root m's).

Usage:
  python exp02_multilayer_dag.py [--d 8] [--T 200] [--kernel-size 5] [--seed 0] [--n-prop 5]
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

# Reuse building blocks from exp01
from exp01_pe_causal_conv import (
    ACTIVATION_CHOICES,
    apply_activation,
    apply_causal_conv,
    build_causal_conv,
    positional_encoding,
)

# Architecture: nodes per layer
LAYERS = [3, 2, 3]


# ── Generator ──────────────────────────────────────────────────────────────────

class MultiLayerDAGGenerator:
    """
    Multi-layer fully-connected DAG of 1D series.

    Seed fixes all convolution kernels and activations.
    Each propagation re-samples only the root m vectors.
    """

    def __init__(self, d: int, T: int, kernel_size: int, seed: int,
                 layers: list[int] | None = None):
        self.d = d
        self.T = T
        self.K = kernel_size
        self.seed = seed
        self.layers = layers or list(LAYERS)
        n_layers = len(self.layers)

        gen_rng = np.random.default_rng(seed)

        # ── Roots (layer 0): each has its own conv d→1 and activation ──
        n_roots = self.layers[0]
        self.root_kernels = []    # list of (d, K)
        self.root_acts = []       # list of str
        for _ in range(n_roots):
            self.root_kernels.append(build_causal_conv(d, kernel_size, gen_rng))
            self.root_acts.append(gen_rng.choice(ACTIVATION_CHOICES))

        # ── Internal layers: each node has conv (n_parents→1) and activation ──
        self.internal_kernels = []  # internal_kernels[l][j] = (n_parents, K)
        self.internal_acts = []     # internal_acts[l][j] = str
        for l_idx in range(1, n_layers):
            n_parents = self.layers[l_idx - 1]
            n_nodes = self.layers[l_idx]
            layer_kernels = []
            layer_acts = []
            for _ in range(n_nodes):
                layer_kernels.append(build_causal_conv(n_parents, kernel_size, gen_rng))
                layer_acts.append(gen_rng.choice(ACTIVATION_CHOICES))
            self.internal_kernels.append(layer_kernels)
            self.internal_acts.append(layer_acts)

        # ── PE for roots ──
        # Roots must produce long enough output so downstream layers can apply
        # their convs without padding.  Each layer shrinks by K-1.
        # Layers after roots: n_layers - 1 internal layers.
        n_internal = n_layers - 1
        self.root_T_in = T + n_internal * (kernel_size - 1) + (kernel_size - 1)
        # root_T_in  =  T  +  (n_internal + 1)*(K-1)
        # root output length = root_T_in - (K-1) = T + n_internal*(K-1)
        self.pe = positional_encoding(self.root_T_in, d)  # (d, root_T_in)

        # RNG for m sampling
        self.sample_rng = np.random.default_rng(gen_rng.integers(0, 2**62))

    def propagate(self) -> list[list[np.ndarray]]:
        """
        Run one forward pass (one set of root m vectors).

        Returns:
            all_series[layer_idx][node_idx] = 1D array of length T_l
            where T_l = T + (n_internal - layer_idx) * (K-1) for internal
            layers, and T for the last layer.
        """
        K = self.K
        n_internal = len(self.layers) - 1

        # ── Layer 0: roots ──
        root_series = []
        for j in range(self.layers[0]):
            m = self.sample_rng.normal(0, 1, size=(self.d,))
            m_rep = np.tile(m[:, None], (1, self.root_T_in))   # (d, root_T_in)
            x = m_rep + self.pe                                 # (d, root_T_in)
            y = apply_causal_conv(x, self.root_kernels[j])      # (root_T_in - K + 1,)
            y = apply_activation(y, self.root_acts[j])
            root_series.append(y)

        all_series = [root_series]

        # ── Internal layers ──
        prev_series = root_series
        for l_idx in range(n_internal):
            layer_series = []
            for j in range(self.layers[l_idx + 1]):
                # Stack parents: (n_parents, L_parent)
                parent_stack = np.stack(prev_series, axis=0)
                y = apply_causal_conv(parent_stack, self.internal_kernels[l_idx][j])
                y = apply_activation(y, self.internal_acts[l_idx][j])
                layer_series.append(y)
            all_series.append(layer_series)
            prev_series = layer_series

        return all_series

    def propagate_batch(self, n: int) -> list[list[np.ndarray]]:
        """
        Run n propagations.

        Returns:
            batch[layer_idx][node_idx] = (n, T_l)
        """
        runs = [self.propagate() for _ in range(n)]
        # Restructure
        n_layers = len(self.layers)
        batch = []
        for l_idx in range(n_layers):
            layer_batch = []
            for j in range(self.layers[l_idx]):
                stacked = np.stack([runs[i][l_idx][j] for i in range(n)], axis=0)
                layer_batch.append(stacked)
            batch.append(layer_batch)
        return batch

    def node_info(self, l_idx: int, j: int) -> str:
        """Short description of a node."""
        if l_idx == 0:
            return f'L0 root {j}  act={self.root_acts[j]}'
        return f'L{l_idx} node {j}  act={self.internal_acts[l_idx-1][j]}'


# ── Visualisation ──────────────────────────────────────────────────────────────

def visualize_dag(gen: MultiLayerDAGGenerator, n_prop: int,
                  save_path: str | None = None):
    """
    One subplot per node, each showing n_prop overlaid series.
    """
    batch = gen.propagate_batch(n_prop)

    total_nodes = sum(gen.layers)
    n_cols = max(gen.layers)
    n_rows = len(gen.layers)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows),
                             squeeze=False)
    cmap = plt.cm.tab10

    for l_idx in range(n_rows):
        n_nodes = gen.layers[l_idx]
        for j in range(n_cols):
            ax = axes[l_idx][j]
            if j >= n_nodes:
                ax.set_visible(False)
                continue
            series_mat = batch[l_idx][j]  # (n_prop, T_l)
            for i in range(n_prop):
                ax.plot(series_mat[i], color=cmap(i % 10), alpha=0.7,
                        linewidth=1.0, label=f'prop {i}' if l_idx == 0 and j == 0 else None)
            ax.set_title(gen.node_info(l_idx, j), fontsize=9)
            ax.grid(True, alpha=0.3)
            if l_idx == n_rows - 1:
                ax.set_xlabel('t')
            if j == 0:
                ax.set_ylabel(f'Layer {l_idx}')

    # Single legend
    handles = [plt.Line2D([0], [0], color=cmap(i % 10), lw=1.2)
               for i in range(n_prop)]
    labels = [f'prop {i}' for i in range(n_prop)]
    fig.legend(handles, labels, loc='upper right', fontsize=8, ncol=n_prop)

    fig.suptitle(
        f'Multi-layer DAG  |  layers={gen.layers}, d={gen.d}, T={gen.T}, '
        f'K={gen.K}, seed={gen.seed}  |  {n_prop} propagations',
        fontsize=11, y=1.02
    )
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Exp 02: Multi-layer DAG')
    parser.add_argument('--d', type=int, default=8, help='Latent dim d (root m)')
    parser.add_argument('--T', type=int, default=200, help='Output time steps')
    parser.add_argument('--kernel-size', type=int, default=5, help='Causal conv kernel size')
    parser.add_argument('--seed', type=int, default=0, help='Generator seed')
    parser.add_argument('--n-prop', type=int, default=5, help='Propagations to visualise')
    args = parser.parse_args()

    gen = MultiLayerDAGGenerator(d=args.d, T=args.T, kernel_size=args.kernel_size,
                                 seed=args.seed)

    save_path = os.path.join(
        os.path.dirname(__file__), 'results',
        f'exp02_d{args.d}_K{args.kernel_size}_seed{args.seed}.png')
    visualize_dag(gen, n_prop=args.n_prop, save_path=save_path)


if __name__ == '__main__':
    main()
