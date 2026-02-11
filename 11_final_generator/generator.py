"""
Dataset generator: builds DAG, assigns per-node operations, propagates, extracts datasets.

Node types and computation:
  ROOT:      dimension d, sampled N(0,std) or U(-a,a) per observation.
  TABULAR:   dim 1 (continuous).  flatten parents → W·x + b + activation → scalar + noise.
  DISCRETE:  dim 1 (discrete→continuous).  flatten parents → nearest-prototype →
             class index → continuous class_value + noise for downstream propagation.
  SERIES:    dim 1×T.
    All series nodes use the same logic:
    - Parent inputs → (n_parents, L), non-series parents are replicated along time.
    - Concatenate normalized time index → (n_parents+1, L).
    - Causal 1D conv (n_parents+1 → 1) + activation → (1, T) + iid noise.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from dag_structure import DAGStructure, build_dag, visualize_dag
from hyperparameters import GeneratorHyperparameters


# ── Activations ────────────────────────────────────────────────────────────────

def apply_activation(z: np.ndarray, name: str) -> np.ndarray:
    if name == 'identity':  return z
    if name == 'log':       return np.sign(z) * np.log1p(np.abs(z))
    if name == 'sigmoid':   return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    if name == 'abs':       return np.abs(z)
    if name == 'sin':       return np.sin(z)
    if name == 'tanh':      return np.tanh(z)
    if name == 'square':    return z ** 2
    if name == 'power':     return np.sign(z) * np.sqrt(np.abs(z))
    if name == 'softplus':  return np.log1p(np.exp(np.clip(z, -500, 500)))
    if name == 'step':      return (z >= 0).astype(z.dtype)
    if name == 'modulo':    return np.mod(z, 1.0)
    return z


# ── Batch causal convolution ──────────────────────────────────────────────────

def batch_causal_conv(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Multi-channel causal conv supporting multi-channel output.

    x:      (N, C_in, L)
    kernel: (C_out, C_in, K)  or  (C_in, K) for single-output (legacy)
    output: (N, C_out, L-K+1)  or  (N, L-K+1) if kernel was 2D
    """
    squeeze = False
    if kernel.ndim == 2:
        kernel = kernel[None, :, :]   # (1, C_in, K)
        squeeze = True
    C_out, C_in, K = kernel.shape
    N = x.shape[0]
    L = x.shape[2]
    out_len = L - K + 1
    y = np.zeros((N, C_out, out_len))
    for t in range(out_len):
        # x[:, :, t:t+K] → (N, C_in, K);  kernel → (C_out, C_in, K)
        y[:, :, t] = np.einsum('nck,ock->no', x[:, :, t:t + K], kernel)
    if squeeze:
        return y[:, 0, :]   # (N, out_len)
    return y


# ── Dataset generator ─────────────────────────────────────────────────────────

class DatasetGenerator:
    """
    Full pipeline: DAG topology → per-node operations → propagation → dataset.

    A single seed fixes everything (DAG, kernels, activations, prototypes).
    Each observation re-samples only the root vector.
    """

    def __init__(self, seed: int, hp: GeneratorHyperparameters | None = None):
        self.hp = hp or GeneratorHyperparameters()
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # 1. Build DAG
        dag_rng = np.random.default_rng(self.rng.integers(0, 2**62))
        self.dag = build_dag(self.hp.dag, self.hp.roles, dag_rng)

        # 2. Sample dataset-level params (T, n_samples, root init)
        self._sample_dataset_params()

        # 3. Build per-node operations (weights, kernels, prototypes, etc.)
        self._build_operations()

        # 4. n_classes is determined by the target discrete node
        self.n_classes = self.node_ops[self.dag.target_node.id]['k']

    # ── 2. Dataset-level params ───────────────────────────────────────────

    def _sample_dataset_params(self):
        hp_d = self.hp.dataset
        hp_p = self.hp.propagation

        self.n_features = len(self.dag.feature_nodes)

        # T (constrained by feat * T <= max)
        t_max = min(hp_d.t_range[1],
                     hp_d.max_feat_times_t // max(1, self.n_features))
        t_min = min(hp_d.t_range[0], t_max)
        self.T = int(self.rng.integers(t_min, t_max + 1))

        # n_samples
        self.n_samples = int(self.rng.integers(hp_d.min_samples, hp_d.max_samples + 1))

        # Root init
        self.root_init = str(self.rng.choice(hp_p.root_init_choices))
        if self.root_init == 'normal':
            self.root_std = float(self.rng.uniform(*hp_p.root_normal_std_range))
        else:
            self.root_a = float(self.rng.uniform(*hp_p.root_uniform_a_range))

    # ── 3. Build per-node operations ─────────────────────────────────────

    def _log_uniform_int(self, lo: int, hi: int) -> int:
        """Sample int from log-uniform distribution (favors smaller values)."""
        if lo == hi:
            return lo
        val = np.exp(self.rng.uniform(np.log(lo), np.log(hi)))
        return int(np.clip(np.round(val), lo, hi))

    def _sample_noise_std(self):
        """Sample per-node noise std, log-uniform to favor small values."""
        lo, hi = self.hp.propagation.noise_std_range
        return float(np.exp(self.rng.uniform(np.log(lo), np.log(hi))))

    def _build_operations(self):
        hp_p = self.hp.propagation
        d = self.dag.root_d
        T = self.T
        acts = hp_p.activation_choices

        self.node_ops: Dict[int, dict] = {}

        for node in self.dag.nodes:
            if node.node_type == 'root':
                continue

            parent_types = [self.dag.nodes[p].node_type for p in node.parents]

            if node.node_type == 'tabular':
                self._build_tabular_ops(node, parent_types, d, T, acts)

            elif node.node_type == 'discrete':
                self._build_discrete_ops(node, parent_types, d, T, hp_p)

            elif node.node_type == 'series':
                self._build_series_ops(node, parent_types, d, acts)

    def _input_dim(self, parent_types, d, T):
        """Total flattened input dimension for a scalar (tabular/discrete) node."""
        return sum(
            d if pt == 'root' else (T if pt == 'series' else 1)
            for pt in parent_types
        )

    def _build_tabular_ops(self, node, parent_types, d, T, acts):
        dim = self._input_dim(parent_types, d, T)
        std = np.sqrt(2.0 / (dim + 1))
        self.node_ops[node.id] = {
            'kind': 'tabular',
            'W': self.rng.normal(0, std, size=(dim,)),
            'b': float(self.rng.normal(0, 0.1)),
            'act': str(self.rng.choice(acts)),
            'noise_std': self._sample_noise_std(),
        }

    def _build_discrete_ops(self, node, parent_types, d, T, hp_p):
        dim = self._input_dim(parent_types, d, T)
        k = self._log_uniform_int(*hp_p.discrete_classes_range)

        # Well-separated prototypes: sample random directions and normalize
        # to unit sphere, then scale by sqrt(dim) so they sit on a sphere
        # with radius proportional to expected norm of inputs.
        raw = self.rng.normal(0, 1.0, size=(k, dim))
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        prototypes = raw / norms * np.sqrt(dim)

        class_values = self.rng.normal(0, 1.0, size=(k,))
        self.node_ops[node.id] = {
            'kind': 'discrete',
            'k': k,
            'prototypes': prototypes,
            'class_values': class_values,
            'noise_std': self._sample_noise_std(),
        }

    def _build_series_ops(self, node, parent_types, d, acts):
        """
        Unified series node builder — mini CNN with L layers.

        Each series node is a stack of causal conv layers:
          Layer 0: (n_parents+1, L_in) → (h, L_in-K0+1) + activation
          Layer 1: (h, ...) → (h, ...) + activation
          ...
          Layer L-1: (h, ...) → (1, T) + activation

        Each layer has its own kernel (with independently sampled size)
        and its own activation. The time index channel gets a random
        continuous activation applied before entering the first layer.
        """
        hp_p = self.hp.propagation
        n_layers = self._log_uniform_int(*hp_p.n_conv_layers_range)
        hidden = self._log_uniform_int(*hp_p.hidden_channels_range)
        c_in = len(parent_types) + 1  # +1 for time index channel

        kernels = []
        layer_acts = []
        for i in range(n_layers):
            ci = c_in if i == 0 else hidden
            co = 1 if i == n_layers - 1 else hidden
            K = self._log_uniform_int(*hp_p.kernel_size_range)
            std = np.sqrt(2.0 / (ci * K))
            kernels.append(self.rng.normal(0, std, size=(co, ci, K)))
            layer_acts.append(str(self.rng.choice(acts)))

        self.node_ops[node.id] = {
            'kind': 'series',
            'kernels': kernels,
            'acts': layer_acts,
            'time_act': str(self.rng.choice(hp_p.time_activation_choices)),
            'noise_std': self._sample_noise_std(),
        }

    # ── Propagation ──────────────────────────────────────────────────────

    def _sample_root(self, n: int) -> np.ndarray:
        d = self.dag.root_d
        if self.root_init == 'normal':
            return self.rng.normal(0, self.root_std, (n, d))
        return self.rng.uniform(-self.root_a, self.root_a, (n, d))

    def propagate(self, n: int):
        """
        Forward-pass for n observations.

        Returns:
            vals:  {node_id: np.ndarray} – continuous values
                   root → (n, d), tabular/discrete → (n,), series → (n, T)
            disc:  {node_id: np.ndarray} – discrete class indices (int)
                   only for discrete nodes → (n,)
        """
        T = self.T
        vals: Dict[int, np.ndarray] = {}
        disc: Dict[int, np.ndarray] = {}
        vals[self.dag.root.id] = self._sample_root(n)

        for l_idx in range(1, self.dag.n_layers):
            for nid in self.dag.layers[l_idx]:
                node = self.dag.nodes[nid]
                ops = self.node_ops[nid]

                if ops['kind'] == 'tabular':
                    vals[nid] = self._prop_tabular(node, ops, vals, n)

                elif ops['kind'] == 'discrete':
                    cont, idx = self._prop_discrete(node, ops, vals, n)
                    vals[nid] = cont
                    disc[nid] = idx

                elif ops['kind'] == 'series':
                    vals[nid] = self._prop_series(node, ops, vals, n, T)

        return vals, disc

    # ── Helper: gather flat input ─────────────────────────────────────────

    def _gather_flat(self, node, vals, n):
        """Flatten parent outputs → (n, input_dim)."""
        parts = []
        for pid in node.parents:
            pt = self.dag.nodes[pid].node_type
            v = vals[pid]
            if pt == 'root':
                parts.append(v)                          # (n, d)
            elif pt in ('tabular', 'discrete'):
                parts.append(v[:, None])                  # (n, 1)
            elif pt == 'series':
                parts.append(v)                           # (n, T)
        return np.concatenate(parts, axis=1)

    # ── Tabular propagation ───────────────────────────────────────────────

    def _prop_tabular(self, node, ops, vals, n):
        x = self._gather_flat(node, vals, n)
        out = np.sum(x * ops['W'][None, :], axis=1) + ops['b']
        out = apply_activation(out, ops['act'])
        out += self.rng.normal(0, ops['noise_std'], size=(n,))
        return out

    # ── Discrete propagation ──────────────────────────────────────────────

    def _prop_discrete(self, node, ops, vals, n):
        x = self._gather_flat(node, vals, n)              # (n, input_dim)
        prototypes = ops['prototypes']                     # (k, input_dim)
        dists = np.sum((x[:, None, :] - prototypes[None, :, :]) ** 2, axis=2)  # (n, k)
        indices = np.argmin(dists, axis=1)                 # (n,) int
        cont_vals = ops['class_values'][indices]           # (n,)
        cont_vals = cont_vals + self.rng.normal(0, ops['noise_std'], size=(n,))
        return cont_vals, indices

    # ── Series propagation (unified mini CNN) ──────────────────────────────

    def _prop_series(self, node, ops, vals, n, T):
        """
        Unified series propagation — mini CNN with L conv layers.

        1. Compute total shrinkage from all layers to determine L_in.
        2. Gather parent channels → (n, n_parents, L_in).
        3. Append time index channel (with activation) → (n, n_parents+1, L_in).
        4. Apply conv layers: conv + activation at each layer.
        5. Final output: (n, T) + iid noise.
        """
        kernels = ops['kernels']
        layer_acts = ops['acts']

        # Total shrinkage = sum of (K_i - 1) across all layers
        total_shrink = sum(k.shape[2] - 1 for k in kernels)
        L_in = T + total_shrink

        channels = []
        for pid in node.parents:
            pt = self.dag.nodes[pid].node_type
            v = vals[pid]
            if pt == 'series':
                channels.append(np.pad(v, ((0, 0), (0, L_in - T)), mode='edge'))
            elif pt in ('tabular', 'discrete'):
                channels.append(np.tile(v[:, None], (1, L_in)))
            elif pt == 'root':
                channels.append(
                    np.tile(v.mean(axis=1, keepdims=True), (1, L_in)))

        # Normalized time index channel with activation
        t_channel = np.linspace(-1.0, 1.0, L_in)
        t_channel = apply_activation(t_channel, ops['time_act'])
        t_batch = np.tile(t_channel[None, :], (n, 1))
        channels.append(t_batch)

        # x: (n, n_parents+1, L_in)
        x = np.stack(channels, axis=1)

        # Apply mini CNN layers
        for kernel, act in zip(kernels, layer_acts):
            x = batch_causal_conv(x, kernel)        # (n, C_out, L') or (n, L') for last
            x = apply_activation(x, act)
            # If output is 2D (last layer, C_out=1, squeezed), break
            if x.ndim == 2:
                break
            # Otherwise keep going as 3D

        # x should be (n, T) at this point
        if x.ndim == 3:
            x = x[:, 0, :]  # squeeze channel dim if still 3D

        out = x
        out += self.rng.normal(0, ops['noise_std'], size=(n, T))
        return out

    # ── Dataset extraction ────────────────────────────────────────────────

    def generate_dataset(self) -> dict | None:
        """
        Generate a complete classification dataset.

        - Propagate 3× the requested n_samples to increase chance of balanced classes.
        - Labels come directly from the target discrete node.
        - Classes with < min_samples_per_class observations are dropped.
        - Subsample back to n_samples after filtering.
        - If fewer than 2 classes survive, returns **None** (caller should skip).
        - Stratified split guarantees all test labels exist in train.
        """
        min_per_class = self.hp.dataset.min_samples_per_class   # 6
        feat_ids = [n.id for n in self.dag.feature_nodes]
        target_id = self.dag.target_node.id

        # Oversample: propagate 3× more, then trim
        n_propagate = min(self.n_samples * 3, 3000)
        vals, disc = self.propagate(n_propagate)
        X = np.stack([vals[fid] for fid in feat_ids], axis=1)
        y = disc[target_id].astype(int)

        # ── Drop rare classes ─────────────────────────────────────────────
        unique, counts = np.unique(y, return_counts=True)
        keep_classes = unique[counts >= min_per_class]

        if len(keep_classes) < 2:
            return None                         # dataset unusable → skip

        mask = np.isin(y, keep_classes)
        X, y = X[mask], y[mask]

        # ── Subsample back to n_samples ───────────────────────────────────
        if len(y) > self.n_samples:
            idx = self.rng.choice(len(y), size=self.n_samples, replace=False)
            X, y = X[idx], y[idx]

        # Re-check after subsample
        unique2, counts2 = np.unique(y, return_counts=True)
        keep2 = unique2[counts2 >= min_per_class]
        if len(keep2) < 2:
            return None

        mask2 = np.isin(y, keep2)
        X, y = X[mask2], y[mask2]

        # Re-index classes to 0..k'-1
        class_map = {c: i for i, c in enumerate(sorted(keep2))}
        y = np.array([class_map[c] for c in y])
        n_classes = len(keep2)

        # ── Stratified train / test split ─────────────────────────────────
        train_ratio = self.hp.dataset.train_ratio
        train_idx, test_idx = [], []
        for cls in range(n_classes):
            cls_mask = np.where(y == cls)[0]
            perm = self.rng.permutation(cls_mask)
            n_train = max(1, int(len(perm) * train_ratio))
            if len(perm) > 1 and n_train == len(perm):
                n_train -= 1
            train_idx.extend(perm[:n_train].tolist())
            test_idx.extend(perm[n_train:].tolist())

        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx) if test_idx else np.array([], dtype=int)

        return {
            'X_train': X[train_idx],
            'X_test': X[test_idx] if len(test_idx) > 0 else np.empty((0, self.n_features, self.T)),
            'y_train': y[train_idx],
            'y_test': y[test_idx] if len(test_idx) > 0 else np.array([], dtype=int),
            'n_classes': n_classes,
            'n_features': self.n_features,
            'T': self.T,
            'n_samples': len(y),
        }

    # ── Summary ──────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            f'seed={self.seed}  n_samples={self.n_samples}  T={self.T}  '
            f'n_features={self.n_features}  n_classes={self.n_classes}',
            f'root: d={self.dag.root_d}, init={self.root_init}',
            self.dag.summary(),
            'Node operations:',
        ]
        for nid, ops in sorted(self.node_ops.items()):
            noise = f', noise_std={ops["noise_std"]:.1e}'
            if ops['kind'] == 'tabular':
                lines.append(
                    f'  node {nid} (tabular): W({len(ops["W"])}), '
                    f'act={ops["act"]}{noise}')
            elif ops['kind'] == 'discrete':
                lines.append(
                    f'  node {nid} (discrete): k={ops["k"]}, '
                    f'proto({ops["prototypes"].shape}){noise}')
            elif ops['kind'] == 'series':
                kshapes = [f'{k.shape}' for k in ops['kernels']]
                acts_str = '→'.join(ops['acts'])
                lines.append(
                    f'  node {nid} (series): {len(ops["kernels"])}L '
                    f'kernels={kshapes}, acts=[{acts_str}], '
                    f't_act={ops["time_act"]}{noise}')
        return '\n'.join(lines)


# ── Visualisation ──────────────────────────────────────────────────────────────

def visualize_dataset(ds: dict, gen: DatasetGenerator, save_path: str,
                      n_per_class: int = 5):
    """
    Grid: columns = classes, rows = observations (n_per_class per class).
    Each cell plots all feature time-series for that observation.
    """
    X = ds['X_train']
    y = ds['y_train']
    classes = np.sort(np.unique(y))
    n_cols = len(classes)
    n_features = ds['n_features']
    cmap = plt.cm.tab10

    fig, axes = plt.subplots(n_per_class, n_cols,
                             figsize=(3.5 * n_cols, 2.2 * n_per_class),
                             squeeze=False)

    for col, cls in enumerate(classes):
        idx = np.where(y == cls)[0]
        n_show = min(n_per_class, len(idx))
        for row in range(n_per_class):
            ax = axes[row][col]
            if row >= n_show:
                ax.set_visible(False)
                continue
            for f in range(n_features):
                ax.plot(X[idx[row], f, :], color=cmap(f % 10), alpha=0.8, lw=0.9)
            ax.grid(True, alpha=0.2)
            if row == 0:
                ax.set_title(f'class {cls} ({len(idx)} train)', fontsize=9)
            if col == 0:
                ax.set_ylabel(f'obs {row}', fontsize=8)
            ax.tick_params(labelsize=6)

    if n_features <= 12:
        handles = [plt.Line2D([0], [0], color=cmap(f % 10), lw=1.2)
                   for f in range(n_features)]
        labels = [f'feat {f}' for f in range(n_features)]
        fig.legend(handles, labels, loc='upper right', fontsize=7,
                   ncol=min(n_features, 6))

    fig.suptitle(
        f'Dataset  seed={gen.seed}  |  {ds["n_samples"]} samples, '
        f'{n_features} feat, T={ds["T"]}, {ds["n_classes"]} classes',
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Generate and visualise datasets')
    parser.add_argument('--n', type=int, default=5, help='Number of datasets')
    parser.add_argument('--seed', type=int, default=0, help='Base seed')
    args = parser.parse_args()

    hp = GeneratorHyperparameters()
    out_dir = os.path.join(os.path.dirname(__file__), 'visualizations', 'datasets')
    dag_dir = os.path.join(os.path.dirname(__file__), 'visualizations', 'dag_structure')

    for i in range(args.n):
        seed = args.seed + i
        gen = DatasetGenerator(seed=seed, hp=hp)
        print(f'\n{"="*70}')
        print(gen.summary())

        ds = gen.generate_dataset()
        if ds is None:
            print('  SKIPPED (< 2 classes with enough samples)')
            continue
        print(f'  X_train: {ds["X_train"].shape}  X_test: {ds["X_test"].shape}')
        if len(ds['y_train']) > 0:
            print(f'  y_train distribution: {np.bincount(ds["y_train"])}')
        if len(ds['y_test']) > 0:
            print(f'  y_test  distribution: {np.bincount(ds["y_test"], minlength=ds["n_classes"])}')

        visualize_dataset(ds, gen,
                          os.path.join(out_dir, f'dataset_seed{seed}.png'))
        visualize_dag(gen.dag,
                      os.path.join(dag_dir, f'dag_seed{seed}.png'))


if __name__ == '__main__':
    main()
