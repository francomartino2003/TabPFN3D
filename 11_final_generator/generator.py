"""
Dataset generator: builds DAG, assigns per-node operations, propagates, extracts datasets.

Node types and computation:
  ROOT LAYER (layer 0):
    Each root node is tabular or discrete.
    Tabular roots: sampled N(0,std) or U(-a,a) per observation → scalar.
    Discrete roots: sample class uniformly → class_value → scalar.
  TABULAR:   dim 1 (continuous).  flatten parents → W·x + b + activation → scalar + noise.
  DISCRETE:  dim 1 (discrete→continuous).  flatten parents → nearest-prototype →
             class index → continuous class_value + noise for downstream propagation.
  SERIES:    dim 1×T.
    All series nodes use the same two-conv logic:
    - Parent inputs → (n_parents, T), non-series parents replicated along time.
    - Add N_t normalized time-index channels (variable per node).
      N_t = 1 + extra if no series parents, else 0 + extra.
    - Conv1 pointwise (K=1): (n_parents + N_t → d_hidden) + act1 → (d_hidden, T).
    - Conv2 temporal (K, dilation D): (d_hidden → 1), left zero-padded + act2 → (1, T).
    - Add iid noise.
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


# ── Batch convolutions ────────────────────────────────────────────────────────

def batch_pointwise_conv(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Pointwise (K=1) convolution — mixes channels independently at each t.

    x:      (N, C_in, T)
    kernel: (C_out, C_in)
    output: (N, C_out, T)
    """
    # einsum: for each n,t: y[n, o, t] = sum_c kernel[o,c] * x[n,c,t]
    return np.einsum('oc,nct->not', kernel, x)


def batch_dilated_causal_conv(x: np.ndarray, kernel: np.ndarray,
                              dilation: int = 1) -> np.ndarray:
    """
    Dilated causal conv with left zero-padding to preserve length T.

    x:      (N, C_in, T)
    kernel: (C_out, C_in, K)
    dilation: int  (spacing between kernel taps)
    output: (N, C_out, T)

    Effective receptive field: D * (K - 1) + 1.
    Left zero-pad of D * (K - 1) ensures causality and same output length.
    """
    C_out, C_in, K = kernel.shape
    N, _, T = x.shape
    pad_len = dilation * (K - 1)

    # Left zero-pad for causality
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad_len, 0)), mode='constant')

    y = np.zeros((N, C_out, T))
    for t in range(T):
        t_p = t + pad_len  # position in padded array
        # Gather K samples at dilation spacing, going backwards from t_p
        indices = [t_p - k * dilation for k in range(K)]
        x_slice = x_padded[:, :, indices]  # (N, C_in, K)
        y[:, :, t] = np.einsum('nck,ock->no', x_slice, kernel)
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
        T = self.T
        acts = hp_p.activation_choices

        self.node_ops: Dict[int, dict] = {}

        for node in self.dag.nodes:
            # Layer 0: root nodes (initialized by sampling, not propagation)
            if node.layer == 0:
                if node.node_type == 'discrete':
                    k = self._log_uniform_int(*hp_p.discrete_classes_range)
                    class_values = self.rng.uniform(-1, 1, size=(k,))
                    self.node_ops[node.id] = {
                        'kind': 'root_discrete',
                        'k': k,
                        'class_values': class_values,
                    }
                else:  # tabular root
                    self.node_ops[node.id] = {'kind': 'root_tabular'}
                continue

            parent_types = [self.dag.nodes[p].node_type for p in node.parents]

            if node.node_type == 'tabular':
                self._build_tabular_ops(node, parent_types, T, acts)

            elif node.node_type == 'discrete':
                self._build_discrete_ops(node, parent_types, T, hp_p)

            elif node.node_type == 'series':
                self._build_series_ops(node, parent_types, acts)

    def _input_dim(self, parent_types, T):
        """Total flattened input dimension for a scalar (tabular/discrete) node."""
        return sum(T if pt == 'series' else 1 for pt in parent_types)

    def _build_tabular_ops(self, node, parent_types, T, acts):
        dim = self._input_dim(parent_types, T)
        std = np.sqrt(2.0 / (dim + 1))
        self.node_ops[node.id] = {
            'kind': 'tabular',
            'W': self.rng.normal(0, std, size=(dim,)),
            'b': float(self.rng.normal(0, 0.1)),
            'act': str(self.rng.choice(acts)),
            'noise_std': self._sample_noise_std(),
        }

    def _build_discrete_ops(self, node, parent_types, T, hp_p):
        dim = self._input_dim(parent_types, T)
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

    def _build_series_ops(self, node, parent_types, acts):
        """
        Series node builder — pointwise conv + activation + dilated causal conv.

        Time-index channels per node:
          - No series parents → 1 mandatory + extra (1..5 total).
          - Has series parents → 0 mandatory + extra (0..4 total).
        """
        hp_p = self.hp.propagation

        has_series_parent = any(pt == 'series' for pt in parent_types)
        n_extra = int(self.rng.integers(
            hp_p.n_extra_time_indices_range[0],
            hp_p.n_extra_time_indices_range[1] + 1,
        ))
        n_time_indices = (0 if has_series_parent else 1) + n_extra
        c_in = len(parent_types) + n_time_indices

        # Conv1: pointwise (K=1), maps c_in → d_hidden
        d_hidden = self._log_uniform_int(*hp_p.series_hidden_channels_range)
        std1 = np.sqrt(2.0 / max(c_in, 1))
        kernel1 = self.rng.normal(0, std1, size=(d_hidden, c_in))

        # Conv2: temporal dilated causal, maps d_hidden → 1
        K2 = self._log_uniform_int(*hp_p.kernel_size_range)
        D2 = self._log_uniform_int(*hp_p.dilation_range)
        std2 = np.sqrt(2.0 / (d_hidden * K2))
        kernel2 = self.rng.normal(0, std2, size=(1, d_hidden, K2))

        act1 = str(self.rng.choice(acts))
        # act2: 50 % identity, 50 % random from bank
        act2 = 'identity' if self.rng.random() < 0.3 else str(self.rng.choice(acts))

        self.node_ops[node.id] = {
            'kind': 'series',
            'kernel1': kernel1,
            'kernel2': kernel2,
            'dilation': D2,
            'act1': act1,
            'act2': act2,
            'noise_std': self._sample_noise_std(),
            'n_time_indices': n_time_indices,
        }

    # ── Propagation ──────────────────────────────────────────────────────

    def propagate(self, n: int):
        """
        Forward-pass for n observations.

        Returns:
            vals:  {node_id: np.ndarray} – continuous values
                   tabular/discrete/root → (n,), series → (n, T)
            disc:  {node_id: np.ndarray} – discrete class indices (int)
                   only for discrete nodes → (n,)
        """
        T = self.T
        vals: Dict[int, np.ndarray] = {}
        disc: Dict[int, np.ndarray] = {}

        # Initialize layer 0 (root nodes)
        for nid in self.dag.layers[0]:
            ops = self.node_ops[nid]
            if ops['kind'] == 'root_discrete':
                k = ops['k']
                idx = self.rng.integers(0, k, size=(n,))
                vals[nid] = ops['class_values'][idx]
                disc[nid] = idx
            else:  # root_tabular
                if self.root_init == 'normal':
                    vals[nid] = self.rng.normal(0, self.root_std, (n,))
                else:
                    vals[nid] = self.rng.uniform(-self.root_a, self.root_a, (n,))

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
            if pt == 'series':
                parts.append(v)                           # (n, T)
            else:  # tabular or discrete
                parts.append(v[:, None])                  # (n, 1)
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

    # ── Series propagation (pointwise + activation + dilated causal) ────────

    def _prop_series(self, node, ops, vals, n, T):
        """
        Series propagation — pointwise conv + act1 + dilated causal conv + act2.

        1. Gather parent channels → (n, n_parents, T).
        2. Append N_t normalized time-index channels.
        3. Conv1 pointwise → (n, d_hidden, T) + act1.
        4. Conv2 dilated causal → (n, 1, T) + act2.
        5. Squeeze + iid noise → (n, T).
        """
        channels = []
        for pid in node.parents:
            pt = self.dag.nodes[pid].node_type
            v = vals[pid]
            if pt == 'series':
                channels.append(v)                                    # (n, T)
            else:  # tabular or discrete
                channels.append(np.tile(v[:, None], (1, T)))          # (n, T)

        # Time-index channels (variable count per node)
        n_time = ops['n_time_indices']
        if n_time > 0:
            t_channel = np.linspace(-1.0, 1.0, T)
            t_batch = np.tile(t_channel[None, :], (n, 1))
            for _ in range(n_time):
                channels.append(t_batch)

        # x: (n, n_parents + n_time, T)
        x = np.stack(channels, axis=1)

        # Conv1 pointwise → (n, d_hidden, T) + act1
        x = batch_pointwise_conv(x, ops['kernel1'])
        x = apply_activation(x, ops['act1'])

        # Conv2 dilated causal → (n, 1, T) + act2
        x = batch_dilated_causal_conv(x, ops['kernel2'], ops['dilation'])
        x = apply_activation(x, ops['act2'])

        # Squeeze to (n, T)
        out = x[:, 0, :]
        out = out + self.rng.normal(0, ops['noise_std'], size=(n, T))
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
            f'root layer: d={self.dag.root_d} nodes, init={self.root_init}',
            self.dag.summary(),
            'Node operations:',
        ]
        for nid, ops in sorted(self.node_ops.items()):
            kind = ops['kind']
            if kind == 'root_tabular':
                lines.append(f'  node {nid} (root tabular): sampled')
            elif kind == 'root_discrete':
                lines.append(
                    f'  node {nid} (root discrete): k={ops["k"]}')
            elif kind == 'tabular':
                noise = f', noise_std={ops["noise_std"]:.1e}'
                lines.append(
                    f'  node {nid} (tabular): W({len(ops["W"])}), '
                    f'act={ops["act"]}{noise}')
            elif kind == 'discrete':
                noise = f', noise_std={ops["noise_std"]:.1e}'
                lines.append(
                    f'  node {nid} (discrete): k={ops["k"]}, '
                    f'proto({ops["prototypes"].shape}){noise}')
            elif kind == 'series':
                noise = f', noise_std={ops["noise_std"]:.1e}'
                d_h = ops['kernel1'].shape[0]
                K2 = ops['kernel2'].shape[2]
                D2 = ops['dilation']
                n_t = ops['n_time_indices']
                lines.append(
                    f'  node {nid} (series): pw({ops["kernel1"].shape[1]}→{d_h}) → '
                    f'{ops["act1"]} → conv(K={K2},D={D2},{d_h}→1) → '
                    f'{ops["act2"]}, t_idx={n_t}{noise}')
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
