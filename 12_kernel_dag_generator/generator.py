"""
Kernel-DAG dataset generator.

Key differences from 11_final_generator:
  ROOT LAYER (layer 0):
    - Tabular roots: scalar N(0,std) or U(-a,a) per observation (unchanged).
    - Discrete roots: uniform class → class_value → scalar (unchanged).
    - Series roots (NEW): sample from GP with kernel K (TxT).
      Kernel bank: Linear, RBF, Periodic — one drawn per series root.
  TABULAR (internal): unchanged — flatten parents → W·x + b + act + noise → scalar.
  DISCRETE (internal): unchanged — flatten parents → nearest-prototype → class index.
  SERIES (internal — redesigned):
    - No time-index channels. Series ancestry guaranteed by DAG structure.
    - Parents (c channels × T) → single Conv1D(c_in=c, c_out=1, K, D) + bias + act + noise.
    - K ∈ {1,7,9,11}, weights ~ N(0,1) then centered, bias ~ U(-1,1).
    - Dilation: d = floor(2^x), x~U(0, log2((T-1)/(K-1))), capped. K=1 → d=1.
    - Padding: Bernoulli(0.5) → causal (left) or centered.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cholesky

from dag_structure import DAGStructure, build_dag, visualize_dag
from hyperparameters import GeneratorHyperparameters


# ── Kumaraswamy warping ─────────────────────────────────────────────────────────

def kumaraswamy_cdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    x = np.clip(x.astype(np.float64), 1e-12, 1.0 - 1e-12)
    return (1.0 - (1.0 - np.power(x, a)) ** b).astype(np.float64)


def apply_kumaraswamy_warp_to_feature(X: np.ndarray, feat_idx: int,
                                      a: float, b: float,
                                      rng: np.random.Generator) -> None:
    n, _, T = X.shape
    for i in range(n):
        x = X[i, feat_idx, :].astype(np.float64)
        x_min, x_max = x.min(), x.max()
        if x_max <= x_min:
            continue
        x_norm = (x - x_min) / (x_max - x_min)
        x_warp = kumaraswamy_cdf(x_norm, a, b)
        X[i, feat_idx, :] = x_min + (x_max - x_min) * x_warp


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
    if name == 'softplus':    return np.log1p(np.exp(np.clip(z, -500, 500)))
    if name == 'smooth_relu': return z / (1.0 + np.exp(-np.clip(z, -500, 500)))  # SiLU
    if name == 'step':        return (z >= 0).astype(z.dtype)
    if name == 'modulo':      return np.mod(z, 1.0)
    return z


# ── GP kernel functions ──────────────────────────────────────────────────────

def build_gp_kernel(T: int, kernel_type: str, params: dict) -> np.ndarray:
    """Build a TxT covariance matrix from the kernel bank.

    Time grid normalized to [0, 1].
    """
    t = np.linspace(0, 1, T)

    if kernel_type == 'linear':
        sigma, c = params['sigma'], params['c']
        K = sigma**2 * np.outer(t - c, t - c)

    elif kernel_type == 'rbf':
        sigma, ell = params['sigma'], params['ell']
        diff = t[:, None] - t[None, :]
        K = sigma**2 * np.exp(-diff**2 / (2 * ell**2))

    elif kernel_type == 'periodic':
        sigma, p, ell = params['sigma'], params['period'], params['ell']
        diff = np.abs(t[:, None] - t[None, :])
        K = sigma**2 * np.exp(-2 * np.sin(np.pi * diff / p)**2 / ell**2)

    else:
        raise ValueError(f'Unknown kernel type: {kernel_type}')

    K += 1e-6 * np.eye(T)
    return K


def sample_gp(K: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n functions from GP(0, K). Returns (n, T)."""
    T = K.shape[0]
    try:
        L = cholesky(K, lower=True)
    except np.linalg.LinAlgError:
        K_fixed = K + 1e-4 * np.eye(T)
        L = cholesky(K_fixed, lower=True)
    z = rng.standard_normal((T, n))
    return (L @ z).T  # (n, T)


# ── Batch Conv1D ──────────────────────────────────────────────────────────────

def batch_conv1d(x: np.ndarray, kernel: np.ndarray, bias: float,
                 dilation: int = 1, padding: str = 'left',
                 pre_pad: bool = True, pad_mode: str = 'constant') -> np.ndarray:
    """
    Conv1D: (N, C_in, T) → (N, 1, T) preserving length.

    kernel   : (C_in, K) — single output channel
    padding  : 'left' (causal) or 'center'
    pre_pad  : True  → pad input BEFORE conv (default, boundary effects distributed)
               False → valid conv (no pre-padding), then pad output to restore T
    pad_mode : 'constant' (zeros) or 'edge' (replicate edge value)
    """
    C_in, K = kernel.shape
    N, _, T = x.shape
    pad_len = dilation * (K - 1)

    if padding == 'left':
        pad_left, pad_right = pad_len, 0
    else:  # center
        pad_left  = ((K - 1) // 2) * dilation
        pad_right = pad_len - pad_left

    # K=1 has no receptive field beyond t=0 — treat as pre-pad (trivially same)
    T_valid = T - pad_len
    if K == 1 or T_valid <= 0:
        pre_pad = True  # fallback: valid range too small

    y = np.zeros((N, 1, T))

    if pre_pad:
        # ── Pre-padding: pad input, then convolve at all T positions ──────
        x_padded = np.pad(x, ((0, 0), (0, 0), (pad_left, pad_right)),
                          mode=pad_mode)
        if padding == 'left':
            offsets = [-k * dilation for k in range(K)]
        else:
            offsets = [-pad_left + k * dilation for k in range(K)]

        for t in range(T):
            t_p      = t + pad_left
            indices  = [t_p + off for off in offsets]
            x_slice  = x_padded[:, :, indices]          # (N, C_in, K)
            y[:, 0, t] = np.einsum('nck,ck->n', x_slice, kernel) + bias

    else:
        # ── Post-padding: valid conv (T_valid outputs), then pad to T ────
        # Causal  valid: output[t_v] = conv(x[t_v + pad_len - k*d], k=0..K-1)
        #   → uses x[t_v], x[t_v+d], ..., x[t_v+pad_len]  (same values, kernel reversed)
        # Centered valid: output[t_v] = conv(x[t_v + k*d], k=0..K-1)
        #   → uses x[t_v], x[t_v+d], ..., x[t_v+pad_len]
        # In both cases the gathered positions are [t_v + k*d for k in range(K)];
        # only the kernel weight order differs (causal: kernel is applied as-is
        # to the reversed-time slice; centered: straight order).
        valid_out = np.zeros((N, 1, T_valid))

        if padding == 'left':
            # Causal: position t (original) = t_v + pad_len
            # x gathered: x[t_v+pad_len - k*d] = x[t_v+pad_len], x[t_v+pad_len-d], ..., x[t_v]
            for t_v in range(T_valid):
                indices = [t_v + pad_len - k * dilation for k in range(K)]
                x_slice = x[:, :, indices]              # (N, C_in, K)
                valid_out[:, 0, t_v] = np.einsum('nck,ck->n', x_slice, kernel) + bias
        else:
            # Centered: position t (original) = t_v + pad_left
            # x gathered: x[t_v + k*d] for k=0..K-1
            for t_v in range(T_valid):
                indices = [t_v + k * dilation for k in range(K)]
                x_slice = x[:, :, indices]              # (N, C_in, K)
                valid_out[:, 0, t_v] = np.einsum('nck,ck->n', x_slice, kernel) + bias

        # Post-pad to restore T (pad_left on left, pad_right on right)
        y = np.pad(valid_out, ((0, 0), (0, 0), (pad_left, pad_right)),
                   mode=pad_mode)

    return y


# ── Dataset generator ─────────────────────────────────────────────────────────

class DatasetGenerator:
    """
    Kernel-DAG pipeline: DAG → per-node ops → propagation → dataset.

    A single seed fixes everything (DAG, kernels, activations, prototypes).
    Each observation re-samples only the root values/functions.
    """

    def __init__(self, seed: int, hp: GeneratorHyperparameters | None = None):
        self.hp = hp or GeneratorHyperparameters()
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        dag_rng = np.random.default_rng(self.rng.integers(0, 2**62))
        self.dag = build_dag(self.hp.dag, self.hp.roles, dag_rng)

        self._sample_dataset_params()
        self._build_operations()

        self.n_classes = self.node_ops[self.dag.target_node.id]['k']

    # ── Dataset-level params ───────────────────────────────────────────

    def _sample_dataset_params(self):
        hp_d = self.hp.dataset

        self.n_features = len(self.dag.feature_nodes)

        t_max = min(hp_d.t_range[1],
                     hp_d.max_feat_times_t // max(1, self.n_features))
        t_min = min(hp_d.t_range[0], t_max)
        self.T = int(self.rng.integers(t_min, t_max + 1))

        self.n_samples = int(self.rng.integers(hp_d.min_samples,
                                                hp_d.max_samples + 1))

        # Variable-length series: sample u_std once per dataset
        if self.rng.random() < hp_d.variable_length_prob:
            lo, hi = hp_d.variable_length_u_std_frac_range
            frac = float(np.exp(self.rng.uniform(np.log(lo), np.log(hi))))
            self.u_std = frac * self.T
        else:
            self.u_std = 0.0  # fixed-length dataset

        # Conv padding strategy (dataset-level flags — shared by ALL series nodes)
        hp_p = self.hp.propagation
        self.causal_padding  = self.rng.random() < hp_p.conv_padding_causal_prob
        self.no_pre_padding  = self.rng.random() < hp_d.no_pre_padding_prob
        self.edge_padding    = self.rng.random() < hp_d.edge_padding_prob

    # ── Helpers ───────────────────────────────────────────────────────

    def _log_uniform_int(self, lo: int, hi: int) -> int:
        if lo == hi:
            return lo
        val = np.exp(self.rng.uniform(np.log(lo + 1), np.log(hi + 1))) - 1
        return int(np.clip(np.round(val), lo, hi))

    def _sample_noise_std(self):
        lo, hi = self.hp.propagation.noise_std_range
        return float(np.exp(self.rng.uniform(np.log(lo), np.log(hi))))

    def _sample_noise_prob(self):
        """Log-uniform noise probability — favors small values (often near 0)."""
        lo, hi = self.hp.propagation.node_noise_prob_range
        return float(np.exp(self.rng.uniform(np.log(lo), np.log(hi))))

    def _sample_series_activation(self) -> str:
        """Sample activation for a temporal (series) node, with 'identity' boosted."""
        acts = list(self.hp.propagation.activation_choices)
        w = self.hp.propagation.series_identity_weight
        weights = np.array([w if a == 'identity' else 1.0 for a in acts])
        weights /= weights.sum()
        return str(self.rng.choice(acts, p=weights))

    # ── Build operations ──────────────────────────────────────────────

    def _build_operations(self):
        hp_p = self.hp.propagation
        T = self.T
        acts = hp_p.activation_choices

        self.node_ops: Dict[int, dict] = {}

        for node in self.dag.nodes:
            if node.layer == 0:
                if node.node_type == 'series':
                    self._build_root_series_ops(node)
                elif node.node_type == 'discrete':
                    k = self._log_uniform_int(*hp_p.discrete_classes_range)
                    class_values = self.rng.uniform(-1, 1, size=(k,))
                    self.node_ops[node.id] = {
                        'kind': 'root_discrete',
                        'k': k,
                        'class_values': class_values,
                    }
                else:  # tabular root
                    std = float(self.rng.uniform(*hp_p.root_normal_std_range))
                    init = str(self.rng.choice(hp_p.root_init_choices))
                    self.node_ops[node.id] = {
                        'kind': 'root_tabular',
                        'init': init,
                        'std': std,
                        'a': float(self.rng.uniform(*hp_p.root_uniform_a_range)),
                    }
                continue

            parent_types = [self.dag.nodes[p].node_type for p in node.parents]

            if node.node_type == 'tabular':
                self._build_tabular_ops(node, parent_types, T, acts)
            elif node.node_type == 'discrete':
                self._build_discrete_ops(node, parent_types, T, hp_p)
            elif node.node_type == 'series':
                self._build_series_ops(node, parent_types, acts)

    def _sample_base_kernel(self, T: int) -> tuple:
        """Sample one base kernel matrix and a description string.

        Returns (K_matrix: (T,T), description: str).
        """
        gp_hp = self.hp.propagation.gp_kernels
        ktype = str(self.rng.choice(gp_hp.kernel_choices))
        if ktype == 'linear':
            params = {
                'sigma': float(self.rng.uniform(*gp_hp.linear_sigma_range)),
                'c':     float(self.rng.uniform(*gp_hp.linear_c_range)),
            }
        elif ktype == 'rbf':
            params = {
                'sigma': float(self.rng.uniform(*gp_hp.rbf_sigma_range)),
                'ell':   float(self.rng.uniform(*gp_hp.rbf_lengthscale_range)),
            }
        else:  # periodic
            params = {
                'sigma':  float(self.rng.uniform(*gp_hp.periodic_sigma_range)),
                'period': float(self.rng.uniform(*gp_hp.periodic_period_range)),
                'ell':    float(self.rng.uniform(*gp_hp.periodic_lengthscale_range)),
            }
        K = build_gp_kernel(T, ktype, params)
        desc = f'{ktype}({", ".join(f"{k}={v:.3g}" for k, v in params.items())})'
        return K, desc

    def _build_root_series_ops(self, node):
        """Build a composed GP kernel for a series root.

        1. Sample J ~ log_uniform(1, 5) base kernels.
        2. While >1 kernel: pick 2, combine with + or ×, replace by result.
        3. The final (T,T) matrix is the covariance for GP(0, K).
        """
        gp_hp = self.hp.propagation.gp_kernels
        T = self.T

        J = self._log_uniform_int(*gp_hp.n_kernels_range)

        kernels = []
        descs = []
        for _ in range(J):
            K_base, d = self._sample_base_kernel(T)
            kernels.append(K_base)
            descs.append(d)

        while len(kernels) > 1:
            # Pick two random distinct indices and combine them
            i, j = self.rng.choice(len(kernels), size=2, replace=False)
            i, j = int(min(i, j)), int(max(i, j))
            op = '+' if self.rng.random() < 0.5 else '*'
            if op == '+':
                combined = kernels[i] + kernels[j]
                combined += 1e-6 * np.eye(T)
            else:
                combined = kernels[i] * kernels[j]
                combined += 1e-6 * np.eye(T)
            desc_combined = f'({descs[i]} {op} {descs[j]})'
            # Remove j first (higher index), then i
            kernels.pop(j); descs.pop(j)
            kernels.pop(i); descs.pop(i)
            kernels.append(combined)
            descs.append(desc_combined)

        K_final = kernels[0]
        kernel_desc = descs[0]

        self.node_ops[node.id] = {
            'kind': 'root_series',
            'kernel_desc': kernel_desc,
            'n_base_kernels': J,
            'cov_matrix': K_final,
        }

    def _input_dim(self, parent_types, T):
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
            'noise_prob': self._sample_noise_prob(),
        }

    def _build_discrete_ops(self, node, parent_types, T, hp_p):
        dim = self._input_dim(parent_types, T)
        k = self._log_uniform_int(*hp_p.discrete_classes_range)

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
            'noise_prob': self._sample_noise_prob(),
        }

    def _build_series_ops(self, node, parent_types, acts):
        """
        Internal series node — single Conv1D.

        c_in = number of parent channels (each parent contributes 1 channel of T).
        Conv1D(c_in, 1, K) with centered weights, bias, dilation, padding, activation.
        """
        hp_p = self.hp.propagation
        T = self.T
        c_in = len(parent_types)

        # Kernel length
        K = int(self.rng.choice(hp_p.conv_kernel_length_choices))

        # Dilation: d = floor(2^x), x ~ U(0, A)
        if K <= 1:
            D = 1
        else:
            A = np.log2(max((T - 1) / (K - 1), 1.0))
            A = min(A, hp_p.conv_max_dilation_exp)
            x_exp = float(self.rng.uniform(0, A))
            D = max(1, int(np.floor(2 ** x_exp)))

        # Weights ~ N(0,1), then centered
        W = self.rng.normal(0, 1.0, size=(c_in, K))
        W = W - W.mean()

        # Bias ~ U(-1, 1)
        bias = float(self.rng.uniform(-1, 1))

        # Padding: dataset-level causal/centered flag
        padding = 'left' if self.causal_padding else 'center'

        # Activation — identity heavily favored for temporal nodes
        act = self._sample_series_activation()

        self.node_ops[node.id] = {
            'kind': 'series',
            'kernel': W,          # (c_in, K)
            'bias': bias,
            'dilation': D,
            'padding': padding,
            'act': act,
            'noise_std': self._sample_noise_std(),
            'noise_prob': self._sample_noise_prob(),
        }

    # ── Propagation ──────────────────────────────────────────────────────

    def propagate(self, n: int, lengths: np.ndarray | None = None):
        """Propagate n observations through the DAG.

        lengths: optional int array of shape (n,) with per-observation effective
                 length T_i = T - u_i.  Root series nodes are zero-padded at
                 positions [T_i:T] before propagation, so all downstream nodes
                 naturally operate on the padded signal.
        """
        T = self.T
        vals: Dict[int, np.ndarray] = {}
        disc: Dict[int, np.ndarray] = {}

        # Layer 0 — roots
        for nid in self.dag.layers[0]:
            ops = self.node_ops[nid]
            if ops['kind'] == 'root_discrete':
                k = ops['k']
                idx = self.rng.integers(0, k, size=(n,))
                vals[nid] = ops['class_values'][idx]
                disc[nid] = idx
            elif ops['kind'] == 'root_series':
                root_vals = sample_gp(ops['cov_matrix'], n, self.rng)  # (n, T)
                if lengths is not None:
                    for i, Li in enumerate(lengths):
                        if Li < T:
                            if self.edge_padding and Li > 0:
                                root_vals[i, Li:] = root_vals[i, Li - 1]
                            else:
                                root_vals[i, Li:] = 0.0
                vals[nid] = root_vals
            else:  # root_tabular
                if ops['init'] == 'normal':
                    vals[nid] = self.rng.normal(0, ops['std'], (n,))
                else:
                    vals[nid] = self.rng.uniform(-ops['a'], ops['a'], (n,))

        # Hidden layers
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
        parts = []
        for pid in node.parents:
            pt = self.dag.nodes[pid].node_type
            v = vals[pid]
            if pt == 'series':
                parts.append(v)                           # (n, T)
            else:
                parts.append(v[:, None])                  # (n, 1)
        return np.concatenate(parts, axis=1)

    # ── Tabular propagation ───────────────────────────────────────────────

    def _prop_tabular(self, node, ops, vals, n):
        x = self._gather_flat(node, vals, n)
        out = np.sum(x * ops['W'][None, :], axis=1) + ops['b']
        out = apply_activation(out, ops['act'])
        if self.rng.random() < ops['noise_prob']:
            out += self.rng.normal(0, ops['noise_std'], size=(n,))
        return out

    # ── Discrete propagation ──────────────────────────────────────────────

    def _prop_discrete(self, node, ops, vals, n):
        x = self._gather_flat(node, vals, n)
        prototypes = ops['prototypes']
        dists = np.sum((x[:, None, :] - prototypes[None, :, :]) ** 2, axis=2)
        indices = np.argmin(dists, axis=1)
        cont_vals = ops['class_values'][indices]
        if self.rng.random() < ops['noise_prob']:
            cont_vals = cont_vals + self.rng.normal(0, ops['noise_std'], size=(n,))
        return cont_vals, indices

    # ── Series propagation (single Conv1D) ────────────────────────────────

    def _prop_series(self, node, ops, vals, n, T):
        """
        1. Gather parent channels → (n, c, T).
           Series parents contribute (n, T), tabular/discrete are tiled.
        2. Conv1D(c_in, 1, K, D) + bias + activation.
        3. Squeeze + iid noise → (n, T).
        """
        channels = []
        for pid in node.parents:
            pt = self.dag.nodes[pid].node_type
            v = vals[pid]
            if pt == 'series':
                channels.append(v)                                    # (n, T)
            else:
                channels.append(np.tile(v[:, None], (1, T)))          # (n, T)

        x = np.stack(channels, axis=1)  # (n, c_in, T)

        pad_mode = 'edge' if self.edge_padding else 'constant'
        x = batch_conv1d(x, ops['kernel'], ops['bias'],
                         ops['dilation'], ops['padding'],
                         pre_pad=not self.no_pre_padding, pad_mode=pad_mode)
        x = apply_activation(x, ops['act'])

        out = x[:, 0, :]  # (n, T)
        if self.rng.random() < ops['noise_prob']:
            out = out + self.rng.normal(0, ops['noise_std'], size=(n, T))
        return out

    # ── Dataset extraction ────────────────────────────────────────────────

    def generate_dataset(self) -> dict | None:
        min_per_class = self.hp.dataset.min_samples_per_class
        feat_ids = [n.id for n in self.dag.feature_nodes]
        target_id = self.dag.target_node.id

        n_propagate = min(self.n_samples * 3, 3000)

        # Variable-length: sample per-observation lengths before propagation
        if self.u_std > 0.0:
            u_raw = np.abs(self.rng.normal(0.0, self.u_std, size=(n_propagate,)))
            u_int = np.clip(np.round(u_raw).astype(int), 0, self.T - 1)
            lengths = self.T - u_int  # T_i = T - u_i, each >= 1
        else:
            lengths = None

        vals, disc = self.propagate(n_propagate, lengths=lengths)
        X = np.stack([vals[fid] for fid in feat_ids], axis=1)  # (n, m, T)
        y = disc[target_id].astype(int)

        hp_d = self.hp.dataset

        if hp_d.warping_prob > 0 and self.rng.random() < hp_d.warping_prob:
            j = self.rng.integers(0, self.n_features)
            a = float(self.rng.uniform(*hp_d.kumaraswamy_a_range))
            b = float(self.rng.uniform(*hp_d.kumaraswamy_b_range))
            apply_kumaraswamy_warp_to_feature(X, j, a, b, self.rng)

        unique, counts = np.unique(y, return_counts=True)
        keep_classes = unique[counts >= min_per_class]
        if len(keep_classes) < 2:
            return None

        mask = np.isin(y, keep_classes)
        X, y = X[mask], y[mask]

        if len(y) > self.n_samples:
            idx = self.rng.choice(len(y), size=self.n_samples, replace=False)
            X, y = X[idx], y[idx]

        unique2, counts2 = np.unique(y, return_counts=True)
        keep2 = unique2[counts2 >= min_per_class]
        if len(keep2) < 2:
            return None

        mask2 = np.isin(y, keep2)
        X, y = X[mask2], y[mask2]

        class_map = {c: i for i, c in enumerate(sorted(keep2))}
        y = np.array([class_map[c] for c in y])
        n_classes = len(keep2)

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
            'X_test': X[test_idx] if len(test_idx) > 0
                      else np.empty((0, self.n_features, self.T)),
            'y_train': y[train_idx],
            'y_test': y[test_idx] if len(test_idx) > 0
                      else np.array([], dtype=int),
            'n_classes': n_classes,
            'n_features': self.n_features,
            'T': self.T,
            'n_samples': len(y),
        }

    # ── Summary ──────────────────────────────────────────────────────────

    def summary(self) -> str:
        var_len_str = (f'  var_len=True(u_std={self.u_std:.1f})'
                       if self.u_std > 0 else '  var_len=False')
        pad_str  = 'causal' if self.causal_padding else 'centered'
        pad_str += '+post' if self.no_pre_padding else '+pre'
        pad_str += '+edge' if self.edge_padding else '+zero'
        lines = [
            f'seed={self.seed}  n_samples={self.n_samples}  T={self.T}  '
            f'n_features={self.n_features}  n_classes={self.n_classes}'
            f'{var_len_str}  conv={pad_str}',
            self.dag.summary(),
            'Node operations:',
        ]
        for nid, ops in sorted(self.node_ops.items()):
            kind = ops['kind']
            if kind == 'root_tabular':
                lines.append(
                    f'  node {nid} (root tabular): {ops["init"]}')
            elif kind == 'root_discrete':
                lines.append(
                    f'  node {nid} (root discrete): k={ops["k"]}')
            elif kind == 'root_series':
                J = ops.get('n_base_kernels', 1)
                desc = ops.get('kernel_desc', '?')
                lines.append(
                    f'  node {nid} (root series): GP(J={J}, K={desc})')
            elif kind == 'tabular':
                np_ = ops.get('noise_prob', 1.0)
                lines.append(
                    f'  node {nid} (tabular): W({len(ops["W"])}), '
                    f'act={ops["act"]}, noise={ops["noise_std"]:.1e}@p={np_:.2f}')
            elif kind == 'discrete':
                np_ = ops.get('noise_prob', 1.0)
                lines.append(
                    f'  node {nid} (discrete): k={ops["k"]}, '
                    f'proto({ops["prototypes"].shape}), '
                    f'noise={ops["noise_std"]:.1e}@p={np_:.2f}')
            elif kind == 'series':
                c_in, K = ops['kernel'].shape
                D = ops['dilation']
                pad = ops['padding']
                np_ = ops.get('noise_prob', 1.0)
                lines.append(
                    f'  node {nid} (series): conv1d({c_in}→1, K={K}, D={D}, '
                    f'pad={pad}) + act={ops["act"]}, '
                    f'noise={ops["noise_std"]:.1e}@p={np_:.2f}')
        return '\n'.join(lines)


# ── Visualisation ──────────────────────────────────────────────────────────────

def visualize_dataset(ds: dict, gen: DatasetGenerator, save_path: str,
                      n_per_class: int = 5):
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
    parser = argparse.ArgumentParser(
        description='Kernel-DAG generator: generate and visualise datasets')
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
            print(f'  y_test  distribution: '
                  f'{np.bincount(ds["y_test"], minlength=ds["n_classes"])}')

        visualize_dataset(ds, gen,
                          os.path.join(out_dir, f'dataset_seed{seed}.png'))
        visualize_dag(gen.dag,
                      os.path.join(dag_dir, f'dag_seed{seed}.png'))


if __name__ == '__main__':
    main()
