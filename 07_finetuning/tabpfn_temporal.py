"""
Temporal-aware TabPFN wrapper (v2 — keeps features_per_group=3).

Strategy:
- Keep the pretrained TabPFN v2.5 architecture EXACTLY as-is (fpg=3, all weights)
- Reorder flattened columns so that each group of 3 contains consecutive timesteps
  from the SAME original feature: (f_j_t, f_j_{t+1}, f_j_{t+2})
- Pad T to a multiple of 3 with zeros on the right if needed
- Replace the generic per-group positional embedding with a structured one:
  * Feature embedding: same pseudo-random subspace embedding for all groups from
    the same original feature (m unique embeddings, reusing COL_EMBEDDING)
  * Temporal PE: sinusoidal encoding based on the group's central timestep index
- NO new parameters, NO encoder changes — pure fine-tuning of existing weights

Column layout after reordering (for m=2, T=6):
  Original flat: [f0_t0, f0_t1, f0_t2, f0_t3, f0_t4, f0_t5, f1_t0, ..., f1_t5]
  Reordered:     [f0_t0, f0_t1, f0_t2, f0_t3, f0_t4, f0_t5, f1_t0, ..., f1_t5]
  Groups (fpg=3): (f0_t0,f0_t1,f0_t2) | (f0_t3,f0_t4,f0_t5) | (f1_t0,...) | ...

  Since numpy reshape(-1) already gives [f0_all_t, f1_all_t, ...], and within each
  feature the timesteps are already consecutive, the default flatten order is ALREADY
  correct for this grouping. No column reordering needed!

Usage:
    model, clf = build_temporal_tabpfn(device="cuda")
    # Before each forward pass:
    set_temporal_info(model, n_features=m, T=T)
"""

import math
import copy
from functools import partial

import numpy as np
import torch
from torch import nn

from tabpfn import TabPFNClassifier


# ─────────────────────────────────────────────────────────────────────────────
# Sinusoidal positional encoding (standard Vaswani et al.)
# ─────────────────────────────────────────────────────────────────────────────

def sinusoidal_pe(length: int, d_model: int, device=None, dtype=None):
    """Sinusoidal PE of shape (length, d_model)."""
    pe = torch.zeros(length, d_model, device=device, dtype=dtype)
    pos = torch.arange(0, length, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
        * -(math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
    return pe.to(dtype=dtype) if dtype is not None else pe


# ─────────────────────────────────────────────────────────────────────────────
# Pad T to multiple of 3
# ─────────────────────────────────────────────────────────────────────────────

def pad_to_group3(X_flat: np.ndarray, m: int, T: int) -> tuple:
    """
    Pad flattened data so T is a multiple of 3 (zero-pad on the right).

    Args:
        X_flat: (n_samples, m*T) flattened array
        m: number of original features
        T: number of timesteps

    Returns:
        X_padded: (n_samples, m*T_padded) with T_padded = ceil(T/3)*3
        T_padded: the new T value
    """
    remainder = T % 3
    if remainder == 0:
        return X_flat, T

    pad_t = 3 - remainder
    T_padded = T + pad_t
    n = X_flat.shape[0]

    # Reshape to (n, m, T), pad, reshape back
    X_3d = X_flat.reshape(n, m, T)
    X_3d_padded = np.pad(X_3d, ((0, 0), (0, 0), (0, pad_t)),
                         mode='constant', constant_values=0.0)
    return X_3d_padded.reshape(n, m * T_padded).astype(np.float32), T_padded


# ─────────────────────────────────────────────────────────────────────────────
# Monkey-patch add_embeddings for temporal awareness (fpg=3)
# ─────────────────────────────────────────────────────────────────────────────

def temporal_add_embeddings(self, x, y, *, data_dags, num_features, seq_len,
                            cache_embeddings=False, use_cached_embeddings=False):
    """
    Replacement for PerFeatureTransformer.add_embeddings.

    With fpg=3, each group of 3 consecutive columns represents 3 consecutive
    timesteps from the same original feature.  Layout: m blocks of T/3 groups.

    For each group we add:
      feature_emb(j)  — same for all groups from feature j (pseudo-random subspace)
      temporal_PE(g)  — based on the group index within its feature

    x shape: (batch, seq_len, n_groups, emsize)
    n_groups = m * (T_padded / 3)
    """
    if use_cached_embeddings and self.cached_feature_positional_embeddings is not None:
        x += self.cached_feature_positional_embeddings[None, None]
        return x, y

    n_groups = x.shape[2]
    emsize = x.shape[3]

    m, T_padded = getattr(self, '_temporal_info', (None, None))
    if m is None or T_padded is None:
        # Fallback: use original subspace embeddings (non-time-series data)
        return _default_subspace_embeddings(self, x, y,
                                            cache_embeddings=cache_embeddings)

    groups_per_feature = T_padded // 3  # number of groups per original feature

    # ── Feature embedding (subspace, reusing pretrained projection) ──
    rng = torch.Generator(device=x.device).manual_seed(self.random_embedding_seed)
    feat_raw = torch.randn(
        (m, emsize // 4),
        device=x.device, dtype=x.dtype,
        generator=rng,
    )
    from tabpfn.architectures.base.transformer import COL_EMBEDDING
    if feat_raw.shape[1] == 48 and self.random_embedding_seed == 42:
        feat_raw[:min(m, 2000)] = COL_EMBEDDING[:min(m, 2000)].to(
            device=feat_raw.device, dtype=feat_raw.dtype
        )
    # Project: (m, 48) → (m, 192)
    feat_emb = self.feature_positional_embedding_embeddings(feat_raw)

    # Repeat each feature embedding for its groups_per_feature groups
    # (m, emsize) → (m * groups_per_feature, emsize) = (n_groups, emsize)
    feat_emb_expanded = feat_emb.unsqueeze(1).expand(
        m, groups_per_feature, emsize).reshape(m * groups_per_feature, emsize)

    # ── Temporal PE (sinusoidal, per group within feature) ──
    temporal_pe = sinusoidal_pe(groups_per_feature, emsize,
                                device=x.device, dtype=x.dtype)
    # Tile for all m features: (groups_per_feature, emsize) → (m * gpf, emsize)
    temporal_pe_tiled = temporal_pe.unsqueeze(0).expand(
        m, groups_per_feature, emsize).reshape(m * groups_per_feature, emsize)

    # ── Combine ──
    combined = feat_emb_expanded + temporal_pe_tiled

    # Handle padding groups if n_groups != m * groups_per_feature
    if combined.shape[0] < n_groups:
        pad = torch.zeros(n_groups - combined.shape[0], emsize,
                          device=x.device, dtype=x.dtype)
        combined = torch.cat([combined, pad], dim=0)
    elif combined.shape[0] > n_groups:
        combined = combined[:n_groups]

    x += combined[None, None]  # broadcast over (batch, seq_len)

    self.cached_embeddings = None
    if cache_embeddings:
        self.cached_embeddings = combined

    return x, y


def _default_subspace_embeddings(self, x, y, cache_embeddings=False):
    """Fallback: use original subspace embeddings when no temporal info."""
    n_groups = x.shape[2]
    emsize = x.shape[3]

    rng = torch.Generator(device=x.device).manual_seed(self.random_embedding_seed)
    embs = torch.randn(
        (n_groups, emsize // 4),
        device=x.device, dtype=x.dtype,
        generator=rng,
    )
    from tabpfn.architectures.base.transformer import COL_EMBEDDING
    if embs.shape[1] == 48 and self.random_embedding_seed == 42:
        embs[:min(n_groups, 2000)] = COL_EMBEDDING[:min(n_groups, 2000)].to(
            device=embs.device, dtype=embs.dtype
        )

    embs = self.feature_positional_embedding_embeddings(embs)
    x += embs[None, None]

    self.cached_embeddings = None
    if cache_embeddings:
        self.cached_embeddings = embs

    return x, y


# ─────────────────────────────────────────────────────────────────────────────
# Build the temporal-aware model (keeps fpg=3, no new params)
# ─────────────────────────────────────────────────────────────────────────────

def build_temporal_tabpfn(device: str = "auto"):
    """
    Build a TabPFN model modified for temporal time-series data.

    Architecture is IDENTICAL to pretrained v2.5 (fpg=3, same encoder, same
    weights). Only add_embeddings is patched to use structured temporal +
    feature embeddings.

    Returns:
        model: the modified PerFeatureTransformer
        clf: the TabPFNClassifier (for preprocessing pipeline access)
    """
    clf = TabPFNClassifier(
        device=device,
        n_estimators=1,
        ignore_pretraining_limits=True,
        fit_mode="batched",
        differentiable_input=False,
        inference_config={"FEATURE_SHIFT_METHOD": None},  # Disable shuffle
    )
    clf._initialize_model_variables()
    model = clf.model_

    print(f"  features_per_group={model.features_per_group} (kept as-is)")
    print(f"  emsize={model.ninp}, nlayers={len(list(model.transformer_encoder.layers))}")

    # Only change: monkey-patch add_embeddings
    import types
    model.add_embeddings = types.MethodType(temporal_add_embeddings, model)
    print(f"  Patched add_embeddings: feature_emb(j) + sinusoidal_PE(group_idx)")
    print(f"  No new parameters — 100% pretrained weights")

    model.to(device if device != "auto" else "cpu")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")

    return model, clf


def set_temporal_info(model, n_features: int, T: int):
    """
    Set temporal metadata on the model before a forward pass.
    T_padded is computed here (pad to multiple of 3).
    """
    T_padded = T if T % 3 == 0 else T + (3 - T % 3)
    model._temporal_info = (n_features, T_padded)


def clear_temporal_info(model):
    """Clear temporal metadata (reverts to default subspace embeddings)."""
    model._temporal_info = (None, None)
