"""
Temporal-aware TabPFN wrapper.

Supports two modes:
  - fpg=3 (v2): keeps pretrained architecture exactly, no new parameters
  - fpg=8 (v3): replaces input encoder Linear(6->192) with Linear(16->192)
    Xavier-initialized from scratch; all other weights remain pretrained

Strategy (both modes):
- Reorder flattened columns so that each group of `fpg` timesteps belongs to
  the SAME original feature: (f_j_t0, f_j_t1, ..., f_j_{t+fpg-1})
- Pad T to a multiple of `fpg` with zeros on the right if needed
- Replace the generic per-group positional embedding with a structured one:
  * Feature embedding: same pseudo-random subspace embedding for all groups
    from the same original feature (m unique embeddings, reusing COL_EMBEDDING)
  * Temporal PE: sinusoidal encoding based on the group index within its feature

Column layout after flattening (for m=2, T=8, fpg=8):
  flat: [f0_t0..f0_t7, f1_t0..f1_t7]
  Groups (fpg=8): (f0_t0..f0_t7) | (f1_t0..f1_t7)

Usage:
    # fpg=3 (backward compatible)
    model, clf = build_temporal_tabpfn(device="cuda")
    set_temporal_info(model, n_features=m, T=T, group_size=3)

    # fpg=8 (new)
    model, clf, new_params = build_temporal_tabpfn_fpg8(device="cuda")
    set_temporal_info(model, n_features=m, T=T, group_size=8)
"""

import math
import types
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
# Pad T to multiple of group_size
# ─────────────────────────────────────────────────────────────────────────────

def pad_to_group(X_flat: np.ndarray, m: int, T: int,
                 group_size: int = 8) -> tuple:
    """
    Pad flattened data so T is a multiple of group_size (zero-pad on the right).

    Args:
        X_flat: (n_samples, m*T) flattened array
        m: number of original features (channels)
        T: number of timesteps
        group_size: number of timesteps per token group

    Returns:
        X_padded: (n_samples, m*T_padded) with T_padded = ceil(T/group_size)*group_size
        T_padded: the new T value
    """
    remainder = T % group_size
    if remainder == 0:
        return X_flat, T

    pad_t = group_size - remainder
    T_padded = T + pad_t
    n = X_flat.shape[0]

    X_3d = X_flat.reshape(n, m, T)
    X_3d_padded = np.pad(X_3d, ((0, 0), (0, 0), (0, pad_t)),
                         mode='constant', constant_values=0.0)
    return X_3d_padded.reshape(n, m * T_padded).astype(np.float32), T_padded


def pad_to_group3(X_flat: np.ndarray, m: int, T: int) -> tuple:
    """Backward-compatible wrapper: pad T to multiple of 3."""
    return pad_to_group(X_flat, m, T, group_size=3)


# ─────────────────────────────────────────────────────────────────────────────
# Monkey-patch add_embeddings for temporal awareness
# ─────────────────────────────────────────────────────────────────────────────

def temporal_add_embeddings(self, x, y, *, data_dags, num_features, seq_len,
                            cache_embeddings=False, use_cached_embeddings=False):
    """
    Replacement for PerFeatureTransformer.add_embeddings.

    Each group of `group_size` consecutive columns represents consecutive
    timesteps from the same original feature. Layout: m blocks of
    T_padded/group_size groups.

    For each group we add:
      feature_emb(j)  -- same for all groups from feature j (pseudo-random subspace)
      temporal_PE(g)  -- based on the group index within its feature

    x shape: (batch, seq_len, n_groups, emsize)
    n_groups = m * (T_padded / group_size)
    """
    if use_cached_embeddings and self.cached_feature_positional_embeddings is not None:
        x += self.cached_feature_positional_embeddings[None, None]
        return x, y

    n_groups = x.shape[2]
    emsize = x.shape[3]

    info = getattr(self, '_temporal_info', (None, None, None))
    if len(info) == 2:
        m, T_padded = info
        group_size = self.features_per_group
    else:
        m, T_padded, group_size = info

    if m is None or T_padded is None:
        return _default_subspace_embeddings(self, x, y,
                                            cache_embeddings=cache_embeddings)

    groups_per_feature = T_padded // group_size

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
    feat_emb = self.feature_positional_embedding_embeddings(feat_raw)

    # (m, emsize) -> (m * groups_per_feature, emsize) = (n_groups, emsize)
    feat_emb_expanded = feat_emb.unsqueeze(1).expand(
        m, groups_per_feature, emsize).reshape(m * groups_per_feature, emsize)

    # ── Temporal PE (sinusoidal, per group within feature) ──
    temporal_pe = sinusoidal_pe(groups_per_feature, emsize,
                                device=x.device, dtype=x.dtype)
    temporal_pe_tiled = temporal_pe.unsqueeze(0).expand(
        m, groups_per_feature, emsize).reshape(m * groups_per_feature, emsize)

    combined = feat_emb_expanded + temporal_pe_tiled

    if combined.shape[0] < n_groups:
        pad = torch.zeros(n_groups - combined.shape[0], emsize,
                          device=x.device, dtype=x.dtype)
        combined = torch.cat([combined, pad], dim=0)
    elif combined.shape[0] > n_groups:
        combined = combined[:n_groups]

    x += combined[None, None]

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
# Build temporal-aware model (fpg=3, backward compatible, no new params)
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
        inference_config={"FEATURE_SHIFT_METHOD": None},
    )
    clf._initialize_model_variables()
    model = clf.model_

    print(f"  features_per_group={model.features_per_group} (kept as-is)")
    print(f"  emsize={model.ninp}, nlayers={len(list(model.transformer_encoder.layers))}")

    model.add_embeddings = types.MethodType(temporal_add_embeddings, model)
    print(f"  Patched add_embeddings: feature_emb(j) + sinusoidal_PE(group_idx)")
    print(f"  No new parameters — 100% pretrained weights")

    model.to(device if device != "auto" else "cpu")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")

    return model, clf


# ─────────────────────────────────────────────────────────────────────────────
# Build temporal-aware model (fpg=8, new input encoder)
# ─────────────────────────────────────────────────────────────────────────────

def build_temporal_tabpfn_fpg8(device: str = "auto"):
    """
    Build a TabPFN model for temporal time-series with features_per_group=8.

    1. Load pretrained TabPFN v2.5 (fpg=3)
    2. Replace features_per_group 3 -> 8
    3. Rebuild the input encoder with correct input dimension (8+8=16 -> 192)
       using Xavier initialization (cannot reuse pretrained 3+3=6 -> 192 weights)
    4. All other weights (transformer layers, y_encoder, decoder, subspace
       projection) remain pretrained.

    Returns:
        model: the modified PerFeatureTransformer
        clf: the TabPFNClassifier
        new_encoder_params: list of nn.Parameter that are freshly initialized
    """
    from tabpfn.architectures.base import get_encoder
    from tabpfn.architectures.base.config import ModelConfig

    clf = TabPFNClassifier(
        device=device,
        n_estimators=1,
        ignore_pretraining_limits=True,
        fit_mode="batched",
        differentiable_input=False,
        inference_config={"FEATURE_SHIFT_METHOD": None},
    )
    clf._initialize_model_variables()
    model = clf.model_

    old_fpg = model.features_per_group
    new_fpg = 8
    emsize = model.ninp
    nlayers = len(list(model.transformer_encoder.layers))

    print(f"  features_per_group: {old_fpg} -> {new_fpg}")
    print(f"  emsize={emsize}, nlayers={nlayers}")

    # ── Change features_per_group ──
    model.features_per_group = new_fpg

    # ── Rebuild encoder with fpg=8 ──
    # The factory uses the same config as the pretrained model but with new fpg
    new_encoder = get_encoder(
        num_features=new_fpg,
        embedding_size=emsize,
        remove_empty_features=True,
        remove_duplicate_features=False,
        nan_handling_enabled=True,
        normalize_on_train_only=True,
        normalize_to_ranking=False,
        normalize_x=True,
        remove_outliers=False,
        normalize_by_used_features=True,
        encoder_use_bias=False,
        encoder_type="linear",
    )

    # Xavier init the new Linear layer
    from tabpfn.architectures.base.encoders import LinearInputEncoderStep
    for step in new_encoder:
        if isinstance(step, LinearInputEncoderStep):
            nn.init.xavier_uniform_(step.layer.weight)
            if step.layer.bias is not None:
                nn.init.zeros_(step.layer.bias)

    # Collect new encoder params before moving to device
    new_encoder_params = list(new_encoder.parameters())

    model.encoder = new_encoder
    print(f"  New encoder: Linear({new_fpg * 2}->{emsize}) Xavier init")

    # ── Patch add_embeddings ──
    model.add_embeddings = types.MethodType(temporal_add_embeddings, model)
    print(f"  Patched add_embeddings: feature_emb(j) + sinusoidal_PE(group_idx)")

    model.to(device if device != "auto" else "cpu")

    n_params = sum(p.numel() for p in model.parameters())
    n_new = sum(p.numel() for p in new_encoder_params)
    n_pretrained = n_params - n_new
    print(f"  Total parameters: {n_params:,}")
    print(f"    Pretrained: {n_pretrained:,}")
    print(f"    New (encoder): {n_new:,}")

    return model, clf, new_encoder_params


# ─────────────────────────────────────────────────────────────────────────────
# Temporal info management
# ─────────────────────────────────────────────────────────────────────────────

def set_temporal_info(model, n_features: int, T: int, group_size: int = 8):
    """
    Set temporal metadata on the model before a forward pass.
    T should already be padded to a multiple of group_size.
    """
    gs = group_size
    T_padded = T if T % gs == 0 else T + (gs - T % gs)
    model._temporal_info = (n_features, T_padded, group_size)


def clear_temporal_info(model):
    """Clear temporal metadata (reverts to default subspace embeddings)."""
    model._temporal_info = (None, None, None)
