"""
Build a TabPFN with overlapping temporal groups (window=16, stride=8).

Architecture:
  - Pretrained TabPFN v2.5 backbone (24 layers, emsize=192, nhead=3, nhid=384)
  - features_per_group = 16
  - Fresh encoder: Linear(32→192) Xavier init
  - Fresh feature positional embedding projection: Xavier init
  - 1 NEW PerFeatureEncoderLayer prepended (Xavier init)
  - 24 pretrained layers (original weights)
  - Pretrained decoder + y_encoder (kept pretrained; reinit breaks in-context learning)

Input preparation (handled externally before model forward):
  1. Pad T to multiple of 16
  2. Extract overlapping windows: window=16, stride=8
     n_groups_per_feat = (T_padded - 16) // 8 + 1
  3. Lay flat: (n, m*T_padded) → (n, m * n_groups_per_feat * 16)
  4. set_temporal_info(model, m_orig, n_groups_per_feat * 16, group_size=16)

The model's standard non-overlapping grouping (einops split by 16) then
correctly picks up each overlapping window as one token.
"""

import types

import numpy as np
import torch
from torch import nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / '00_TabPFN' / 'src'))

from tabpfn import TabPFNClassifier
from tabpfn.architectures.base import get_encoder
from tabpfn.architectures.base.config import ModelConfig
from tabpfn.architectures.base.layer import PerFeatureEncoderLayer
from tabpfn.architectures.base.transformer import LayerStack

from tabpfn_temporal import (
    _reinit_module,
    temporal_add_embeddings,
)

WINDOW = 16
STRIDE = 8


# ─────────────────────────────────────────────────────────────────────────────
# Overlap expansion helpers
# ─────────────────────────────────────────────────────────────────────────────

def pad_T_for_overlap(T, window=WINDOW):
    """Compute the padded T so that overlapping windows are valid.

    Pads to next multiple of window (16) and ensures T >= window.
    """
    T_pad = max(window, T)
    remainder = T_pad % window
    if remainder != 0:
        T_pad += window - remainder
    return T_pad


def expand_overlap_flat(X_flat, m, T, window=WINDOW, stride=STRIDE):
    """Expand (n, m*T) → (n, m * n_groups * window) with overlapping windows.

    T must be >= window and (T - window) % stride == 0.
    Returns (X_expanded_float32, n_groups_per_feat).
    """
    n = X_flat.shape[0]
    X_3d = X_flat.reshape(n, m, T)

    n_groups = (T - window) // stride + 1

    starts = np.arange(n_groups) * stride
    offsets = np.arange(window)
    indices = starts[:, None] + offsets[None, :]  # (n_groups, window)

    # X_3d[:, :, indices] → (n, m, n_groups, window)
    X_windows = X_3d[:, :, indices]

    # Feature-major layout: groups from same feature are contiguous
    # (n, m, n_groups, window) → (n, m * n_groups * window)
    X_expanded = X_windows.reshape(n, m * n_groups * window)

    return X_expanded.astype(np.float32), n_groups


def expand_overlap_flat_torch(X_flat, m, T, window=WINDOW, stride=STRIDE):
    """Torch version: (n, m*T) → (n, m * n_groups * window)."""
    n = X_flat.shape[0]
    X_3d = X_flat.reshape(n, m, T)

    n_groups = (T - window) // stride + 1

    starts = torch.arange(n_groups, device=X_flat.device) * stride
    offsets = torch.arange(window, device=X_flat.device)
    indices = starts[:, None] + offsets[None, :]

    X_windows = X_3d[:, :, indices]  # (n, m, n_groups, window)
    X_expanded = X_windows.reshape(n, m * n_groups * window)

    return X_expanded, n_groups


def pad_and_expand_overlap(X_flat_np, m, T, window=WINDOW, stride=STRIDE):
    """Pad T to valid overlap size, then expand overlapping windows.

    Returns (X_expanded, T_padded, n_groups_per_feat).
    """
    T_pad = pad_T_for_overlap(T, window)

    if T_pad > T:
        n = X_flat_np.shape[0]
        X_3d = X_flat_np.reshape(n, m, T)
        X_3d = np.pad(X_3d, ((0, 0), (0, 0), (0, T_pad - T)),
                       mode='constant', constant_values=0.0)
        X_flat_np = X_3d.reshape(n, m * T_pad).astype(np.float32)

    X_exp, n_groups = expand_overlap_flat(X_flat_np, m, T_pad, window, stride)
    return X_exp, T_pad, n_groups


# ─────────────────────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_overlap_model(device="auto"):
    """Build TabPFN with overlapping groups: 1 fresh layer + 24 pretrained.

    Returns (model, clf, fresh_params, pretrained_params).
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load pretrained TabPFN ──
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

    emsize = model.ninp          # 192
    nhid_factor = model.nhid_factor  # 2
    nhid = emsize * nhid_factor  # 384
    pretrained_layers = list(model.transformer_encoder.layers)
    nlayers_pretrained = len(pretrained_layers)  # 24
    nhead = pretrained_layers[0].self_attn_between_items._nhead  # 3

    print(f"  Pretrained TabPFN: emsize={emsize}, nhead={nhead}, "
          f"nhid_factor={nhid_factor}, nlayers={nlayers_pretrained}")
    print(f"  Overlap config: window={WINDOW}, stride={STRIDE}")

    # ── 1. Change features_per_group to 16 ──
    model.features_per_group = WINDOW
    print(f"  features_per_group: 3 → {WINDOW}")

    # ── 2. Rebuild + reinit input encoder: Linear(32→192) ──
    new_encoder = get_encoder(
        num_features=WINDOW,
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
    _reinit_module(new_encoder)
    model.encoder = new_encoder
    print(f"  [fresh] encoder: Linear({WINDOW * 2}→{emsize}) Xavier")

    # ── 3. Reinit feature embedding projection ──
    _reinit_module(model.feature_positional_embedding_embeddings)
    print(f"  [fresh] feature_positional_embedding_embeddings: Xavier")

    # y_encoder: KEEP PRETRAINED (reinit breaks in-context learning for 24 pretrained layers)

    # ── 4. Create 1 new PerFeatureEncoderLayer ──
    layer_config = ModelConfig(
        max_num_classes=10,
        num_buckets=0,
        emsize=emsize,
        nhead=nhead,
        nlayers=1,
        nhid_factor=nhid_factor,
        features_per_group=WINDOW,
        dropout=0.0,
        recompute_layer=True,
        recompute_attn=False,
    )

    new_layer = PerFeatureEncoderLayer(
        config=layer_config,
        dim_feedforward=nhid,
        activation="gelu",
        zero_init=False,
    )
    _reinit_module(new_layer)
    print(f"  [fresh] 1 new PerFeatureEncoderLayer: Xavier")

    # ── 5. Prepend new layer → 25 total layers ──
    all_layers = [new_layer] + pretrained_layers
    model.transformer_encoder = LayerStack(
        layers=all_layers,
        min_num_layers_layer_dropout=len(all_layers),
    )
    print(f"  LayerStack: {len(all_layers)} layers "
          f"(1 fresh + {nlayers_pretrained} pretrained)")

    # ── 6. Patch add_embeddings ──
    model.add_embeddings = types.MethodType(temporal_add_embeddings, model)
    print(f"  Patched add_embeddings: feature_emb(j) + sinusoidal_PE(group_idx)")

    # ── 7. Collect parameter groups ──
    fresh_modules = [
        model.encoder,
        model.feature_positional_embedding_embeddings,
        new_layer,
    ]
    fresh_ids = set()
    fresh_params = []
    for mod in fresh_modules:
        for p in mod.parameters():
            if id(p) not in fresh_ids:
                fresh_ids.add(id(p))
                fresh_params.append(p)

    pretrained_params = [p for p in model.parameters() if id(p) not in fresh_ids]

    model.to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_fresh = sum(p.numel() for p in fresh_params)
    n_pretrained = n_total - n_fresh
    print(f"  Total parameters: {n_total:,}")
    print(f"    Fresh:      {n_fresh:,}  ({100 * n_fresh / n_total:.1f}%)")
    print(f"    Pretrained: {n_pretrained:,}  ({100 * n_pretrained / n_total:.1f}%)")

    return model, clf, fresh_params, pretrained_params


if __name__ == "__main__":
    model, clf, fresh, pretrained = build_overlap_model(device="cpu")
    print(f"\nFresh params:      {sum(p.numel() for p in fresh):,}")
    print(f"Pretrained params: {sum(p.numel() for p in pretrained):,}")

    # Quick test: expand a dummy input
    m, T = 2, 64
    X = np.random.randn(5, m * T).astype(np.float32)
    X_exp, T_pad, n_groups = pad_and_expand_overlap(X, m, T)
    print(f"\nDummy: m={m}, T={T} → T_pad={T_pad}, n_groups/feat={n_groups}")
    print(f"  Input:    (5, {m * T})")
    print(f"  Expanded: {X_exp.shape}")
