"""
Overlap TabPFN model: MLP encoder with overlapping temporal groups.

Architecture summary
--------------------
- Pretrained TabPFN v2.5 backbone (24 layers, emsize=192, nhead=3, nhid=384).
- features_per_group = 16  (each token = 1 channel × 16 timesteps).
- Fresh-initialized (Xavier):
    * Encoder: MLP(32→96→GELU→96→GELU→192) via TabPFN's get_encoder pipeline
      (includes NanHandlingEncoderStep, InputNormalizationEncoderStep,
       VariableNumFeaturesEncoderStep).
- Everything else is unchanged from the pretrained checkpoint:
    * add_embeddings: TabPFN's built-in "subspace" mode, seed=42.
      Each token gets a unique COL_EMBEDDING-aligned 48-d vector projected
      through the pretrained Linear(48→192).  No monkey-patching needed.
    * 24 transformer layers, y_encoder, decoder — all pretrained.

Tokenization (overlap expansion, done externally before every forward pass)
---------------------------------------------------------------------------
  1. Pad T so (T_pad - WINDOW) % STRIDE == 0 and T_pad >= WINDOW.
  2. Extract overlapping windows: window=16, stride=12.
     n_groups = (T_padded - 16) // 12 + 1
  3. Flat layout (feature-major): groups from same channel are contiguous.
     [ch0_g0(16), ch0_g1(16), ..., ch1_g0(16), ch1_g1(16), ...]
  TabPFN's einops split by features_per_group=16 picks up each window.

Usage
-----
  model, clf, fresh_params, pretrained_params = build_overlap_model(device="cuda")
  X_exp, T_pad, n_groups = pad_and_expand_overlap(X_flat, m, T)
  # … forward pass (no set_temporal_info needed) …
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent / "00_TabPFN" / "src"))

from tabpfn import TabPFNClassifier
from tabpfn.architectures.base import get_encoder

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

WINDOW: int = 16           # temporal window size per token (features_per_group)
STRIDE: int = 12           # temporal stride between windows
ENCODER_HIDDEN: int = 96   # MLP hidden dim (first two layers)
ENCODER_LAYERS: int = 3    # MLP depth: 32→96,GELU→96,GELU→192 (no GELU after last)


# ─────────────────────────────────────────────────────────────────────────────
# Weight (re-)initialisation helper
# ─────────────────────────────────────────────────────────────────────────────

def _reinit_module(module: nn.Module) -> None:
    """Xavier-reinitialise all Linear and LayerNorm submodules in-place."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)


# ─────────────────────────────────────────────────────────────────────────────
# Overlap expansion helpers
# ─────────────────────────────────────────────────────────────────────────────

def pad_T_for_overlap(T: int, window: int = WINDOW, stride: int = STRIDE) -> int:
    """Pad T so that (T_pad - window) % stride == 0 and T_pad >= window.

    For stride=12, window=16 the old 'pad to multiple of window' formula is
    wrong (12 does not divide 16). This correctly pads T to the smallest valid
    T_pad >= max(T, window) satisfying the overlap condition.
    """
    T_pad = max(window, T)
    remainder = (T_pad - window) % stride
    if remainder != 0:
        T_pad += stride - remainder
    return T_pad


def expand_overlap_flat(
    X_flat: np.ndarray, m: int, T: int,
    window: int = WINDOW, stride: int = STRIDE,
) -> tuple[np.ndarray, int]:
    """Expand (n, m*T) → (n, m * n_groups * window) with overlapping windows.

    T must be >= window and (T - window) % stride == 0.
    Returns (X_expanded_float32, n_groups_per_feat).
    """
    n = X_flat.shape[0]
    X_3d = X_flat.reshape(n, m, T)
    n_groups = (T - window) // stride + 1
    starts = np.arange(n_groups) * stride
    offsets = np.arange(window)
    indices = starts[:, None] + offsets[None, :]   # (n_groups, window)
    X_windows = X_3d[:, :, indices]                # (n, m, n_groups, window)
    X_expanded = X_windows.reshape(n, m * n_groups * window)
    return X_expanded.astype(np.float32), n_groups


def pad_and_expand_overlap(
    X_flat_np: np.ndarray, m: int, T: int,
    window: int = WINDOW, stride: int = STRIDE,
) -> tuple[np.ndarray, int, int]:
    """Pad T to valid overlap size, then expand overlapping windows.

    Returns (X_expanded, T_padded, n_groups_per_feat).
    """
    T_pad = pad_T_for_overlap(T, window, stride)
    if T_pad > T:
        n = X_flat_np.shape[0]
        X_3d = X_flat_np.reshape(n, m, T)
        X_3d = np.pad(X_3d, ((0, 0), (0, 0), (0, T_pad - T)),
                       mode="constant", constant_values=0.0)
        X_flat_np = X_3d.reshape(n, m * T_pad).astype(np.float32)
    X_exp, n_groups = expand_overlap_flat(X_flat_np, m, T_pad, window, stride)
    return X_exp, T_pad, n_groups


def pad_to_group(
    X_flat: np.ndarray, m: int, T: int, group_size: int = 8,
) -> tuple[np.ndarray, int]:
    """Zero-pad T to next multiple of group_size (right-pad).

    Returns (X_padded, T_padded).
    """
    remainder = T % group_size
    if remainder == 0:
        return X_flat, T
    pad_t = group_size - remainder
    T_padded = T + pad_t
    n = X_flat.shape[0]
    X_3d = X_flat.reshape(n, m, T)
    X_3d_padded = np.pad(X_3d, ((0, 0), (0, 0), (0, pad_t)),
                         mode="constant", constant_values=0.0)
    return X_3d_padded.reshape(n, m * T_padded).astype(np.float32), T_padded


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compat stubs (called by inference.py / worker_evaluator_v2.py;
# no longer affect embeddings since we use TabPFN's built-in add_embeddings)
# ─────────────────────────────────────────────────────────────────────────────

def set_temporal_info(model, n_features: int, T: int, group_size: int = WINDOW) -> None:
    """No-op kept for backward compatibility with inference.py / worker_evaluator_v2.py."""


def clear_temporal_info(model) -> None:
    """No-op kept for backward compatibility."""


# ─────────────────────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_overlap_model(device: str = "auto") -> tuple:
    """Build TabPFN with overlapping groups: custom MLP encoder, everything else unchanged.

    The only architectural change vs vanilla TabPFN is:
      - features_per_group: 3 → WINDOW (16)
      - encoder: replaced with MLP(32→96→GELU→96→GELU→192, no bias)
        via TabPFN's get_encoder pipeline (NanHandling, InputNorm, VariableNumFeatures).

    Everything else (embeddings, all 24 transformer layers, y_encoder, decoder)
    is kept from the pretrained checkpoint without modification.

    Returns:
        model             — modified PerFeatureTransformer
        clf               — TabPFNClassifier (for fit_from_preprocessed API)
        fresh_params      — list of fresh nn.Parameter objects (encoder only)
        pretrained_params — list of pretrained nn.Parameter objects
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

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

    emsize = model.ninp
    nlayers = len(list(model.transformer_encoder.layers))
    nhead = list(model.transformer_encoder.layers)[0].self_attn_between_items._nhead

    print(f"  Pretrained TabPFN: emsize={emsize}, nhead={nhead}, nlayers={nlayers}")
    print(f"  Overlap config: window={WINDOW}, stride={STRIDE}")

    # ── 1. features_per_group = 16 (1 channel × 16 timesteps) ──
    model.features_per_group = WINDOW
    print(f"  features_per_group: 3 → {WINDOW}")

    # ── 2. Replace encoder with MLP(32→96,GELU→96,GELU→192, no bias) ──
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
        encoder_type="mlp",
        encoder_mlp_hidden_dim=ENCODER_HIDDEN,
        encoder_mlp_num_layers=ENCODER_LAYERS,
    )
    _reinit_module(new_encoder)
    model.encoder = new_encoder
    print(f"  [fresh] encoder: MLP({WINDOW * 2}→{ENCODER_HIDDEN},GELU"
          f"{'→' + str(ENCODER_HIDDEN) + ',GELU' * (ENCODER_LAYERS - 2)}→{emsize}, no bias) Xavier")

    # Embeddings, add_embeddings, seed, and all transformer layers are left
    # exactly as loaded from the pretrained checkpoint.
    emb_seed = model.random_embedding_seed
    emb_type = model.feature_positional_embedding
    print(f"  [pretrained] embeddings: type={emb_type!r}  seed={emb_seed}  (unchanged)")

    # ── 3. Collect parameter groups ──
    #   Only the encoder is fresh; everything else is pretrained.
    fresh_ids: set = set()
    fresh_params: list = []
    for p in model.encoder.parameters():
        if id(p) not in fresh_ids:
            fresh_ids.add(id(p))
            fresh_params.append(p)

    pretrained_params = [p for p in model.parameters() if id(p) not in fresh_ids]

    model.to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_fresh = sum(p.numel() for p in fresh_params)
    print(f"  Total params: {n_total:,}  fresh: {n_fresh:,}  pretrained: {n_total - n_fresh:,}")

    return model, clf, fresh_params, pretrained_params


# ─────────────────────────────────────────────────────────────────────────────
# Legacy builder (fpg=8, used only in final_evaluation.py for old checkpoints)
# ─────────────────────────────────────────────────────────────────────────────

def build_temporal_tabpfn_fpg8(
    device: str = "auto", n_fresh_transformer_layers: int = 4
) -> tuple:
    """Build a TabPFN with features_per_group=8 (V3 architecture, non-overlap).

    Fresh-initialized:
      1. model.encoder — Linear(16→192), Xavier
      2. model.feature_positional_embedding_embeddings — Xavier
      3. transformer_encoder.layers[:n_fresh_transformer_layers] — Xavier

    Returns (model, clf, fresh_params).
    Only needed to load V3 checkpoints in final_evaluation.py.
    """
    from tabpfn.architectures.base.encoders import LinearInputEncoderStep

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

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

    new_fpg = 8
    emsize = model.ninp
    layers = list(model.transformer_encoder.layers)
    n_fresh = min(n_fresh_transformer_layers, len(layers))

    model.features_per_group = new_fpg

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
    for step in new_encoder:
        if isinstance(step, LinearInputEncoderStep):
            nn.init.xavier_uniform_(step.layer.weight)
            if step.layer.bias is not None:
                nn.init.zeros_(step.layer.bias)
    model.encoder = new_encoder

    _reinit_module(model.feature_positional_embedding_embeddings)
    for layer in layers[:n_fresh]:
        _reinit_module(layer)

    seen_ids: set = set()
    fresh_params: list = []
    for mod in [model.encoder, model.feature_positional_embedding_embeddings] + list(layers[:n_fresh]):
        for p in mod.parameters():
            if id(p) not in seen_ids:
                seen_ids.add(id(p))
                fresh_params.append(p)

    model.to(device)
    return model, clf, fresh_params


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model, clf, fresh, pretrained = build_overlap_model(device="cpu")
    print(f"\nFresh params:      {sum(p.numel() for p in fresh):,}")
    print(f"Pretrained params: {sum(p.numel() for p in pretrained):,}")

    m, T = 2, 64
    X = np.random.randn(5, m * T).astype(np.float32)
    X_exp, T_pad, n_groups = pad_and_expand_overlap(X, m, T)
    print(f"\nDummy: m={m}, T={T} → T_pad={T_pad}, n_groups/feat={n_groups}")
    print(f"  Input:    (5, {m * T})")
    print(f"  Expanded: {X_exp.shape}")
