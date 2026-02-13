"""
Temporal-aware TabPFN wrapper.

Modifies the TabPFN v2.5 model to support flattened time-series data by:
1. Setting features_per_group=1 (each column = one token)
2. Replacing the generic feature positional embedding with a structured one:
   - Feature embedding: same pseudo-random embedding for all timesteps of the
     same original feature (reused from TabPFN's "subspace" mechanism)
   - Temporal PE: sinusoidal positional encoding per timestep
3. Loading pretrained weights with strict=False (only encoder linear changes)

Usage:
    model, clf = build_temporal_tabpfn(device="cuda")
    # model is ready for fine-tuning with temporal awareness
"""

import math
import copy
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tabpfn import TabPFNClassifier


# ─────────────────────────────────────────────────────────────────────────────
# Sinusoidal positional encoding (standard Vaswani et al.)
# ─────────────────────────────────────────────────────────────────────────────

def sinusoidal_pe(T: int, d_model: int, device: torch.device = None,
                  dtype: torch.dtype = None) -> torch.Tensor:
    """
    Generate sinusoidal positional encoding of shape (T, d_model).

    PE(t, 2i)   = sin(t / 10000^(2i/d_model))
    PE(t, 2i+1) = cos(t / 10000^(2i/d_model))
    """
    pe = torch.zeros(T, d_model, device=device, dtype=dtype)
    position = torch.arange(0, T, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
        * -(math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
    return pe.to(dtype=dtype) if dtype is not None else pe


# ─────────────────────────────────────────────────────────────────────────────
# Monkey-patch add_embeddings
# ─────────────────────────────────────────────────────────────────────────────

def temporal_add_embeddings(self, x, y, *, data_dags, num_features, seq_len,
                            cache_embeddings=False, use_cached_embeddings=False):
    """
    Replacement for PerFeatureTransformer.add_embeddings that adds:
    1. Feature embedding: pseudo-random subspace embedding, one per original
       feature m, repeated for all T timesteps of that feature.
    2. Temporal PE: sinusoidal, one per timestep t, shared across features.

    Requires self._temporal_info = (n_original_features, T) to be set before
    the forward pass.

    x shape: (batch, seq_len, n_groups, emsize) where n_groups = m*T
    """
    if use_cached_embeddings and self.cached_feature_positional_embeddings is not None:
        x += self.cached_feature_positional_embeddings[None, None]
        return x, y

    n_groups = x.shape[2]   # total columns = m * T
    emsize = x.shape[3]

    # ── Determine m and T ──
    m, T = getattr(self, '_temporal_info', (None, None))
    if m is None or T is None:
        # Fallback: no temporal info, use default subspace embedding
        # (for evaluation on non-time-series data)
        return _default_subspace_embeddings(self, x, y,
                                            cache_embeddings=cache_embeddings)

    # ── Feature embedding (subspace mechanism from TabPFN) ──
    # Generate m pseudo-random vectors (one per original feature)
    rng = torch.Generator(device=x.device).manual_seed(self.random_embedding_seed)
    feat_raw = torch.randn(
        (m, emsize // 4),
        device=x.device, dtype=x.dtype,
        generator=rng,
    )
    # Use hardcoded COL_EMBEDDING when available (matches pretrained)
    from tabpfn.architectures.base.transformer import COL_EMBEDDING
    if feat_raw.shape[1] == 48 and self.random_embedding_seed == 42:
        feat_raw[:min(m, 2000)] = COL_EMBEDDING[:min(m, 2000)].to(
            device=feat_raw.device, dtype=feat_raw.dtype
        )

    # Project through pretrained nn.Linear(48, 192)
    feat_emb = self.feature_positional_embedding_embeddings(feat_raw)  # (m, emsize)

    # Repeat each feature embedding T times: (m, emsize) → (m*T, emsize)
    # Column order is [f0_t0, f0_t1, ..., f0_{T-1}, f1_t0, ..., f_{m-1}_{T-1}]
    feat_emb_repeated = feat_emb.unsqueeze(1).expand(m, T, emsize).reshape(m * T, emsize)

    # ── Temporal PE (sinusoidal, no learned params) ──
    temporal_pe = sinusoidal_pe(T, emsize, device=x.device, dtype=x.dtype)  # (T, emsize)
    # Tile for all m features: (T,) → (m*T,)
    temporal_pe_tiled = temporal_pe.unsqueeze(0).expand(m, T, emsize).reshape(m * T, emsize)

    # ── Combine and add ──
    # Pad or truncate if n_groups != m*T (due to zero-padding to fpg multiple)
    combined = feat_emb_repeated + temporal_pe_tiled  # (m*T, emsize)
    if combined.shape[0] < n_groups:
        # Pad with zeros for any extra padding columns
        pad = torch.zeros(n_groups - combined.shape[0], emsize,
                          device=x.device, dtype=x.dtype)
        combined = torch.cat([combined, pad], dim=0)
    elif combined.shape[0] > n_groups:
        combined = combined[:n_groups]

    x += combined[None, None]  # broadcast over (batch, seq_len)

    # Cache if needed
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
# Build the temporal-aware model
# ─────────────────────────────────────────────────────────────────────────────

def build_temporal_tabpfn(device: str = "auto"):
    """
    Build a TabPFN model modified for temporal time-series data.

    Returns:
        model: the modified PerFeatureTransformer (ready for fine-tuning)
        clf: the TabPFNClassifier (for preprocessing pipeline access)
    """
    # ── Step 1: Load pretrained model normally ──
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

    # Save pretrained state dict
    pretrained_sd = copy.deepcopy(model.state_dict())

    # ── Step 2: Record original config info ──
    original_fpg = model.features_per_group
    emsize = model.ninp
    print(f"  Original features_per_group={original_fpg}, emsize={emsize}")

    # ── Step 3: Change features_per_group to 1 ──
    model.features_per_group = 1

    # ── Step 4: Rebuild the encoder for num_features=1 ──
    # The encoder has internal steps (NanHandling, VariableNumFeatures, etc.)
    # that are all parameterized by num_features=fpg. We need to rebuild
    # the entire encoder for num_features=1.
    from tabpfn.architectures.base import get_encoder
    new_encoder = get_encoder(
        num_features=1,  # features_per_group=1
        embedding_size=emsize,
        remove_empty_features=True,   # match v2.5 config
        remove_duplicate_features=False,
        nan_handling_enabled=True,
        normalize_on_train_only=True,
        normalize_to_ranking=False,
        normalize_x=True,
        remove_outliers=False,
        normalize_by_used_features=True,
        encoder_use_bias=False,  # v2.5 has encoder_use_bias=False
        encoder_type="linear",
    )
    model.encoder = new_encoder
    print(f"  Rebuilt encoder for num_features=1 (was {original_fpg})")

    new_num_features = 2  # 1 feature + 1 NaN indicator

    # ── Step 5: Monkey-patch add_embeddings ──
    import types
    model.add_embeddings = types.MethodType(temporal_add_embeddings, model)
    print(f"  Patched add_embeddings with temporal PE + feature embedding")

    # ── Step 6: Load pretrained weights ──
    # Remove keys with shape mismatch (encoder) before loading
    current_sd = model.state_dict()
    filtered_sd = {}
    skipped = []
    for k, v in pretrained_sd.items():
        if k in current_sd and current_sd[k].shape != v.shape:
            skipped.append(f"{k}: {v.shape} → {current_sd[k].shape}")
        elif k not in current_sd:
            skipped.append(f"{k}: not in new model")
        else:
            filtered_sd[k] = v

    if skipped:
        print(f"  Skipped keys (shape mismatch or missing):")
        for s in skipped:
            print(f"    {s}")

    missing, unexpected = model.load_state_dict(filtered_sd, strict=False)
    if missing:
        print(f"  Re-initialized keys: {missing}")

    # ── Step 7: Xavier-init the new encoder linear ──
    from tabpfn.architectures.base.encoders import LinearInputEncoderStep
    for step in model.encoder.children():
        if isinstance(step, LinearInputEncoderStep):
            nn.init.xavier_uniform_(step.layer.weight)
            if step.layer.bias is not None:
                nn.init.zeros_(step.layer.bias)
            break

    # ── Step 8: Enable gradient checkpointing for attention & MLP ──
    # With fpg=1, the number of feature tokens is 3x larger than fpg=3,
    # causing O(n^2) attention to use ~9x more memory during backprop.
    # Wrapping attention and MLP forwards with checkpoint() trades ~30-40%
    # more compute for much less peak memory.
    from functools import partial
    from torch.utils.checkpoint import checkpoint as torch_checkpoint
    from tabpfn.architectures.base.attention.full_attention import MultiHeadAttention
    from tabpfn.architectures.base.mlp import MLP

    n_wrapped = 0
    for module in model.modules():
        if isinstance(module, (MultiHeadAttention, MLP)):
            # Only wrap if not already wrapped (check if forward is already a partial)
            if not isinstance(module.forward, partial):
                module.forward = partial(
                    torch_checkpoint, module.forward, use_reentrant=False)
                n_wrapped += 1
    print(f"  Enabled gradient checkpointing on {n_wrapped} attention/MLP modules")

    model.to(device if device != "auto" else "cpu")

    n_params = sum(p.numel() for p in model.parameters())
    n_new = new_num_features * emsize  # the only re-initialized params
    print(f"  Total parameters: {n_params:,}")
    print(f"  Re-initialized parameters: {n_new} (encoder linear)")

    return model, clf


def set_temporal_info(model, n_features: int, T: int):
    """Set temporal metadata on the model before a forward pass."""
    model._temporal_info = (n_features, T)


def clear_temporal_info(model):
    """Clear temporal metadata (reverts to default subspace embeddings)."""
    model._temporal_info = (None, None)
