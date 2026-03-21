"""
Overlap TabPFN model: patch encoder + global Conv1D encoders.

Architecture summary
--------------------
- Pretrained TabPFN v2.5 backbone (24 layers, emsize=192, nhead=3, nhid=384).
- features_per_group = 16  (each token = 1 channel × 16 timesteps).
- Fresh-initialized (Xavier):
    * Patch encoder: MLP(32→96→GELU→192) via TabPFN's get_encoder
      pipeline (NanHandling, VariableNumFeatures).
      normalize_x=False — per-position normalization is disabled.
    * Global Conv1D encoders: 4 kernels (3, 7, 9, 11), each Conv1d(2→192)+GELU.
      Mean/max pool over T, then (if m>1) mean/max pool over m.
      → 8 tokens (univariate) or 16 tokens (multivariate).
- Pretrained (unchanged):
    * add_embeddings: TabPFN's "subspace" mode, seed=42, COL_EMBEDDING.
      Patched to inject global tokens alongside patch tokens.
    * 24 transformer layers, y_encoder, decoder.

Normalisation strategy
----------------------
  per_channel_normalize() is called BEFORE overlap expansion.  Each channel
  is centred to mean=0, std=1 using train-only statistics (ignoring NaN).
  This single normalisation step feeds BOTH the patch encoder and the global
  Conv1D encoder, replacing TabPFN's default per-position InputNormalization
  which would destroy temporal continuity within patches.

Token layout for the transformer
---------------------------------
  [patch_0, patch_1, ..., patch_{m*G-1}, global_0, ..., global_{7or15}]
  All get unique COL_EMBEDDING vectors via TabPFN's subspace embeddings.

Usage
-----
  model, clf, fresh_params, pretrained_params = build_overlap_model(device="cuda")
  X_tr_3d_n, X_te_3d_n = per_channel_normalize(X_tr_3d, X_te_3d)
  X_exp, T_pad, n_groups = pad_and_expand_overlap(X_flat_norm, m, T)
  set_global_input(model, X_tr_3d_n, X_te_3d_n)
  # … forward pass …
"""

from __future__ import annotations

import types
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
ENCODER_HIDDEN: int = 96   # MLP hidden dim (192/2)
ENCODER_LAYERS: int = 2    # MLP depth: 32→96,GELU→192 (no GELU after last)
GLOBAL_KERNEL_SIZES: tuple[int, ...] = (3, 7, 9, 11)


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
# Global Conv1D encoder
# ─────────────────────────────────────────────────────────────────────────────

class GlobalConvEncoder(nn.Module):
    """Extract global summary tokens via multi-scale Conv1D + pooling.

    For each kernel size k in GLOBAL_KERNEL_SIZES:
      Conv1d(2, emsize, k, padding=k//2) + GELU  over each channel independently.
      Input channels: [value, nan_indicator].
      Then:
        - mean pool over T → (n, m, emsize)
        - max  pool over T → (n, m, emsize)
      If m > 1, additionally:
        - mean/max pool over m for each T-pooled result.

    Univariate  (m=1): 4 kernels × 2 T-pools             = 8 tokens of dim emsize
    Multivariate(m>1): 4 kernels × 2 T-pools × 2 m-pools = 16 tokens of dim emsize
    """

    def __init__(self, emsize: int = 192, kernel_sizes: tuple[int, ...] = GLOBAL_KERNEL_SIZES):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=2, out_channels=emsize, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.act = nn.GELU()

    def forward(self, x_3d: torch.Tensor, nan_mask: torch.Tensor) -> torch.Tensor:
        """Compute global tokens from raw time series data.

        Args:
            x_3d:     (n, m, T) — values with NaN replaced by 0.
            nan_mask: (n, m, T) — 1.0 where original was NaN, 0.0 otherwise.

        Returns:
            (n, n_global, emsize) where n_global = 8 if m==1, 16 if m>1.
        """
        n, m, T = x_3d.shape
        emsize = self.convs[0].out_channels

        x_in = torch.stack([x_3d, nan_mask], dim=2)   # (n, m, 2, T)
        x_in = x_in.reshape(n * m, 2, T)               # (n*m, 2, T)

        tokens: list[torch.Tensor] = []
        for conv in self.convs:
            h = self.act(conv(x_in))             # (n*m, emsize, T')
            mean_t = h.mean(dim=2)               # (n*m, emsize)
            max_t  = h.max(dim=2).values         # (n*m, emsize)

            mean_t = mean_t.reshape(n, m, emsize)
            max_t  = max_t.reshape(n, m, emsize)

            if m == 1:
                tokens.append(mean_t[:, 0])      # (n, emsize)
                tokens.append(max_t[:, 0])
            else:
                tokens.append(mean_t.mean(dim=1))       # mean_m(mean_t)
                tokens.append(mean_t.max(dim=1).values)  # max_m(mean_t)
                tokens.append(max_t.mean(dim=1))         # mean_m(max_t)
                tokens.append(max_t.max(dim=1).values)   # max_m(max_t)

        return torch.stack(tokens, dim=1)        # (n, n_global, emsize)


# ─────────────────────────────────────────────────────────────────────────────
# Global-token injection: store raw data + patch add_embeddings
# ─────────────────────────────────────────────────────────────────────────────

def per_channel_normalize(
    X_tr: np.ndarray, X_te: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize each channel to mean=0, std=1 using train-only statistics.

    Statistics are computed over all (n_train × T) values per channel,
    ignoring NaN.  This preserves the temporal shape of each channel while
    removing amplitude differences between channels and datasets.

    Must be called BEFORE overlap expansion so that both the patch encoder
    and the global Conv1D encoder see the same normalised data.  The patch
    encoder's ``InputNormalizationEncoderStep`` is disabled (``normalize_x=False``)
    precisely because this function handles normalisation.

    Args:
        X_tr: (n_train, m, T)
        X_te: (n_test,  m, T)

    Returns:
        X_tr_norm, X_te_norm — float32, NaN positions unchanged.
    """
    m = X_tr.shape[1]
    X_tr_n = X_tr.copy().astype(np.float64)
    X_te_n = X_te.copy().astype(np.float64)

    for j in range(m):
        train_vals = X_tr_n[:, j, :].ravel()
        finite = train_vals[np.isfinite(train_vals)]
        if len(finite) == 0:
            continue
        mu  = finite.mean()
        std = finite.std()
        if std < 1e-8:
            std = 1.0
        X_tr_n[:, j, :] = (X_tr_n[:, j, :] - mu) / std
        X_te_n[:, j, :] = (X_te_n[:, j, :] - mu) / std

    return X_tr_n.astype(np.float32), X_te_n.astype(np.float32)


def set_global_input(model, X_tr_3d: np.ndarray, X_te_3d: np.ndarray) -> None:
    """Prepare and store global tokens from pre-normalized 3D data.

    Expects data already per-channel normalised by the caller via
    ``per_channel_normalize``.  Pipeline here:
      1. NaN → 0  (NaN positions get a neutral value).
      2. Build NaN indicator mask (1.0 where original was NaN).
      3. Pass [value, nan_indicator] to GlobalConvEncoder.

    Args:
        model:    PerFeatureTransformer with model.global_conv_encoder attached.
        X_tr_3d:  (n_train, m, T) numpy — pre-normalised, may contain NaN.
        X_te_3d:  (n_test, m, T) numpy — pre-normalised, may contain NaN.
    """
    device = next(model.parameters()).device

    X_all = np.concatenate([X_tr_3d, X_te_3d], axis=0)
    nan_mask = np.isnan(X_all).astype(np.float32)
    X_clean = np.nan_to_num(X_all, nan=0.0).astype(np.float32)

    x_t = torch.as_tensor(X_clean, dtype=torch.float32, device=device)
    m_t = torch.as_tensor(nan_mask, dtype=torch.float32, device=device)
    model._global_tokens = model.global_conv_encoder(x_t, m_t)


def clear_global_input(model) -> None:
    """Remove stored global tokens."""
    model._global_tokens = None


def _make_injecting_add_embeddings(orig_cls_method):
    """Wrap TabPFN's add_embeddings to concat global tokens before embedding."""

    def _injecting_add_embeddings(
        self, x, y, *, data_dags, num_features, seq_len,
        cache_embeddings=False, use_cached_embeddings=False,
    ):
        gt = getattr(self, "_global_tokens", None)
        if gt is not None:
            # gt: (n_all, n_global, emsize), needs (batch=1, seq_len, n_global, emsize)
            x = torch.cat([x, gt.unsqueeze(0)], dim=2)
            self._global_tokens = None   # consumed

        return orig_cls_method(
            self, x, y,
            data_dags=data_dags, num_features=num_features, seq_len=seq_len,
            cache_embeddings=cache_embeddings,
            use_cached_embeddings=use_cached_embeddings,
        )

    return _injecting_add_embeddings


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compat stubs
# ─────────────────────────────────────────────────────────────────────────────

def set_temporal_info(model, n_features: int, T: int, group_size: int = WINDOW) -> None:
    """No-op kept for backward compatibility with inference.py / worker_evaluator_v2.py."""


def clear_temporal_info(model) -> None:
    """No-op kept for backward compatibility."""


# ─────────────────────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_overlap_model(device: str = "auto") -> tuple:
    """Build TabPFN with patch encoder + global Conv1D encoders.

    Architectural changes vs vanilla TabPFN:
      - features_per_group: 3 → WINDOW (16)
      - encoder: replaced with MLP(32→96→GELU→192, no bias)
        via TabPFN's get_encoder pipeline (NanHandling, VariableNumFeatures).
        normalize_x=False — caller pre-normalises per-channel via per_channel_normalize().
      - global_conv_encoder: 4 Conv1D kernels (3,7,9,11), each Conv1d(2→192)+GELU,
        producing 8 (univariate) or 16 (multivariate) extra global tokens.
      - add_embeddings: monkey-patched to inject global tokens alongside patch tokens,
        so all tokens receive COL_EMBEDDING vectors from the pretrained projection.

    Returns:
        model             — modified PerFeatureTransformer
        clf               — TabPFNClassifier (for fit_from_preprocessed API)
        fresh_params      — list of fresh nn.Parameter objects (encoder + global_conv_encoder)
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

    # ── 2. Replace encoder with MLP(32→96,GELU→192, no bias) ──
    # normalize_x=False: we pre-normalize per-channel before overlap expansion
    # so that temporal dynamics within patches are preserved.
    new_encoder = get_encoder(
        num_features=WINDOW,
        embedding_size=emsize,
        remove_empty_features=True,
        remove_duplicate_features=False,
        nan_handling_enabled=True,
        normalize_on_train_only=True,
        normalize_to_ranking=False,
        normalize_x=False,
        remove_outliers=False,
        normalize_by_used_features=True,
        encoder_use_bias=False,
        encoder_type="mlp",
        encoder_mlp_hidden_dim=ENCODER_HIDDEN,
        encoder_mlp_num_layers=ENCODER_LAYERS,
    )
    _reinit_module(new_encoder)
    model.encoder = new_encoder
    print(f"  [fresh] encoder: MLP({WINDOW * 2}→{ENCODER_HIDDEN}→GELU→{emsize}, no bias) Xavier")

    # ── 3. Global Conv1D encoder ──
    global_enc = GlobalConvEncoder(emsize=emsize, kernel_sizes=GLOBAL_KERNEL_SIZES)
    _reinit_module(global_enc)
    model.global_conv_encoder = global_enc
    n_global_params = sum(p.numel() for p in global_enc.parameters())
    print(f"  [fresh] global_conv_encoder: kernels={GLOBAL_KERNEL_SIZES}  "
          f"Conv1d(2→{emsize})+GELU  ({n_global_params:,} params)")

    # ── 4. Patch add_embeddings to inject global tokens ──
    orig_add_emb = type(model).add_embeddings
    model.add_embeddings = types.MethodType(
        _make_injecting_add_embeddings(orig_add_emb), model,
    )
    model._global_tokens = None
    print(f"  [patched] add_embeddings: injects global tokens alongside patch tokens")

    # Embeddings, seed, and all transformer layers are left unchanged.
    emb_seed = model.random_embedding_seed
    emb_type = model.feature_positional_embedding
    print(f"  [pretrained] embeddings: type={emb_type!r}  seed={emb_seed}  (unchanged)")

    # ── 5. Collect parameter groups ──
    fresh_ids: set = set()
    fresh_params: list = []
    for mod in [model.encoder, model.global_conv_encoder]:
        for p in mod.parameters():
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
    n_samples = 5
    X = np.random.randn(n_samples, m * T).astype(np.float32)

    X_3d = X.reshape(n_samples, m, T)
    X_tr_3d, X_te_3d = X_3d[:3], X_3d[3:]
    X_tr_n, X_te_n = per_channel_normalize(X_tr_3d, X_te_3d)
    print(f"\nper_channel_normalize: train {X_tr_n.shape}, test {X_te_n.shape}")
    ch_means = [f"{X_tr_n[:, j, :].mean():.4f}" for j in range(m)]
    print(f"  train channel means: {ch_means}")

    X_flat_n = np.concatenate([X_tr_n, X_te_n], axis=0).reshape(n_samples, m * T)
    X_exp, T_pad, n_groups = pad_and_expand_overlap(X_flat_n, m, T)
    print(f"\nDummy: m={m}, T={T} → T_pad={T_pad}, n_groups/feat={n_groups}")
    print(f"  Input:    ({n_samples}, {m * T})")
    print(f"  Expanded: {X_exp.shape}")

    set_global_input(model, X_tr_n, X_te_n)
    gt = model._global_tokens
    print(f"  Global tokens: {gt.shape}  (expect ({n_samples}, 16, 192) for m>1)")

    m1, T1 = 1, 48
    X1 = np.random.randn(4, m1 * T1).astype(np.float32)
    X1_3d = X1.reshape(4, m1, T1)
    X1_tr_n, X1_te_n = per_channel_normalize(X1_3d[:2], X1_3d[2:])
    set_global_input(model, X1_tr_n, X1_te_n)
    gt1 = model._global_tokens
    print(f"  Univariate tokens: {gt1.shape}  (expect (4, 8, 192) for m=1)")
