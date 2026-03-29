"""
Inference utilities for the overlap TabPFN model.

forward_single_dataset()
    Core bypass-preprocessing forward pass used during both training
    (worker_trainer_v2) and evaluation (worker_evaluator_v2, final_evaluation).
    Bypasses all sklearn preprocessing: data goes directly to the MLP encoder
    (which includes TabPFN's NanHandling, InputNormalization, and
    VariableNumFeatures steps internally).

deserialize_batch()
    Load a .npz batch file written by worker_generator into a list of dicts.

evaluate_ensemble()
    Multi-iteration ensemble inference for final evaluation:
    applies channel/class permutations per iteration then calls the model.
    Returns averaged probability matrix.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "00_TabPFN" / "src"))

from tabpfn.preprocessing.configs import ClassifierEnsembleConfig, PreprocessorConfig
from model import (
    WINDOW,
    STRIDE,
    pad_to_group,
    pad_and_expand_overlap,
    set_temporal_info,
    set_global_input,
    per_channel_normalize,
)

SOFTMAX_TEMPERATURE: float = 0.9


# ─────────────────────────────────────────────────────────────────────────────
# Batch (de)serialisation
# ─────────────────────────────────────────────────────────────────────────────

def deserialize_batch(path: str) -> List[Dict]:
    """Load a .npz batch file back into a list of dataset dicts."""
    data = np.load(path, allow_pickle=False)
    n = int(data["__n__"])
    batch = []
    for i in range(n):
        batch.append({
            "X_train":         data[f"{i}_X_train"],
            "X_test":          data[f"{i}_X_test"],
            "y_train":         data[f"{i}_y_train"],
            "y_test":          data[f"{i}_y_test"],
            "n_classes":       int(data[f"{i}_n_classes"]),
            "n_features":      int(data[f"{i}_n_features"]),
            "n_features_orig": int(data[f"{i}_n_features_orig"]),
            "T":               int(data[f"{i}_T"]),
            "n_samples":       int(data[f"{i}_n_samples"]),
        })
    return batch


# ─────────────────────────────────────────────────────────────────────────────
# Single-dataset forward pass (bypasses sklearn preprocessing)
# ─────────────────────────────────────────────────────────────────────────────

def _make_dummy_ensemble_config() -> ClassifierEnsembleConfig:
    """Minimal ensemble config: no preprocessing, no shifts, no fingerprint."""
    return ClassifierEnsembleConfig(
        preprocess_config=PreprocessorConfig("none", categorical_name="numeric"),
        feature_shift_count=0,
        class_permutation=None,
        add_fingerprint_feature=False,
        polynomial_features="no",
        feature_shift_decoder=None,
        subsample_ix=None,
        _model_index=0,
    )


def forward_single_dataset(model, clf, device: str, data: Dict) -> Optional[Dict]:
    """One forward pass for a single dataset through the overlap model.

    Applies overlap expansion, sets temporal info, then calls clf.fit_from_preprocessed
    to bypass sklearn preprocessing entirely.

    Args:
        model:  PerFeatureTransformer (overlap architecture)
        clf:    TabPFNClassifier (for fit_from_preprocessed / forward API)
        device: torch device string
        data:   dataset dict (keys: X_train, X_test, y_train, y_test,
                n_classes, n_features_orig, T)

    Returns dict with 'loss' (scalar tensor, requires_grad=True) and 'accuracy',
    or None if the forward pass fails.
    """
    try:
        m = data["n_features_orig"]
        T = data["T"]

        X_tr_3d = data["X_train"].reshape(-1, m, T)
        X_te_3d = data["X_test"].reshape(-1, m, T)
        X_tr_3d_n, X_te_3d_n = per_channel_normalize(X_tr_3d, X_te_3d)

        X_tr_flat = X_tr_3d_n.reshape(-1, m * T)
        X_te_flat = X_te_3d_n.reshape(-1, m * T)

        X_tr_p, T_pad, n_groups = pad_and_expand_overlap(X_tr_flat, m, T)
        X_te_p, _, _ = pad_and_expand_overlap(X_te_flat, m, T)
        T_eff = n_groups * WINDOW
        set_temporal_info(model, m, T_eff, group_size=WINDOW)

        set_global_input(model, X_tr_3d_n, X_te_3d_n)

        y_test_t = torch.tensor(data["y_test"], dtype=torch.long, device=device)

        X_tr_t = torch.as_tensor(X_tr_p, dtype=torch.float32, device=device).unsqueeze(0)
        y_tr_t = torch.as_tensor(data["y_train"], dtype=torch.float32, device=device).unsqueeze(0)
        X_te_t = torch.as_tensor(X_te_p, dtype=torch.float32, device=device).unsqueeze(0)

        dummy_cfg = _make_dummy_ensemble_config()
        clf.n_classes_ = data["n_classes"]
        clf.fit_from_preprocessed(
            [X_tr_t], [y_tr_t],
            cat_ix=[[[]]],
            configs=[[dummy_cfg]],
        )
        logits = clf.forward([X_te_t], return_raw_logits=True)

        if logits.ndim == 2:
            logits_out = logits
        elif logits.ndim == 3:
            logits_out = logits.squeeze(1)
        elif logits.ndim == 4:
            logits_out = logits.mean(dim=(1, 2))
        else:
            return None

        if y_test_t.min() < 0 or y_test_t.max() >= logits_out.shape[-1]:
            return None

        loss = F.cross_entropy(logits_out, y_test_t)
        with torch.no_grad():
            acc = (logits_out.argmax(dim=-1) == y_test_t).float().mean()

        return {"loss": loss, "accuracy": acc}

    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble inference (final evaluation)
# ─────────────────────────────────────────────────────────────────────────────

def _soft_clip(x: np.ndarray, B: float = 3.0) -> np.ndarray:
    return x / np.sqrt(1.0 + (x / B) ** 2)


def _temporal_squashing_scaler(
    X_tr: np.ndarray, X_te: np.ndarray, m: int, T: int,
    max_abs: float = 3.0, q_low: float = 25.0, q_high: float = 75.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature robust scaling (IQR) + soft clip applied to train and test.

    Mirrors TabPFN's SquashingScaler but applied per-channel×timestep.
    Used on odd ensemble iterations to add diversity.
    """
    X_tr, X_te = X_tr.copy(), X_te.copy()
    for j in range(m):
        c0, c1 = j * T, j * T + T
        vals = X_tr[:, c0:c1].ravel()
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            X_tr[:, c0:c1] = 0.0
            X_te[:, c0:c1] = 0.0
            continue
        median = np.median(finite)
        q_lo = np.percentile(finite, q_low)
        q_hi = np.percentile(finite, q_high)
        if q_hi != q_lo:
            scale = 1.0 / (q_hi - q_lo)
        else:
            vmin, vmax = np.min(finite), np.max(finite)
            if vmax != vmin:
                scale = 2.0 / (vmax - vmin)
            else:
                X_tr[:, c0:c1] = 0.0
                X_te[:, c0:c1] = 0.0
                continue
        X_tr[:, c0:c1] = _soft_clip((X_tr[:, c0:c1] - median) * scale, max_abs)
        X_te[:, c0:c1] = _soft_clip((X_te[:, c0:c1] - median) * scale, max_abs)
    return X_tr.astype(np.float32), X_te.astype(np.float32)


def _shuffle_features(X_flat: np.ndarray, m: int, T: int, perm: np.ndarray) -> np.ndarray:
    n = X_flat.shape[0]
    return X_flat.reshape(n, m, T)[:, perm, :].reshape(n, m * T)


MAX_M_TIMES_T: int = 2000   # hard cap from PFN filter (m*T ≤ 2000)
GROUP_SIZE = 8              # non-overlap group size for fpg8 checkpoints


def _global_pool(
    X_flat: np.ndarray, m: int, T: int, K: int = 16, S: int = 8
) -> Optional[tuple]:
    """Sliding-window pooling: mean/max/min over windows of size K, stride S.

    Returns (X_pooled, m_new, T_new) where m_new = 3*m and T_new = number of windows,
    or None if T < K.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    if T < K:
        return None
    n = X_flat.shape[0]
    X_3d = X_flat.reshape(n, m, T)
    windows = sliding_window_view(X_3d, K, axis=2)[:, :, ::S, :]  # (n, m, T_new, K)
    T_new = windows.shape[2]
    mean_ch = windows.mean(axis=3).astype(np.float32)
    max_ch  = windows.max(axis=3).astype(np.float32)
    min_ch  = windows.min(axis=3).astype(np.float32)
    X_out = np.concatenate([mean_ch, max_ch, min_ch], axis=1).reshape(n, 3 * m * T_new)
    return X_out, 3 * m, T_new


def evaluate_ensemble(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
    m: int,
    T: int,
    device: str,
    n_iters: int = 1,
    seed: int = 42,
    use_overlap: bool = True,
    use_global_pool: bool = True,
) -> Optional[np.ndarray]:
    """Multi-iteration ensemble: returns averaged probability matrix or None.

    Each iteration applies:
      - Channel permutation
      - Class permutation (undone after logits)
      - Odd iterations: robust squashing scaler (IQR + soft clip)
      - Every other pair of iterations (iters 2,3,6,7,...): global sliding-window
        pooling (mean/max/min) → triples the channel count, shortens T
      - Per-channel normalisation (on effective m_eff, T_eff after optional pool)
      - Overlap expansion (if use_overlap) or group-pad (if fpg8 model)
    """
    rng = np.random.RandomState(seed)
    proba_sum = np.zeros((X_test.shape[0], n_classes), dtype=np.float64)
    n_valid = 0

    for it in range(n_iters):
        try:
            feat_perm = rng.permutation(m)
            X_tr_p = _shuffle_features(X_train, m, T, feat_perm)
            X_te_p = _shuffle_features(X_test, m, T, feat_perm)

            class_perm = rng.permutation(n_classes)
            y_tr_p = class_perm[y_train]

            if it % 2 == 1:
                X_tr_p, X_te_p = _temporal_squashing_scaler(X_tr_p, X_te_p, m, T)

            m_eff, T_eff = m, T
            if use_global_pool and ((it // 2) % 2 == 1) and T > 96:
                pooled = _global_pool(X_tr_p, m, T)
                if pooled is not None:
                    X_tr_pooled, m_new, T_new = pooled
                    if m_new * T_new <= MAX_M_TIMES_T:
                        pooled_te = _global_pool(X_te_p, m, T)
                        if pooled_te is not None:
                            X_tr_p = X_tr_pooled
                            X_te_p = pooled_te[0]
                            m_eff, T_eff = m_new, T_new

            X_tr_3d = X_tr_p.reshape(-1, m_eff, T_eff)
            X_te_3d = X_te_p.reshape(-1, m_eff, T_eff)
            X_tr_3d_n, X_te_3d_n = per_channel_normalize(X_tr_3d, X_te_3d)
            X_tr_p = X_tr_3d_n.reshape(-1, m_eff * T_eff)
            X_te_p = X_te_3d_n.reshape(-1, m_eff * T_eff)

            if use_overlap:
                X_tr_pad, _, n_groups = pad_and_expand_overlap(X_tr_p, m_eff, T_eff)
                X_te_pad, _, _ = pad_and_expand_overlap(X_te_p, m_eff, T_eff)
                T_exp = n_groups * WINDOW
                set_temporal_info(model, m_eff, T_exp, group_size=WINDOW)
            else:
                X_tr_pad, T_padded = pad_to_group(X_tr_p, m_eff, T_eff, group_size=GROUP_SIZE)
                X_te_pad, _ = pad_to_group(X_te_p, m_eff, T_eff, group_size=GROUP_SIZE)
                set_temporal_info(model, m_eff, T_padded, group_size=GROUP_SIZE)

            if hasattr(model, "global_conv_encoder"):
                set_global_input(model, X_tr_3d_n, X_te_3d_n)

            X_tr_t = torch.as_tensor(X_tr_pad, dtype=torch.float32, device=device)
            y_tr_t = torch.as_tensor(y_tr_p, dtype=torch.float32, device=device)
            X_te_t = torch.as_tensor(X_te_pad, dtype=torch.float32, device=device)

            X_full = torch.cat([X_tr_t, X_te_t], dim=0).unsqueeze(1)
            y_in = y_tr_t.unsqueeze(1)

            output = model(X_full, y_in, only_return_standard_out=True, categorical_inds=[[]])
            logits = output.squeeze(1) if output.ndim == 3 else output
            logits = logits[:, :n_classes]
            logits = logits[:, class_perm]   # undo class permutation
            proba = torch.softmax(logits / SOFTMAX_TEMPERATURE, dim=-1).cpu().numpy()
            proba_sum += proba
            n_valid += 1

            del X_tr_t, y_tr_t, X_te_t, X_full, y_in, output, logits, proba

        except Exception as e:
            print(f"    [Ensemble iter {it}] FAILED: {type(e).__name__}: {str(e)[:100]}")
            continue

    return proba_sum / n_valid if n_valid > 0 else None
