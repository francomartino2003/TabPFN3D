#!/usr/bin/env python3
"""
Full 4-way evaluation: finetuned V3 vs original TabPFN vs AEON benchmarks.

Models evaluated:
  A) TabPFN original — bypass (no preprocessing, no ensemble, no temporal PE)
  B) TabPFN original — standard (built-in preprocessing + ensemble, n_estimators=4)
  C) V3 finetuned  — bypass (temporal PE, no scaler, no ensemble)
  D) V3 finetuned  — manual ensemble (4 iters: alternating SquashingScaler/none
     + feature channel shuffle + class label permutation, all with temporal PE)

Comparisons:
  - Bypass vs bypass: A vs C  (does finetuning + temporal PE help?)
  - Ensemble vs ensemble: B vs D  (does our ensemble beat TabPFN's native one?)
  - All 4 vs AEON benchmark (full ranking)

Usage:
    python evaluate_v3_with_ensemble.py <checkpoint_path>
    python evaluate_v3_with_ensemble.py checkpoints_v3/v3_bypass_lr2e4_f/checkpoint_step300.pt
"""

import sys
import argparse
import pickle
import time
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / '11_final_generator'))
sys.path.insert(0, str(Path(__file__).parent.parent / '00_TabPFN' / 'src'))

from tabpfn import TabPFNClassifier
from tabpfn_temporal import (
    build_temporal_tabpfn,
    set_temporal_info,
    pad_to_group3,
)

# ═════════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════════

MAX_SAMPLES = 1000
MAX_FLAT_FEATURES = 500
MAX_CLASSES = 10
N_ENSEMBLE_ITERS = 4
ENSEMBLE_SEED = 42
SOFTMAX_TEMPERATURE = 0.9  # TabPFN default calibration temperature

BENCHMARKS_DIR = Path(__file__).parent.parent / "01_real_data" / "AEON" / "benchmarks"
REAL_DATA_PKL = (Path(__file__).parent.parent / "01_real_data" / "AEON"
                 / "data" / "classification_datasets.pkl")


# ═════════════════════════════════════════════════════════════════════════════
# Temporal SquashingScaler (per-feature-m, across all t)
# ═════════════════════════════════════════════════════════════════════════════

def _soft_clip(x, B=3.0):
    """z / sqrt(1 + (z/B)^2) — maps (-inf,inf) → (-B, B)."""
    return x / np.sqrt(1.0 + (x / B) ** 2)


def temporal_squashing_scaler(X_train_flat, X_test_flat, m, T,
                              max_abs=3.0, q_low=25.0, q_high=75.0):
    """
    Squashing scaler that normalizes per original feature m, pooling
    statistics across all T timesteps of that feature.

    For feature j, we gather columns [j*T .. j*T+T-1] from X_train,
    pool all their values to compute a single median and IQR, then apply
    robust scaling + soft clipping to those columns in both X_train and X_test.
    """
    X_tr = X_train_flat.copy()
    X_te = X_test_flat.copy()

    for j in range(m):
        col_start = j * T
        col_end = j * T + T

        vals = X_tr[:, col_start:col_end].ravel()
        finite = vals[np.isfinite(vals)]

        if len(finite) == 0:
            X_tr[:, col_start:col_end] = 0.0
            X_te[:, col_start:col_end] = 0.0
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
                X_tr[:, col_start:col_end] = 0.0
                X_te[:, col_start:col_end] = 0.0
                continue

        X_tr[:, col_start:col_end] = (X_tr[:, col_start:col_end] - median) * scale
        X_te[:, col_start:col_end] = (X_te[:, col_start:col_end] - median) * scale

        X_tr[:, col_start:col_end] = _soft_clip(X_tr[:, col_start:col_end], max_abs)
        X_te[:, col_start:col_end] = _soft_clip(X_te[:, col_start:col_end], max_abs)

    return X_tr.astype(np.float32), X_te.astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# Feature shuffle (permute m channels, keep T order within each)
# ═════════════════════════════════════════════════════════════════════════════

def shuffle_features(X_flat, m, T, perm):
    """
    Permute the m feature channels in flattened data.

    X_flat: (n_samples, m*T)
    Layout: [f0_t0..f0_{T-1}, f1_t0..f1_{T-1}, ..., f{m-1}_t0..f{m-1}_{T-1}]

    perm: permutation array of length m, e.g. [2, 0, 1]
    Result: blocks reordered so position p gets feature perm[p]
    """
    n = X_flat.shape[0]
    X_3d = X_flat.reshape(n, m, T)         # (n, m, T)
    X_3d_perm = X_3d[:, perm, :]           # permute channels
    return X_3d_perm.reshape(n, m * T)     # flatten back


# ═════════════════════════════════════════════════════════════════════════════
# Data loading
# ═════════════════════════════════════════════════════════════════════════════

def load_real_datasets():
    """Load real datasets with constraints (same as training)."""
    real_data_dir = Path(__file__).resolve().parent.parent / '01_real_data'
    if str(real_data_dir) not in sys.path:
        sys.path.insert(0, str(real_data_dir))

    from src.data_loader import TimeSeriesDataset  # noqa: F401

    if not REAL_DATA_PKL.exists():
        print(f"ERROR: Data not found at {REAL_DATA_PKL}")
        return []

    with open(REAL_DATA_PKL, 'rb') as f:
        datasets_list = pickle.load(f)

    valid = []
    for ds in datasets_list:
        try:
            X_train, X_test = ds.X_train, ds.X_test
            if X_train is None or X_test is None:
                continue
            if X_train.ndim == 2:
                X_train = X_train[:, np.newaxis, :]
            if X_test.ndim == 2:
                X_test = X_test[:, np.newaxis, :]

            # (n, m, T) — already in aeon format, no transpose
            n = X_train.shape[0] + X_test.shape[0]
            m = X_train.shape[1]
            T = X_train.shape[2]
            flat = m * T

            if n > MAX_SAMPLES or flat > MAX_FLAT_FEATURES:
                continue
            n_classes = len(np.unique(ds.y_train))
            if n_classes > MAX_CLASSES or n_classes < 2:
                continue

            X_tr = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
            X_te = X_test.reshape(X_test.shape[0], -1).astype(np.float32)

            # NaN handling (train-based)
            if np.any(np.isnan(X_tr)):
                col_means = np.nan_to_num(np.nanmean(X_tr, axis=0), nan=0.0)
                for i in range(X_tr.shape[1]):
                    X_tr[:, i] = np.where(np.isnan(X_tr[:, i]), col_means[i], X_tr[:, i])
                    X_te[:, i] = np.where(np.isnan(X_te[:, i]), col_means[i], X_te[:, i])

            le = LabelEncoder()
            le.fit(ds.y_train)

            valid.append({
                'name': ds.name,
                'X_train': X_tr,
                'X_test': X_te,
                'y_train': le.transform(ds.y_train).astype(np.int64),
                'y_test': le.transform(ds.y_test).astype(np.int64),
                'n_classes': n_classes,
                'n_features': flat,
                'm': m,
                'T': T,
            })
        except Exception:
            continue

    return valid


# ═════════════════════════════════════════════════════════════════════════════
# Model forward pass (bypass preprocessing)
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_single_bypass(model, X_train, y_train, X_test, n_classes, m, T, device):
    """Single forward pass (no ensemble). Returns proba (n_test, n_classes)."""
    # Pad
    X_tr_padded, T_padded = pad_to_group3(X_train, m, T)
    X_te_padded, _ = pad_to_group3(X_test, m, T)

    set_temporal_info(model, m, T_padded)

    X_tr_t = torch.as_tensor(X_tr_padded, dtype=torch.float32, device=device)
    y_tr_t = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    X_te_t = torch.as_tensor(X_te_padded, dtype=torch.float32, device=device)

    X_full = torch.cat([X_tr_t, X_te_t], dim=0).unsqueeze(1)  # (n_all, 1, feat)
    y_in = y_tr_t.unsqueeze(1)  # (n_train, 1)

    output = model(
        X_full, y_in,
        only_return_standard_out=True,
        categorical_inds=[[]],
    )

    if output.ndim == 3:
        logits = output.squeeze(1)
    else:
        logits = output
    logits = logits[:, :n_classes]

    proba = torch.softmax(logits / SOFTMAX_TEMPERATURE, dim=-1).cpu().numpy()
    return proba


def evaluate_ensemble(model, X_train, y_train, X_test, n_classes, m, T, device,
                      n_iters=N_ENSEMBLE_ITERS, seed=ENSEMBLE_SEED):
    """
    Manual ensemble: n_iters iterations.
    - Even iterations (0, 2): temporal SquashingScaler
    - Odd iterations (1, 3): no scaler
    - Each iteration: random permutation of m feature channels
    - Each iteration: random class label permutation (undone on output)
    - Softmax temperature T=0.9 applied to logits
    - Average probabilities across all iterations

    Returns proba (n_test, n_classes).
    """
    rng = np.random.RandomState(seed)
    proba_sum = np.zeros((X_test.shape[0], n_classes), dtype=np.float64)
    n_valid = 0

    for it in range(n_iters):
        try:
            # --- Feature channel permutation ---
            feat_perm = rng.permutation(m)
            X_tr_perm = shuffle_features(X_train, m, T, feat_perm)
            X_te_perm = shuffle_features(X_test, m, T, feat_perm)

            # --- Class label permutation (like TabPFN default) ---
            # perm[i] = "original class i is trained as class perm[i]"
            # To undo on output: result[j] = logit[perm[j]], i.e. logits[:, perm]
            class_perm = rng.permutation(n_classes)
            y_train_perm = class_perm[y_train]

            # --- Scaler: alternate SquashingScaler / none ---
            if it % 2 == 0:
                # SquashingScaler (temporal: per-feature-m, across all t)
                X_tr_proc, X_te_proc = temporal_squashing_scaler(
                    X_tr_perm, X_te_perm, m, T)
            else:
                # No scaler
                X_tr_proc = X_tr_perm.copy()
                X_te_proc = X_te_perm.copy()

            # --- Pad to group3 ---
            X_tr_padded, T_padded = pad_to_group3(X_tr_proc, m, T)
            X_te_padded, _ = pad_to_group3(X_te_proc, m, T)

            # --- Set temporal info (same m, shuffled doesn't change m/T) ---
            set_temporal_info(model, m, T_padded)

            # --- Forward pass ---
            X_tr_t = torch.as_tensor(X_tr_padded, dtype=torch.float32, device=device)
            y_tr_t = torch.as_tensor(y_train_perm, dtype=torch.float32, device=device)
            X_te_t = torch.as_tensor(X_te_padded, dtype=torch.float32, device=device)

            X_full = torch.cat([X_tr_t, X_te_t], dim=0).unsqueeze(1)
            y_in = y_tr_t.unsqueeze(1)

            output = model(
                X_full, y_in,
                only_return_standard_out=True,
                categorical_inds=[[]],
            )

            if output.ndim == 3:
                logits = output.squeeze(1)
            else:
                logits = output
            logits = logits[:, :n_classes]

            # Undo class permutation on logits, then apply temperature
            logits = logits[:, class_perm]
            proba = torch.softmax(logits / SOFTMAX_TEMPERATURE, dim=-1).cpu().numpy()
            proba_sum += proba
            n_valid += 1

            del X_tr_t, y_tr_t, X_te_t, X_full, y_in, output, logits, proba

        except Exception as e:
            print(f"    [Ensemble iter {it}] FAILED: {type(e).__name__}: {str(e)[:100]}")
            continue

    if n_valid == 0:
        return None

    return proba_sum / n_valid


# ═════════════════════════════════════════════════════════════════════════════
# Original TabPFN evaluation (no finetuning, no temporal PE)
# ═════════════════════════════════════════════════════════════════════════════

def build_vanilla_tabpfn(device):
    """Build a fresh, unmodified TabPFN model (no temporal PE patch)."""
    clf = TabPFNClassifier(
        device=device,
        n_estimators=1,
        ignore_pretraining_limits=True,
        fit_mode="batched",
        differentiable_input=False,
        inference_config={"FEATURE_SHIFT_METHOD": None},  # no feature shift
    )
    clf._initialize_model_variables()
    model = clf.model_
    model.eval()
    return model, clf


def evaluate_vanilla_bypass(model, X_train, y_train, X_test, n_classes, device):
    """
    TabPFN original with bypass preprocessing (no scaler, no ensemble,
    no temporal PE). Direct model forward pass with default subspace embeddings.
    Returns proba (n_test, n_classes).
    """
    X_tr_t = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    y_tr_t = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    X_te_t = torch.as_tensor(X_test, dtype=torch.float32, device=device)

    X_full = torch.cat([X_tr_t, X_te_t], dim=0).unsqueeze(1)  # (n_all, 1, feat)
    y_in = y_tr_t.unsqueeze(1)  # (n_train, 1)

    output = model(
        X_full, y_in,
        only_return_standard_out=True,
        categorical_inds=[[]],
    )

    if output.ndim == 3:
        logits = output.squeeze(1)
    else:
        logits = output
    logits = logits[:, :n_classes]

    proba = torch.softmax(logits / SOFTMAX_TEMPERATURE, dim=-1).cpu().numpy()
    return proba


def evaluate_vanilla_standard(X_train, y_train, X_test, n_classes, device,
                              n_estimators=N_ENSEMBLE_ITERS):
    """
    TabPFN original with its full built-in preprocessing + ensemble.
    Uses clf.fit() / clf.predict_proba() which goes through:
    SquashingScaler, SVD, feature shifting, class permutations, etc.
    Returns proba (n_test, n_classes).
    """
    clf = TabPFNClassifier(
        device=device,
        n_estimators=n_estimators,
        ignore_pretraining_limits=True,
    )
    clf.fit(X_train, y_train)

    with torch.no_grad():
        proba = clf.predict_proba(X_test)

    del clf
    return proba


# ═════════════════════════════════════════════════════════════════════════════
# Metrics
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, proba, n_classes):
    """Compute accuracy and AUC from probabilities."""
    preds = proba.argmax(axis=1)
    acc = accuracy_score(y_true, preds)
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, proba[:, 1])
        else:
            auc = roc_auc_score(y_true, proba, multi_class='ovr')
    except Exception:
        auc = None
    return acc, auc


# ═════════════════════════════════════════════════════════════════════════════
# AEON benchmark loading
# ═════════════════════════════════════════════════════════════════════════════

def load_aeon_benchmarks():
    """Load all AEON benchmark AUROC files."""
    benchmarks = {}
    for filepath in BENCHMARKS_DIR.glob("*_auroc.csv"):
        model_name = filepath.stem.replace("_auroc", "")
        try:
            df = pd.read_csv(filepath, index_col=0)
            benchmarks[model_name] = df.mean(axis=1)  # avg across resamples
        except Exception as e:
            print(f"  Warning: Could not load {filepath.name}: {e}")
    return benchmarks


# ═════════════════════════════════════════════════════════════════════════════
# Main evaluation
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Full 4-way evaluation: V3 finetuned vs original TabPFN vs AEON')
    parser.add_argument('checkpoint', type=str,
                        help='Path to checkpoint .pt file')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, mps, cuda, cpu')
    parser.add_argument('--n-ensemble', type=int, default=N_ENSEMBLE_ITERS,
                        help='Number of ensemble iterations (default: 4)')
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        checkpoint_path = Path(__file__).parent / args.checkpoint
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return

    # Device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    n_ens = args.n_ensemble

    print("=" * 80)
    print("FULL 4-WAY EVALUATION")
    print("=" * 80)
    print(f"  A) TabPFN original — bypass  (no preproc, no ensemble, no temporal PE)")
    print(f"  B) TabPFN original — standard (built-in preproc + ensemble, n_est={n_ens})")
    print(f"  C) V3 finetuned   — bypass  (temporal PE, no scaler, no ensemble)")
    print(f"  D) V3 finetuned   — ensemble ({n_ens} iters: scaler/none + feat shuffle + class perm)")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device:     {device}")
    print(f"Constraints: samples≤{MAX_SAMPLES}, features≤{MAX_FLAT_FEATURES}, "
          f"classes≤{MAX_CLASSES}")

    # ── Load datasets ──
    print("\nLoading real datasets...")
    datasets = load_real_datasets()
    print(f"  {len(datasets)} datasets meet constraints")

    # ── Build vanilla TabPFN (for A and B) ──
    print("\nBuilding vanilla TabPFN (no temporal PE)...")
    vanilla_model, vanilla_clf = build_vanilla_tabpfn(device=device)

    # ── Build temporal TabPFN + load checkpoint (for C and D) ──
    print("\nBuilding temporal TabPFN + loading checkpoint...")
    ft_model, ft_clf = build_temporal_tabpfn(device=device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ft_model.load_state_dict(ckpt['model_state_dict'])
    ft_model.eval()
    print(f"  Checkpoint step: {ckpt.get('step', '?')}, "
          f"version: {ckpt.get('version', '?')}")

    # ── Run all 4 evaluations ──
    results = []
    t0 = time.time()

    for i, ds in enumerate(datasets):
        name = ds['name']
        m, T = ds['m'], ds['T']
        nc = ds['n_classes']
        print(f"\n  [{i+1}/{len(datasets)}] {name} "
              f"(m={m}, T={T}, flat={m*T}, cls={nc})", flush=True)

        res = {'name': name, 'm': m, 'T': T, 'n_classes': nc,
               'n_features': ds['n_features']}

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── A) TabPFN original — bypass ──
        try:
            with torch.no_grad():
                proba_a = evaluate_vanilla_bypass(
                    vanilla_model, ds['X_train'], ds['y_train'], ds['X_test'],
                    nc, device)
            acc_a, auc_a = compute_metrics(ds['y_test'], proba_a, nc)
            res['orig_bypass_acc'] = acc_a
            res['orig_bypass_auc'] = auc_a
            del proba_a
        except Exception as e:
            print(f"    [A] FAILED: {type(e).__name__}: {str(e)[:80]}")
            res['orig_bypass_acc'] = None
            res['orig_bypass_auc'] = None

        # ── B) TabPFN original — standard (built-in preprocessing + ensemble) ──
        try:
            proba_b = evaluate_vanilla_standard(
                ds['X_train'], ds['y_train'], ds['X_test'],
                nc, device, n_estimators=n_ens)
            acc_b, auc_b = compute_metrics(ds['y_test'], proba_b, nc)
            res['orig_standard_acc'] = acc_b
            res['orig_standard_auc'] = auc_b
            del proba_b
        except Exception as e:
            print(f"    [B] FAILED: {type(e).__name__}: {str(e)[:80]}")
            res['orig_standard_acc'] = None
            res['orig_standard_auc'] = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── C) V3 finetuned — bypass ──
        try:
            with torch.no_grad():
                proba_c = evaluate_single_bypass(
                    ft_model, ds['X_train'], ds['y_train'], ds['X_test'],
                    nc, m, T, device)
            acc_c, auc_c = compute_metrics(ds['y_test'], proba_c, nc)
            res['v3_bypass_acc'] = acc_c
            res['v3_bypass_auc'] = auc_c
            del proba_c
        except Exception as e:
            print(f"    [C] FAILED: {type(e).__name__}: {str(e)[:80]}")
            res['v3_bypass_acc'] = None
            res['v3_bypass_auc'] = None

        # ── D) V3 finetuned — manual ensemble ──
        try:
            with torch.no_grad():
                proba_d = evaluate_ensemble(
                    ft_model, ds['X_train'], ds['y_train'], ds['X_test'],
                    nc, m, T, device, n_iters=n_ens)
            if proba_d is not None:
                acc_d, auc_d = compute_metrics(ds['y_test'], proba_d, nc)
                res['v3_ensemble_acc'] = acc_d
                res['v3_ensemble_auc'] = auc_d
            else:
                res['v3_ensemble_acc'] = None
                res['v3_ensemble_auc'] = None
            del proba_d
        except Exception as e:
            print(f"    [D] FAILED: {type(e).__name__}: {str(e)[:80]}")
            res['v3_ensemble_acc'] = None
            res['v3_ensemble_auc'] = None

        # Print row summary
        def _f(v):
            return f"{v:.4f}" if v is not None else "FAIL"

        print(f"    A={_f(res.get('orig_bypass_auc'))}  "
              f"B={_f(res.get('orig_standard_auc'))}  "
              f"C={_f(res.get('v3_bypass_auc'))}  "
              f"D={_f(res.get('v3_ensemble_auc'))}")

        results.append(res)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - t0

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    df = pd.DataFrame(results)

    # Column mapping for convenience
    cols = {
        'A': ('orig_bypass_auc', 'orig_bypass_acc',
              'TabPFN orig bypass'),
        'B': ('orig_standard_auc', 'orig_standard_acc',
              f'TabPFN orig standard (n_est={n_ens})'),
        'C': ('v3_bypass_auc', 'v3_bypass_acc',
              'V3 finetuned bypass'),
        'D': ('v3_ensemble_auc', 'v3_ensemble_acc',
              f'V3 finetuned ensemble ({n_ens} iters)'),
    }

    print("\n\n" + "=" * 80)
    print("SUMMARY: 4-WAY COMPARISON")
    print("=" * 80)
    print(f"Evaluation time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"Datasets: {len(datasets)}")

    for key, (auc_col, acc_col, label) in cols.items():
        valid = df[df[auc_col].notna()]
        if len(valid) > 0:
            print(f"\n  {key}) {label}")
            print(f"     Mean AUC: {valid[auc_col].mean():.4f}   "
                  f"Mean Acc: {valid[acc_col].mean():.4f}   "
                  f"({len(valid)}/{len(datasets)} OK)")
        else:
            print(f"\n  {key}) {label}  — NO VALID RESULTS")

    # ── Bypass vs bypass: A vs C ──
    print("\n" + "-" * 70)
    print("BYPASS vs BYPASS: A (orig) vs C (V3 finetuned)")
    print("-" * 70)
    common_ac = df[df['orig_bypass_auc'].notna() & df['v3_bypass_auc'].notna()]
    if len(common_ac) > 0:
        d_auc = common_ac['v3_bypass_auc'].mean() - common_ac['orig_bypass_auc'].mean()
        d_acc = common_ac['v3_bypass_acc'].mean() - common_ac['orig_bypass_acc'].mean()
        print(f"  ΔAUC (C-A): {d_auc:+.4f}   ΔAcc: {d_acc:+.4f}")
        wins = (common_ac['v3_bypass_auc'] > common_ac['orig_bypass_auc'] + 0.001).sum()
        ties = ((common_ac['v3_bypass_auc'] >= common_ac['orig_bypass_auc'] - 0.001) &
                (common_ac['v3_bypass_auc'] <= common_ac['orig_bypass_auc'] + 0.001)).sum()
        losses = (common_ac['v3_bypass_auc'] < common_ac['orig_bypass_auc'] - 0.001).sum()
        print(f"  Win/Tie/Loss (C vs A): {wins}/{ties}/{losses}")

    # ── Ensemble vs ensemble: B vs D ──
    print("\n" + "-" * 70)
    print("ENSEMBLE vs ENSEMBLE: B (orig standard) vs D (V3 manual ensemble)")
    print("-" * 70)
    common_bd = df[df['orig_standard_auc'].notna() & df['v3_ensemble_auc'].notna()]
    if len(common_bd) > 0:
        d_auc = common_bd['v3_ensemble_auc'].mean() - common_bd['orig_standard_auc'].mean()
        d_acc = common_bd['v3_ensemble_acc'].mean() - common_bd['orig_standard_acc'].mean()
        print(f"  ΔAUC (D-B): {d_auc:+.4f}   ΔAcc: {d_acc:+.4f}")
        wins = (common_bd['v3_ensemble_auc'] > common_bd['orig_standard_auc'] + 0.001).sum()
        ties = ((common_bd['v3_ensemble_auc'] >= common_bd['orig_standard_auc'] - 0.001) &
                (common_bd['v3_ensemble_auc'] <= common_bd['orig_standard_auc'] + 0.001)).sum()
        losses = (common_bd['v3_ensemble_auc'] < common_bd['orig_standard_auc'] - 0.001).sum()
        print(f"  Win/Tie/Loss (D vs B): {wins}/{ties}/{losses}")

    # ── Finetuned bypass vs finetuned ensemble: C vs D ──
    print("\n" + "-" * 70)
    print("FINETUNED: C (bypass) vs D (ensemble)")
    print("-" * 70)
    common_cd = df[df['v3_bypass_auc'].notna() & df['v3_ensemble_auc'].notna()]
    if len(common_cd) > 0:
        d_auc = common_cd['v3_ensemble_auc'].mean() - common_cd['v3_bypass_auc'].mean()
        d_acc = common_cd['v3_ensemble_acc'].mean() - common_cd['v3_bypass_acc'].mean()
        print(f"  ΔAUC (D-C): {d_auc:+.4f}   ΔAcc: {d_acc:+.4f}")
        wins = (common_cd['v3_ensemble_auc'] > common_cd['v3_bypass_auc'] + 0.001).sum()
        ties = ((common_cd['v3_ensemble_auc'] >= common_cd['v3_bypass_auc'] - 0.001) &
                (common_cd['v3_ensemble_auc'] <= common_cd['v3_bypass_auc'] + 0.001)).sum()
        losses = (common_cd['v3_ensemble_auc'] < common_cd['v3_bypass_auc'] - 0.001).sum()
        print(f"  Win/Tie/Loss (D vs C): {wins}/{ties}/{losses}")

    # ── Per-dataset table ──
    print("\n" + "-" * 70)
    print("PER-DATASET AUC")
    print("-" * 70)
    print(f"  {'Dataset':<30} {'A:Orig':>8} {'B:Orig':>8} {'C:V3':>8} {'D:V3':>8} "
          f"{'C-A':>7} {'D-B':>7}")
    print(f"  {'':30} {'bypass':>8} {'std+ens':>8} {'bypass':>8} {'ensemb':>8}")
    print(f"  {'-'*80}")

    for _, row in df.iterrows():
        def _v(col):
            v = row.get(col)
            return f"{v:.4f}" if pd.notna(v) else "  N/A "
        ca_delta = ""
        db_delta = ""
        if pd.notna(row.get('v3_bypass_auc')) and pd.notna(row.get('orig_bypass_auc')):
            ca_delta = f"{row['v3_bypass_auc'] - row['orig_bypass_auc']:+.4f}"
        if pd.notna(row.get('v3_ensemble_auc')) and pd.notna(row.get('orig_standard_auc')):
            db_delta = f"{row['v3_ensemble_auc'] - row['orig_standard_auc']:+.4f}"

        print(f"  {row['name']:<30} "
              f"{_v('orig_bypass_auc'):>8} {_v('orig_standard_auc'):>8} "
              f"{_v('v3_bypass_auc'):>8} {_v('v3_ensemble_auc'):>8} "
              f"{ca_delta:>7} {db_delta:>7}")

    # ══════════════════════════════════════════════════════════════════════
    # AEON BENCHMARK COMPARISON
    # ══════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 80)
    print("AEON BENCHMARK COMPARISON (all 4 variants + AEON models)")
    print("=" * 80)

    print("\nLoading AEON benchmarks...")
    benchmarks = load_aeon_benchmarks()
    print(f"  Loaded {len(benchmarks)} AEON models")

    aeon_df = pd.DataFrame(benchmarks)
    aeon_datasets = set(aeon_df.index)
    our_datasets_set = set(df['name'].values)
    common_datasets = sorted(aeon_datasets & our_datasets_set)

    print(f"  Our datasets: {len(our_datasets_set)}")
    print(f"  AEON datasets: {len(aeon_datasets)}")
    print(f"  Common: {len(common_datasets)}")

    if len(common_datasets) == 0:
        print("  No common datasets — cannot compare with AEON")
    else:
        aeon_common = aeon_df.loc[common_datasets].copy()
        our_indexed = df.set_index('name')

        # Add all 4 of our variants
        variant_map = {
            'TabPFN_orig_bypass':   'orig_bypass_auc',
            'TabPFN_orig_standard': 'orig_standard_auc',
            'TabPFN_V3_bypass':     'v3_bypass_auc',
            'TabPFN_V3_ensemble':   'v3_ensemble_auc',
        }
        for vname, vcol in variant_map.items():
            vals = our_indexed.loc[common_datasets, vcol].values
            aeon_common[vname] = vals

        # Drop rows where ALL our variants are NaN
        our_cols = list(variant_map.keys())
        aeon_common = aeon_common.dropna(subset=our_cols, how='all')

        # Rankings
        rankings = aeon_common.rank(axis=1, ascending=False, method='min')
        n_models = len(aeon_common.columns)
        n_ds = len(aeon_common)

        all_stats = []
        for model_name in aeon_common.columns:
            model_ranks = rankings[model_name]
            model_aucs = aeon_common[model_name]
            valid_ranks = model_ranks.dropna()

            mean_rank = valid_ranks.mean() if len(valid_ranks) > 0 else float('inf')
            rank1 = (valid_ranks == 1).sum()
            rank1_strict = 0
            for dataset in valid_ranks.index:
                if valid_ranks[dataset] == 1:
                    if (rankings.loc[dataset] == 1).sum() == 1:
                        rank1_strict += 1

            mean_auc = model_aucs.dropna().mean() if model_aucs.notna().any() else None

            all_stats.append({
                'model': model_name,
                'mean_rank': mean_rank,
                'rank1': int(rank1),
                'rank1_strict': int(rank1_strict),
                'mean_auc': mean_auc,
                'n_datasets': int(model_aucs.notna().sum()),
            })

        stats_df = pd.DataFrame(all_stats).sort_values('mean_rank')

        # Print ranking table
        print(f"\n{'Model':<28} {'Mean Rank':>10} {'Rank1':>7} {'SOTA':>6} "
              f"{'Mean AUC':>10} {'N':>4}")
        print("-" * 70)

        for _, row in stats_df.iterrows():
            is_ours = 'TabPFN' in row['model']
            marker = ">>>" if is_ours else "   "
            auc_str = f"{row['mean_auc']:.4f}" if row['mean_auc'] is not None else "N/A"
            print(f"{marker} {row['model']:<25} {row['mean_rank']:>10.2f} "
                  f"{row['rank1']:>7} {row['rank1_strict']:>6} "
                  f"{auc_str:>10} {row['n_datasets']:>4}")

        # Our variants summary
        print(f"\n{'='*70}")
        print(f"TabPFN VARIANTS SUMMARY ({n_ds} datasets, {n_models} total models)")
        print(f"{'='*70}")

        for vname in our_cols:
            row = stats_df[stats_df['model'] == vname]
            if len(row) > 0:
                row = row.iloc[0]
                pos = (stats_df['mean_rank'] < row['mean_rank']).sum() + 1
                auc_str = f"{row['mean_auc']:.4f}" if row['mean_auc'] is not None else "N/A"
                print(f"\n  {vname}:")
                print(f"    Position: {pos}/{n_models} (mean rank {row['mean_rank']:.2f})")
                print(f"    Rank 1: {row['rank1']}/{n_ds}  "
                      f"(strict SOTA: {row['rank1_strict']}/{n_ds})")
                print(f"    Mean AUC: {auc_str}")

        # Per-dataset: our 4 variants + best AEON
        print(f"\n\n{'Dataset':<28} {'A:Orig':>7} {'B:Orig':>7} {'C:V3':>7} {'D:V3':>7} "
              f"{'Best AEON':>10} {'Best Model':<18}")
        print(f"{'':28} {'bypass':>7} {'std':>7} {'bypass':>7} {'ens':>7}")
        print("-" * 100)

        for dataset in common_datasets:
            a = our_indexed.loc[dataset, 'orig_bypass_auc']
            b = our_indexed.loc[dataset, 'orig_standard_auc']
            c = our_indexed.loc[dataset, 'v3_bypass_auc']
            d = our_indexed.loc[dataset, 'v3_ensemble_auc']

            aeon_row = aeon_df.loc[dataset] if dataset in aeon_df.index else None
            if aeon_row is not None and aeon_row.notna().any():
                best_val = aeon_row.max()
                best_model = aeon_row.idxmax()
            else:
                best_val = None
                best_model = "?"

            def _f2(v):
                return f"{v:.4f}" if pd.notna(v) else "  N/A "
            bv_str = f"{best_val:.4f}" if best_val is not None else "?"

            print(f"  {dataset:<26} {_f2(a):>7} {_f2(b):>7} "
                  f"{_f2(c):>7} {_f2(d):>7} "
                  f"{bv_str:>10} {best_model:<18}")

        # Save results
        output_dir = Path(__file__).parent / "comparison_results"
        output_dir.mkdir(exist_ok=True)

        step_str = checkpoint_path.stem.replace('checkpoint_', '')
        run_name = checkpoint_path.parent.name

        df.to_csv(output_dir / f"v3_4way_{run_name}_{step_str}.csv", index=False)
        stats_df.to_csv(output_dir / f"v3_4way_aeon_rankings_{run_name}_{step_str}.csv",
                        index=False)

        print(f"\nResults saved to {output_dir}")

    print(f"\nDone! Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
