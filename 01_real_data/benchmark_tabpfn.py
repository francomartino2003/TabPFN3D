#!/usr/bin/env python3
"""
Benchmark TabPFN on UCR and UEA datasets that pass the PFN filter.

Three variants, each run N_RUNS times (different random seeds) and averaged:

  e1           — TabPFN, n_estimators=1  (standard sklearn pipeline)
  e8           — TabPFN, n_estimators=8  (standard sklearn pipeline)
  e8_ours_inf  — TabPFN pretrained model (no finetuning) + OUR inference pipeline:
                   • per-channel normalisation (not per-position)
                   • 8 iterations: channel permutation + class permutation
                   • fit_from_preprocessed — no feature subsampling, no fingerprint
                   • temperature = 0.9
                 Isolates the effect of our inference strategy from finetuning.

Results are saved to 01_real_data/benchmark_results/{ucr,uea}_benchmark_tabpfn.csv
and loaded by 04_evaluation/final_evaluation.py.

Usage:
  python 01_real_data/benchmark_tabpfn.py
  python 01_real_data/benchmark_tabpfn.py --device cuda --n-runs 30
  python 01_real_data/benchmark_tabpfn.py --dataset ItalyPowerDemand
"""
from __future__ import annotations

import argparse
import gc
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parent
TABPFN_SRC = PROJECT_ROOT / "00_TabPFN" / "src"

if str(TABPFN_SRC) not in sys.path:
    sys.path.insert(0, str(TABPFN_SRC))

from tabpfn import TabPFNClassifier

DATA_UCR = HERE / "data" / "ucr"
DATA_UEA = HERE / "data" / "uea"
SUMMARY_CSV = HERE / "datasets_summary.csv"
OUT_DIR = HERE / "benchmark_results"
N_RUNS_DEFAULT = 30
SUBSAMPLE_N = 1_000
SUBSAMPLE_SEED = 0
SOFTMAX_TEMP = 0.9   # matches TabPFN default and our inference.py


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()


def _compute_metrics(y_test, proba, n_classes):
    y_pred = proba.argmax(axis=1)
    acc = float(accuracy_score(y_test, y_pred))
    try:
        if n_classes == 2:
            auc = float(roc_auc_score(y_test, proba[:, 1]))
        else:
            auc = float(roc_auc_score(y_test, proba, multi_class="ovr"))
    except Exception:
        auc = float("nan")
    return acc, auc


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_and_prepare(name: str, collection: str, subsample_train: bool):
    """Load train/test from npz, optionally subsample train, flatten to 2D.

    Returns (X_train, y_train, X_test, y_test, n_classes, m, T) or None.
    m = number of channels, T = series length (kept for e8_ours_inf).
    """
    data_dir = DATA_UCR if collection == "UCR" else DATA_UEA
    tr_path = data_dir / f"{name}_train.npz"
    te_path = data_dir / f"{name}_test.npz"
    if not tr_path.exists() or not te_path.exists():
        return None
    try:
        with np.load(tr_path, allow_pickle=True) as d:
            X_tr = np.asarray(d["X"], dtype=np.float32)
            y_tr = np.asarray(d["y"]).ravel()
        with np.load(te_path, allow_pickle=True) as d:
            X_te = np.asarray(d["X"], dtype=np.float32)
            y_te = np.asarray(d["y"]).ravel()
    except Exception:
        return None

    if subsample_train and X_tr.shape[0] >= SUBSAMPLE_N:
        rng = np.random.RandomState(SUBSAMPLE_SEED)
        idx = rng.choice(X_tr.shape[0], SUBSAMPLE_N, replace=False)
        idx.sort()
        X_tr, y_tr = X_tr[idx], y_tr[idx]

    # Record m and T before flattening (needed for per-channel normalisation)
    n_tr = X_tr.shape[0]
    m    = X_tr.shape[1] if X_tr.ndim == 3 else 1
    T    = X_tr.shape[2] if X_tr.ndim == 3 else X_tr.shape[1]
    n_te = X_te.shape[0]

    X_tr = X_tr.reshape(n_tr, -1)
    X_te = X_te.reshape(n_te, -1)

    np.putmask(X_tr, np.isinf(X_tr), np.nan)
    np.putmask(X_te, np.isinf(X_te), np.nan)

    all_y = np.concatenate([y_tr, y_te])
    if all_y.dtype.kind in ("i", "u") or (all_y.dtype.kind == "f" and np.all(np.isfinite(all_y))):
        uniq = np.unique(all_y)
        if len(uniq) > 100:
            return None
        le = LabelEncoder()
        le.fit(all_y)
        y_tr = le.transform(y_tr).astype(np.int64)
        y_te = le.transform(y_te).astype(np.int64)
    else:
        uniq = sorted(np.unique(all_y), key=str)
        label_to_idx = {v: i for i, v in enumerate(uniq)}
        y_tr = np.array([label_to_idx[v] for v in y_tr], dtype=np.int64)
        y_te = np.array([label_to_idx[v] for v in y_te], dtype=np.int64)

    n_classes = int(len(np.unique(np.concatenate([y_tr, y_te]))))
    return X_tr, y_tr, X_te, y_te, n_classes, m, T


# ─────────────────────────────────────────────────────────────────────────────
# Single run
# ─────────────────────────────────────────────────────────────────────────────

def _run_tabpfn(X_tr, y_tr, X_te, y_te, n_classes, n_estimators, seed, device):
    """One fit+predict with TabPFN; falls back to CPU on OOM."""
    def _try(dev):
        clf = TabPFNClassifier(
            device=dev,
            n_estimators=n_estimators,
            random_state=seed,
            ignore_pretraining_limits=True,
        )
        clf.fit(X_tr, y_tr)
        with torch.no_grad():
            proba = clf.predict_proba(X_te)
        del clf
        return _compute_metrics(y_te, proba, n_classes)

    try:
        return _try(device)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and device != "cpu":
            _free_memory()
            try:
                return _try("cpu")
            except Exception:
                return float("nan"), float("nan")
        return float("nan"), float("nan")
    except Exception:
        return float("nan"), float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for e8_ours_inf (our inference pipeline on pretrained TabPFN)
# ─────────────────────────────────────────────────────────────────────────────

def _pcn(X_tr: np.ndarray, X_te: np.ndarray, m: int, T: int):
    """Per-channel normalisation: mean=0, std=1 using train-only statistics."""
    n_tr, n_te = X_tr.shape[0], X_te.shape[0]
    X_tr3 = X_tr.reshape(n_tr, m, T).copy().astype(np.float64)
    X_te3 = X_te.reshape(n_te, m, T).copy().astype(np.float64)
    for j in range(m):
        vals = X_tr3[:, j, :].ravel()
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            continue
        mu  = finite.mean()
        std = finite.std()
        if std < 1e-8:
            std = 1.0
        X_tr3[:, j, :] = (X_tr3[:, j, :] - mu) / std
        X_te3[:, j, :] = (X_te3[:, j, :] - mu) / std
    return (X_tr3.reshape(n_tr, m * T).astype(np.float32),
            X_te3.reshape(n_te, m * T).astype(np.float32))


def _run_tabpfn_our_inference(X_tr, y_tr, X_te, y_te,
                              n_classes, m, T, seed, device, n_iters=8):
    """Standard pretrained TabPFN + our inference pipeline.

    Uses the pretrained model weights unchanged (features_per_group=3,
    original encoder) but applies our diversity strategy:
      - Per-channel normalisation (not TabPFN's per-position z-score)
      - Channel permutation + class permutation per iteration
      - fit_from_preprocessed: bypasses SVD, subsampling, fingerprint
      - temperature = 0.9

    This ablation isolates how much of our improvement comes from the
    inference pipeline vs. the finetuned weights.
    """
    from tabpfn.preprocessing.configs import ClassifierEnsembleConfig, PreprocessorConfig

    # Build the standard pretrained model once (not our overlap model)
    clf = TabPFNClassifier(
        device=device,
        n_estimators=1,
        ignore_pretraining_limits=True,
        fit_mode="batched",
        differentiable_input=False,
        inference_config={"FEATURE_SHIFT_METHOD": None},
    )
    clf._initialize_model_variables()

    dummy_cfg = ClassifierEnsembleConfig(
        preprocess_config=PreprocessorConfig("none", categorical_name="numeric"),
        feature_shift_count=0,
        class_permutation=None,
        add_fingerprint_feature=False,
        polynomial_features="no",
        feature_shift_decoder=None,
        subsample_ix=None,
        _model_index=0,
    )

    rng = np.random.RandomState(seed)
    proba_sum = np.zeros((X_te.shape[0], n_classes), dtype=np.float64)
    n_valid   = 0

    for it in range(n_iters):
        try:
            feat_perm = rng.permutation(m)
            n_tr_s    = X_tr.shape[0]
            X_tr_it   = X_tr.reshape(n_tr_s, m, T)[:, feat_perm, :].reshape(n_tr_s, m * T)
            X_te_it   = X_te.reshape(X_te.shape[0], m, T)[:, feat_perm, :].reshape(X_te.shape[0], m * T)

            class_perm = rng.permutation(n_classes)
            y_tr_it    = class_perm[y_tr]

            X_tr_it, X_te_it = _pcn(X_tr_it, X_te_it, m, T)

            X_tr_t = torch.as_tensor(X_tr_it, dtype=torch.float32, device=device).unsqueeze(0)
            y_tr_t = torch.as_tensor(y_tr_it.astype(np.float32), device=device).unsqueeze(0)
            X_te_t = torch.as_tensor(X_te_it, dtype=torch.float32, device=device).unsqueeze(0)

            clf.n_classes_ = n_classes
            clf.fit_from_preprocessed(
                [X_tr_t], [y_tr_t],
                cat_ix=[[[]]],
                configs=[[dummy_cfg]],
            )
            with torch.no_grad():
                logits = clf.forward([X_te_t], return_raw_logits=True)

            if logits.ndim == 2:
                logits_out = logits
            elif logits.ndim == 3:
                logits_out = logits.squeeze(1)
            elif logits.ndim == 4:
                logits_out = logits.mean(dim=(1, 2))
            else:
                continue

            logits_out = logits_out[:, :n_classes]
            logits_out = logits_out[:, class_perm]     # undo class permutation
            proba = torch.softmax(logits_out / SOFTMAX_TEMP, dim=-1).cpu().numpy()
            proba_sum += proba
            n_valid   += 1

            del X_tr_t, y_tr_t, X_te_t, logits, logits_out, proba

        except Exception:
            continue

    del clf
    _free_memory()

    if n_valid == 0:
        return float("nan"), float("nan")

    return _compute_metrics(y_te, proba_sum / n_valid, n_classes)


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset benchmark
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_dataset(name, collection, subsample_train, n_runs, device):
    data = load_and_prepare(name, collection, subsample_train)
    if data is None:
        return None
    X_tr, y_tr, X_te, y_te, n_classes, m, T = data

    acc1, auc1 = [], []
    acc8, auc8 = [], []
    acc8oi, auc8oi = [], []  # e8_ours_inf

    for run in range(n_runs):
        seed = 42 + run

        _free_memory()
        a, u = _run_tabpfn(X_tr, y_tr, X_te, y_te, n_classes, 1, seed, device)
        acc1.append(a); auc1.append(u)

        _free_memory()
        a, u = _run_tabpfn(X_tr, y_tr, X_te, y_te, n_classes, 8, seed, device)
        acc8.append(a); auc8.append(u)

        _free_memory()
        a, u = _run_tabpfn_our_inference(
            X_tr, y_tr, X_te, y_te, n_classes, m, T, seed, device
        )
        acc8oi.append(a); auc8oi.append(u)

    _free_memory()
    return {
        "dataset":                name,
        "collection":             collection,
        "e1_acc_mean":            float(np.nanmean(acc1)),
        "e1_acc_std":             float(np.nanstd(acc1)),
        "e1_auc_mean":            float(np.nanmean(auc1)),
        "e1_auc_std":             float(np.nanstd(auc1)),
        "e8_acc_mean":            float(np.nanmean(acc8)),
        "e8_acc_std":             float(np.nanstd(acc8)),
        "e8_auc_mean":            float(np.nanmean(auc8)),
        "e8_auc_std":             float(np.nanstd(auc8)),
        "e8_ours_inf_acc_mean":   float(np.nanmean(acc8oi)),
        "e8_ours_inf_acc_std":    float(np.nanstd(acc8oi)),
        "e8_ours_inf_auc_mean":   float(np.nanmean(auc8oi)),
        "e8_ours_inf_auc_std":    float(np.nanstd(auc8oi)),
        "n_runs":                 n_runs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark TabPFN (1-ens and 8-ens) on UCR/UEA PFN-filtered datasets."
    )
    parser.add_argument("--ucr-only", action="store_true")
    parser.add_argument("--uea-only", action="store_true")
    parser.add_argument("--n-runs", type=int, default=N_RUNS_DEFAULT,
                        help=f"Runs per dataset per variant (default {N_RUNS_DEFAULT})")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto (cuda→mps→cpu), cpu, cuda, mps")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Evaluate a single dataset by name (for testing)")
    args = parser.parse_args()

    if not SUMMARY_CSV.exists():
        print(f"Missing {SUMMARY_CSV}. Run 01_real_data/download.py first.")
        sys.exit(1)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    df = pd.read_csv(SUMMARY_CSV)
    passes = df["passes_pfn_filters"]
    if passes.dtype == object:
        passes = passes.map(lambda x: x is True or (isinstance(x, str) and x.lower() == "true"))
    df = df.loc[passes].copy()
    if df.empty:
        print("No datasets pass PFN filters.")
        sys.exit(0)

    if args.dataset:
        df = df[df["dataset"] == args.dataset]
        if df.empty:
            print(f"Dataset '{args.dataset}' not found or does not pass PFN filters.")
            sys.exit(1)

    do_ucr = not args.uea_only
    do_uea = not args.ucr_only
    ucr_rows = df[df["collection"] == "UCR"] if do_ucr else pd.DataFrame()
    uea_rows = df[df["collection"] == "UEA"] if do_uea else pd.DataFrame()

    n_total = len(ucr_rows) + len(uea_rows)
    n_runs = args.n_runs
    print(f"Datasets: {len(ucr_rows)} UCR + {len(uea_rows)} UEA  ({n_total} total)")
    print(f"Runs per dataset: {n_runs}  (e1 + e8 + e8_ours_inf each run)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def _run_collection(rows, collection_name, out_filename):
        if rows.empty:
            return
        names = list(rows["dataset"])
        sub_map = dict(zip(rows["dataset"], rows["subsample_train"]))
        out_path = OUT_DIR / out_filename

        if out_path.exists():
            done = set(pd.read_csv(out_path)["dataset"].tolist())
            remaining = [n for n in names if n not in done]
            print(f"\n{collection_name}: {len(remaining)}/{len(names)} remaining "
                  f"({len(done)} already done, resuming {out_path.name})")
        else:
            done = set()
            remaining = names
            print(f"\n{collection_name}: {len(names)} datasets")

        for name in tqdm(remaining, desc=collection_name):
            r = benchmark_dataset(
                name, collection_name,
                bool(sub_map.get(name, False)),
                n_runs, device,
            )
            if r is not None:
                row_df = pd.DataFrame([r])
                row_df.to_csv(out_path, mode="a",
                              header=not out_path.exists() or out_path.stat().st_size == 0,
                              index=False)
                tqdm.write(
                    f"  {name}: "
                    f"e1 acc={r['e1_acc_mean']:.4f}  "
                    f"e8 acc={r['e8_acc_mean']:.4f}  "
                    f"e8_ours_inf acc={r['e8_ours_inf_acc_mean']:.4f}"
                )
            else:
                tqdm.write(f"  {name}: SKIP (load failed)")
        print(f"  Saved {out_path}")

    _run_collection(ucr_rows, "UCR", "ucr_benchmark_tabpfn.csv")
    _run_collection(uea_rows, "UEA", "uea_benchmark_tabpfn.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
