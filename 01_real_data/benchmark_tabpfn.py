#!/usr/bin/env python3
"""
Benchmark TabPFN on UCR and UEA datasets that pass the PFN filter.

Two variants evaluated, each run N_RUNS times (different random_state seeds)
and averaged:

  tabpfn_1   — TabPFN with n_estimators=1  (single forward pass)
  tabpfn_8   — TabPFN with n_estimators=8  (8 built-in ensemble members)

Both use TabPFN's default preprocessing (SVD, fingerprint, feature subsampling,
squashing scaler, feature/class shifts) with no modifications.

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

    Returns (X_train, y_train, X_test, y_test, n_classes) or None.
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

    n_tr, n_te = X_tr.shape[0], X_te.shape[0]
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
    return X_tr, y_tr, X_te, y_te, n_classes


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
# Per-dataset benchmark
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_dataset(name, collection, subsample_train, n_runs, device):
    data = load_and_prepare(name, collection, subsample_train)
    if data is None:
        return None
    X_tr, y_tr, X_te, y_te, n_classes = data

    acc1, auc1 = [], []
    acc8, auc8 = [], []

    for run in range(n_runs):
        seed = 42 + run

        _free_memory()
        a, u = _run_tabpfn(X_tr, y_tr, X_te, y_te, n_classes, 1, seed, device)
        acc1.append(a)
        auc1.append(u)

        _free_memory()
        a, u = _run_tabpfn(X_tr, y_tr, X_te, y_te, n_classes, 8, seed, device)
        acc8.append(a)
        auc8.append(u)

    _free_memory()
    return {
        "dataset":       name,
        "collection":    collection,
        "e1_acc_mean":   float(np.nanmean(acc1)),
        "e1_acc_std":    float(np.nanstd(acc1)),
        "e1_auc_mean":   float(np.nanmean(auc1)),
        "e1_auc_std":    float(np.nanstd(auc1)),
        "e8_acc_mean":   float(np.nanmean(acc8)),
        "e8_acc_std":    float(np.nanstd(acc8)),
        "e8_auc_mean":   float(np.nanmean(auc8)),
        "e8_auc_std":    float(np.nanstd(auc8)),
        "n_runs":        n_runs,
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
    print(f"Runs per dataset: {n_runs}  (1-ens + 8-ens each run)")

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
                    f"1-ens acc={r['e1_acc_mean']:.4f} auc={r['e1_auc_mean']:.4f}  "
                    f"8-ens acc={r['e8_acc_mean']:.4f} auc={r['e8_auc_mean']:.4f}"
                )
            else:
                tqdm.write(f"  {name}: SKIP (load failed)")
        print(f"  Saved {out_path}")

    _run_collection(ucr_rows, "UCR", "ucr_benchmark_tabpfn.csv")
    _run_collection(uea_rows, "UEA", "uea_benchmark_tabpfn.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
