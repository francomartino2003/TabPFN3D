#!/usr/bin/env python3
"""
Benchmark our finetuned model across four configurations.

Configurations evaluated on every PFN-eligible dataset (30 random seeds each):

  D1_e8       — Phase-1 best.pt,  8 ensemble iterations
  D2_e1       — Phase-2 best.pt,  1 ensemble iteration
  D2_e8       — Phase-2 best.pt,  8 ensemble iterations
  D2_e8_prep  — Phase-2 best.pt,  8 ensemble iterations
                + step-repeat preprocessing
                (applied when m*T < 1000 AND T < 96, up to 3 doublings)

Each configuration is evaluated N_RUNS times (default 30) with different
random seeds and results are averaged.  The 30 seeds control the random
channel/class permutations inside evaluate_ensemble.

Output CSV (one row per dataset):
  dataset, collection, m, T, n_classes,
  D1_e8_acc_mean,  D1_e8_acc_std,  D1_e8_auc_mean,  D1_e8_auc_std,
  D2_e1_acc_mean,  D2_e1_acc_std,  D2_e1_auc_mean,  D2_e1_auc_std,
  D2_e8_acc_mean,  D2_e8_acc_std,  D2_e8_auc_mean,  D2_e8_auc_std,
  D2_e8_prep_acc_mean, D2_e8_prep_acc_std,
  D2_e8_prep_auc_mean, D2_e8_prep_auc_std,
  n_runs

Usage:
  python 04_evaluation/benchmark_ours.py \\
      --phase1-ckpt 03_finetuning/checkpoints/phase1/best.pt \\
      --phase2-ckpt 03_finetuning/checkpoints/phase2/best.pt \\
      --collection  ucr

  python 04_evaluation/benchmark_ours.py ... --n-runs 30 --device cuda
"""

import argparse
import gc
import logging
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)
logging.getLogger("posthog").setLevel(logging.CRITICAL)

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

HERE = Path(__file__).parent
ROOT = HERE.parent

for _p in [
    str(ROOT / "00_TabPFN" / "src"),
    str(ROOT / "03_finetuning"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import build_overlap_model
from inference import evaluate_ensemble

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

REAL_DATA_ROOT  = ROOT / "01_real_data"
SUBSAMPLE_N     = 1_000
SUBSAMPLE_SEED  = 0

# Step-repeat thresholds (same as preprocessing_eval.py)
PREP_MT_THRESH  = 1_000
PREP_T_THRESH   = 96
PREP_MAX_ROUNDS = 3


# ─────────────────────────────────────────────────────────────────────────────
# Memory helpers
# ─────────────────────────────────────────────────────────────────────────────

def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_datasets(collection_filter: str | None = None) -> list:
    """Load all PFN-eligible datasets, optionally filtered by collection."""
    summary  = pd.read_csv(REAL_DATA_ROOT / "datasets_summary.csv")
    eligible = summary[summary["passes_pfn_filters"]].copy()
    if collection_filter:
        eligible = eligible[
            eligible["collection"].str.lower() == collection_filter.lower()
        ]

    valid, skipped = [], []
    for _, row in eligible.iterrows():
        name       = row["dataset"]
        collection = row["collection"].lower()
        data_dir   = REAL_DATA_ROOT / "data" / collection
        tr_path    = data_dir / f"{name}_train.npz"
        te_path    = data_dir / f"{name}_test.npz"
        if not tr_path.exists() or not te_path.exists():
            skipped.append(name)
            continue
        try:
            tr = np.load(tr_path, allow_pickle=False)
            te = np.load(te_path, allow_pickle=False)
            X_tr_3d  = tr["X"].astype(np.float32)
            y_tr_raw = tr["y"]
            X_te_3d  = te["X"].astype(np.float32)
            y_te_raw = te["y"]
            n_tr, m, _ = X_tr_3d.shape
            T          = X_te_3d.shape[2]
            n_classes  = int(row["n_classes"])

            if bool(row.get("subsample_train", False)) and n_tr >= SUBSAMPLE_N:
                rng = np.random.RandomState(SUBSAMPLE_SEED)
                idx = rng.choice(n_tr, SUBSAMPLE_N, replace=False)
                idx.sort()
                X_tr_3d  = X_tr_3d[idx]
                y_tr_raw = y_tr_raw[idx]

            le = LabelEncoder()
            le.fit(y_tr_raw)
            y_tr = le.transform(y_tr_raw).astype(np.int64)
            y_te = le.transform(y_te_raw).astype(np.int64)

            X_tr = X_tr_3d.reshape(X_tr_3d.shape[0], -1).astype(np.float32)
            X_te = X_te_3d.reshape(X_te_3d.shape[0], -1).astype(np.float32)
            np.putmask(X_tr, ~np.isfinite(X_tr), np.nan)
            np.putmask(X_te, ~np.isfinite(X_te), np.nan)

            valid.append({
                "name": name, "collection": row["collection"].upper(),
                "X_train": X_tr, "X_test": X_te,
                "y_train": y_tr, "y_test": y_te,
                "n_classes": n_classes, "m": m, "T": T,
            })
        except Exception as e:
            skipped.append(name)
            print(f"  [load] {name}: {str(e)[:80]}")

    if skipped:
        print(f"  [load] Skipped {len(skipped)}: "
              f"{skipped[:5]}{'...' if len(skipped) > 5 else ''}")
    return valid


# ─────────────────────────────────────────────────────────────────────────────
# Step-repeat preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def _step_repeat_once(X_flat: np.ndarray, m: int, T: int) -> np.ndarray:
    return X_flat.reshape(X_flat.shape[0], m, T).repeat(2, axis=2).reshape(X_flat.shape[0], m * 2 * T)


def apply_step_repeat(X_tr, X_te, m, T):
    """Return (X_tr_out, X_te_out, T_out, n_rounds)."""
    T_cur = T
    for r in range(PREP_MAX_ROUNDS):
        if m * T_cur >= PREP_MT_THRESH or T_cur >= PREP_T_THRESH:
            return X_tr, X_te, T_cur, r
        X_tr = _step_repeat_once(X_tr, m, T_cur)
        X_te = _step_repeat_once(X_te, m, T_cur)
        T_cur *= 2
    return X_tr, X_te, T_cur, PREP_MAX_ROUNDS


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, proba, n_classes):
    preds = proba.argmax(axis=1)
    acc   = float(accuracy_score(y_true, preds))
    try:
        auc = (float(roc_auc_score(y_true, proba[:, 1]))
               if n_classes == 2
               else float(roc_auc_score(y_true, proba, multi_class="ovr")))
    except Exception:
        auc = None
    return acc, auc


# ─────────────────────────────────────────────────────────────────────────────
# Single-config evaluation: N_RUNS seeds, return mean/std acc+auc
# ─────────────────────────────────────────────────────────────────────────────

def eval_config(model, X_tr, y_tr, X_te, y_te, n_classes, m, T,
                device, n_iters, n_runs, label):
    """Run evaluate_ensemble n_runs times with seeds 0..n_runs-1, return stats."""
    accs, aucs = [], []
    for seed in range(n_runs):
        try:
            model.to(device)
            with torch.no_grad():
                proba = evaluate_ensemble(
                    model, X_tr, y_tr, X_te,
                    n_classes, m, T, device,
                    n_iters=n_iters, seed=seed, use_overlap=True,
                )
            if proba is None:
                raise RuntimeError("evaluate_ensemble returned None")
            acc, auc = compute_metrics(y_te, proba, n_classes)
            accs.append(acc)
            if auc is not None:
                aucs.append(auc)
            del proba
        except Exception as e:
            # OOM: retry on CPU once
            _free()
            try:
                model.to("cpu")
                with torch.no_grad():
                    proba = evaluate_ensemble(
                        model, X_tr, y_tr, X_te,
                        n_classes, m, T, "cpu",
                        n_iters=n_iters, seed=seed, use_overlap=True,
                    )
                model.to(device)
                if proba is None:
                    continue
                acc, auc = compute_metrics(y_te, proba, n_classes)
                accs.append(acc)
                if auc is not None:
                    aucs.append(auc)
                del proba
            except Exception as e2:
                print(f"      [{label} seed={seed}] CPU fallback failed: "
                      f"{type(e2).__name__}: {str(e2)[:60]}", flush=True)
        _free()

    if not accs:
        return None, None, None, None, 0
    acc_arr = np.array(accs)
    auc_arr = np.array(aucs) if aucs else None
    return (float(acc_arr.mean()), float(acc_arr.std()),
            float(auc_arr.mean()) if auc_arr is not None else None,
            float(auc_arr.std())  if auc_arr is not None else None,
            len(accs))


# ─────────────────────────────────────────────────────────────────────────────
# Load checkpoint helper
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model, _, _, _ = build_overlap_model(device=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    step = ckpt.get("step", "?")
    print(f"  Loaded {ckpt_path.name}  step={step}  version={ckpt.get('version','?')}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark loop
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(datasets, phase1_path, phase2_path, device, n_runs, output_dir):
    results = []

    # ── Phase 1: D1_e8 ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"Phase 1 — D1_e8  ({n_runs} runs × 8 iters each)")
    print("=" * 70)
    model1 = load_model(phase1_path, device)

    d1_e8 = {}
    for i, ds in enumerate(datasets):
        name, m, T, nc = ds["name"], ds["m"], ds["T"], ds["n_classes"]
        t0 = time.time()
        print(f"  [{i+1:3d}/{len(datasets)}] {name:<35} m={m:3d} T={T:5d} cls={nc}", end="", flush=True)
        mu_a, sd_a, mu_u, sd_u, n_ok = eval_config(
            model1, ds["X_train"], ds["y_train"], ds["X_test"], ds["y_test"],
            nc, m, T, device, n_iters=8, n_runs=n_runs, label="D1_e8",
        )
        d1_e8[name] = (mu_a, sd_a, mu_u, sd_u, n_ok)
        tag = f"acc={mu_a:.4f}" if mu_a is not None else "FAIL"
        print(f"  {tag}  ({time.time()-t0:.0f}s  {n_ok}/{n_runs} runs)", flush=True)

    del model1
    _free()

    # ── Phase 2: D2_e1 / D2_e8 / D2_e8_prep ────────────────────────────────
    print("\n" + "=" * 70)
    print(f"Phase 2 — D2_e1 / D2_e8 / D2_e8_prep  ({n_runs} runs each)")
    print("=" * 70)
    model2 = load_model(phase2_path, device)

    d2_e1 = {}
    d2_e8 = {}
    d2_e8_prep = {}

    for i, ds in enumerate(datasets):
        name, m, T, nc = ds["name"], ds["m"], ds["T"], ds["n_classes"]
        print(f"\n  [{i+1:3d}/{len(datasets)}] {name}  m={m} T={T} cls={nc}", flush=True)

        # D2_e1
        t0 = time.time()
        mu_a, sd_a, mu_u, sd_u, n_ok = eval_config(
            model2, ds["X_train"], ds["y_train"], ds["X_test"], ds["y_test"],
            nc, m, T, device, n_iters=1, n_runs=n_runs, label="D2_e1",
        )
        d2_e1[name] = (mu_a, sd_a, mu_u, sd_u, n_ok)
        tag = f"acc={mu_a:.4f}" if mu_a is not None else "FAIL"
        print(f"    D2_e1:       {tag}  ({time.time()-t0:.0f}s)", flush=True)

        # D2_e8
        t0 = time.time()
        mu_a, sd_a, mu_u, sd_u, n_ok = eval_config(
            model2, ds["X_train"], ds["y_train"], ds["X_test"], ds["y_test"],
            nc, m, T, device, n_iters=8, n_runs=n_runs, label="D2_e8",
        )
        d2_e8[name] = (mu_a, sd_a, mu_u, sd_u, n_ok)
        tag = f"acc={mu_a:.4f}" if mu_a is not None else "FAIL"
        print(f"    D2_e8:       {tag}  ({time.time()-t0:.0f}s)", flush=True)

        # D2_e8_prep: apply step-repeat, then run e8
        X_tr_p, X_te_p, T_p, n_rds = apply_step_repeat(
            ds["X_train"], ds["X_test"], m, T
        )
        t0 = time.time()
        if n_rds > 0:
            mu_a, sd_a, mu_u, sd_u, n_ok = eval_config(
                model2, X_tr_p, ds["y_train"], X_te_p, ds["y_test"],
                nc, m, T_p, device, n_iters=8, n_runs=n_runs, label="D2_e8_prep",
            )
        else:
            # No preprocessing applies → same as D2_e8
            mu_a, sd_a, mu_u, sd_u, n_ok = d2_e8[name]
        d2_e8_prep[name] = (mu_a, sd_a, mu_u, sd_u, n_ok, n_rds, T_p)
        tag = f"acc={mu_a:.4f}" if mu_a is not None else "FAIL"
        note = f"T:{T}→{T_p} ({n_rds}x)" if n_rds > 0 else "no prep"
        print(f"    D2_e8_prep:  {tag}  [{note}]  ({time.time()-t0:.0f}s)", flush=True)

        del X_tr_p, X_te_p

    del model2
    _free()

    # ── Assemble CSV ─────────────────────────────────────────────────────────
    rows = []
    for ds in datasets:
        name = ds["name"]
        r1 = d1_e8.get(name,      (None,)*5)
        r2 = d2_e1.get(name,      (None,)*5)
        r3 = d2_e8.get(name,      (None,)*5)
        r4 = d2_e8_prep.get(name, (None,)*7)

        rows.append({
            "dataset":    name,
            "collection": ds["collection"],
            "m":          ds["m"],
            "T":          ds["T"],
            "n_classes":  ds["n_classes"],
            # D1_e8
            "D1_e8_acc_mean": r1[0], "D1_e8_acc_std": r1[1],
            "D1_e8_auc_mean": r1[2], "D1_e8_auc_std": r1[3],
            "D1_e8_n_runs":   r1[4],
            # D2_e1
            "D2_e1_acc_mean": r2[0], "D2_e1_acc_std": r2[1],
            "D2_e1_auc_mean": r2[2], "D2_e1_auc_std": r2[3],
            "D2_e1_n_runs":   r2[4],
            # D2_e8
            "D2_e8_acc_mean": r3[0], "D2_e8_acc_std": r3[1],
            "D2_e8_auc_mean": r3[2], "D2_e8_auc_std": r3[3],
            "D2_e8_n_runs":   r3[4],
            # D2_e8_prep
            "D2_e8_prep_acc_mean": r4[0], "D2_e8_prep_acc_std": r4[1],
            "D2_e8_prep_auc_mean": r4[2], "D2_e8_prep_auc_std": r4[3],
            "D2_e8_prep_n_runs":   r4[4],
            "D2_e8_prep_T_after":  r4[6] if len(r4) > 6 else None,
            "D2_e8_prep_n_repeats": r4[5] if len(r4) > 5 else 0,
        })

    df = pd.DataFrame(rows)

    # Quick summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for col, label in [
        ("D1_e8_acc_mean",       "D1_e8      acc"),
        ("D2_e1_acc_mean",       "D2_e1      acc"),
        ("D2_e8_acc_mean",       "D2_e8      acc"),
        ("D2_e8_prep_acc_mean",  "D2_e8_prep acc"),
    ]:
        v = df[col].dropna()
        if len(v):
            print(f"  {label}: mean={v.mean():.4f}  median={v.median():.4f}  "
                  f"n={len(v)}/{len(df)}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark phase1/phase2 model across 4 configs (30 runs each)"
    )
    parser.add_argument(
        "--phase1-ckpt", type=str,
        default="03_finetuning/checkpoints/phase1/best.pt",
        help="Path to Phase-1 best.pt",
    )
    parser.add_argument(
        "--phase2-ckpt", type=str,
        default="03_finetuning/checkpoints/phase2/best.pt",
        help="Path to Phase-2 best.pt",
    )
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto / cuda / mps / cpu")
    parser.add_argument("--collection", type=str, default=None,
                        choices=["ucr", "uea"],
                        help="Restrict to UCR or UEA (default: both)")
    parser.add_argument("--n-runs", type=int, default=30,
                        help="Number of seeds per config (default: 30)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: 04_evaluation/results/benchmark_ours/)")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    # Paths
    phase1_path = Path(args.phase1_ckpt)
    phase2_path = Path(args.phase2_ckpt)
    if not phase1_path.is_absolute():
        phase1_path = ROOT / phase1_path
    if not phase2_path.is_absolute():
        phase2_path = ROOT / phase2_path
    for p in [phase1_path, phase2_path]:
        if not p.exists():
            print(f"ERROR: checkpoint not found: {p}")
            sys.exit(1)

    # Output dir
    out_root = Path(args.out_dir) if args.out_dir else HERE / "results" / "benchmark_ours"
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BENCHMARK — our model (phase1 / phase2)")
    print("=" * 70)
    print(f"  Phase-1 ckpt : {phase1_path}")
    print(f"  Phase-2 ckpt : {phase2_path}")
    print(f"  Device       : {device}")
    print(f"  Collection   : {args.collection or 'all'}")
    print(f"  N runs       : {args.n_runs}")
    print(f"  Output       : {out_root}")

    # Collections to run
    collections = (
        [args.collection]
        if args.collection
        else ["ucr", "uea"]
    )

    for coll in collections:
        print(f"\n\n{'='*70}")
        print(f"Loading datasets — collection: {coll.upper()}")
        print(f"{'='*70}")
        datasets = load_datasets(collection_filter=coll)
        if not datasets:
            print(f"  No datasets found for {coll}. Skipping.")
            continue
        print(f"  {len(datasets)} datasets loaded")

        df = run_benchmark(datasets, phase1_path, phase2_path,
                           device, args.n_runs, out_root)

        out_csv = out_root / f"{coll}_benchmark_ours.csv"
        df.to_csv(out_csv, index=False)
        print(f"\n  Saved: {out_csv}")

    print("\n\nDONE")


if __name__ == "__main__":
    main()
