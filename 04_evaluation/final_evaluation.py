#!/usr/bin/env python3
"""
Final Evaluation — comparing our finetuned overlap model against TabPFN baselines and SOTA.

Test 1: Direct comparison  (e1 / e8  vs  D_e1 / D_e8)
--------------------------------------------------------
  e1   — TabPFN standard, 1 ensemble  (precomputed by benchmark_tabpfn.py)
  e8   — TabPFN standard, 8 ensembles (precomputed by benchmark_tabpfn.py)
  D_e1 — Our overlap finetuned model, 1 ensemble iteration  (run live)
  D_e8 — Our overlap finetuned model, 8 ensemble iterations (run live)

  Fair apples-to-apples comparisons:  D_e1 vs e1  and  D_e8 vs e8.

  Precomputed baselines loaded from 01_real_data/benchmark_results/:
    ucr_benchmark_tabpfn.csv  /  uea_benchmark_tabpfn.csv
  Column names accepted: e1_acc_mean / e8_acc_mean (new) or B_acc_mean / C_acc_mean (legacy).

  Output: test1_comparison.csv, test1_scatter.png

Test 2: SOTA comparison  (D_e8 vs HC2 benchmarks)
---------------------------------------------------
  Compares D_e8 against the HC2 accuracy benchmarks from
  01_real_data/benchmarks_hc2/ on the intersection of PFN-filtered datasets
  and the HC2 benchmark datasets.

  Output: test2_sota_rankings.csv

Usage:
  python 04_evaluation/final_evaluation.py 03_finetuning/checkpoints/phase2/best.pt
  python 04_evaluation/final_evaluation.py <ckpt.pt> --device cuda
  python 04_evaluation/final_evaluation.py <ckpt.pt> --skip-test1
  python 04_evaluation/final_evaluation.py <ckpt.pt> --collection uea
  python 04_evaluation/final_evaluation.py <ckpt.pt> --with-history
"""

import argparse
import gc
import json
import logging
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)
logging.getLogger("posthog").setLevel(logging.CRITICAL)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats as sp_stats
from sklearn.metrics import accuracy_score, roc_auc_score

HERE = Path(__file__).parent
ROOT = HERE.parent

for _p in [
    str(ROOT / "00_TabPFN" / "src"),
    str(ROOT / "03_finetuning"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import build_overlap_model, build_temporal_tabpfn_fpg8
from inference import evaluate_ensemble

# ─────────────────────────────────────────────────────────────────────────────
# Paths and constants
# ─────────────────────────────────────────────────────────────────────────────

REAL_DATA_ROOT      = ROOT / "01_real_data"
BENCHMARKS_HC2_DIR  = REAL_DATA_ROOT / "benchmarks_hc2"
BENCHMARK_RESULTS_DIR = REAL_DATA_ROOT / "benchmark_results"

ENSEMBLE_SEED = 42
SUBSAMPLE_N    = 1_000
SUBSAMPLE_SEED = 0


# ─────────────────────────────────────────────────────────────────────────────
# Memory management
# ─────────────────────────────────────────────────────────────────────────────

def _free_memory():
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

def load_datasets(names: list | None = None) -> list:
    """Load all PFN-eligible datasets from datasets_summary.csv.

    Applies subsample_train if flagged (fixed seed for reproducibility).
    Returns list of dicts: name, collection, X_train, X_test, y_train, y_test,
    n_classes, m (channels), T (timesteps).
    """
    from sklearn.preprocessing import LabelEncoder

    summary = pd.read_csv(REAL_DATA_ROOT / "datasets_summary.csv")
    eligible = summary[summary["passes_pfn_filters"]].copy()
    if names is not None:
        eligible = eligible[eligible["dataset"].isin(names)]

    valid, skipped = [], []
    for _, row in eligible.iterrows():
        name       = row["dataset"]
        collection = row["collection"].lower()
        data_dir   = REAL_DATA_ROOT / "data" / collection
        tr_path    = data_dir / f"{name}_train.npz"
        te_path    = data_dir / f"{name}_test.npz"
        if not tr_path.exists() or not te_path.exists():
            skipped.append((name, "no NPZ"))
            continue
        try:
            tr = np.load(tr_path, allow_pickle=False)
            te = np.load(te_path, allow_pickle=False)
            X_tr_3d  = tr["X"].astype(np.float32)
            y_tr_raw = tr["y"]
            X_te_3d  = te["X"].astype(np.float32)
            y_te_raw = te["y"]
            n_tr, m, _ = X_tr_3d.shape
            n_classes  = int(row["n_classes"])
            T          = X_te_3d.shape[2]   # use test T for model dimensioning

            if bool(row.get("subsample_train", False)) and n_tr >= SUBSAMPLE_N:
                rng = np.random.RandomState(SUBSAMPLE_SEED)
                idx = rng.choice(n_tr, SUBSAMPLE_N, replace=False)
                idx.sort()
                X_tr_3d  = X_tr_3d[idx]
                y_tr_raw = y_tr_raw[idx]

            X_tr = X_tr_3d.reshape(X_tr_3d.shape[0], -1)
            X_te = X_te_3d.reshape(X_te_3d.shape[0], -1)
            np.putmask(X_tr, ~np.isfinite(X_tr), np.nan)
            np.putmask(X_te, ~np.isfinite(X_te), np.nan)

            le = LabelEncoder()
            le.fit(y_tr_raw)
            y_tr = le.transform(y_tr_raw).astype(np.int64)
            y_te = le.transform(y_te_raw).astype(np.int64)

            valid.append({
                "name": name, "collection": row["collection"],
                "X_train": X_tr, "X_test": X_te,
                "y_train": y_tr, "y_test": y_te,
                "n_classes": n_classes, "m": m, "T": T,
            })
        except Exception as e:
            skipped.append((name, str(e)[:60]))

    if skipped:
        print(f"  [load] Skipped {len(skipped)}: "
              f"{[s[0] for s in skipped[:5]]}{'...' if len(skipped) > 5 else ''}")
    return valid


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble inference with CPU fallback on OOM
# ─────────────────────────────────────────────────────────────────────────────

def _eval_with_fallback(ft_model, X_tr, y_tr, X_te, nc, m, T, device,
                        n_iters, seed, use_overlap, label=""):
    """Call evaluate_ensemble; if result is None and not on CPU, retry on CPU.

    evaluate_ensemble catches exceptions internally (printing them) and returns
    None when all iterations fail — OOM never propagates out as an exception.
    Retrying on CPU handles both OOM and other device-specific failures.
    """
    def _try(dev):
        ft_model.to(dev)
        with torch.no_grad():
            return evaluate_ensemble(
                ft_model, X_tr, y_tr, X_te,
                nc, m, T, dev, n_iters=n_iters, seed=seed,
                use_overlap=use_overlap,
            )

    proba = _try(device)
    if proba is None and device != "cpu":
        _free_memory()
        print(f"    [{label}] retrying on CPU...", flush=True)
        proba = _try("cpu")
        ft_model.to(device)
    return proba


# ─────────────────────────────────────────────────────────────────────────────
# Precomputed baselines loader
# ─────────────────────────────────────────────────────────────────────────────

def load_precomputed_baselines() -> dict:
    """Load pre-computed e1 / e8 baseline results from benchmark CSVs.

    Reads columns: e1_acc_mean, e1_auc_mean, e8_acc_mean, e8_auc_mean.
    Falls back to legacy column names: B_acc_mean → e1, C_acc_mean → e8.

    Returns {dataset_name: {'e1_acc', 'e1_auc', 'e8_acc', 'e8_auc'}}.
    """
    out = {}
    for fname in ["ucr_benchmark_tabpfn.csv", "uea_benchmark_tabpfn.csv"]:
        path = BENCHMARK_RESULTS_DIR / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        cols = set(df.columns)
        for _, row in df.iterrows():
            entry = {}
            for new_col, legacy_col, key in [
                ("e1_acc_mean", "B_acc_mean", "e1_acc"),
                ("e1_auc_mean", "B_auc_mean", "e1_auc"),
                ("e8_acc_mean", "C_acc_mean", "e8_acc"),
                ("e8_auc_mean", "C_auc_mean", "e8_auc"),
            ]:
                src = new_col if new_col in cols else legacy_col if legacy_col in cols else None
                if src and pd.notna(row.get(src)):
                    entry[key] = float(row[src])
            if entry:
                out[row["dataset"]] = entry
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, proba, n_classes):
    preds = proba.argmax(axis=1)
    acc = accuracy_score(y_true, preds)
    try:
        auc = (roc_auc_score(y_true, proba[:, 1])
               if n_classes == 2
               else roc_auc_score(y_true, proba, multi_class="ovr"))
    except Exception:
        auc = None
    return acc, auc


def _wilcoxon_summary(a, b, label_a, label_b):
    diffs = a - b
    stat, pval = sp_stats.wilcoxon(a, b, alternative="greater")
    wins   = (diffs > 0.001).sum()
    ties   = ((diffs >= -0.001) & (diffs <= 0.001)).sum()
    losses = (diffs < -0.001).sum()
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
    print(f"    Mean Δ: {diffs.mean():+.4f}   Median: {np.median(diffs):+.4f}")
    print(f"    Range:  {diffs.min():+.4f} … {diffs.max():+.4f}")
    print(f"    Win/Tie/Loss: {wins}/{ties}/{losses}")
    print(f"    Wilcoxon ({label_a} > {label_b}): stat={stat:.1f}, p={pval:.6f}  {sig}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — Direct comparison: e1/e8 vs D_e1/D_e8
# ─────────────────────────────────────────────────────────────────────────────

def run_test1(datasets, ft_model, device, output_dir,
              use_overlap: bool = True,
              precomputed: dict | None = None):
    """Run our model with 1 and 8 ensemble iters and compare against precomputed baselines.

    Columns in output CSV:
      name, m, T, n_classes,
      e1_acc, e1_auc,   e8_acc, e8_auc,       ← precomputed TabPFN baselines
      D_e1_acc, D_e1_auc,  D_e8_acc, D_e8_auc  ← our model (live)
    """
    pre = precomputed or {}
    has_e1 = any("e1_acc" in v for v in pre.values())
    has_e8 = any("e8_acc" in v for v in pre.values())
    baseline_str = ("e1+e8 precomputed" if has_e1 and has_e8
                    else "e1 precomputed" if has_e1
                    else "e8 precomputed" if has_e8
                    else "no precomputed baselines")

    print("\n" + "=" * 80)
    print(f"TEST 1: Comparison  e1/e8 vs D_e1/D_e8  [{baseline_str}]")
    print("=" * 80)

    results = []
    t0 = time.time()
    _f = lambda v: f"{v:.4f}" if v is not None else "FAIL"

    for i, ds in enumerate(datasets):
        name, m, T, nc = ds["name"], ds["m"], ds["T"], ds["n_classes"]
        print(f"  [{i+1}/{len(datasets)}] {name}  (m={m}, T={T}, cls={nc})", flush=True)
        res = {"name": name, "m": m, "T": T, "n_classes": nc}
        entry = pre.get(name, {})

        # ── Precomputed baselines ──
        res["e1_acc"] = entry.get("e1_acc")
        res["e1_auc"] = entry.get("e1_auc")
        res["e8_acc"] = entry.get("e8_acc")
        res["e8_auc"] = entry.get("e8_auc")

        # ── D_e1 (1 ensemble iter) ──
        _free_memory()
        try:
            proba = _eval_with_fallback(
                ft_model, ds["X_train"], ds["y_train"], ds["X_test"],
                nc, m, T, device, n_iters=1, seed=ENSEMBLE_SEED,
                use_overlap=use_overlap, label="D_e1",
            )
            res["D_e1_acc"], res["D_e1_auc"] = (
                compute_metrics(ds["y_test"], proba, nc) if proba is not None
                else (None, None)
            )
            del proba
        except Exception as e:
            print(f"    [D_e1] FAILED: {type(e).__name__}: {str(e)[:80]}")
            res["D_e1_acc"], res["D_e1_auc"] = None, None
        _free_memory()

        # ── D_e8 (8 ensemble iters) ──
        try:
            proba = _eval_with_fallback(
                ft_model, ds["X_train"], ds["y_train"], ds["X_test"],
                nc, m, T, device, n_iters=8, seed=ENSEMBLE_SEED,
                use_overlap=use_overlap, label="D_e8",
            )
            res["D_e8_acc"], res["D_e8_auc"] = (
                compute_metrics(ds["y_test"], proba, nc) if proba is not None
                else (None, None)
            )
            del proba
        except Exception as e:
            print(f"    [D_e8] FAILED: {type(e).__name__}: {str(e)[:80]}")
            res["D_e8_acc"], res["D_e8_auc"] = None, None
        _free_memory()

        print(f"    e1:   acc={_f(res['e1_acc'])}  auc={_f(res['e1_auc'])}")
        print(f"    e8:   acc={_f(res['e8_acc'])}  auc={_f(res['e8_auc'])}")
        print(f"    D_e1: acc={_f(res['D_e1_acc'])}  auc={_f(res['D_e1_auc'])}")
        print(f"    D_e8: acc={_f(res['D_e8_acc'])}  auc={_f(res['D_e8_auc'])}")
        results.append(res)

    elapsed = time.time() - t0
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "test1_comparison.csv", index=False)

    print(f"\n  Datasets evaluated: {len(df)}  ({elapsed:.0f}s)")

    # ── Statistical summaries ──
    comparisons = [
        ("D_e1", "e1",  "D_e1 vs e1  (1-ens)"),
        ("D_e8", "e8",  "D_e8 vs e8  (8-ens)"),
        ("D_e8", "e1",  "D_e8 vs e1  (8-ens vs 1-ens baseline)"),
    ]
    for x, y, label in comparisons:
        for metric in ["acc", "auc"]:
            cx, cy = f"{x}_{metric}", f"{y}_{metric}"
            if cx not in df.columns or cy not in df.columns:
                continue
            v = df.dropna(subset=[cx, cy])
            if len(v) < 2:
                continue
            print(f"\n  {metric.upper()} — {label}  (n={len(v)}):")
            _wilcoxon_summary(v[cx].values, v[cy].values, x, y)

    # ── Scatter plots ──
    pairs = [("D_e1", "e1"), ("D_e8", "e8")]
    labels = {
        "e1":   "TabPFN 1-ens",
        "e8":   "TabPFN 8-ens",
        "D_e1": "Our model 1-ens",
        "D_e8": "Our model 8-ens",
    }
    fig, axes = plt.subplots(2, len(pairs), figsize=(7 * len(pairs), 12))
    if len(pairs) == 1:
        axes = axes.reshape(2, 1)

    for col, (x, y) in enumerate(pairs):
        for row, (metric, mtitle) in enumerate([("auc", "AUC"), ("acc", "Accuracy")]):
            ax = axes[row][col]
            cx, cy = f"{x}_{metric}", f"{y}_{metric}"
            if cx not in df.columns or cy not in df.columns:
                ax.set_visible(False)
                continue
            v = df.dropna(subset=[cx, cy])
            if len(v) < 2:
                ax.set_visible(False)
                continue
            x_vals = v[cy].values   # baseline on x-axis
            y_vals = v[cx].values   # our model on y-axis
            diffs  = y_vals - x_vals
            _, pval = sp_stats.wilcoxon(y_vals, x_vals, alternative="greater")
            lo = min(x_vals.min(), y_vals.min()) - 0.02
            hi = max(x_vals.max(), y_vals.max()) + 0.02
            ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, lw=1)
            pt_colors = ["#2ecc71" if d > 0.001 else "#e74c3c" if d < -0.001
                         else "#95a5a6" for d in diffs]
            ax.scatter(x_vals, y_vals, c=pt_colors, s=40, alpha=0.7,
                       edgecolors="white", linewidth=0.5)
            ax.set_xlabel(f"{labels[y]} — {mtitle}", fontsize=10)
            ax.set_ylabel(f"{labels[x]} — {mtitle}", fontsize=10)
            ax.set_title(f"{mtitle}: {x} vs {y}  (Δ={diffs.mean():+.4f}, p={pval:.3f})",
                         fontsize=11, fontweight="bold")
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
            wins = (diffs > 0.001).sum(); losses = (diffs < -0.001).sum()
            ax.text(0.05, 0.95,
                    f"{x} wins: {wins}\n{y} wins: {losses}",
                    transform=ax.transAxes, fontsize=9, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    fig.savefig(output_dir / "test1_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: test1_comparison.csv, test1_scatter.png")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — SOTA comparison: D_e8 vs HC2 benchmarks
# ─────────────────────────────────────────────────────────────────────────────

def _load_hc2(collection: str) -> dict:
    """Load HC2 accuracy table: {model_name → pd.Series(dataset → accuracy)}."""
    path = BENCHMARKS_HC2_DIR / f"{collection}_sota_average_over_30.csv"
    if not path.exists():
        print(f"  [HC2] Not found: {path}")
        return {}
    try:
        df = pd.read_csv(path, index_col=0)
        df.index.name = "dataset"
        return {col: df[col].dropna() for col in df.columns}
    except Exception as e:
        print(f"  [HC2] Failed to load: {e}")
        return {}


def run_test2(test1_df: pd.DataFrame, output_dir: Path, collection: str = "ucr"):
    """Compare D_e8 against HC2 SOTA on the intersection of PFN-filtered + HC2 datasets."""
    print("\n\n" + "=" * 80)
    print(f"TEST 2: D_e8 vs HC2 SOTA  (collection={collection})")
    print("=" * 80)

    our_name = "TabPFN_Overlap"
    benchmarks = _load_hc2(collection)
    if not benchmarks:
        print("  No HC2 benchmarks found. Run 01_real_data/download_hc2_benchmarks.py first.")
        return

    aeon_df  = pd.DataFrame(benchmarks)
    our_indexed = test1_df.set_index("name")

    # Intersection: PFN-filtered AND present in HC2 AND D_e8 evaluated successfully
    common = [d for d in sorted(set(aeon_df.index) & set(our_indexed.index))
              if "D_e8_acc" in our_indexed.columns
              and pd.notna(our_indexed.loc[d, "D_e8_acc"])]
    if not common:
        print("  No common datasets. Skipping.")
        return

    aeon_common = aeon_df.loc[common].copy()
    aeon_common[our_name] = our_indexed.loc[common, "D_e8_acc"].values
    aeon_common = aeon_common.dropna(axis=1, how="all")

    n_models = len(aeon_common.columns)
    n_ds     = len(common)
    rankings = aeon_common.rank(axis=1, ascending=False, method="min")

    all_stats = []
    for model_name in aeon_common.columns:
        model_ranks = rankings[model_name].dropna()
        model_vals  = aeon_common[model_name].dropna()
        rank1_strict = sum(
            1 for ds_name in model_ranks.index
            if model_ranks[ds_name] == 1 and (rankings.loc[ds_name] == 1).sum() == 1
        )
        all_stats.append({
            "model":         model_name,
            "mean_rank":     model_ranks.mean() if len(model_ranks) else float("inf"),
            "rank1":         int((model_ranks == 1).sum()),
            "rank1_strict":  rank1_strict,
            "mean_accuracy": float(model_vals.mean()) if len(model_vals) else None,
            "n_datasets":    int(model_vals.notna().sum()),
        })

    stats_df = pd.DataFrame(all_stats).sort_values("mean_rank")
    stats_df.to_csv(output_dir / "test2_sota_rankings.csv", index=False)

    print(f"\n  {'Model':<28} {'Mean Rank':>10} {'Rank 1':>7} {'Strict':>7} "
          f"{'Mean Acc':>10} {'N':>4}")
    print(f"  {'-'*72}")
    for _, row in stats_df.iterrows():
        marker = ">>>" if row["model"] == our_name else "   "
        v_str  = f"{row['mean_accuracy']:.4f}" if row["mean_accuracy"] is not None else "N/A"
        print(f"  {marker} {row['model']:<25} {row['mean_rank']:>10.2f} "
              f"{int(row['rank1']):>7} {int(row['rank1_strict']):>7} "
              f"{v_str:>10} {int(row['n_datasets']):>4}")

    our_row = stats_df[stats_df["model"] == our_name]
    if len(our_row):
        r = our_row.iloc[0]
        pos = int((stats_df["mean_rank"] < r["mean_rank"]).sum()) + 1
        print(f"\n  >>> {our_name}:")
        print(f"      Position: {pos}/{n_models}  (mean rank {r['mean_rank']:.2f})")
        print(f"      Rank 1:   {r['rank1']}/{n_ds}  (strict: {r['rank1_strict']}/{n_ds})")
        if r["mean_accuracy"] is not None:
            print(f"      Mean Accuracy: {r['mean_accuracy']:.4f}")

    our_ranks = rankings.get(our_name)
    if our_ranks is not None:
        print(f"\n  {'Dataset':<30} {'D_e8 Acc':>8} {'Rank':>8} "
              f"{'Best HC2':>9} {'Best Model':<20}")
        print(f"  {'-'*82}")
        for ds_name in common:
            our_val  = aeon_common.loc[ds_name, our_name]
            our_rank = int(our_ranks[ds_name])
            others   = aeon_common.loc[ds_name].drop(our_name).dropna()
            best_val = others.max() if len(others) else None
            best_mod = others.idxmax() if len(others) else "?"
            ov = f"{our_val:.4f}" if pd.notna(our_val) else "N/A"
            bv = f"{best_val:.4f}" if best_val is not None else "N/A"
            print(f"  {ds_name:<30} {ov:>8} {our_rank:>5}/{n_models}  "
                  f"{bv:>9} {best_mod:<20}")

    print(f"\n  Saved: test2_sota_rankings.csv")


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL: Training history diagnostics (--with-history)
# ─────────────────────────────────────────────────────────────────────────────

def run_training_history(log_dir: Path, output_dir: Path):
    """Plot synth-eval vs real-eval curves from worker_evaluator_v2 history.json."""
    history_path = log_dir / "history.json"
    if not history_path.exists():
        print(f"  No history.json found in {log_dir}. Skipping.")
        return

    with open(history_path) as f:
        history = json.load(f)
    df = pd.DataFrame(history).sort_values("step").reset_index(drop=True)
    print(f"  Loaded {len(df)} evaluation checkpoints")
    df.to_csv(output_dir / "training_history.csv", index=False)

    pairs = [
        ("synth_loss", "real_acc",  "SynthLoss vs Real Acc"),
        ("synth_loss", "real_auc",  "SynthLoss vs Real AUC"),
        ("synth_acc",  "real_acc",  "SynthAcc  vs Real Acc"),
        ("synth_acc",  "real_auc",  "SynthAcc  vs Real AUC"),
    ]
    for col_x, col_y, label in pairs:
        if col_x not in df or col_y not in df:
            continue
        rho, pval = sp_stats.spearmanr(df[col_x].dropna(), df[col_y].dropna())
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
        print(f"    {label}: ρ={rho:+.4f}  p={pval:.6f}  {sig}")

    steps = df["step"].values
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    c_sl, c_ra, c_rau, c_sa = "#e74c3c", "#2ecc71", "#9b59b6", "#3498db"

    ax1 = axes[0]
    ax1.plot(steps, df["synth_loss"], color=c_sl, alpha=0.3, lw=1)
    ax1.plot(steps, df["synth_loss"].rolling(5, center=True).mean(),
             color=c_sl, lw=2.5, label="SynthEval Loss")
    ax1.set_ylabel("SynthEval Loss", color=c_sl, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=c_sl)
    ax1.invert_yaxis()
    ax1b = ax1.twinx()
    ax1b.plot(steps, df["real_auc"], color=c_rau, alpha=0.3, lw=1)
    ax1b.plot(steps, df["real_auc"].rolling(5, center=True).mean(),
              color=c_rau, lw=2.5, label="Real AUC")
    ax1b.set_ylabel("Real AUC", color=c_rau, fontsize=12)
    ax1b.tick_params(axis="y", labelcolor=c_rau)
    lines  = ax1.get_legend_handles_labels()[0] + ax1b.get_legend_handles_labels()[0]
    labels_h = ax1.get_legend_handles_labels()[1] + ax1b.get_legend_handles_labels()[1]
    ax1.legend(lines, labels_h, loc="lower left", fontsize=10)
    ax1.set_title("SynthEval Loss ↓  ↔  Real AUC ↑", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.2)

    ax2 = axes[1]
    ax2.plot(steps, df["synth_acc"], color=c_sa, alpha=0.3, lw=1)
    ax2.plot(steps, df["synth_acc"].rolling(5, center=True).mean(),
             color=c_sa, lw=2.5, label="SynthEval Acc")
    ax2.set_ylabel("SynthEval Acc", color=c_sa, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=c_sa)
    ax2b = ax2.twinx()
    ax2b.plot(steps, df["real_acc"], color=c_ra, alpha=0.3, lw=1)
    ax2b.plot(steps, df["real_acc"].rolling(5, center=True).mean(),
              color=c_ra, lw=2.5, label="Real Acc")
    ax2b.plot(steps, df["real_auc"], color=c_rau, alpha=0.3, lw=1)
    ax2b.plot(steps, df["real_auc"].rolling(5, center=True).mean(),
              color=c_rau, lw=2.5, linestyle="--", label="Real AUC")
    ax2b.set_ylabel("Real Eval", fontsize=12)
    lines  = ax2.get_legend_handles_labels()[0] + ax2b.get_legend_handles_labels()[0]
    labels_h = ax2.get_legend_handles_labels()[1] + ax2b.get_legend_handles_labels()[1]
    ax2.legend(lines, labels_h, loc="lower left", fontsize=10)
    ax2.set_title("SynthEval Acc  ↔  Real Acc & AUC", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Training Step", fontsize=12)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(output_dir / "training_history.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: training_history.csv, training_history.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Final evaluation: D_e1/D_e8 vs e1/e8 baselines + SOTA"
    )
    parser.add_argument("checkpoint", type=str,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto (cuda→mps→cpu), cpu, cuda, mps")
    parser.add_argument("--dataset-names", type=str, default=None,
                        help="Comma-separated dataset names or path to .txt file")
    parser.add_argument("--collection", type=str, default="ucr",
                        choices=["ucr", "uea"],
                        help="HC2 collection for Test 2 (default: ucr)")
    parser.add_argument("--skip-test1", action="store_true",
                        help="Skip Test 1 and load from existing test1_comparison.csv")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output dir (default: 04_evaluation/results/<ckpt_stem>/)")
    parser.add_argument("--with-history", action="store_true",
                        help="Also plot training history (needs history.json near checkpoint)")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory with history.json (default: checkpoint parent)")
    args = parser.parse_args()

    # ── Checkpoint ──
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        checkpoint_path = HERE / args.checkpoint
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return

    # ── Device ──
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    # ── Output dir ──
    output_dir = Path(args.out_dir) if args.out_dir else HERE / "results" / checkpoint_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FINAL EVALUATION — Test 1 (D_e1/D_e8 vs e1/e8) + Test 2 (SOTA)")
    print("=" * 80)
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Device     : {device}")
    print(f"  Output     : {output_dir}")

    # ── Datasets ──
    print("\nLoading datasets...")
    if args.dataset_names:
        p = Path(args.dataset_names)
        raw = (p.read_text().strip() if p.exists() and p.suffix in (".txt", ".csv")
               else args.dataset_names)
        names = [n.strip() for n in raw.replace("\n", ",").split(",") if n.strip()]
        datasets = load_datasets(names=names)
        print(f"  {len(datasets)}/{len(names)} datasets loaded (custom list)")
    else:
        datasets = load_datasets()
        print(f"  {len(datasets)} datasets loaded (all PFN-filtered)")

    # ── Load checkpoint ──
    print("\nLoading checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    version = ckpt.get("version", "unknown")

    if version in ("overlap_v1", "overlap_v2"):
        ft_model, _, _, _ = build_overlap_model(device=device)
        use_overlap = True
    else:
        ft_model, _, _ = build_temporal_tabpfn_fpg8(device=device)
        use_overlap = False

    ft_model.load_state_dict(ckpt["model_state_dict"])
    ft_model.eval()
    print(f"  Version: {version}  Step: {ckpt.get('step', '?')}")

    # ── Precomputed baselines ──
    print("\nLooking for pre-computed baselines (e1/e8)...")
    precomputed = load_precomputed_baselines()
    if precomputed:
        n_e1 = sum(1 for v in precomputed.values() if "e1_acc" in v)
        n_e8 = sum(1 for v in precomputed.values() if "e8_acc" in v)
        print(f"  Found: {n_e1} datasets with e1, {n_e8} with e8  (from CSV)")
    else:
        print(f"  Not found in {BENCHMARK_RESULTS_DIR}")
        print(f"  Run benchmark_tabpfn.sbatch to pre-compute baselines.")

    # ── TEST 1 ──
    test1_csv = output_dir / "test1_comparison.csv"
    if args.skip_test1 and test1_csv.exists():
        print(f"\n  Skipping Test 1 — loading from {test1_csv}")
        test1_df = pd.read_csv(test1_csv)
    else:
        test1_df = run_test1(
            datasets, ft_model, device, output_dir,
            use_overlap=use_overlap, precomputed=precomputed,
        )

    # ── TEST 2 ──
    run_test2(test1_df, output_dir, collection=args.collection)

    # ── OPTIONAL: training history ──
    if args.with_history:
        print("\n\n" + "=" * 80)
        print("TRAINING HISTORY")
        print("=" * 80)
        log_dir = Path(args.log_dir) if args.log_dir else checkpoint_path.parent
        run_training_history(log_dir, output_dir)

    print("\n\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)
    print(f"  Results: {output_dir}")


if __name__ == "__main__":
    main()
