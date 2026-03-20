#!/usr/bin/env python3
"""
Preprocessing Evaluation — step-repeat augmentation ablation.

For PFN-filtered datasets where  m*T < 1000  AND  T < 96:
  Apply step-repeat preprocessing: repeat each time point so T doubles.
  Continue while both conditions hold, up to 3 applications (max 8× original T).

  Example: T=20, m=3  → m*T=60 < 1000, T=20 < 96
    round 1: T=40,  m*T=120  (still qualifies)
    round 2: T=80,  m*T=240  (still qualifies)
    round 3: T=160, m*T=480  (stop: reached max 3 rounds)

Comparison:
  D_raw  — our model (1 ensemble iter), original data
  D_prep — our model (1 ensemble iter), step-repeated data

Output:
  preprocessing_comparison.csv  — per-dataset results
    columns: name, m, T_orig, T_after, n_repeats, D_raw_acc, D_raw_auc,
             D_prep_acc, D_prep_auc

Usage:
  python 04_evaluation/preprocessing_eval.py 03_finetuning/checkpoints/phase2/best.pt
  python 04_evaluation/preprocessing_eval.py <ckpt.pt> --device cuda
  python 04_evaluation/preprocessing_eval.py <ckpt.pt> --out-dir my_results/
"""

import argparse
import gc
import logging
import sys
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
# Constants
# ─────────────────────────────────────────────────────────────────────────────

REAL_DATA_ROOT  = ROOT / "01_real_data"
SUBSAMPLE_N     = 1_000
SUBSAMPLE_SEED  = 0
ENSEMBLE_SEED   = 42

# Preprocessing conditions
PREP_MT_THRESH  = 1_000   # apply while m*T < this
PREP_T_THRESH   = 96      # apply while T < this
PREP_MAX_ROUNDS = 3       # max doublings (up to 8×)


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
# Step-repeat preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def _step_repeat(X_flat: np.ndarray, m: int, T: int) -> np.ndarray:
    """Double T by repeating each time point once.

    Input:  (n_samples, m*T)  flat layout [ch0_t0, ch0_t1, ..., ch1_t0, ...]
    Output: (n_samples, m*2T) same layout with each time point repeated
    """
    n = X_flat.shape[0]
    X_3d = X_flat.reshape(n, m, T)                  # (n, m, T)
    X_rep = np.repeat(X_3d, 2, axis=2)              # (n, m, 2T)
    return X_rep.reshape(n, m * 2 * T)


def apply_step_repeat(
    X_tr: np.ndarray, X_te: np.ndarray, m: int, T: int
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Apply step-repeat preprocessing up to PREP_MAX_ROUNDS times.

    Applies while: m * T < PREP_MT_THRESH  AND  T < PREP_T_THRESH.
    Returns (X_tr_out, X_te_out, T_out, n_rounds_applied).
    """
    n_rounds = 0
    T_cur = T
    X_tr_cur, X_te_cur = X_tr, X_te

    for _ in range(PREP_MAX_ROUNDS):
        if m * T_cur >= PREP_MT_THRESH or T_cur >= PREP_T_THRESH:
            break
        X_tr_cur = _step_repeat(X_tr_cur, m, T_cur)
        X_te_cur = _step_repeat(X_te_cur, m, T_cur)
        T_cur   *= 2
        n_rounds += 1

    return X_tr_cur, X_te_cur, T_cur, n_rounds


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_datasets() -> list:
    """Load all PFN-eligible datasets, returning 3D X arrays (not yet flattened).

    Returns list of dicts with keys:
      name, collection, X_train_3d, X_test_3d, y_train, y_test,
      n_classes, m, T
    """
    from sklearn.preprocessing import LabelEncoder

    summary  = pd.read_csv(REAL_DATA_ROOT / "datasets_summary.csv")
    eligible = summary[summary["passes_pfn_filters"]].copy()

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
            T = X_te_3d.shape[2]
            n_classes = int(row["n_classes"])

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

            # Flatten and replace inf with nan
            X_tr_flat = X_tr_3d.reshape(X_tr_3d.shape[0], -1)
            X_te_flat = X_te_3d.reshape(X_te_3d.shape[0], -1)
            np.putmask(X_tr_flat, ~np.isfinite(X_tr_flat), np.nan)
            np.putmask(X_te_flat, ~np.isfinite(X_te_flat), np.nan)

            valid.append({
                "name": name, "collection": row["collection"],
                "X_train": X_tr_flat, "X_test": X_te_flat,
                "y_train": y_tr, "y_test": y_te,
                "n_classes": n_classes, "m": m, "T": T,
            })
        except Exception as e:
            skipped.append(name)
            print(f"  [load] {name}: {str(e)[:60]}")

    if skipped:
        print(f"  [load] Skipped {len(skipped)} datasets (no NPZ or error)")
    return valid


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, proba, n_classes):
    preds = proba.argmax(axis=1)
    acc = float(accuracy_score(y_true, preds))
    try:
        auc = (float(roc_auc_score(y_true, proba[:, 1]))
               if n_classes == 2
               else float(roc_auc_score(y_true, proba, multi_class="ovr")))
    except Exception:
        auc = None
    return acc, auc


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
        ft_model.to(device)   # move model back to original device
    return proba


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_preprocessing_eval(datasets, ft_model, device, output_dir, use_overlap: bool = True):
    """Evaluate D_e1 with and without step-repeat preprocessing.

    For each dataset:
      - Compute how many step-repeat rounds apply (0 = not eligible).
      - Run D_e1 on the original data (D_raw).
      - If any rounds were applied, also run D_e1 on preprocessed data (D_prep).
      - If no rounds, D_prep == D_raw (preprocessing is identity).
    """
    print("\n" + "=" * 80)
    print("PREPROCESSING EVALUATION — step-repeat ablation")
    print(f"  Condition: m*T < {PREP_MT_THRESH} AND T < {PREP_T_THRESH}  (max {PREP_MAX_ROUNDS} rounds = {2**PREP_MAX_ROUNDS}×)")
    print("=" * 80)

    results = []
    n_prep = 0

    for i, ds in enumerate(datasets):
        name, m, T, nc = ds["name"], ds["m"], ds["T"], ds["n_classes"]

        # ── Determine preprocessing ──
        X_tr_prep, X_te_prep, T_prep, n_rounds = apply_step_repeat(
            ds["X_train"], ds["X_test"], m, T
        )
        if n_rounds > 0:
            n_prep += 1

        print(f"  [{i+1}/{len(datasets)}] {name}  "
              f"m={m}  T={T}→{T_prep}  rounds={n_rounds}", flush=True)

        res = {
            "name": name, "m": m, "T_orig": T, "T_after": T_prep,
            "n_repeats": n_rounds,
        }

        # ── D_raw (no preprocessing, original T) ──
        _free_memory()
        try:
            proba = _eval_with_fallback(
                ft_model, ds["X_train"], ds["y_train"], ds["X_test"],
                nc, m, T, device, n_iters=1, seed=ENSEMBLE_SEED,
                use_overlap=use_overlap, label="D_raw",
            )
            res["D_raw_acc"], res["D_raw_auc"] = (
                compute_metrics(ds["y_test"], proba, nc) if proba is not None
                else (None, None)
            )
            del proba
        except Exception as e:
            print(f"    [D_raw] FAILED: {type(e).__name__}: {str(e)[:80]}")
            res["D_raw_acc"], res["D_raw_auc"] = None, None
        _free_memory()

        # ── D_prep (with step-repeat, possibly different T) ──
        if n_rounds > 0:
            try:
                proba = _eval_with_fallback(
                    ft_model, X_tr_prep, ds["y_train"], X_te_prep,
                    nc, m, T_prep, device, n_iters=1, seed=ENSEMBLE_SEED,
                    use_overlap=use_overlap, label="D_prep",
                )
                res["D_prep_acc"], res["D_prep_auc"] = (
                    compute_metrics(ds["y_test"], proba, nc) if proba is not None
                    else (None, None)
                )
                del proba
            except Exception as e:
                print(f"    [D_prep] FAILED: {type(e).__name__}: {str(e)[:80]}")
                res["D_prep_acc"], res["D_prep_auc"] = None, None
            _free_memory()
        else:
            # Preprocessing didn't apply — copy raw results
            res["D_prep_acc"] = res["D_raw_acc"]
            res["D_prep_auc"] = res["D_raw_auc"]

        _f = lambda v: f"{v:.4f}" if v is not None else "FAIL"
        if n_rounds > 0:
            raw_acc, prep_acc = res["D_raw_acc"], res["D_prep_acc"]
            raw_auc, prep_auc = res["D_raw_auc"], res["D_prep_auc"]
            delta_acc = f"{prep_acc - raw_acc:+.4f}" if raw_acc is not None and prep_acc is not None else "N/A"
            delta_auc = f"{prep_auc - raw_auc:+.4f}" if raw_auc is not None and prep_auc is not None else "N/A"
            print(f"    raw:  acc={_f(raw_acc)}  auc={_f(raw_auc)}")
            print(f"    prep: acc={_f(prep_acc)}  auc={_f(prep_auc)}  "
                  f"Δacc={delta_acc}  Δauc={delta_auc}")
        else:
            print(f"    (no preprocessing)  acc={_f(res['D_raw_acc'])}  auc={_f(res['D_raw_auc'])}")

        results.append(res)

    df = pd.DataFrame(results)
    out_csv = output_dir / "preprocessing_comparison.csv"
    df.to_csv(out_csv, index=False)

    # ── Summary stats ──
    df_prep = df[df["n_repeats"] > 0].copy()
    print(f"\n  Total datasets: {len(df)}")
    print(f"  Preprocessing applied: {len(df_prep)} datasets")

    # Overall means across ALL datasets (raw uses original, prep uses preprocessed where applied)
    print(f"\n  ── Overall means (all {len(df)} datasets) ──")
    for metric in ["acc", "auc"]:
        v_all = df.dropna(subset=[f"D_raw_{metric}", f"D_prep_{metric}"])
        if len(v_all) == 0:
            continue
        raw_mean  = v_all[f"D_raw_{metric}"].mean()
        prep_mean = v_all[f"D_prep_{metric}"].mean()
        print(f"    {metric.upper()}:  raw={raw_mean:.4f}  prep={prep_mean:.4f}  "
              f"Δ={prep_mean - raw_mean:+.4f}  (n={len(v_all)})")

    if len(df_prep) >= 2:
        print(f"\n  ── On preprocessed-only datasets ({len(df_prep)}) ──")
        for metric in ["acc", "auc"]:
            v = df_prep.dropna(subset=[f"D_raw_{metric}", f"D_prep_{metric}"])
            if len(v) < 2:
                continue
            raw_vals  = v[f"D_raw_{metric}"].values
            prep_vals = v[f"D_prep_{metric}"].values
            diffs = prep_vals - raw_vals
            stat, pval = sp_stats.wilcoxon(prep_vals, raw_vals, alternative="greater")
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
            wins   = (diffs > 0.001).sum()
            losses = (diffs < -0.001).sum()
            ties   = len(diffs) - wins - losses
            print(f"\n  {metric.upper()} — D_prep vs D_raw  (n={len(v)}):")
            print(f"    Mean:  raw={raw_vals.mean():.4f}  prep={prep_vals.mean():.4f}  "
                  f"Δ={diffs.mean():+.4f}   Median Δ: {np.median(diffs):+.4f}")
            print(f"    Range Δ: {diffs.min():+.4f} … {diffs.max():+.4f}")
            print(f"    Win/Tie/Loss: {wins}/{ties}/{losses}")
            print(f"    Wilcoxon (prep > raw): stat={stat:.1f}, p={pval:.6f}  {sig}")

        # ── Scatter plot ──
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, (metric, mtitle) in zip(axes, [("acc", "Accuracy"), ("auc", "AUC")]):
            v = df_prep.dropna(subset=[f"D_raw_{metric}", f"D_prep_{metric}"])
            if len(v) < 2:
                ax.set_visible(False)
                continue
            x_vals = v[f"D_raw_{metric}"].values
            y_vals = v[f"D_prep_{metric}"].values
            diffs  = y_vals - x_vals
            _, pval = sp_stats.wilcoxon(y_vals, x_vals, alternative="greater")
            lo = min(x_vals.min(), y_vals.min()) - 0.02
            hi = max(x_vals.max(), y_vals.max()) + 0.02
            ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, lw=1)
            pt_colors = ["#2ecc71" if d > 0.001 else "#e74c3c" if d < -0.001
                         else "#95a5a6" for d in diffs]
            ax.scatter(x_vals, y_vals, c=pt_colors, s=50, alpha=0.7,
                       edgecolors="white", linewidth=0.5)
            # Annotate outliers
            for j in range(len(v)):
                if abs(diffs[j]) > 0.05:
                    ax.annotate(v.iloc[j]["name"], (x_vals[j], y_vals[j]),
                                fontsize=6, alpha=0.7,
                                xytext=(4, 4), textcoords="offset points")
            ax.set_xlabel(f"D_raw {mtitle}  (no preprocessing)", fontsize=11)
            ax.set_ylabel(f"D_prep {mtitle}  (step-repeat)", fontsize=11)
            ax.set_title(f"{mtitle}: prep vs raw  (Δ={diffs.mean():+.4f}, p={pval:.3f})",
                         fontsize=12, fontweight="bold")
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
            wins = (diffs > 0.001).sum(); losses = (diffs < -0.001).sum()
            ax.text(0.05, 0.95,
                    f"prep wins: {wins}\nraw  wins: {losses}",
                    transform=ax.transAxes, fontsize=9, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        plt.tight_layout()
        fig.savefig(output_dir / "preprocessing_scatter.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Saved: preprocessing_scatter.png")

    print(f"  Saved: preprocessing_comparison.csv")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ablation: step-repeat preprocessing vs raw on D_e1"
    )
    parser.add_argument("checkpoint", type=str,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto (cuda→mps→cpu), cpu, cuda, mps")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output dir (default: 04_evaluation/results/<ckpt_stem>/prep/)")
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
    if args.out_dir:
        output_dir = Path(args.out_dir)
    else:
        output_dir = HERE / "results" / checkpoint_path.stem / "prep"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PREPROCESSING EVALUATION")
    print("=" * 80)
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Device     : {device}")
    print(f"  Output     : {output_dir}")

    # ── Datasets ──
    print("\nLoading datasets...")
    datasets = load_datasets()
    print(f"  {len(datasets)} datasets loaded (all PFN-filtered)")

    qualifying = sum(1 for ds in datasets
                     if ds["m"] * ds["T"] < PREP_MT_THRESH and ds["T"] < PREP_T_THRESH)
    print(f"  {qualifying} datasets qualify for preprocessing "
          f"(m*T < {PREP_MT_THRESH} AND T < {PREP_T_THRESH})")

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

    # ── Run ──
    run_preprocessing_eval(datasets, ft_model, device, output_dir, use_overlap=use_overlap)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"  Results: {output_dir}")


if __name__ == "__main__":
    main()
