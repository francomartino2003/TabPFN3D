#!/usr/bin/env python3
"""
Analyze distributions of PFN-eligible datasets (raw, before pooling).

Reports for each dimension: min, p25, median, p75, max, mean.
Also inspects NPZ files to measure:
  - Actual NaN fractions (missing values + variable-length padding)
  - Fraction of padded (shorter-than-max) series per variable-length dataset
  - Min samples per class (train+test combined)

Saves: 01_real_data/analysis/distribution_stats.csv
       01_real_data/analysis/distributions.png  (unless --no-plots)

Usage:
  python 01_real_data/analyze_distributions.py
  python 01_real_data/analyze_distributions.py --no-plots
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE        = Path(__file__).parent
SUMMARY_CSV = HERE / "datasets_summary.csv"
DATA_DIR    = HERE / "data"
OUT_DIR     = HERE / "analysis"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_shape(s: str) -> tuple[int, int, int]:
    parts = re.split(r"[×x*]", str(s).strip())
    return int(parts[0]), int(parts[1]), int(parts[2])


def stats_row(label: str, arr) -> dict:
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {"metric": label, "n": 0}
    p = np.percentile(a, [0, 25, 50, 75, 100])
    return {
        "metric": label, "n": len(a),
        "min": p[0], "p25": p[1], "median": p[2], "p75": p[3], "max": p[4],
        "mean": float(a.mean()),
    }


def print_stats(label: str, arr):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        print(f"  {label}: (no data)")
        return
    p = np.percentile(a, [0, 25, 50, 75, 100])
    print(f"  {label:30s}  n={len(a):3d}  "
          f"min={p[0]:.1f}  p25={p[1]:.1f}  median={p[2]:.1f}  "
          f"p75={p[3]:.1f}  max={p[4]:.1f}  mean={a.mean():.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# NPZ inspection
# ─────────────────────────────────────────────────────────────────────────────

def _effective_length(series: np.ndarray) -> int:
    """Length of the non-NaN prefix (for variable-length detection via right-NaN-padding)."""
    nan_mask = np.isnan(series)
    if not nan_mask.any():
        return len(series)
    # Find last non-NaN position
    non_nan = np.where(~nan_mask)[0]
    return int(non_nan[-1] + 1) if len(non_nan) > 0 else 0


def inspect_npz(name: str, collection: str):
    """
    Load NPZ and return:
      nan_frac        : fraction of all X values that are NaN
      vl_frac         : fraction of series shorter than T (variable-length)
                        detected via right-NaN-padding OR right-zero-padding
                        (aeon sometimes zero-pads instead of NaN-pads)
      min_class_count : minimum samples per class (train+test)
    Returns None if NPZ not found.
    """
    data_dir = DATA_DIR / collection.lower()
    tr_path  = data_dir / f"{name}_train.npz"
    te_path  = data_dir / f"{name}_test.npz"
    if not tr_path.exists() or not te_path.exists():
        return None

    try:
        tr = np.load(tr_path, allow_pickle=True)
        te = np.load(te_path, allow_pickle=True)
        X_tr = np.asarray(tr["X"], dtype=np.float32)   # (n_tr, m, T)
        X_te = np.asarray(te["X"], dtype=np.float32)
        y_tr = tr["y"].ravel()
        y_te = te["y"].ravel()
    except Exception:
        return None

    # ── NaN fraction (all positions) ──────────────────────────────────────────
    total = X_tr.size + X_te.size
    nan_count = int(np.isnan(X_tr).sum()) + int(np.isnan(X_te).sum())
    nan_frac = nan_count / total if total > 0 else 0.0

    # ── Variable-length: fraction of series with a shorter effective length ──
    # Check channel 0 for right-NaN-padding (true variable-length from download.py)
    n_tr, m, T = X_tr.shape
    n_te = X_te.shape[0]

    shorter_count = 0
    total_series  = n_tr + n_te

    for X in (X_tr, X_te):
        for i in range(X.shape[0]):
            ch0 = X[i, 0]
            eff = _effective_length(ch0)
            if eff < T:
                shorter_count += 1

    vl_frac = shorter_count / total_series if total_series > 0 else 0.0

    # ── Min samples per class ─────────────────────────────────────────────────
    all_y = np.concatenate([y_tr, y_te])
    _, counts = np.unique(all_y, return_counts=True)
    min_class = int(counts.min())

    return {
        "nan_frac": nan_frac,
        "vl_frac":  vl_frac,
        "min_class": min_class,
        "T": T,
        "shorter_count": shorter_count,
        "total_series": total_series,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(SUMMARY_CSV)
    passes = df["passes_pfn_filters"].map(lambda x: str(x).lower() == "true")
    df = df[passes].copy().reset_index(drop=True)

    n_ucr = (df["collection"] == "UCR").sum()
    n_uea = (df["collection"] == "UEA").sum()
    print(f"\n{'='*72}")
    print(f"  PFN-eligible: {len(df)} datasets  (UCR: {n_ucr}  UEA: {n_uea})")
    print(f"{'='*72}")

    # ── Parse shapes ──────────────────────────────────────────────────────────
    trains = df["train_shape"].apply(parse_shape)
    tests  = df["test_shape"].apply(parse_shape)

    n_train = np.array([s[0] for s in trains])
    n_test  = np.array([s[0] for s in tests])
    m_arr   = np.array([s[1] for s in trains])
    T_arr   = np.array([s[2] for s in trains])
    n_total = n_train + n_test
    n_cls   = df["n_classes"].values.astype(int)
    mT_arr  = m_arr * T_arr

    is_univ  = m_arr == 1
    is_var   = df["is_variable_length"].map(lambda x: str(x).lower() == "true").values
    has_miss = df["has_missings"].map(lambda x: str(x).lower() == "true").values

    print("\n── Distributions (min | p25 | median | p75 | max | mean) ──────────")
    print_stats("n_total",      n_total)
    print_stats("n_train",      n_train)
    print_stats("n_test",       n_test)
    print_stats("m (channels)", m_arr)
    print_stats("T (timesteps)", T_arr)
    print_stats("m×T",          mT_arr)
    print_stats("n_classes",    n_cls)

    print("\n── Categorical flags ──────────────────────────────────────────────")
    print(f"  Univariate  (m=1):       {is_univ.sum():3d} / {len(df)}  ({100*is_univ.mean():.1f}%)")
    print(f"  Multivariate (m>1):      {(~is_univ).sum():3d} / {len(df)}  ({100*(~is_univ).mean():.1f}%)")
    print(f"  Variable-length flag:    {is_var.sum():3d} / {len(df)}  ({100*is_var.mean():.1f}%)")
    print(f"  Has missing flag:        {has_miss.sum():3d} / {len(df)}  ({100*has_miss.mean():.1f}%)")

    # ── NPZ inspection ────────────────────────────────────────────────────────
    print(f"\n── Inspecting NPZ files ({len(df)} datasets) …")
    nan_fracs_all    = []
    nan_fracs_flagged = []   # only datasets flagged as var-len or has_missing
    vl_fracs         = []   # only datasets flagged as var-len
    min_class_counts = []
    npz_failed       = []

    for _, row in df.iterrows():
        res = inspect_npz(row["dataset"], row["collection"])
        if res is None:
            npz_failed.append(row["dataset"])
            continue

        nan_fracs_all.append(res["nan_frac"])
        if row["is_variable_length"] or row["has_missings"]:
            nan_fracs_flagged.append(res["nan_frac"])
        if row["is_variable_length"]:
            vl_fracs.append(res["vl_frac"])
        min_class_counts.append(res["min_class"])

        if res["nan_frac"] > 0 or res["vl_frac"] > 0:
            print(f"  {row['dataset']:35s}  NaN={res['nan_frac']:.4f}  "
                  f"vl_frac={res['vl_frac']:.4f}  "
                  f"({res['shorter_count']}/{res['total_series']} shorter than T={res['T']})")

    if npz_failed:
        print(f"\n  NPZ not found for {len(npz_failed)}: "
              f"{npz_failed[:5]}{'...' if len(npz_failed)>5 else ''}")

    print(f"\n── NaN fractions (all {len(nan_fracs_all)} datasets) ─────────────")
    print_stats("nan_frac (all)",      nan_fracs_all)
    nonzero = [f for f in nan_fracs_all if f > 0]
    print(f"  Datasets with any NaN: {len(nonzero)} / {len(nan_fracs_all)}")
    if nan_fracs_flagged:
        print_stats("nan_frac (flagged)",  nan_fracs_flagged)

    if vl_fracs:
        print(f"\n── Variable-length: fraction of shorter series ─────────────────")
        print_stats("vl_frac (var-len datasets)", vl_fracs)
        nonzero_vl = [f for f in vl_fracs if f > 0]
        print(f"  Datasets with actual shorter series: {len(nonzero_vl)} / {len(vl_fracs)}")

    print(f"\n── Min samples per class (train+test) ──────────────────────────")
    print_stats("min_class_count", min_class_counts)
    for t in [5, 6, 8, 10, 15, 20, 30]:
        n_below = sum(c <= t for c in min_class_counts)
        print(f"  ≤{t:2d} samples/class:  {n_below:3d} datasets ({100*n_below/len(min_class_counts):.1f}%)")

    # ── Key ranges summary ────────────────────────────────────────────────────
    print(f"\n── Full ranges (relevant for hyperparameter.py) ────────────────")
    for label, arr in [
        ("n_total",      n_total),
        ("n_train",      n_train),
        ("m",            m_arr),
        ("T",            T_arr),
        ("m×T",          mT_arr),
        ("n_classes",    n_cls),
        ("min_class",    min_class_counts),
    ]:
        a = np.asarray(arr, dtype=float)
        print(f"  {label:12s}  [{a.min():.0f}, {a.max():.0f}]  "
              f"(median={np.median(a):.0f}  mean={a.mean():.0f})")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(exist_ok=True)
    rows = [
        stats_row("n_total",      n_total),
        stats_row("n_train",      n_train),
        stats_row("n_test",       n_test),
        stats_row("m_channels",   m_arr),
        stats_row("T_timesteps",  T_arr),
        stats_row("mxT",          mT_arr),
        stats_row("n_classes",    n_cls),
        stats_row("min_class",    min_class_counts),
        stats_row("nan_frac_all", nan_fracs_all),
    ]
    if nan_fracs_flagged:
        rows.append(stats_row("nan_frac_flagged", nan_fracs_flagged))
    if vl_fracs:
        rows.append(stats_row("vl_frac", vl_fracs))
    pd.DataFrame(rows).to_csv(OUT_DIR / "distribution_stats.csv", index=False)
    print(f"\n  Saved: {OUT_DIR}/distribution_stats.csv")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if args.no_plots:
        print()
        return

    fig, axes = plt.subplots(2, 4, figsize=(24, 11))
    fig.suptitle(f"Real dataset distributions — {len(df)} PFN-eligible datasets  "
                 f"(UCR: {n_ucr}  UEA: {n_uea})",
                 fontsize=14, fontweight="bold")

    def _hist(ax, data, title, color, log_x=False):
        a = np.asarray(data, dtype=float)
        a = a[np.isfinite(a)]
        ax.hist(a, bins=30, color=color, alpha=0.85, edgecolor="white", lw=0.5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("count", fontsize=9)
        if log_x:
            ax.set_xscale("log")
        med = np.median(a)
        ax.axvline(med, color="black", lw=1.5, ls="--", label=f"med={med:.0f}")
        ax.axvline(a.min(), color="#e74c3c", lw=1, ls=":", label=f"min={a.min():.0f}")
        ax.axvline(a.max(), color="#2ecc71", lw=1, ls=":", label=f"max={a.max():.0f}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    _hist(axes[0][0], n_total,   "n_total",       "#3498db", log_x=True)
    _hist(axes[0][1], m_arr,     "m (channels)",  "#9b59b6")
    _hist(axes[0][2], T_arr,     "T (timesteps)", "#e67e22", log_x=True)
    _hist(axes[0][3], mT_arr,    "m × T",         "#e74c3c", log_x=True)
    _hist(axes[1][0], n_cls,     "n_classes",     "#2ecc71")
    _hist(axes[1][1], min_class_counts, "min samples/class", "#1abc9c", log_x=True)

    # UCR vs UEA breakdown
    ax_bar = axes[1][2]
    collections = ["UCR", "UEA"]
    c_counts    = [(df["collection"] == c).sum() for c in collections]
    bars = ax_bar.bar(collections, c_counts, color=["#3498db", "#e67e22"],
                      alpha=0.85, edgecolor="white")
    for b, cnt in zip(bars, c_counts):
        ax_bar.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                    str(cnt), ha="center", fontsize=11, fontweight="bold")
    ax_bar.set_title("UCR vs UEA", fontsize=11, fontweight="bold")
    ax_bar.set_ylabel("count", fontsize=9)
    ax_bar.grid(True, alpha=0.3, axis="y")

    # Flags summary
    ax_flags = axes[1][3]
    flag_labels = ["Univariate", "Multivariate", "Variable-\nlength", "Has\nmissings"]
    flag_counts = [is_univ.sum(), (~is_univ).sum(), is_var.sum(), has_miss.sum()]
    flag_colors = ["#3498db", "#e67e22", "#9b59b6", "#e74c3c"]
    bars2 = ax_flags.bar(flag_labels, flag_counts, color=flag_colors,
                          alpha=0.85, edgecolor="white")
    for b, cnt in zip(bars2, flag_counts):
        ax_flags.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                      str(cnt), ha="center", fontsize=10, fontweight="bold")
    ax_flags.set_title("Dataset flags", fontsize=11, fontweight="bold")
    ax_flags.set_ylabel("count", fontsize=9)
    ax_flags.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "distributions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR}/distributions.png\n")


if __name__ == "__main__":
    main()
