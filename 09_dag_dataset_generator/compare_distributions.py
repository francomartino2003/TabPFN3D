#!/usr/bin/env python3
"""
Compare distributions of real AEON datasets vs synthetic datasets from dag_dataset_generator.
Generates:
  1. Console table with side-by-side statistics
  2. PNG with histogram comparisons for each metric
  3. TabPFN benchmark comparison (accuracy, AUC) on both real and synthetic
"""

import json
import argparse
import pickle
import numpy as np
import sys
import time
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent))
from dag_dataset_generator import generate_dataset, GeneratedDataset, HYPER_CONFIG


# ============================================================
# CONSTANTS / CONSTRAINTS
# ============================================================

MAX_SAMPLES   = HYPER_CONFIG["max_samples"]          # 1000
MAX_FEATURES  = HYPER_CONFIG["max_features"]         # 12
MAX_TIMESTEPS = HYPER_CONFIG["max_timesteps"]        # 1000
MAX_FEAT_T    = HYPER_CONFIG["max_feat_times_t"]     # 500
MAX_CLASSES   = 10

AEON_STATS_PATH = (
    Path(__file__).parent.parent
    / "01_real_data" / "AEON" / "data" / "classification_stats.json"
)
AEON_PKL_PATH = (
    Path(__file__).parent.parent
    / "01_real_data" / "AEON" / "data" / "classification_datasets.pkl"
)


# ============================================================
# LOAD REAL DATA
# ============================================================

def load_real_stats():
    """Load AEON classification_stats.json and filter by constraints."""
    with open(AEON_STATS_PATH) as f:
        all_stats = json.load(f)

    filtered = []
    for ds in all_stats:
        n_samples   = ds["n_samples"]
        n_features  = ds.get("n_dimensions", 1)
        timesteps   = ds["length"]
        n_classes   = ds["n_classes"]
        feat_x_t    = n_features * timesteps
        balance     = ds.get("class_balance", 1.0)
        missing     = ds.get("missing_pct", 0.0)

        if (n_samples <= MAX_SAMPLES
                and n_features <= MAX_FEATURES
                and timesteps <= MAX_TIMESTEPS
                and feat_x_t <= MAX_FEAT_T
                and n_classes <= MAX_CLASSES):

            avg_per_class = n_samples / n_classes
            min_spc = int(balance * avg_per_class)

            filtered.append({
                "name": ds["name"],
                "n_samples": n_samples,
                "n_features": n_features,
                "timesteps": timesteps,
                "feat_x_t": feat_x_t,
                "n_classes": n_classes,
                "class_balance": balance,
                "missing_pct": missing,
                "min_samples_per_class": min_spc,
                "train_size": ds.get("train_size", 0),
                "test_size": ds.get("test_size", 0),
            })
    return filtered


_AEON_CACHE = None

def load_aeon_pkl():
    """Load AEON datasets from pkl (cached)."""
    global _AEON_CACHE
    if _AEON_CACHE is not None:
        return _AEON_CACHE
    # Need the TimeSeriesDataset class
    sys.path.insert(0, str(Path(__file__).parent.parent / "01_real_data"))
    with open(AEON_PKL_PATH, "rb") as f:
        datasets = pickle.load(f)
    _AEON_CACHE = {ds.name: ds for ds in datasets}
    return _AEON_CACHE


def load_aeon_dataset(name):
    """Load a single AEON dataset by name. Returns (X_train, y_train, X_test, y_test) or Nones."""
    try:
        cache = load_aeon_pkl()
        if name not in cache:
            return None, None, None, None
        ds = cache[name]
        return ds.X_train, ds.y_train, ds.X_test, ds.y_test
    except Exception:
        return None, None, None, None


# ============================================================
# GENERATE SYNTHETIC DATA
# ============================================================

def generate_synthetic_datasets(n_datasets=100, seed=42):
    """Generate synthetic datasets. Returns list of GeneratedDataset."""
    rng = np.random.default_rng(seed)
    datasets = []
    for i in range(n_datasets):
        try:
            ds = generate_dataset(rng)
            datasets.append(ds)
        except Exception as e:
            print(f"  Dataset {i}: FAILED ({e})")
    return datasets


def compute_stats(datasets_info):
    """datasets_info: list of dicts with keys n_samples, n_features, timesteps, etc."""
    return datasets_info  # already structured


# ============================================================
# TABPFN BENCHMARK
# ============================================================

def flatten_3d(X):
    """(n, channels, length) -> (n, channels*length)"""
    if X.ndim == 3:
        return X.reshape(X.shape[0], -1)
    return X


def run_tabpfn(tabpfn, X_train, y_train, X_test, y_test, n_classes):
    """Run TabPFN, return (accuracy, auc) or (None, None)."""
    from sklearn.metrics import accuracy_score, roc_auc_score

    X_tr = flatten_3d(X_train)
    X_te = flatten_3d(X_test)

    # Ensure labels are int
    if isinstance(y_train[0], str):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

    # Skip if flattened features exceed TabPFN limit
    if X_tr.shape[1] > 500:
        return None, None

    try:
        tabpfn.fit(X_tr, y_train)
        y_pred = tabpfn.predict(X_te)
        y_proba = tabpfn.predict_proba(X_te)

        acc = accuracy_score(y_test, y_pred)

        if n_classes == 2:
            auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            try:
                auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
            except Exception:
                auc = None
        return acc, auc
    except Exception as e:
        print(f"    TabPFN error: {e}")
        return None, None


def _data_diagnostics(X, label=""):
    """Print min/max/mean/std/inf/nan diagnostics for a data array."""
    flat = X.ravel()
    n_inf = np.isinf(flat).sum()
    n_nan = np.isnan(flat).sum()
    finite = flat[np.isfinite(flat)]
    if len(finite) == 0:
        return f"{label} ALL inf/nan ({n_inf} inf, {n_nan} nan)"
    return (f"{label} range=[{finite.min():.2e}, {finite.max():.2e}] "
            f"mean={finite.mean():.2e} std={finite.std():.2e} "
            f"inf={n_inf} nan={n_nan}")


def benchmark_synthetic(synth_datasets, tabpfn):
    """Run TabPFN on synthetic datasets. Returns list of result dicts."""
    results = []
    for i, ds in enumerate(synth_datasets):
        n_classes = len(np.unique(np.concatenate([ds.y_train, ds.y_test])))

        n_total = len(ds.y_train) + len(ds.y_test)
        n_feat = ds.X_train.shape[1]
        t_out = ds.X_train.shape[2]

        # Diagnostics: check data range
        X_flat = flatten_3d(ds.X_train)
        diag = _data_diagnostics(X_flat, "  X_train")
        has_extreme = (np.abs(X_flat[np.isfinite(X_flat)]).max() > 1e6
                       if np.isfinite(X_flat).any() else True)
        if has_extreme or np.isinf(X_flat).any() or np.isnan(X_flat).any():
            print(f"  [{i+1}/{len(synth_datasets)}] synth_{i:03d}  "
                  f"({n_total} samp, {n_feat}x{t_out})  EXTREME VALUES")
            print(f"    {diag}")
            # Also print hyper info
            h = ds.hyper
            print(f"    V={h.V} M={h.M} d={h.d} T={h.T} K={h.kernel_size} "
                  f"gain={h.gain:.2f} init={h.init_type} "
                  f"std={h.root_std:.3f} a={h.root_a:.3f}")

        t0 = time.time()
        acc, auc = run_tabpfn(tabpfn, ds.X_train, ds.y_train, ds.X_test, ds.y_test, n_classes)
        dt = time.time() - t0

        acc_s = f"acc={acc:.3f}" if acc is not None else "skip"
        auc_s = f"auc={auc:.3f}" if auc is not None else ""
        print(f"  [{i+1}/{len(synth_datasets)}] synth_{i:03d}  "
              f"({n_total} samp, {n_feat}x{t_out})  {acc_s} {auc_s}  ({dt:.1f}s)")

        results.append({
            "name": f"synth_{i:03d}",
            "n_samples": n_total,
            "n_features": n_feat,
            "timesteps": t_out,
            "feat_x_t": n_feat * t_out,
            "n_classes": n_classes,
            "accuracy": acc,
            "auc": auc,
        })
    return results


def benchmark_real(real_stats, tabpfn):
    """Run TabPFN on real AEON datasets. Returns list of result dicts."""
    results = []
    for i, info in enumerate(real_stats):
        X_train, y_train, X_test, y_test = load_aeon_dataset(info["name"])
        if X_train is None:
            print(f"  [{i+1}/{len(real_stats)}] {info['name']}  SKIPPED (not in pkl)")
            continue

        t0 = time.time()
        acc, auc = run_tabpfn(tabpfn, X_train, y_train, X_test, y_test, info["n_classes"])
        dt = time.time() - t0

        acc_s = f"acc={acc:.3f}" if acc is not None else "skip"
        auc_s = f"auc={auc:.3f}" if auc is not None else ""
        print(f"  [{i+1}/{len(real_stats)}] {info['name']}  "
              f"({info['n_samples']} samp, {info['n_features']}x{info['timesteps']})  "
              f"{acc_s} {auc_s}  ({dt:.1f}s)")

        results.append({
            **info,
            "accuracy": acc,
            "auc": auc,
        })
    return results


# ============================================================
# TEXT OUTPUT
# ============================================================

def _row(label, vals):
    a = np.array(vals, dtype=float)
    return (f"  {label:25s}  min={a.min():<8.1f} mean={a.mean():<8.1f} "
            f"median={np.median(a):<8.1f} max={a.max():<8.1f} std={a.std():<8.1f}")


def _dist_line(vals, label):
    counts = Counter(vals)
    return f"  {label:25s}  {dict(sorted(counts.items()))}"


def print_comparison(real, synth):
    print("=" * 90)
    print(f"DISTRIBUTION COMPARISON: {len(real)} real  vs  {len(synth)} synthetic")
    print(f"Constraints: samples<={MAX_SAMPLES}  features<={MAX_FEATURES}  "
          f"timesteps<={MAX_TIMESTEPS}  feat*t<={MAX_FEAT_T}  classes<={MAX_CLASSES}")
    print("=" * 90)

    metrics = [
        ("n_samples",            "n_samples"),
        ("n_features",           "n_features"),
        ("timesteps",            "timesteps"),
        ("feat_x_t",             "feat_x_t"),
        ("n_classes",            "n_classes"),
        ("class_balance",        "class_balance"),
        ("missing_pct",          "missing_pct"),
        ("min_samples_per_class","min_samples_per_class"),
    ]

    for tag, group in [("REAL", real), ("SYNTHETIC", synth)]:
        print(f"\n--- {tag} ({len(group)} datasets) ---")
        if not group:
            print("  (none)")
            continue
        for label, key in metrics:
            vals = [s.get(key, 0) for s in group]
            print(_row(label, vals))
        print(_dist_line([s["n_features"] for s in group], "n_features dist"))
        print(_dist_line([s["n_classes"] for s in group],  "n_classes dist"))

    # Constraint checks
    print("\n--- CONSTRAINT CHECK (synthetic) ---")
    checks = {
        f"n_samples > {MAX_SAMPLES}":   sum(1 for s in synth if s["n_samples"]  > MAX_SAMPLES),
        f"n_features > {MAX_FEATURES}": sum(1 for s in synth if s["n_features"] > MAX_FEATURES),
        f"timesteps > {MAX_TIMESTEPS}": sum(1 for s in synth if s["timesteps"]  > MAX_TIMESTEPS),
        f"feat_x_t > {MAX_FEAT_T}":     sum(1 for s in synth if s["feat_x_t"]   > MAX_FEAT_T),
        f"n_classes > {MAX_CLASSES}":    sum(1 for s in synth if s["n_classes"]   > MAX_CLASSES),
    }
    all_ok = True
    for k, v in checks.items():
        status = "OK" if v == 0 else f"VIOLATION ({v})"
        if v > 0:
            all_ok = False
        print(f"  {k:25s}  {status}")
    if all_ok:
        print("  ALL CONSTRAINTS SATISFIED")


def print_benchmark_results(synth_bench, real_bench):
    """Print benchmark comparison."""
    print("\n" + "=" * 90)
    print("TABPFN BENCHMARK COMPARISON")
    print("=" * 90)

    for tag, results in [("SYNTHETIC", synth_bench), ("REAL", real_bench)]:
        accs = [r["accuracy"] for r in results if r["accuracy"] is not None]
        aucs = [r["auc"] for r in results if r["auc"] is not None]
        print(f"\n--- {tag} ({len(results)} datasets, {len(accs)} benchmarked) ---")
        if accs:
            a = np.array(accs)
            print(f"  {'accuracy':25s}  min={a.min():.3f}  mean={a.mean():.3f}  "
                  f"median={np.median(a):.3f}  max={a.max():.3f}  std={a.std():.3f}")
        if aucs:
            a = np.array(aucs)
            print(f"  {'auc':25s}  min={a.min():.3f}  mean={a.mean():.3f}  "
                  f"median={np.median(a):.3f}  max={a.max():.3f}  std={a.std():.3f}")

    # Detailed per-dataset
    for tag, results in [("SYNTHETIC", synth_bench), ("REAL", real_bench)]:
        with_acc = [r for r in results if r["accuracy"] is not None]
        if not with_acc:
            continue
        print(f"\n--- {tag} per-dataset ---")
        print(f"  {'Name':<30s} {'Samp':>6} {'Feat':>5} {'T':>5} {'Cls':>4} {'Acc':>7} {'AUC':>7}")
        print("  " + "-" * 75)
        sorted_r = sorted(with_acc, key=lambda x: x["accuracy"], reverse=True)
        for r in sorted_r:
            auc_str = f"{r['auc']:.3f}" if r["auc"] is not None else "  N/A"
            print(f"  {r['name']:<30s} {r['n_samples']:>6} {r['n_features']:>5} "
                  f"{r['timesteps']:>5} {r['n_classes']:>4} {r['accuracy']:>7.3f} {auc_str:>7}")


# ============================================================
# PLOT
# ============================================================

def plot_comparison(real, synth, out_path, synth_bench=None, real_bench=None):
    """Side-by-side histograms for each metric + optional benchmark histograms."""
    metrics = [
        ("n_samples",             "n_samples",             False),
        ("n_features",            "n_features",            True),
        ("timesteps",             "timesteps",             False),
        ("feat_x_t",              "feat x timesteps",      False),
        ("n_classes",             "n_classes",             True),
        ("class_balance",         "class balance (min/max)",False),
        ("min_samples_per_class", "min samples / class",   False),
    ]

    has_bench = synth_bench and real_bench
    n_metrics = len(metrics) + (2 if has_bench else 0)  # +acc +auc
    ncols = 3
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4 * nrows))
    axes = axes.ravel()

    plot_idx = 0
    for key, title, is_discrete in metrics:
        ax = axes[plot_idx]
        plot_idx += 1
        r_vals = np.array([s.get(key, 0) for s in real],  dtype=float)
        s_vals = np.array([s.get(key, 0) for s in synth], dtype=float)

        if is_discrete:
            all_vals = sorted(set(r_vals.astype(int)).union(set(s_vals.astype(int))))
            x = np.arange(len(all_vals))
            width = 0.35
            r_counts = [np.sum(r_vals == v) for v in all_vals]
            s_counts = [np.sum(s_vals == v) for v in all_vals]
            r_frac = np.array(r_counts) / max(len(r_vals), 1)
            s_frac = np.array(s_counts) / max(len(s_vals), 1)
            ax.bar(x - width / 2, r_frac, width, label="Real", alpha=0.7, color="steelblue")
            ax.bar(x + width / 2, s_frac, width, label="Synth", alpha=0.7, color="coral")
            ax.set_xticks(x)
            ax.set_xticklabels([str(v) for v in all_vals])
            ax.set_ylabel("fraction")
        else:
            lo = min(r_vals.min(), s_vals.min())
            hi = max(r_vals.max(), s_vals.max())
            if hi - lo < 1e-12:
                ax.bar(["Real", "Synth"], [len(r_vals), len(s_vals)],
                       color=["steelblue", "coral"], alpha=0.7)
                ax.set_ylabel("count")
                ax.text(0.5, 0.5, f"all = {lo:.2g}", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, color="gray")
            else:
                bins = np.linspace(lo, hi, 20)
                ax.hist(r_vals, bins=bins, alpha=0.6, density=True, label="Real", color="steelblue")
                ax.hist(s_vals, bins=bins, alpha=0.6, density=True, label="Synth", color="coral")
                ax.set_ylabel("density")

        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)

    # Benchmark histograms
    if has_bench:
        for metric_key, metric_title in [("accuracy", "TabPFN Accuracy"), ("auc", "TabPFN AUC")]:
            ax = axes[plot_idx]
            plot_idx += 1
            r_vals = np.array([r[metric_key] for r in real_bench if r[metric_key] is not None])
            s_vals = np.array([r[metric_key] for r in synth_bench if r[metric_key] is not None])
            if len(r_vals) == 0 and len(s_vals) == 0:
                ax.set_visible(False)
                continue
            lo = min(r_vals.min() if len(r_vals) else 1, s_vals.min() if len(s_vals) else 1)
            hi = max(r_vals.max() if len(r_vals) else 0, s_vals.max() if len(s_vals) else 0)
            bins = np.linspace(max(0, lo - 0.05), min(1, hi + 0.05), 15)
            if len(r_vals):
                ax.hist(r_vals, bins=bins, alpha=0.6, density=True, label=f"Real (n={len(r_vals)})", color="steelblue")
            if len(s_vals):
                ax.hist(s_vals, bins=bins, alpha=0.6, density=True, label=f"Synth (n={len(s_vals)})", color="coral")
            # Add mean lines
            if len(r_vals):
                ax.axvline(r_vals.mean(), color="steelblue", ls="--", lw=2, label=f"Real mean={r_vals.mean():.3f}")
            if len(s_vals):
                ax.axvline(s_vals.mean(), color="coral", ls="--", lw=2, label=f"Synth mean={s_vals.mean():.3f}")
            ax.set_title(metric_title, fontsize=10)
            ax.legend(fontsize=7)
            ax.set_ylabel("density")

    for idx in range(plot_idx, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"Real ({len(real)}) vs Synthetic ({len(synth)}) — "
        f"feat*t<={MAX_FEAT_T}, samples<={MAX_SAMPLES}, features<={MAX_FEATURES}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to {out_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Compare real vs synthetic dataset distributions + TabPFN benchmark")
    parser.add_argument("--n-synthetic", type=int, default=55, help="Number of synthetic datasets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="results/comparison")
    parser.add_argument("--no-benchmark", action="store_true", help="Skip TabPFN benchmark")
    args = parser.parse_args()

    # --- Distribution stats ---
    print("Loading real AEON dataset stats...")
    real_stats = load_real_stats()
    print(f"  {len(real_stats)} datasets meet constraints")

    print(f"\nGenerating {args.n_synthetic} synthetic datasets (seed={args.seed})...")
    synth_datasets = generate_synthetic_datasets(n_datasets=args.n_synthetic, seed=args.seed)
    print(f"  {len(synth_datasets)} generated successfully")

    # Build synth stats
    synth_stats = []
    for i, ds in enumerate(synth_datasets):
        X = ds.X_train
        y_all = np.concatenate([ds.y_train, ds.y_test])
        counts = Counter(int(c) for c in y_all)
        min_c = min(counts.values())
        max_c = max(counts.values())
        synth_stats.append({
            "name": f"synth_{i:03d}",
            "n_samples": len(ds.y_train) + len(ds.y_test),
            "n_features": X.shape[1],
            "timesteps": X.shape[2],
            "feat_x_t": X.shape[1] * X.shape[2],
            "n_classes": len(counts),
            "class_balance": min_c / max_c if max_c > 0 else 1.0,
            "missing_pct": 0.0,
            "min_samples_per_class": min_c,
        })

    print_comparison(real_stats, synth_stats)

    # --- TabPFN Benchmark ---
    synth_bench = None
    real_bench = None

    if not args.no_benchmark:
        print("\n" + "=" * 90)
        print("TABPFN BENCHMARK")
        print("=" * 90)

        print("\nInitializing TabPFN...")
        from tabpfn import TabPFNClassifier
        tabpfn = TabPFNClassifier()

        print(f"\nBenchmarking {len(synth_datasets)} synthetic datasets...")
        synth_bench = benchmark_synthetic(synth_datasets, tabpfn)

        print(f"\nBenchmarking {len(real_stats)} real datasets...")
        real_bench = benchmark_real(real_stats, tabpfn)

        print_benchmark_results(synth_bench, real_bench)

    # --- Plot ---
    plot_comparison(real_stats, synth_stats,
                    f"{args.out_dir}/distributions.png",
                    synth_bench, real_bench)


if __name__ == "__main__":
    main()
