#!/usr/bin/env python3
"""
Compare distributions of real AEON datasets vs synthetic datasets from the
final generator (folder 11).

Produces:
  1. Console table with side-by-side statistics (distributions)
  2. PNG histogram comparison per metric
  3. TabPFN benchmark comparison (accuracy, AUC) on both real and synthetic
  4. Per-dataset benchmark table sorted by accuracy
"""

import json
import argparse
import pickle
import time
import sys
import numpy as np
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from generator import DatasetGenerator, visualize_dataset
from hyperparameters import GeneratorHyperparameters


# ============================================================
# CONSTANTS / CONSTRAINTS (from hyperparameters)
# ============================================================

_HP = GeneratorHyperparameters()

MAX_SAMPLES   = _HP.dataset.max_samples          # 1000
MAX_FEATURES  = _HP.roles.n_features_range[1]      # 12
MAX_FEAT_T    = _HP.dataset.max_feat_times_t       # 500
MAX_CLASSES   = 10
MAX_TIMESTEPS = _HP.dataset.t_range[1]             # 500

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
    # Need the TimeSeriesDataset class from 01_real_data
    sys.path.insert(0, str(Path(__file__).parent.parent / "01_real_data"))
    with open(AEON_PKL_PATH, "rb") as f:
        datasets = pickle.load(f)
    _AEON_CACHE = {ds.name: ds for ds in datasets}
    return _AEON_CACHE


def load_aeon_dataset(name):
    """Load a single AEON dataset by name."""
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

def generate_synthetic_datasets(n_datasets=30, base_seed=0):
    """Generate synthetic datasets using the folder-11 generator.

    Returns list of (dict_with_data, DatasetGenerator) tuples.
    Datasets that return None (< 2 classes) are silently skipped.
    """
    hp = GeneratorHyperparameters()
    results = []
    skipped = 0

    for i in range(n_datasets):
        seed = base_seed + i
        try:
            gen = DatasetGenerator(seed=seed, hp=hp)
            ds = gen.generate_dataset()

            if ds is None:
                skipped += 1
                continue

            results.append((ds, gen))
        except Exception as e:
            skipped += 1
            print(f"  [seed={seed}] FAILED: {e}")

    if skipped:
        print(f"  {skipped}/{n_datasets} seeds skipped (< 2 classes or error)")
    return results


def synth_to_stats(synth_list):
    """Convert list of (ds_dict, gen) into list of stat dicts."""
    stats = []
    for i, (ds, gen) in enumerate(synth_list):
        y_all = np.concatenate([ds["y_train"], ds["y_test"]])
        counts = Counter(int(c) for c in y_all)
        min_c = min(counts.values())
        max_c = max(counts.values())

        stats.append({
            "name": f"synth_{i:03d}",
            "n_samples": ds["n_samples"],
            "n_features": ds["n_features"],
            "timesteps": ds["T"],
            "feat_x_t": ds["n_features"] * ds["T"],
            "n_classes": ds["n_classes"],
            "class_balance": min_c / max_c if max_c > 0 else 1.0,
            "missing_pct": 0.0,
            "min_samples_per_class": min_c,
        })
    return stats


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

    X_tr = flatten_3d(np.asarray(X_train, dtype=np.float32))
    X_te = flatten_3d(np.asarray(X_test, dtype=np.float32))

    # Replace inf/nan with 0 to avoid TabPFN crashes
    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
    X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)

    # Clip extreme values
    clip_val = 1e6
    X_tr = np.clip(X_tr, -clip_val, clip_val)
    X_te = np.clip(X_te, -clip_val, clip_val)

    # Encode labels if string
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    if y_train.dtype.kind in ('U', 'S', 'O'):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

    # Skip if flattened features exceed TabPFN limit
    if X_tr.shape[1] > 500:
        return None, None

    # Need at least 2 classes in train
    if len(np.unique(y_train)) < 2:
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


def benchmark_synthetic(synth_list, tabpfn):
    """Run TabPFN on synthetic datasets. Returns list of result dicts."""
    results = []
    for i, (ds, gen) in enumerate(synth_list):
        n_classes = ds["n_classes"]
        n_total = ds["n_samples"]
        n_feat = ds["n_features"]
        T = ds["T"]

        t0 = time.time()
        acc, auc = run_tabpfn(tabpfn,
                              ds["X_train"], ds["y_train"],
                              ds["X_test"], ds["y_test"],
                              n_classes)
        dt = time.time() - t0

        acc_s = f"acc={acc:.3f}" if acc is not None else "skip"
        auc_s = f"auc={auc:.3f}" if auc is not None else ""
        print(f"  [{i+1:3d}/{len(synth_list)}] synth_{i:03d}  "
              f"({n_total:4d} samp, {n_feat}x{T})  {acc_s} {auc_s}  ({dt:.1f}s)",
              flush=True)

        results.append({
            "name": f"synth_{i:03d}",
            "n_samples": n_total,
            "n_features": n_feat,
            "timesteps": T,
            "feat_x_t": n_feat * T,
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
            print(f"  [{i+1:3d}/{len(real_stats)}] {info['name']}  SKIPPED (not in pkl)",
                  flush=True)
            continue

        t0 = time.time()
        acc, auc = run_tabpfn(tabpfn, X_train, y_train, X_test, y_test, info["n_classes"])
        dt = time.time() - t0

        acc_s = f"acc={acc:.3f}" if acc is not None else "skip"
        auc_s = f"auc={auc:.3f}" if auc is not None else ""
        print(f"  [{i+1:3d}/{len(real_stats)}] {info['name']:30s}  "
              f"({info['n_samples']:4d} samp, {info['n_features']}x{info['timesteps']})  "
              f"{acc_s} {auc_s}  ({dt:.1f}s)", flush=True)

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
    print("\n" + "=" * 90)
    print(f"DISTRIBUTION COMPARISON: {len(real)} real  vs  {len(synth)} synthetic")
    print(f"Constraints: samples<={MAX_SAMPLES}  features<={MAX_FEATURES}  "
          f"timesteps<={MAX_TIMESTEPS}  feat*t<={MAX_FEAT_T}  classes<={MAX_CLASSES}")
    print("=" * 90)

    metrics = [
        ("n_samples",             "n_samples"),
        ("n_features",            "n_features"),
        ("timesteps",             "timesteps"),
        ("feat_x_t",              "feat_x_t"),
        ("n_classes",             "n_classes"),
        ("class_balance",         "class_balance"),
        ("missing_pct",           "missing_pct"),
        ("min_samples_per_class", "min_samples_per_class"),
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
        print(_dist_line([s["n_classes"]  for s in group], "n_classes dist"))

    # Constraint check on synthetic
    print("\n--- CONSTRAINT CHECK (synthetic) ---")
    checks = {
        f"n_samples > {MAX_SAMPLES}":    sum(1 for s in synth if s["n_samples"]  > MAX_SAMPLES),
        f"n_features > {MAX_FEATURES}":  sum(1 for s in synth if s["n_features"] > MAX_FEATURES),
        f"timesteps > {MAX_TIMESTEPS}":  sum(1 for s in synth if s["timesteps"]  > MAX_TIMESTEPS),
        f"feat_x_t > {MAX_FEAT_T}":      sum(1 for s in synth if s["feat_x_t"]   > MAX_FEAT_T),
        f"n_classes > {MAX_CLASSES}":     sum(1 for s in synth if s["n_classes"]   > MAX_CLASSES),
    }
    all_ok = True
    for k, v in checks.items():
        status = "OK" if v == 0 else f"VIOLATION ({v})"
        if v > 0:
            all_ok = False
        print(f"  {k:25s}  {status}")
    if all_ok:
        print("  ALL CONSTRAINTS SATISFIED")


def _print_bench_section(tag, results):
    """Print summary + per-dataset table for one benchmark group."""
    accs = [r["accuracy"] for r in results if r["accuracy"] is not None]
    aucs = [r["auc"]      for r in results if r["auc"]      is not None]

    print(f"\n--- {tag} SUMMARY ({len(results)} datasets, {len(accs)} benchmarked) ---")
    if accs:
        a = np.array(accs)
        print(f"  {'accuracy':25s}  min={a.min():.3f}  mean={a.mean():.3f}  "
              f"median={np.median(a):.3f}  max={a.max():.3f}  std={a.std():.3f}")
    if aucs:
        a = np.array(aucs)
        print(f"  {'auc':25s}  min={a.min():.3f}  mean={a.mean():.3f}  "
              f"median={np.median(a):.3f}  max={a.max():.3f}  std={a.std():.3f}")

    with_acc = [r for r in results if r["accuracy"] is not None]
    if with_acc:
        print(f"\n  {'Name':<30s} {'Samp':>6} {'Feat':>5} {'T':>5} {'Cls':>4} {'Acc':>7} {'AUC':>7}")
        print("  " + "-" * 75)
        sorted_r = sorted(with_acc, key=lambda x: x["accuracy"], reverse=True)
        for r in sorted_r:
            auc_str = f"{r['auc']:.3f}" if r["auc"] is not None else "  N/A"
            print(f"  {r['name']:<30s} {r['n_samples']:>6} {r['n_features']:>5} "
                  f"{r['timesteps']:>5} {r['n_classes']:>4} {r['accuracy']:>7.3f} {auc_str:>7}")


def print_benchmark_results(synth_bench, real_bench):
    """Print combined comparison header, then each section."""
    print("\n" + "=" * 90)
    print("TABPFN BENCHMARK — FINAL COMPARISON")
    print("=" * 90)
    if synth_bench:
        _print_bench_section("SYNTHETIC", synth_bench)
    if real_bench:
        _print_bench_section("REAL", real_bench)


# ============================================================
# PLOT
# ============================================================

def plot_comparison(real, synth, out_path, synth_bench=None, real_bench=None):
    """Side-by-side histograms for each metric + optional benchmark histograms."""
    metrics = [
        ("n_samples",             "n_samples",               False),
        ("n_features",            "n_features",              True),
        ("timesteps",             "timesteps",               False),
        ("feat_x_t",              "feat x timesteps",        False),
        ("n_classes",             "n_classes",               True),
        ("class_balance",         "class balance (min/max)", False),
        ("min_samples_per_class", "min samples / class",     False),
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
            lo = min(r_vals.min(), s_vals.min()) if len(r_vals) and len(s_vals) else 0
            hi = max(r_vals.max(), s_vals.max()) if len(r_vals) and len(s_vals) else 1
            if hi - lo < 1e-12:
                ax.bar(["Real", "Synth"], [len(r_vals), len(s_vals)],
                       color=["steelblue", "coral"], alpha=0.7)
                ax.set_ylabel("count")
            else:
                bins = np.linspace(lo, hi, 20)
                if len(r_vals):
                    ax.hist(r_vals, bins=bins, alpha=0.6, density=True,
                            label="Real", color="steelblue")
                if len(s_vals):
                    ax.hist(s_vals, bins=bins, alpha=0.6, density=True,
                            label="Synth", color="coral")
                ax.set_ylabel("density")

        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)

    # Benchmark histograms (accuracy, AUC)
    if has_bench:
        for metric_key, metric_title in [("accuracy", "TabPFN Accuracy"),
                                          ("auc", "TabPFN AUC")]:
            ax = axes[plot_idx]
            plot_idx += 1
            r_vals = np.array([r[metric_key] for r in real_bench
                               if r.get(metric_key) is not None])
            s_vals = np.array([r[metric_key] for r in synth_bench
                               if r.get(metric_key) is not None])
            if len(r_vals) == 0 and len(s_vals) == 0:
                ax.set_visible(False)
                continue

            lo = min(r_vals.min() if len(r_vals) else 1,
                     s_vals.min() if len(s_vals) else 1)
            hi = max(r_vals.max() if len(r_vals) else 0,
                     s_vals.max() if len(s_vals) else 0)
            bins = np.linspace(max(0, lo - 0.05), min(1, hi + 0.05), 15)

            if len(r_vals):
                ax.hist(r_vals, bins=bins, alpha=0.6, density=True,
                        label=f"Real (n={len(r_vals)})", color="steelblue")
            if len(s_vals):
                ax.hist(s_vals, bins=bins, alpha=0.6, density=True,
                        label=f"Synth (n={len(s_vals)})", color="coral")
            # Mean lines
            if len(r_vals):
                ax.axvline(r_vals.mean(), color="steelblue", ls="--", lw=2,
                           label=f"Real mean={r_vals.mean():.3f}")
            if len(s_vals):
                ax.axvline(s_vals.mean(), color="coral", ls="--", lw=2,
                           label=f"Synth mean={s_vals.mean():.3f}")
            ax.set_title(metric_title, fontsize=10)
            ax.legend(fontsize=7)
            ax.set_ylabel("density")

    # Hide unused axes
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
# SAVE RESULTS TO JSON
# ============================================================

def save_results(real_stats, synth_stats, synth_bench, real_bench, out_path):
    """Persist all results in a JSON file."""
    obj = {
        "real_stats": real_stats,
        "synth_stats": synth_stats,
    }
    if synth_bench is not None:
        obj["synth_benchmark"] = synth_bench
    if real_bench is not None:
        obj["real_benchmark"] = real_bench
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"Results JSON saved to {out_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare real vs synthetic dataset distributions + TabPFN benchmark")
    parser.add_argument("--n-synthetic", type=int, default=50,
                        help="Number of synthetic seeds to try")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed for synthetic generation")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: results/comparison)")
    parser.add_argument("--no-benchmark", action="store_true",
                        help="Skip TabPFN benchmark (only compare distributions)")
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = str(Path(__file__).parent / "results" / "comparison")

    print("=" * 90)
    print("Distribution + Benchmark Comparison: AEON (real) vs Final Generator (synthetic)")
    print("=" * 90)

    # ── 1. Load real AEON stats ─────────────────────────────────────────
    print("\nLoading real AEON dataset stats...")
    real_stats = load_real_stats()
    print(f"  {len(real_stats)} datasets meet constraints")

    # ── 2. Generate synthetic datasets ──────────────────────────────────
    print(f"\nGenerating {args.n_synthetic} synthetic datasets (seed={args.seed})...")
    synth_list = generate_synthetic_datasets(
        n_datasets=args.n_synthetic, base_seed=args.seed)
    print(f"  {len(synth_list)} generated successfully")
    synth_stats = synth_to_stats(synth_list)

    # ── 2b. Dataset visualizations ───────────────────────────────────────
    vis_dir = Path(args.out_dir) / "dataset_visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    n_vis = min(15, len(synth_list))
    if n_vis > 0:
        print(f"\nSaving {n_vis} dataset visualizations to {vis_dir}...")
        for i in range(n_vis):
            ds, gen = synth_list[i]
            save_path = vis_dir / f"synth_{i:03d}_seed{gen.seed}.png"
            try:
                visualize_dataset(ds, gen, str(save_path), n_per_class=5)
            except Exception as e:
                print(f"  [vis synth_{i:03d}] {e}")
        print(f"  Done.")

    # ── 3. Print distribution comparison ────────────────────────────────
    print_comparison(real_stats, synth_stats)

    # ── 4. TabPFN Benchmark ─────────────────────────────────────────────
    synth_bench = None
    real_bench = None

    if not args.no_benchmark:
        print("\n" + "=" * 90)
        print("TABPFN BENCHMARK")
        print("=" * 90)

        print("\nInitializing TabPFN...")
        from tabpfn import TabPFNClassifier
        tabpfn = TabPFNClassifier(n_estimators=4, ignore_pretraining_limits=True)

        # ── Synthetic first (so results print immediately) ────────────
        print(f"\nBenchmarking {len(synth_list)} synthetic datasets...")
        synth_bench = benchmark_synthetic(synth_list, tabpfn)
        _print_bench_section("SYNTHETIC", synth_bench)

        # ── Real second ───────────────────────────────────────────────
        print(f"\nBenchmarking {len(real_stats)} real datasets...")
        real_bench = benchmark_real(real_stats, tabpfn)
        _print_bench_section("REAL", real_bench)

        # ── Final side-by-side ────────────────────────────────────────
        print_benchmark_results(synth_bench, real_bench)

    # ── 5. Plots ────────────────────────────────────────────────────────
    plot_path = f"{args.out_dir}/distributions.png"
    plot_comparison(real_stats, synth_stats, plot_path, synth_bench, real_bench)

    # ── 6. Save JSON ────────────────────────────────────────────────────
    json_path = f"{args.out_dir}/results.json"
    save_results(real_stats, synth_stats, synth_bench, real_bench, json_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
