#!/usr/bin/env python3
"""
Visualize one dataset: columns = classes, rows = features.
Each cell shows multiple overlaid series from that class,
so you can see what makes classes different.
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from dag_dataset_generator import generate_dataset, GeneratedDataset

CLASS_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def visualize_by_class(
    ds: GeneratedDataset, path: Path, n_series_per_class: int = 8,
) -> None:
    """
    PNG: rows = features, columns = classes.
    Each cell plots up to n_series_per_class overlaid series from that class.
    """
    X = ds.X_train
    y = ds.y_train
    n_feat = X.shape[1]
    T_out = X.shape[2]

    classes = sorted(np.unique(y).tolist())
    n_cls = len(classes)
    t = np.arange(T_out)

    fig, axes = plt.subplots(
        n_feat, n_cls,
        figsize=(4.0 * n_cls, 2.0 * n_feat),
        sharex=True, sharey="row",
        squeeze=False,
    )

    for col, cls in enumerate(classes):
        color = CLASS_COLORS[int(cls) % len(CLASS_COLORS)]
        mask = y == cls
        X_cls = X[mask]  # (n_in_class, n_feat, T_out)
        n_avail = X_cls.shape[0]

        # Pick up to n_series_per_class random samples
        pick = min(n_avail, n_series_per_class)
        idxs = np.random.choice(n_avail, size=pick, replace=False) if n_avail > 0 else []

        for row in range(n_feat):
            ax = axes[row, col]
            for i in idxs:
                ax.plot(t, X_cls[i, row, :], color=color, alpha=0.5, linewidth=0.7)
            ax.grid(True, alpha=0.2)

            if row == 0:
                ax.set_title(f"class {int(cls)}  (n={n_avail})", fontsize=9,
                             color=color, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"feat_{row}", fontsize=9)
            if row == n_feat - 1:
                ax.set_xlabel("t", fontsize=8)

    h = ds.hyper
    fig.suptitle(
        f"V={h.V} M={h.M} d={h.d} T={h.T} K={h.kernel_size} "
        f"gain={h.gain:.2f} init={h.init_type} | "
        f"{h.n_features} feats, {h.n_classes} cls, {h.n_samples} samp\n"
        f"rows=features, cols=classes, {n_series_per_class} series/class",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    p = argparse.ArgumentParser(description="Visualize datasets by class")
    p.add_argument("--n-datasets", type=int, default=5)
    p.add_argument("--n-series", type=int, default=8, help="Series per class per cell")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="results/by_class")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)

    for i in range(args.n_datasets):
        ds = generate_dataset(rng)
        path = out_dir / f"dataset_{i:03d}.png"
        visualize_by_class(ds, path, n_series_per_class=args.n_series)
        h = ds.hyper
        print(f"  [{i}] V={h.V} M={h.M} d={h.d} T={h.T} K={h.kernel_size} "
              f"n_feat={h.n_features} n_cls={h.n_classes} n_samp={h.n_samples}")


if __name__ == "__main__":
    main()
