#!/usr/bin/env python3
"""
Visualize generated DAG datasets.
For each dataset: one PNG with 5 observations (columns), features as rows.
Each series is colored by the target class of that observation.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from dag_dataset_generator import generate_dataset, GeneratedDataset

# Distinct colors for classes (up to 10)
CLASS_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def visualize_dataset(ds: GeneratedDataset, path: Path, n_obs: int = 5) -> None:
    """
    Create a PNG for one dataset.
    Rows = features, columns = observations (samples).
    Each series colored by target class.
    """
    # Use train set; pick n_obs samples (one per class if possible, then fill)
    X = ds.X_train
    y = ds.y_train
    n_train = X.shape[0]
    n_feat = X.shape[1]
    T_out = X.shape[2]

    # Pick diverse observations: one per class then random
    chosen = []
    classes_seen = set()
    indices = np.arange(n_train)
    np.random.shuffle(indices)
    for idx in indices:
        c = int(y[idx])
        if c not in classes_seen and len(chosen) < n_obs:
            chosen.append(idx)
            classes_seen.add(c)
        if len(chosen) >= n_obs:
            break
    # Fill remaining
    for idx in indices:
        if idx not in chosen and len(chosen) < n_obs:
            chosen.append(idx)
        if len(chosen) >= n_obs:
            break
    chosen = chosen[:n_obs]

    t = np.arange(T_out)
    fig, axes = plt.subplots(
        n_feat, n_obs, figsize=(4.5 * n_obs, 1.8 * n_feat),
        sharex=True, sharey="row",
    )
    if n_feat == 1:
        axes = axes.reshape(1, -1)
    if n_obs == 1:
        axes = axes.reshape(-1, 1)

    for col, obs_idx in enumerate(chosen):
        cls = int(y[obs_idx])
        color = CLASS_COLORS[cls % len(CLASS_COLORS)]
        for row in range(n_feat):
            ax = axes[row, col]
            ax.plot(t, X[obs_idx, row, :], color=color, linewidth=0.9)
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(f"obs {obs_idx} (cls={cls})", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"feat_{row}", fontsize=9)
            if row == n_feat - 1:
                ax.set_xlabel("t", fontsize=8)

    h = ds.hyper
    fig.suptitle(
        f"V={h.V} M={h.M} d={h.d} T={h.T} K={h.kernel_size} "
        f"init={h.init_type} | {h.n_features} feats, {h.n_classes} cls, "
        f"{h.n_samples} samp",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    p = argparse.ArgumentParser(description="Visualize generated DAG datasets")
    p.add_argument("--n-datasets", type=int, default=10, help="Number of datasets to generate and visualize")
    p.add_argument("--n-obs", type=int, default=5, help="Observations per PNG (columns)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="results/datasets")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.n_datasets):
        ds = generate_dataset(rng)
        path = out_dir / f"dataset_{i:03d}.png"
        visualize_dataset(ds, path, n_obs=args.n_obs)
        h = ds.hyper
        print(
            f"  [{i}] V={h.V} M={h.M} d={h.d} T={h.T} K={h.kernel_size} "
            f"n_feat={h.n_features} n_cls={h.n_classes} n_samp={h.n_samples}"
        )


if __name__ == "__main__":
    main()
