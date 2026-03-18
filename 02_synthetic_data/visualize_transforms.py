#!/usr/bin/env python3
"""
Visualize how various transforms change synthetic time series.

For each of several random generator seeds, generates one dataset, picks
the first feature channel (first series), and plots:
  - Original
  - Log: log(|x|+1)*sign(x)
  - Exp: sign(x)*(exp(|x|)-1)
  - KDI with several alpha values
  - SquashingScaler (robust scaling + soft clip)
  - Kumaraswamy warp (min-max to [0,1], then CDF warp, with various a,b)

Usage:
    python visualize_transforms.py
    python visualize_transforms.py --n-series 6 --seed 42
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / '00_TabPFN' / 'src'))

from generator import DatasetGenerator
from hyperparameters import GeneratorHyperparameters


# ── Transform functions ──────────────────────────────────────────────────────

def safe_log(x):
    """log(|x| + 1) * sign(x)."""
    return np.sign(x) * np.log1p(np.abs(x))


def safe_exp(x):
    """sign(x) * (exp(|x|) - 1), clipped to avoid overflow."""
    clipped = np.clip(np.abs(x), 0, 10)
    return np.sign(x) * np.expm1(clipped)


def kdi_transform_series(x_1d, alpha=1.0, output_distribution="normal"):
    """Apply KDI transform to a 1D series."""
    try:
        from tabpfn.preprocessing.steps.kdi_transformer import KDITransformerWithNaN
        kdi = KDITransformerWithNaN(alpha=alpha, output_distribution=output_distribution)
        col = x_1d.reshape(-1, 1).astype(np.float64)
        kdi.fit(col)
        out = kdi.transform(col).ravel().astype(np.float32)
        if np.all(np.isfinite(out)):
            return out
    except Exception as e:
        print(f"  KDI(alpha={alpha}) failed: {e}")
    return x_1d


def squashing_scaler(x_1d, max_abs=3.0, q_low=25.0, q_high=75.0):
    """Robust scaling + soft clipping: z / sqrt(1 + (z/B)^2)."""
    x = x_1d.copy().astype(np.float64)
    finite = x[np.isfinite(x)]
    if len(finite) == 0:
        return x.astype(np.float32)

    median = np.median(finite)
    q_lo = np.percentile(finite, q_low)
    q_hi = np.percentile(finite, q_high)

    if q_hi != q_lo:
        scale = 1.0 / (q_hi - q_lo)
    else:
        vmin, vmax = np.min(finite), np.max(finite)
        if vmax != vmin:
            scale = 2.0 / (vmax - vmin)
        else:
            return np.zeros_like(x, dtype=np.float32)

    z = (x - median) * scale
    # Soft clip: z / sqrt(1 + (z/B)^2)
    out = z / np.sqrt(1.0 + (z / max_abs) ** 2)
    return out.astype(np.float32)


def kumaraswamy_warp(x_1d, a, b):
    """Min-max scale to [0,1], then apply Kumaraswamy CDF: F(x) = 1 - (1 - x^a)^b."""
    x = x_1d.copy().astype(np.float64)
    xmin, xmax = x.min(), x.max()
    if xmax <= xmin:
        return x.astype(np.float32)

    # Scale to [0, 1]
    x_norm = (x - xmin) / (xmax - xmin)
    x_norm = np.clip(x_norm, 1e-12, 1.0 - 1e-12)

    # Kumaraswamy CDF
    warped = 1.0 - (1.0 - np.power(x_norm, a)) ** b

    # Scale back to original range
    out = warped * (xmax - xmin) + xmin
    return out.astype(np.float32)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize transforms on synthetic series")
    parser.add_argument("--n-series", type=int, default=6,
                        help="Number of different generator seeds to visualize")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    args = parser.parse_args()

    hp = GeneratorHyperparameters()

    kdi_alphas = [0.3, 1.0, 3.0]
    kuma_params = [(0.5, 0.5), (2.0, 0.5), (0.5, 2.0)]

    # Columns layout:
    # Original | Log | Exp | Squash | KDI a=0.3 | KDI a=1 | KDI a=3 | Kuma(.5,.5) | Kuma(2,.5) | Kuma(.5,2)
    col_specs = (
        [("Original", '#2196F3')]
        + [("Log", '#4CAF50')]
        + [("Exp", '#FF5722')]
        + [("Squash", '#607D8B')]
        + [(f"KDI α={a}", c) for a, c in zip(kdi_alphas, ['#9C27B0', '#FF9800', '#795548'])]
        + [(f"Kuma({a},{b})", c)
           for (a, b), c in zip(kuma_params, ['#E91E63', '#00BCD4', '#8BC34A'])]
    )

    n_cols = len(col_specs)
    n_rows = args.n_series

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.8 * n_rows),
                             squeeze=False)

    for c, (title, _) in enumerate(col_specs):
        axes[0, c].set_title(title, fontsize=10, fontweight='bold')

    series_idx = 0
    seed = args.seed

    while series_idx < n_rows:
        seed += 1
        try:
            gen = DatasetGenerator(seed=seed, hp=hp)
            ds = gen.generate_dataset()
            if ds is None:
                continue
        except Exception:
            continue

        X_3d = ds['X_train']  # (n_samples, n_features, T)
        if X_3d.shape[0] == 0 or X_3d.shape[2] < 5:
            continue

        # Pick first sample, first feature channel
        series = X_3d[0, 0, :].astype(np.float32)
        T = len(series)
        t_axis = np.arange(T)

        # Row label
        axes[series_idx, 0].set_ylabel(
            f"Seed {seed}\nT={T}, m={ds['n_features']}", fontsize=8)

        # Compute all transforms
        transforms = [
            series,                                 # Original
            safe_log(series),                       # Log
            safe_exp(series),                       # Exp
            squashing_scaler(series),               # Squash
        ]
        for alpha in kdi_alphas:
            transforms.append(kdi_transform_series(series, alpha=alpha))
        for a, b in kuma_params:
            transforms.append(kumaraswamy_warp(series, a, b))

        # Plot each column
        for c, ((_title, color), data) in enumerate(zip(col_specs, transforms)):
            axes[series_idx, c].plot(t_axis, data, color=color, linewidth=0.8)
            axes[series_idx, c].tick_params(labelsize=6)
            if series_idx < n_rows - 1:
                axes[series_idx, c].set_xticklabels([])

        series_idx += 1

    plt.suptitle("Synthetic Time Series: Original vs Transforms",
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    out_path = Path(__file__).parent / "transform_visualizations.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
