#!/usr/bin/env python3
"""
Visualize many PE-based series with the same hyperparameters:
each series uses a different sampled latent m and different linear weights w, b.

Use --d-values to generate one PNG per latent dimension d (T fixed at 500).
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from generate_pe_series import generate_series

T_FIXED = 500


def run_one(d: int, T: int, n: int, cols: int, out_path: Path, rng: np.random.Generator,
            init: str, std: float, a: float, gain: float) -> None:
    """Generate n series with given d, T and save to out_path."""
    cols = min(cols, n)
    rows = (n + cols - 1) // cols

    series_list = []  # (x, act_name)
    for _ in range(n):
        x, _, _, _, act_name = generate_series(
            d=d,
            T=T,
            latent_init=init,
            rng=rng,
            std=std,
            a=a,
            gain=gain,
        )
        series_list.append((x, act_name))

    t = np.arange(T)
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 1.8 * rows), sharex=True, sharey=False)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (x, act_name) in enumerate(series_list):
        r, c = idx // cols, idx % cols
        ax = axes[r, c]
        ax.plot(t, x, color="tab:blue", linewidth=0.9, alpha=0.9)
        ax.set_title(act_name, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(n, rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].set_visible(False)

    fig.suptitle(
        f"PE-based series (d={d}, T={T}, init={init}) — {n} series, random act each",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)


def main():
    p = argparse.ArgumentParser(description="Generate many PE series (same hyperparams, different m, w, b)")
    p.add_argument("--d", type=int, default=16, help="Latent dimension (used if --d-values not set)")
    p.add_argument("--d-values", type=int, nargs="+", default=None,
                   help="One PNG per d (T fixed at 500). e.g. --d-values 4 8 16 32 64")
    p.add_argument("--T", type=int, default=200, help="Time series length (ignored if --d-values set)")
    p.add_argument("--n", type=int, default=24, help="Number of series to generate")
    p.add_argument("--init", choices=("normal", "uniform"), default="normal")
    p.add_argument("--std", type=float, default=0.5)
    p.add_argument("--a", type=float, default=1.0)
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="results/many_series.png",
                   help="Output path (single d) or directory prefix for --d-values (e.g. results/many_series)")
    p.add_argument("--cols", type=int, default=6, help="Number of columns in grid")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out).parent
    out_stem = Path(args.out).stem

    if args.d_values is not None:
        T = T_FIXED
        for d in args.d_values:
            path = out_dir / f"{out_stem}_d{d}.png"
            run_one(
                d=d,
                T=T,
                n=args.n,
                cols=args.cols,
                out_path=path,
                rng=rng,
                init=args.init,
                std=args.std,
                a=args.a,
                gain=args.gain,
            )
        return

    run_one(
        d=args.d,
        T=args.T,
        n=args.n,
        cols=args.cols,
        out_path=Path(args.out),
        rng=rng,
        init=args.init,
        std=args.std,
        a=args.a,
        gain=args.gain,
    )


if __name__ == "__main__":
    main()
