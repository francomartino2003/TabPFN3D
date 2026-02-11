#!/usr/bin/env python3
"""
Same as many_series but only the latent m is re-sampled; w, b, and activation are fixed.
One PNG per value of d (T=500).
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from generate_pe_series import (
    generate_series,
    generate_series_fixed_weights,
    sample_latent,
)

T_FIXED = 500


def run_one(
    d: int,
    T: int,
    n: int,
    cols: int,
    out_path: Path,
    rng: np.random.Generator,
    init: str,
    std: float,
    a: float,
    gain: float,
) -> None:
    # Sample w, b, activation ONCE (from one full generate_series call)
    _, _, w, b, act_name = generate_series(
        d=d,
        T=T,
        latent_init=init,
        rng=rng,
        std=std,
        a=a,
        gain=gain,
    )

    # Generate n series, each with a new m (same w, b, act)
    series_list = []
    for _ in range(n):
        m_init = sample_latent(d, init, rng, std=std, a=a)
        x = generate_series_fixed_weights(d, T, m_init, w, b, act_name)
        series_list.append(x)

    cols = min(cols, n)
    rows = (n + cols - 1) // cols

    t = np.arange(T)
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 1.8 * rows), sharex=True, sharey=False)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, x in enumerate(series_list):
        r, c = idx // cols, idx % cols
        ax = axes[r, c]
        ax.plot(t, x, color="tab:blue", linewidth=0.9, alpha=0.9)
        ax.set_title(f"m_{idx + 1}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(n, rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].set_visible(False)

    fig.suptitle(
        f"Only latent m varies (d={d}, T={T}, act={act_name}) — same w, b",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)


def main():
    p = argparse.ArgumentParser(description="Many series with fixed w, b, act; only m re-sampled")
    p.add_argument("--d-values", type=int, nargs="+", required=True,
                   help="One PNG per d (e.g. --d-values 4 8 16 32 64)")
    p.add_argument("--n", type=int, default=24, help="Number of series (latents) per figure")
    p.add_argument("--init", choices=("normal", "uniform"), default="normal")
    p.add_argument("--std", type=float, default=0.5)
    p.add_argument("--a", type=float, default=1.0)
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="results/vary_latent_only",
                   help="Output path prefix (e.g. results/vary_latent_only -> ..._d4.png, ..._d8.png)")
    p.add_argument("--cols", type=int, default=6)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out).parent
    out_stem = Path(args.out).stem
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


if __name__ == "__main__":
    main()
