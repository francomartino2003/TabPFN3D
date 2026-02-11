#!/usr/bin/env python3
"""
Generate univariate time series from a latent vector + positional encoding.

Process:
  1. Sample latent m ~ Normal(0, std) or Uniform(-a, a)  [dim d]
  2. Sample weights w (d,) and bias b (Xavier-style)
  3. For each t = 0, 1, ..., T-1:
       PE(t, 2i)   = sin(t / 10000^(2i/d))
       PE(t, 2i+1) = cos(t / 10000^(2i/d))
       m_t = m + PE(t)
       x_t = activation(w @ m_t + b)   # activation sampled at random
  4. Series is (x_0, x_1, ..., x_{T-1}); visualize.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Same activations as random_nn_generator, except rank (identity, log, sigmoid, abs, sin, tanh, square, power, softplus, step, modulo)
ACTIVATION_CHOICES = (
    "identity",
    "log",
    "sigmoid",
    "abs",
    "sin",
    "tanh",
    "square",
    "power",
    "softplus",
    "step",
    "modulo",
)


def apply_activation(x: np.ndarray, act: str) -> np.ndarray:
    """Apply activation to 1d array x (element-wise, except rank which uses full series)."""
    if act == "identity":
        return x
    if act == "log":
        return np.sign(x) * np.log1p(np.abs(x))
    if act == "sigmoid":
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    if act == "abs":
        return np.abs(x)
    if act == "sin":
        return np.sin(x)
    if act == "tanh":
        return np.tanh(x)
    if act == "square":
        return x ** 2
    if act == "power":
        return np.sign(x) * np.sqrt(np.abs(x))
    if act == "softplus":
        return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -500, 20))))
    if act == "step":
        return np.where(x >= 0, 1.0, 0.0)
    if act == "modulo":
        return np.mod(x, 1.0)
    raise ValueError(f"Unknown activation: {act!r}")


def positional_encoding(t: int, d_model: int) -> np.ndarray:
    """
    Standard sinusoidal PE of dimension d_model at time t.

    PE(t, 2i)   = sin(t / 10000^(2i/d_model))
    PE(t, 2i+1) = cos(t / 10000^(2i/d_model))

    Returns:
        array of shape (d_model,)
    """
    pe = np.zeros(d_model)
    for i in range(d_model):
        # PE(t, 2k) = sin(t/10000^(2k/d)), PE(t, 2k+1) = cos(t/10000^(2k/d))
        if i % 2 == 0:
            exp = i / d_model  # 2k/d with k = i/2 -> 2*(i/2)/d = i/d
            pe[i] = np.sin(t / (10000 ** exp))
        else:
            exp = (i - 1) / d_model
            pe[i] = np.cos(t / (10000 ** exp))
    return pe


def sample_latent(d: int, init: str, rng: np.random.Generator, *, std: float = 1.0, a: float = 1.0) -> np.ndarray:
    """Sample initial latent m of dimension d."""
    if init == "normal":
        return rng.normal(0, std, size=d).astype(np.float64)
    elif init == "uniform":
        return rng.uniform(-a, a, size=d).astype(np.float64)
    raise ValueError(f"init must be 'normal' or 'uniform', got {init!r}")


def sample_weights_xavier(d: int, rng: np.random.Generator, gain: float = 1.0) -> tuple[np.ndarray, float]:
    """
    Sample w (d,) and b for x_t = w @ m_t + b.
    Xavier: Var(out) ~ Var(in). Linear layer 1 output, d inputs -> std = gain * sqrt(2/(d+1)).
    Bias often 0 or small.
    """
    std = gain * np.sqrt(2.0 / (d + 1))
    w = rng.normal(0, std, size=d).astype(np.float64)
    b = float(rng.normal(0, std))  # same scale as one weight
    return w, b


def generate_series(
    d: int,
    T: int,
    latent_init: str,
    rng: np.random.Generator,
    *,
    std: float = 1.0,
    a: float = 1.0,
    gain: float = 1.0,
    activation: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Generate the time series and return x, latent trajectory, w, b, activation name.

    If activation is None, one is sampled at random from ACTIVATION_CHOICES.

    Returns:
        x: (T,) univariate series (after activation)
        m_trajectory: (T, d) latent over time
        w: (d,) weights
        b: scalar bias
        act_name: name of the activation used
    """
    m_init = sample_latent(d, latent_init, rng, std=std, a=a)
    w, b = sample_weights_xavier(d, rng, gain=gain)
    act_name = activation if activation is not None else rng.choice(ACTIVATION_CHOICES).item()

    m_trajectory = np.zeros((T, d))
    x_raw = np.zeros(T)

    for t in range(T):
        pe_t = positional_encoding(t, d)
        m_t = m_init + pe_t
        m_trajectory[t] = m_t
        x_raw[t] = np.dot(w, m_t) + b

    x = apply_activation(x_raw, act_name)
    return x, m_trajectory, w, b, act_name


def generate_series_fixed_weights(
    d: int,
    T: int,
    m_init: np.ndarray,
    w: np.ndarray,
    b: float,
    act_name: str,
) -> np.ndarray:
    """
    Generate one series with fixed w, b, activation; only the latent m_init varies.
    Used when comparing effect of different latents with same linear combo.
    """
    x_raw = np.zeros(T)
    for t in range(T):
        pe_t = positional_encoding(t, d)
        m_t = m_init + pe_t
        x_raw[t] = np.dot(w, m_t) + b
    return apply_activation(x_raw, act_name)


def main():
    p = argparse.ArgumentParser(description="Generate PE-based time series")
    p.add_argument("--d", type=int, default=16, help="Latent dimension (d_model)")
    p.add_argument("--T", type=int, default=200, help="Time series length")
    p.add_argument("--init", choices=("normal", "uniform"), default="normal", help="Latent init")
    p.add_argument("--std", type=float, default=0.5, help="Std for normal init")
    p.add_argument("--a", type=float, default=1.0, help="Half-range for uniform init (-a, a)")
    p.add_argument("--gain", type=float, default=1.0, help="Xavier gain for w,b")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=None, help="Output figure path")
    p.add_argument("--show", action="store_true", help="Show plot interactively")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    x, m_traj, w, b, act_name = generate_series(
        d=args.d,
        T=args.T,
        latent_init=args.init,
        rng=rng,
        std=args.std,
        a=args.a,
        gain=args.gain,
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    t = np.arange(args.T)

    axes[0].plot(t, x, color="tab:blue", linewidth=1.2)
    axes[0].set_ylabel("$x_t$")
    axes[0].set_title(f"Series: act(w$^\\top$m_t+b)  (d={args.d}, T={args.T}, init={args.init}, act={act_name})")
    axes[0].grid(True, alpha=0.3)

    # Plot first few latent dimensions
    n_show = min(4, args.d)
    for i in range(n_show):
        axes[1].plot(t, m_traj[:, i], label=f"$m_t[{i}]$", alpha=0.8)
    axes[1].set_xlabel("$t$")
    axes[1].set_ylabel("latent dims")
    axes[1].set_title("Latent trajectory (first {} dims)".format(n_show))
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print("Saved:", out_path)
    if args.show:
        plt.show()
    else:
        plt.close()

    print(f"d={args.d} T={args.T} init={args.init} act={act_name} -> x range [{x.min():.3f}, {x.max():.3f}]")


if __name__ == "__main__":
    main()
