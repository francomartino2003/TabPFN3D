"""
Experiment 01: Latent m + Positional Encoding + Random Causal Conv → 1D series.

Pipeline:
  1. Sample m ~ N(0, 1) of dimension d
  2. Replicate m across T time steps → (d, T)
  3. Add sinusoidal positional encoding:
       PE(t, 2i)   = sin(t / 10000^(2i/d))
       PE(t, 2i+1) = cos(t / 10000^(2i/d))
  4. Apply a random causal 1D convolution (d channels → 1 channel)
     and a random activation (both fixed by seed).
  5. Output: 1×T series.

Visualisation: 5 series from the same generator (same kernel + activation),
each with a different m vector.

Usage:
  python exp01_pe_causal_conv.py [--d 8] [--T 200] [--kernel-size 5] [--seed 42] [--n-series 5]
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


# ── Positional Encoding ────────────────────────────────────────────────────────

def positional_encoding(T: int, d: int) -> np.ndarray:
    """
    Standard sinusoidal positional encoding.

    Returns: (d, T) array where
        row 2i   = sin(t / 10000^(2i/d))
        row 2i+1 = cos(t / 10000^(2i/d))
    for t = 0, 1, …, T-1.
    """
    pe = np.zeros((d, T))
    t = np.arange(T, dtype=np.float64)          # (T,)
    for i in range(d // 2):
        freq = 1.0 / (10000.0 ** (2.0 * i / d))
        pe[2 * i, :]     = np.sin(t * freq)
        pe[2 * i + 1, :] = np.cos(t * freq)
    # If d is odd, last dimension gets one more sin
    if d % 2 == 1:
        i = d // 2
        freq = 1.0 / (10000.0 ** (2.0 * i / d))
        pe[d - 1, :] = np.sin(t * freq)
    return pe


# ── Random Causal Convolution ──────────────────────────────────────────────────

def build_causal_conv(d_in: int, kernel_size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample a random causal 1D convolution kernel.

    Returns: (d_in, kernel_size) array.
    The convolution maps d_in channels → 1 channel by convolving each input
    channel with its own 1D kernel and summing.  "Causal" means the kernel
    only looks at the current and past time steps.
    """
    # Xavier-like initialisation
    std = np.sqrt(2.0 / (d_in * kernel_size))
    kernel = rng.normal(0, std, size=(d_in, kernel_size))
    return kernel


# ── Activation ─────────────────────────────────────────────────────────────────

ACTIVATION_CHOICES = (
    'identity',   # f(x) = x
    'log',        # f(x) = sign(x) * log(1 + |x|)
    'sigmoid',    # f(x) = 1/(1+exp(-x))
    'abs',        # f(x) = |x|
    'sin',        # f(x) = sin(x)
    'tanh',       # f(x) = tanh(x)
    'square',     # f(x) = x^2
    'power',      # f(x) = sign(x) * |x|^0.5
    'softplus',   # f(x) = log(1 + exp(x)) - smooth ReLU
    'step',       # f(x) = 0 if x < 0 else 1
    'modulo',     # f(x) = x mod 1
)


def apply_activation(z: np.ndarray, name: str) -> np.ndarray:
    """Apply activation element-wise. z: (T,) or (n, T)."""
    if name == 'identity':
        return z
    if name == 'log':
        return np.sign(z) * np.log1p(np.abs(z))
    if name == 'sigmoid':
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    if name == 'abs':
        return np.abs(z)
    if name == 'sin':
        return np.sin(z)
    if name == 'tanh':
        return np.tanh(z)
    if name == 'square':
        return z ** 2
    if name == 'power':
        return np.sign(z) * np.sqrt(np.abs(z))
    if name == 'softplus':
        return np.log1p(np.exp(np.clip(z, -500, 500)))
    if name == 'step':
        return (z >= 0).astype(z.dtype)
    if name == 'modulo':
        return np.mod(z, 1.0)
    return z


def apply_causal_conv(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply causal convolution with no padding (valid).

    Input x: (d_in, L) with L >= K. Output: (L - K + 1,) so that
    y[t] = sum over c,k of kernel[c,k] * x[c, t+k] (causal: only past and current).
    Caller must pass x of length T + K - 1 to get output length T.
    """
    d_in, L = x.shape
    K = kernel.shape[1]
    out_len = L - K + 1
    y = np.zeros(out_len)
    for t in range(out_len):
        y[t] = np.sum(kernel * x[:, t : t + K])
    return y


# ── Generator ─────────────────────────────────────────────────────────────────

class PECausalConvGenerator:
    """
    A generator defined by (d, T, kernel, activation).

    Given a seed the kernel and activation are fixed.
    Each call to `sample()` draws a new m ~ N(0,1) and produces a 1×T series.
    """

    def __init__(self, d: int, T: int, kernel_size: int, seed: int):
        self.d = d
        self.T = T
        self.kernel_size = kernel_size
        self.seed = seed

        # Fixed parts (kernel + activation sampled once)
        gen_rng = np.random.default_rng(seed)
        self.kernel = build_causal_conv(d, kernel_size, gen_rng)   # (d, K)
        self.activation = gen_rng.choice(ACTIVATION_CHOICES)
        T_in = T + kernel_size - 1  # replicate m/PE this long so conv needs no padding
        self.pe = positional_encoding(T_in, d)                     # (d, T_in)

        # Separate RNG for sampling m (not consumed by kernel/activation)
        self.sample_rng = np.random.default_rng(gen_rng.integers(0, 2**62))

    def sample(self) -> np.ndarray:
        """Sample one series of length T."""
        # Replicate m (and PE) for T + K - 1 steps so causal conv needs no padding
        T_in = self.T + self.kernel_size - 1
        m = self.sample_rng.normal(0, 1, size=(self.d,))       # (d,)
        m_rep = np.tile(m[:, None], (1, T_in))                  # (d, T_in)
        x = m_rep + self.pe                                     # (d, T_in)
        y = apply_causal_conv(x, self.kernel)                   # (T,) = (T_in - K + 1,)
        y = apply_activation(y, self.activation)
        return y

    def sample_batch(self, n: int) -> np.ndarray:
        """Sample n series. Returns (n, T)."""
        return np.stack([self.sample() for _ in range(n)], axis=0)


# ── Visualisation ──────────────────────────────────────────────────────────────

def visualize(series: np.ndarray, d: int, T: int, kernel_size: int, seed: int,
              activation: str, save_path: str | None = None):
    """
    Plot n overlaid series and save to PNG (no display).
    series: (n, T)
    """
    n = series.shape[0]
    fig, ax = plt.subplots(figsize=(12, 4))
    cmap = plt.cm.tab10
    for i in range(n):
        ax.plot(series[i], color=cmap(i % 10), alpha=0.8, linewidth=1.2,
                label=f'series {i}')
    ax.set_xlabel('t')
    ax.set_ylabel('y(t)')
    ax.set_title(
        f'PE + Causal Conv  |  d={d}, T={T}, K={kernel_size}, act={activation}, '
        f'seed={seed}  |  {n} series'
    )
    ax.legend(fontsize=8, ncol=n, loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Exp 01: PE + Causal Conv generator')
    parser.add_argument('--d', type=int, default=8, help='Latent dimension d')
    parser.add_argument('--T', type=int, default=200, help='Time steps')
    parser.add_argument('--kernel-size', type=int, default=5, help='Causal conv kernel size')
    parser.add_argument('--seed', type=int, default=42, help='Generator seed (defines the conv)')
    parser.add_argument('--n-series', type=int, default=5, help='Number of series to sample')
    args = parser.parse_args()

    gen = PECausalConvGenerator(d=args.d, T=args.T, kernel_size=args.kernel_size,
                                seed=args.seed)

    series = gen.sample_batch(args.n_series)  # (n, T)

    save_path = os.path.join(os.path.dirname(__file__), 'results',
                             f'exp01_d{args.d}_K{args.kernel_size}_seed{args.seed}.png')
    visualize(series, d=args.d, T=args.T, kernel_size=args.kernel_size,
              seed=args.seed, activation=gen.activation, save_path=save_path)


if __name__ == '__main__':
    main()
