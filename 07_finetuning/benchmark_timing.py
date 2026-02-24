#!/usr/bin/env python3
"""
Benchmark timing for finetuning components on engaging.

Measures:
  1. Batch generation: N/4 unique synthetic datasets + 3 augmented copies each
     → total batch_size datasets (e.g. 24 orig + 72 aug = 96)
  2. Synthetic eval: forward pass on 2 fixed batches of batch_size each
  3. Real eval: forward pass on all real datasets (~74)

Usage:
    python benchmark_timing.py --batch-size 96 --n-repeats 5
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / '12_kernel_dag_generator'))
sys.path.insert(0, str(Path(__file__).parent.parent / '00_TabPFN' / 'src'))

import numpy as np
import torch

from finetune_tabpfn_v3 import (
    FinetuneConfig,
    SyntheticDataGenerator,
    augment_dataset,
    load_real_datasets,
    TabPFNTemporalFineTuner,
)


def main():
    parser = argparse.ArgumentParser(description='Benchmark finetuning timing')
    parser.add_argument('--batch-size', type=int, default=96,
                        help='Batch size (N/4 orig + 3*N/4 aug)')
    parser.add_argument('--n-repeats', type=int, default=5,
                        help='Number of repeats for each benchmark')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    config = FinetuneConfig(batch_size=args.batch_size)
    n_unique = args.batch_size // 4

    print("=" * 60)
    print("BENCHMARK TIMING — Finetuning components")
    print("=" * 60)
    print(f"  Device:     {device}")
    print(f"  Batch size: {args.batch_size}  (n_unique={n_unique}, +3 aug each)")
    print(f"  Repeats:    {args.n_repeats}")
    print()

    # ── 1. Batch generation ───────────────────────────────────────────────
    print("[1] Batch generation (N/4 orig + 3 aug each)")
    synth_gen = SyntheticDataGenerator(config, seed=42)
    aug_rng = np.random.RandomState(42 + 7777)

    gen_times = []
    for rep in range(args.n_repeats):
        t0 = time.perf_counter()
        originals = []
        for _ in range(n_unique):
            try:
                originals.append(synth_gen.generate_one())
            except Exception:
                pass
        batch = list(originals)
        for orig in originals:
            for _ in range(3):
                try:
                    batch.append(augment_dataset(orig, aug_rng))
                except Exception:
                    batch.append(orig)
        dt = time.perf_counter() - t0
        gen_times.append(dt)
        print(f"    rep {rep+1}: {dt:.2f}s  ({len(batch)} datasets)")

    gen_mean = np.mean(gen_times)
    gen_std = np.std(gen_times)
    print(f"  → Mean: {gen_mean:.2f}s ± {gen_std:.2f}s  per batch of {args.batch_size}")
    print()

    # ── 2. Build model + synth eval batches ───────────────────────────────
    print("[2] Building model and synth eval batches...")
    trainer = TabPFNTemporalFineTuner(config)
    trainer.model.eval()

    synth_eval_rng = np.random.RandomState(config.seed + 9999)
    synth_eval_gen = SyntheticDataGenerator(config, seed=config.seed + 5000)
    synth_eval_batches = []
    for _ in range(2):
        originals_eval = []
        for _ in range(n_unique):
            try:
                originals_eval.append(synth_eval_gen.generate_one())
            except Exception:
                pass
        eval_batch = list(originals_eval)
        for orig in originals_eval:
            for _ in range(3):
                try:
                    eval_batch.append(augment_dataset(orig, synth_eval_rng))
                except Exception:
                    eval_batch.append(orig)
        synth_eval_batches.append(eval_batch)
    n_synth_eval = sum(len(b) for b in synth_eval_batches)
    print(f"  Synth eval: 2 batches × ~{args.batch_size} = {n_synth_eval} datasets")
    print()

    # ── 3. Synthetic eval (forward only) ───────────────────────────────────
    print("[3] Synthetic eval (forward pass, no grad)")
    synth_times = []
    for rep in range(args.n_repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = trainer.evaluate_synthetic(synth_eval_batches)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        synth_times.append(dt)
        print(f"    rep {rep+1}: {dt:.2f}s")

    synth_mean = np.mean(synth_times)
    synth_std = np.std(synth_times)
    print(f"  → Mean: {synth_mean:.2f}s ± {synth_std:.2f}s  "
          f"({n_synth_eval} datasets, ~{synth_mean/n_synth_eval*1000:.1f}ms/dataset)")
    print()

    # ── 4. Real eval ───────────────────────────────────────────────────────
    print("[4] Real eval (forward pass on AEON datasets)")
    real_datasets = load_real_datasets(config)
    n_real = len(real_datasets)
    print(f"  {n_real} real datasets")

    real_times = []
    for rep in range(args.n_repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = trainer.evaluate_all(real_datasets)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        real_times.append(dt)
        print(f"    rep {rep+1}: {dt:.2f}s")

    real_mean = np.mean(real_times)
    real_std = np.std(real_times)
    print(f"  → Mean: {real_mean:.2f}s ± {real_std:.2f}s  "
          f"({n_real} datasets, ~{real_mean/n_real*1000:.1f}ms/dataset)")
    print()

    # ── Summary ───────────────────────────────────────────────────────────
    print("=" * 60)
    print("SUMMARY (per step)")
    print("=" * 60)
    print(f"  Batch generation:  {gen_mean:.2f}s ± {gen_std:.2f}s")
    print(f"  Synth eval (2×{args.batch_size}): {synth_mean:.2f}s ± {synth_std:.2f}s")
    print(f"  Real eval ({n_real} ds):  {real_mean:.2f}s ± {real_std:.2f}s")
    print()
    print("  Per training step (gen + train + synth_eval):")
    print(f"    ~{gen_mean:.1f}s gen + [train] + {synth_mean:.1f}s synth_eval")
    print("  Per eval_every step (+ real eval):")
    print(f"    + {real_mean:.1f}s real_eval")
    print()


if __name__ == "__main__":
    main()
