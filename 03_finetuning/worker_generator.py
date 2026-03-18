"""
Worker: Dataset generator (GPU 0).

Generates batches of synthetic datasets (N/4 originals + 3N/4 augmented copies),
serialises each batch as a .npz file, and writes a DONE sentinel when finished.
Throttles output if too many unprocessed files accumulate.
"""

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "02_synthetic_data"))
sys.path.insert(0, str(Path(__file__).parent.parent / "00_TabPFN" / "src"))

from data_utils import FinetuneConfig, SyntheticDataGenerator
from augmentation import augment_dataset


def serialize_batch(batch: list, path: str) -> None:
    """Save a list of dataset dicts to a single .npz file."""
    flat: dict = {"__n__": np.array(len(batch))}
    for i, ds in enumerate(batch):
        for key in ("X_train", "X_test", "y_train", "y_test"):
            flat[f"{i}_{key}"] = ds[key]
        flat[f"{i}_n_classes"] = np.array(ds["n_classes"])
        flat[f"{i}_n_features"] = np.array(ds["n_features"])
        flat[f"{i}_n_features_orig"] = np.array(ds["n_features_orig"])
        flat[f"{i}_T"] = np.array(ds["T"])
        flat[f"{i}_n_samples"] = np.array(ds["n_samples"])
    np.savez(path, **flat)


def main() -> None:
    parser = argparse.ArgumentParser(description="Worker: generate dataset batches")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-datasets", type=int, default=2_000_000,
                        help="Stop after this many total datasets (originals + augmented)")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pending", type=int, default=50,
                        help="Throttle if more than this many .npz files are pending")
    parser.add_argument("--max-T", type=int, default=2000)
    parser.add_argument("--max-m", type=int, default=200)
    parser.add_argument("--max-m-times-T", type=int, default=2000)
    parser.add_argument("--group-size", type=int, default=8)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = FinetuneConfig(
        batch_size=args.batch_size,
        max_T=args.max_T,
        max_m=args.max_m,
        max_m_times_T=args.max_m_times_T,
        group_size=args.group_size,
    )
    synth_gen = SyntheticDataGenerator(cfg, seed=args.seed)
    aug_rng = np.random.RandomState(args.seed + 7777)

    n_unique = args.batch_size // 4
    total_datasets = 0
    batch_counter = 0

    print(
        f"[Generator] Starting: batch_size={args.batch_size}, "
        f"max_datasets={args.max_datasets}, output={out_dir}",
        flush=True,
    )

    while total_datasets < args.max_datasets:
        # Throttle: wait if too many files are pending consumption by the trainer
        while len(glob.glob(str(out_dir / "batch_*.npz"))) >= args.max_pending:
            time.sleep(5)

        t0 = time.time()
        originals = []
        for _ in range(n_unique):
            try:
                originals.append(synth_gen.generate_one())
            except Exception as e:
                print(f"  [Gen] Error: {e}", flush=True)

        if not originals:
            continue

        batch = list(originals)
        for orig in originals:
            for _ in range(3):
                try:
                    batch.append(
                        augment_dataset(
                            orig, aug_rng,
                            max_T=cfg.max_T,
                            max_m=cfg.max_m,
                            max_m_times_T=cfg.max_m_times_T,
                            group_size=cfg.group_size,
                        )
                    )
                except Exception:
                    batch.append(orig)

        fname = out_dir / f"batch_{batch_counter:08d}.npz"
        serialize_batch(batch, str(fname))

        total_datasets += len(batch)
        batch_counter += 1
        dt = time.time() - t0

        if batch_counter % 10 == 0 or batch_counter <= 3:
            print(
                f"  [Gen] batch={batch_counter} datasets={total_datasets} "
                f"({len(batch)} in {dt:.1f}s)",
                flush=True,
            )

    (out_dir / "DONE").touch()
    print(
        f"[Generator] DONE: {total_datasets} datasets in {batch_counter} batches",
        flush=True,
    )


if __name__ == "__main__":
    main()
