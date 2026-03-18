"""
Training configuration, synthetic data generation, and real dataset loading.

SyntheticDataGenerator wraps the DAG generator from 02_synthetic_data
and produces flattened (n, m*T) training/test splits that are ready for
the overlap model.

load_real_datasets() reads from the new NPZ format written by
01_real_data/download.py (data/{ucr,uea}/<Name>_{train,test}.npz) and
uses datasets_summary.csv for metadata-based filtering.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.preprocessing import LabelEncoder

# Paths relative to project root
HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parent
REAL_DATA_ROOT = PROJECT_ROOT / "01_real_data"

sys.path.insert(0, str(PROJECT_ROOT / "02_synthetic_data"))
sys.path.insert(0, str(PROJECT_ROOT / "00_TabPFN" / "src"))

from model import pad_to_group   # noqa: E402  (model.py in same folder)


# ─────────────────────────────────────────────────────────────────────────────
# Training configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FinetuneConfig:
    """Hyper-parameters and paths for fine-tuning the overlap model."""

    # Optimiser
    lr: float = 7e-5
    lr_min: float = 1e-7
    warmup_steps: int = 10
    weight_decay: float = 1e-4
    batch_size: int = 128
    n_steps: int = 400
    grad_clip: float = 1.0

    # Evaluation
    eval_every: int = 1

    # Data constraints — consistent with PFN filter (m*T<=2000, labels<=10).
    # Must match worker_generator.py defaults so evaluator synthetic data
    # has the same scale as training data.
    max_samples: int = 10000
    max_T: int = 2000
    max_m: int = 200
    max_m_times_T: int = 2000
    max_classes: int = 10
    group_size: int = 8

    # Dual-LR: pretrained lr = fresh_lr / encoder_lr_mult (50% → mult=2)
    encoder_lr_mult: float = 2.0
    n_fresh_transformer_layers: int = 4

    # Checkpoint / log directories
    run_name: str = "default"
    checkpoint_dir: str = ""
    log_dir: str = ""

    # Device and seed
    device: str = "auto"
    n_estimators_eval: int = 4
    seed: int = 42

    def __post_init__(self):
        if not self.checkpoint_dir:
            self.checkpoint_dir = f"checkpoints_v3/{self.run_name}"
        if not self.log_dir:
            self.log_dir = f"logs_v3/{self.run_name}"


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data generator
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticDataGenerator:
    """Wraps the kernel-DAG generator to produce training-ready flattened dicts."""

    def __init__(self, config: FinetuneConfig, seed: int = 42):
        from hyperparameters import GeneratorHyperparameters  # from 02_synthetic_data

        self.config = config
        self.seed = seed
        self.hp = GeneratorHyperparameters()
        self.dataset_count = 0

    def generate_one(self, max_retries: int = 20) -> Dict[str, Any]:
        """Generate one valid synthetic dataset.

        Returns dict with keys:
            X_train, X_test  — (n, m*T_padded) float32
            y_train, y_test  — (n,) int64
            n_classes        — int
            n_features       — int  (= m * T_padded, flat size)
            n_features_orig  — int  (= m, original channel count)
            T                — int  (padded timesteps)
            n_samples        — int  (train + test)
        """
        from generator import DatasetGenerator  # from 02_synthetic_data

        for retry in range(max_retries):
            self.dataset_count += 1
            seed = self.seed + self.dataset_count * 100 + retry
            try:
                gen = DatasetGenerator(seed=seed, hp=self.hp)
                ds = gen.generate_dataset()
                if ds is None:
                    continue

                X_tr_3d = ds["X_train"]   # (n_train, m, T)
                X_te_3d = ds["X_test"]
                y_tr = ds["y_train"]
                y_te = ds["y_test"]
                n_classes = ds["n_classes"]
                m = ds["n_features"]
                T = ds["T"]

                if len(y_tr) == 0 or len(y_te) == 0:
                    continue
                if not (2 <= n_classes <= self.config.max_classes):
                    continue

                cfg = self.config
                if T > cfg.max_T or m > cfg.max_m or m * T > cfg.max_m_times_T:
                    continue

                # Replace inf/-inf before casting to float32 (overflow otherwise)
                X_tr_3d = np.where(np.isinf(X_tr_3d), np.nan, X_tr_3d)
                X_te_3d = np.where(np.isinf(X_te_3d), np.nan, X_te_3d)

                X_tr = X_tr_3d.reshape(X_tr_3d.shape[0], -1).astype(np.float32)
                X_te = X_te_3d.reshape(X_te_3d.shape[0], -1).astype(np.float32)

                X_tr, T_pad = pad_to_group(X_tr, m, T, group_size=cfg.group_size)
                X_te, _ = pad_to_group(X_te, m, T, group_size=cfg.group_size)
                T = T_pad

                X_tr = np.clip(np.nan_to_num(X_tr, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)
                X_te = np.clip(np.nan_to_num(X_te, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)

                if np.all(np.std(X_tr, axis=0) < 1e-8):
                    continue

                return {
                    "X_train": X_tr,
                    "X_test": X_te,
                    "y_train": y_tr.astype(np.int64),
                    "y_test": y_te.astype(np.int64),
                    "n_classes": n_classes,
                    "n_features": X_tr.shape[1],
                    "n_features_orig": m,
                    "T": T,
                    "n_samples": X_tr.shape[0] + X_te.shape[0],
                }
            except Exception as e:
                if retry == max_retries - 1:
                    print(f"  [Generator] failed after {max_retries} retries: {e}", flush=True)
                continue

        raise RuntimeError(f"Could not generate valid dataset after {max_retries} retries")


# ─────────────────────────────────────────────────────────────────────────────
# Real dataset loading (NPZ format from 01_real_data/download.py)
# ─────────────────────────────────────────────────────────────────────────────

SUBSAMPLE_N = 1_000
SUBSAMPLE_SEED = 0   # fixed seed for reproducible subsampling


def _subsample_train(X_tr_3d, y_tr_raw, n_train: int):
    """Subsample train to SUBSAMPLE_N with fixed seed (reproducible)."""
    rng = np.random.RandomState(SUBSAMPLE_SEED)
    idx = rng.choice(n_train, SUBSAMPLE_N, replace=False)
    idx.sort()
    return X_tr_3d[idx], y_tr_raw[idx]


def load_real_datasets(config: FinetuneConfig) -> List[Dict[str, Any]]:
    """Load real datasets from 01_real_data/data/{ucr,uea}/ NPZ files.

    Applies the same PFN filter constraints as FinetuneConfig.
    Skips datasets with is_variable_length or has_missings (the evaluator
    during training uses simple flatten; variable-length data needs
    special handling in final_evaluation.py).

    Returns a list of dicts with the same schema as SyntheticDataGenerator.generate_one().
    """
    import pandas as pd

    summary_path = REAL_DATA_ROOT / "datasets_summary.csv"
    if not summary_path.exists():
        print(f"[load_real_datasets] datasets_summary.csv not found at {summary_path}")
        return []

    summary = pd.read_csv(summary_path)
    eligible = summary[
        summary["passes_pfn_filters"]
        & ~summary["is_variable_length"]
        & ~summary["has_missings"]
    ]

    valid: List[Dict[str, Any]] = []
    cfg = config

    for _, row in eligible.iterrows():
        name = row["dataset"]
        collection = row["collection"].lower()   # "ucr" or "uea"
        data_dir = REAL_DATA_ROOT / "data" / collection

        tr_path = data_dir / f"{name}_train.npz"
        te_path = data_dir / f"{name}_test.npz"
        if not tr_path.exists() or not te_path.exists():
            continue

        try:
            tr = np.load(tr_path, allow_pickle=False)
            te = np.load(te_path, allow_pickle=False)
            X_tr_3d = tr["X"].astype(np.float32)   # (n_train, m, T)
            y_tr_raw = tr["y"]
            X_te_3d = te["X"].astype(np.float32)
            y_te_raw = te["y"]

            n_tr, m, T = X_tr_3d.shape
            n_te = X_te_3d.shape[0]

            # Subsample train to SUBSAMPLE_N if flagged
            if bool(row.get("subsample_train", False)) and n_tr >= SUBSAMPLE_N:
                X_tr_3d, y_tr_raw = _subsample_train(X_tr_3d, y_tr_raw, n_tr)
                n_tr = SUBSAMPLE_N

            n_total = n_tr + n_te

            if T > cfg.max_T or m > cfg.max_m or m * T > cfg.max_m_times_T:
                continue

            n_classes = int(row["n_classes"])
            if not (2 <= n_classes <= cfg.max_classes):
                continue

            # Flatten and pad
            X_tr = X_tr_3d.reshape(n_tr, -1)
            X_te = X_te_3d.reshape(n_te, -1)
            X_tr, T_pad = pad_to_group(X_tr, m, T, group_size=cfg.group_size)
            X_te, _ = pad_to_group(X_te, m, T, group_size=cfg.group_size)
            T = T_pad

            # Encode labels
            le = LabelEncoder()
            le.fit(y_tr_raw)
            y_tr = le.transform(y_tr_raw).astype(np.int64)
            y_te = le.transform(y_te_raw).astype(np.int64)

            valid.append({
                "name": name,
                "X_train": X_tr,
                "X_test": X_te,
                "y_train": y_tr,
                "y_test": y_te,
                "n_classes": n_classes,
                "n_features": X_tr.shape[1],
                "n_features_orig": m,
                "T": T,
                "n_samples": n_total,
            })
        except Exception as e:
            print(f"  [load_real_datasets] skipping {name}: {e}")
            continue

    return valid


def load_all_pfn_datasets(group_size: int = 8) -> List[Dict[str, Any]]:
    """Load ALL PFN-eligible datasets from datasets_summary.csv.

    - Includes variable-length and missing-value datasets (NaN passed through).
    - No FinetuneConfig size constraints — only passes_pfn_filters gate.
    - Subsamples train to SUBSAMPLE_N if subsample_train is True.

    Used by worker_evaluator_v2 to evaluate on the full real-data benchmark.

    Returns a list of dicts with the same schema as SyntheticDataGenerator.generate_one().
    """
    import pandas as pd

    summary_path = REAL_DATA_ROOT / "datasets_summary.csv"
    if not summary_path.exists():
        print(f"[load_all_pfn_datasets] datasets_summary.csv not found at {summary_path}")
        return []

    summary = pd.read_csv(summary_path)
    eligible = summary[summary["passes_pfn_filters"] == True]

    valid: List[Dict[str, Any]] = []

    for _, row in eligible.iterrows():
        name = row["dataset"]
        collection = row["collection"].lower()
        data_dir = REAL_DATA_ROOT / "data" / collection

        tr_path = data_dir / f"{name}_train.npz"
        te_path = data_dir / f"{name}_test.npz"
        if not tr_path.exists() or not te_path.exists():
            continue

        try:
            tr = np.load(tr_path, allow_pickle=False)
            te = np.load(te_path, allow_pickle=False)
            X_tr_3d = tr["X"].astype(np.float32)   # (n_train, m, T)
            y_tr_raw = tr["y"]
            X_te_3d = te["X"].astype(np.float32)
            y_te_raw = te["y"]

            n_tr, m, T = X_tr_3d.shape
            n_te = X_te_3d.shape[0]

            n_classes = int(row["n_classes"])
            if not (2 <= n_classes <= 10):
                continue

            # Subsample train to SUBSAMPLE_N if flagged
            if bool(row.get("subsample_train", False)) and n_tr >= SUBSAMPLE_N:
                X_tr_3d, y_tr_raw = _subsample_train(X_tr_3d, y_tr_raw, n_tr)
                n_tr = SUBSAMPLE_N

            n_total = n_tr + n_te

            # Flatten to 2D and pad T to a multiple of group_size.
            X_tr = X_tr_3d.reshape(n_tr, -1)
            X_te = X_te_3d.reshape(n_te, -1)
            X_tr, T_pad = pad_to_group(X_tr, m, T, group_size=group_size)
            X_te, _ = pad_to_group(X_te, m, T, group_size=group_size)
            T = T_pad

            # Convert inf → nan so TabPFN treats them as missing.
            np.putmask(X_tr, ~np.isfinite(X_tr), np.nan)
            np.putmask(X_te, ~np.isfinite(X_te), np.nan)

            # Encode labels contiguously from 0.
            le = LabelEncoder()
            le.fit(y_tr_raw)
            y_tr = le.transform(y_tr_raw).astype(np.int64)
            y_te = le.transform(y_te_raw).astype(np.int64)

            valid.append({
                "name": name,
                "X_train": X_tr,
                "X_test": X_te,
                "y_train": y_tr,
                "y_test": y_te,
                "n_classes": n_classes,
                "n_features": X_tr.shape[1],
                "n_features_orig": m,
                "T": T,
                "n_samples": n_total,
            })
        except Exception as e:
            print(f"  [load_all_pfn_datasets] skipping {name}: {e}")
            continue

    return valid
