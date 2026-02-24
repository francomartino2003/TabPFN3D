"""
Fine-tune TabPFN with Temporal Positional Encoding (V3 — fpg=8).

Key changes:
- features_per_group=8: each token = 8 consecutive timesteps from same feature
- Input encoder Linear(16->192) Xavier-initialized (pretrained used fpg=3)
- All other weights (24 transformer layers, y_encoder, decoder) pretrained
- Dual-LR optimizer: higher LR for new encoder, lower for pretrained weights
- Supports T<=1024, m<=10, m*T<=1200 (~85% of AEON datasets)
- Feature shuffle disabled (temporal order must be preserved)
- Pads T to multiple of 8 so groups align to consecutive timesteps
- Structured group embeddings: feature_emb(j) + sinusoidal_PE(group_idx)
- Synthetic data from 12_kernel_dag_generator
- BYPASSES sklearn preprocessing entirely
- Data augmentation: each batch = N/4 originals + 3*N/4 augmented copies
- Eval uses softmax temperature T=0.9 (matching TabPFN default)
- Per-dataset gradient clipping (no single outlier dataset dominates)
- Fixed synthetic eval batches (2 x batch_size) for tracking synth metrics
- LR schedule: linear warmup + cosine annealing to lr_min

Usage:
    python finetune_tabpfn_v3.py --n-steps 300 --eval-every 1
    python finetune_tabpfn_v3.py --debug
"""

import argparse
import sys
import os
import time
import json
import math
import signal
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
import pickle
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add generator to path
sys.path.insert(0, str(Path(__file__).parent.parent / '12_kernel_dag_generator'))
sys.path.insert(0, str(Path(__file__).parent.parent / '00_TabPFN' / 'src'))

from tabpfn import TabPFNClassifier
from tabpfn.preprocessing.configs import ClassifierEnsembleConfig, PreprocessorConfig

# Import kernel-DAG generator (folder 12)
from generator import DatasetGenerator
from hyperparameters import GeneratorHyperparameters

# Import temporal TabPFN wrapper
from tabpfn_temporal import (
    build_temporal_tabpfn_fpg8,
    set_temporal_info,
    clear_temporal_info,
    pad_to_group,
)

# Global for signal handling
_trainer_instance = None

# Softmax temperature for evaluation (TabPFN default calibration)
SOFTMAX_TEMPERATURE = 0.9


# ============================================================================
# Data Augmentation (feature transforms for augmented copy)
# ============================================================================

# Transform choices and parameter sampling:
#   none     — identity
#   log      — sign(x) * log(|x| + 1)
#   exp      — sign(x) * (exp(|x|) - 1), |x| clipped to 10
#   squash   — robust scaling (median/IQR) + soft clip to [-B, B]
#   kdi      — Kernel Density Integral transform; alpha ~ LogNormal(0, 0.8²)
#              clipped to [0.05, 10]; output_distribution ~ Uniform({normal, uniform})
#   kuma     — Kumaraswamy CDF warp; a, b ~ LogNormal(0, 0.7²) clipped to [0.2, 5]
#              min-max scaled to [0,1], warped, scaled back
#
# For KDI and Kumaraswamy, parameters are sampled per feature channel:
#   log(alpha) ~ N(0, 0.8²)  →  alpha ∈ ~[0.05, 10]  (median=1)
#   log(a), log(b) ~ N(0, 0.7²)  →  a,b ∈ ~[0.2, 5]  (median=1)
#
# This is data augmentation for training, NOT inference.
# Statistics are computed on the FULL dataset (train+test pooled)
# and transforms are applied consistently to the entire dataset.

AUGMENT_TRANSFORM_CHOICES = ['none', 'log', 'exp', 'squash', 'kdi', 'kuma']
# Weights: give 'none' higher weight so ~1/3 of features stay untouched
AUGMENT_TRANSFORM_WEIGHTS = np.array([3.0, 1.0, 1.0, 1.0, 1.0, 1.0])
AUGMENT_TRANSFORM_WEIGHTS = AUGMENT_TRANSFORM_WEIGHTS / AUGMENT_TRANSFORM_WEIGHTS.sum()


def _safe_log_transform(x: np.ndarray) -> np.ndarray:
    """log(|x| + 1) * sign(x) — safe for any real values."""
    return np.sign(x) * np.log1p(np.abs(x))


def _safe_exp_transform(x: np.ndarray) -> np.ndarray:
    """sign(x) * (exp(|x|) - 1), clipped to avoid overflow."""
    clipped = np.clip(np.abs(x), 0, 10)  # exp(10) ≈ 22k, safe
    return np.sign(x) * np.expm1(clipped)


def _squash_feature(X_block: np.ndarray, max_abs: float = 3.0) -> np.ndarray:
    """Robust scaling + soft clip for one feature channel (all T timesteps).

    X_block: shape (n_samples, T). Stats computed on entire block.
    """
    X = X_block.copy().astype(np.float64)

    vals = X.ravel()
    finite = vals[np.isfinite(vals)]
    if len(finite) == 0:
        return X.astype(np.float32)

    median = np.median(finite)
    q_lo = np.percentile(finite, 25.0)
    q_hi = np.percentile(finite, 75.0)

    if q_hi != q_lo:
        scale = 1.0 / (q_hi - q_lo)
    else:
        vmin, vmax = np.min(finite), np.max(finite)
        if vmax != vmin:
            scale = 2.0 / (vmax - vmin)
        else:
            return np.zeros_like(X, dtype=np.float32)

    X = (X - median) * scale
    # Soft clip: z / sqrt(1 + (z/B)^2)
    X = X / np.sqrt(1.0 + (X / max_abs) ** 2)
    return X.astype(np.float32)


def _kdi_feature(X_block: np.ndarray, alpha: float,
                 output_dist: str) -> np.ndarray:
    """KDI transform for one feature channel (all T timesteps).

    Fits on all data pooled, transforms each timestep column.
    X_block: shape (n_samples, T).
    """
    T = X_block.shape[1]
    X = X_block.copy()

    try:
        from tabpfn.preprocessing.steps.kdi_transformer import KDITransformerWithNaN
        all_vals = X.ravel().reshape(-1, 1).astype(np.float64)
        kdi = KDITransformerWithNaN(alpha=alpha, output_distribution=output_dist)
        kdi.fit(all_vals)
        for t in range(T):
            col = X[:, t:t + 1].astype(np.float64)
            out = kdi.transform(col).astype(np.float32)
            if np.all(np.isfinite(out)):
                X[:, t:t + 1] = out
    except Exception:
        pass  # keep original on failure
    return X.astype(np.float32)


def _kuma_feature(X_block: np.ndarray, a: float, b: float) -> np.ndarray:
    """Kumaraswamy warp for one feature channel (all T timesteps).

    Min-max scales to [0,1] using full data stats, applies CDF warp,
    scales back to original range.
    X_block: shape (n_samples, T).
    """
    X = X_block.copy().astype(np.float64)

    vals = X.ravel()
    finite = vals[np.isfinite(vals)]
    if len(finite) == 0:
        return X.astype(np.float32)

    xmin, xmax = np.min(finite), np.max(finite)
    if xmax <= xmin:
        return X.astype(np.float32)

    # Scale to [0, 1]
    X_norm = np.clip((X - xmin) / (xmax - xmin), 1e-12, 1.0 - 1e-12)

    # Kumaraswamy CDF: F(x) = 1 - (1 - x^a)^b
    X_warp = 1.0 - (1.0 - np.power(X_norm, a)) ** b

    # Scale back to original range
    X_out = X_warp * (xmax - xmin) + xmin

    if np.all(np.isfinite(X_out)):
        return X_out.astype(np.float32)
    return X_block.astype(np.float32)


def augment_dataset(data: dict, rng: np.random.RandomState) -> dict:
    """Create an augmented copy of a synthetic dataset.

    Augmentations applied:
    1. Feature channel permutation (permute m channels, keep T order)
    2. Class label permutation (shuffle class indices)
    3. Random per-feature-channel transform chosen from:
       none, log, exp, squash, kdi, kuma
       Transforms are applied consistently to the entire dataset
       (train+test concatenated, then split back).
    """
    m = data['n_features_orig']
    T = data['T']
    n_classes = data['n_classes']
    n_tr = data['X_train'].shape[0]

    # Concatenate train+test for consistent transforms
    X_all = np.concatenate([data['X_train'], data['X_test']], axis=0).copy()
    y_train = data['y_train'].copy()
    y_test = data['y_test'].copy()

    # 1. Feature channel permutation
    n_all = X_all.shape[0]
    feat_perm = rng.permutation(m)
    X_all = X_all.reshape(n_all, m, T)[:, feat_perm, :].reshape(n_all, m * T)

    # 2. Class label permutation
    class_perm = rng.permutation(n_classes)
    y_train = class_perm[y_train]
    y_test = class_perm[y_test]

    # 3. Random per-feature-channel transform
    for j in range(m):
        col_s = j * T
        col_e = j * T + T
        transform = rng.choice(AUGMENT_TRANSFORM_CHOICES,
                               p=AUGMENT_TRANSFORM_WEIGHTS)

        if transform == 'none':
            continue

        elif transform == 'log':
            X_all[:, col_s:col_e] = _safe_log_transform(X_all[:, col_s:col_e])

        elif transform == 'exp':
            X_all[:, col_s:col_e] = _safe_exp_transform(X_all[:, col_s:col_e])

        elif transform == 'squash':
            X_all[:, col_s:col_e] = _squash_feature(X_all[:, col_s:col_e])

        elif transform == 'kdi':
            # alpha ~ LogNormal(0, 0.8²), clipped to [0.05, 10]
            alpha = float(np.clip(np.exp(rng.randn() * 0.8), 0.05, 10.0))
            output_dist = rng.choice(['normal', 'uniform'])
            X_all[:, col_s:col_e] = _kdi_feature(
                X_all[:, col_s:col_e], alpha, output_dist)

        elif transform == 'kuma':
            # log(a), log(b) ~ N(0, 0.7²), clipped to [0.2, 5]
            a = float(np.clip(np.exp(rng.randn() * 0.7), 0.2, 5.0))
            b = float(np.clip(np.exp(rng.randn() * 0.7), 0.2, 5.0))
            X_all[:, col_s:col_e] = _kuma_feature(
                X_all[:, col_s:col_e], a, b)

    # Sanitise after transforms
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=1e6, neginf=-1e6)
    X_all = np.clip(X_all, -1e6, 1e6).astype(np.float32)

    # Split back into train and test
    return {
        'X_train': X_all[:n_tr],
        'X_test': X_all[n_tr:],
        'y_train': y_train.astype(np.int64),
        'y_test': y_test.astype(np.int64),
        'n_classes': n_classes,
        'n_features': data['n_features'],
        'n_features_orig': m,
        'T': T,
        'n_samples': data['n_samples'],
        '_augmented': True,
    }


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class FinetuneConfig:
    """Configuration for fine-tuning temporal TabPFN."""

    # Training
    lr: float = 7e-5            # Peak LR (reached after warmup)
    lr_min: float = 1e-7        # Final LR at end of cosine annealing
    warmup_steps: int = 10      # Linear warmup steps (0 → lr)
    weight_decay: float = 0.0001
    batch_size: int = 128       # Number of datasets per gradient update
    n_steps: int = 400
    grad_clip: float = 1.0
    freeze_layers: int = 0      # Freeze first N transformer layers (0 = all trainable)

    # Evaluation
    eval_every: int = 1         # Evaluate on real datasets every N steps

    # Data constraints
    max_samples: int = 1000
    max_T: int = 1024
    max_m: int = 10
    max_m_times_T: int = 1200
    max_classes: int = 10
    group_size: int = 8
    encoder_lr_mult: float = 10.0  # LR multiplier for fresh (new-init) params
    n_fresh_transformer_layers: int = 4  # first N transformer layers reinited from scratch

    # Paths (run_name allows parallel runs without overwriting)
    run_name: str = "default"
    checkpoint_dir: str = ""   # set from run_name
    log_dir: str = ""          # set from run_name
    real_data_path: str = "../01_real_data/AEON/data/classification_datasets.pkl"

    def __post_init__(self):
        if not self.checkpoint_dir:
            self.checkpoint_dir = f"checkpoints_v3/{self.run_name}"
        if not self.log_dir:
            self.log_dir = f"logs_v3/{self.run_name}"

    # Device
    device: str = "auto"

    # Estimators
    n_estimators_finetune: int = 1   # Must be 1 for finetuning
    n_estimators_eval: int = 4

    # Seed
    seed: int = 42


# ============================================================================
# Synthetic Data Generation (using 12_kernel_dag_generator)
# ============================================================================

class SyntheticDataGenerator:
    """Generates synthetic flattened datasets using the DAG generator (folder 11)."""

    def __init__(self, config: FinetuneConfig, seed: int = 42):
        self.config = config
        self.seed = seed
        self.hp = GeneratorHyperparameters()
        self.dataset_count = 0

    def generate_one(self, max_retries: int = 20) -> Dict[str, Any]:
        """Generate one synthetic dataset (flattened, train/test split).

        Returns dict with X_train, X_test, y_train, y_test, n_classes,
        n_features_orig (original m), T (timesteps), n_features (flat).
        """
        for retry in range(max_retries):
            self.dataset_count += 1
            current_seed = self.seed + self.dataset_count * 100 + retry

            try:
                gen = DatasetGenerator(seed=current_seed, hp=self.hp)
                ds = gen.generate_dataset()

                # generate_dataset returns None if < 2 classes survived
                if ds is None:
                    continue

                X_train_3d = ds['X_train']   # (n_train, n_features, T)
                X_test_3d  = ds['X_test']
                y_train    = ds['y_train']
                y_test     = ds['y_test']
                n_classes  = ds['n_classes']
                n_features_orig = ds['n_features']  # original m
                T = ds['T']                          # timesteps

                if len(y_train) == 0 or len(y_test) == 0:
                    continue

                # Constraints on raw dimensions
                if T > self.config.max_T:
                    continue
                if n_features_orig > self.config.max_m:
                    continue
                if n_features_orig * T > self.config.max_m_times_T:
                    continue
                if n_classes > self.config.max_classes or n_classes < 2:
                    continue

                # Flatten 3D -> 2D: (n, m, T) -> (n, m*T)
                n_train = X_train_3d.shape[0]
                n_test  = X_test_3d.shape[0]
                X_train = X_train_3d.reshape(n_train, -1).astype(np.float32)
                X_test  = X_test_3d.reshape(n_test, -1).astype(np.float32)

                gs = self.config.group_size
                X_train, T_padded = pad_to_group(X_train, n_features_orig, T, group_size=gs)
                X_test, _         = pad_to_group(X_test, n_features_orig, T, group_size=gs)
                T = T_padded

                # Sanitise
                X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
                X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=1e6, neginf=-1e6)
                X_train = np.clip(X_train, -1e6, 1e6)
                X_test  = np.clip(X_test,  -1e6, 1e6)

                # Skip constant datasets
                if np.all(np.std(X_train, axis=0) < 1e-8):
                    continue

                return {
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train.astype(np.int64),
                    'y_test': y_test.astype(np.int64),
                    'n_classes': n_classes,
                    'n_features': X_train.shape[1],  # flat = m*T
                    'n_features_orig': n_features_orig,  # original m
                    'T': T,
                    'n_samples': n_train + n_test,
                }

            except Exception as e:
                if retry == max_retries - 1:
                    print(f"  Generation failed after {max_retries} retries: {e}",
                          flush=True)
                continue

        raise RuntimeError(
            f"Could not generate valid dataset after {max_retries} retries")


# ============================================================================
# Real Data Loading
# ============================================================================

def load_real_datasets(config: FinetuneConfig) -> List[Dict[str, Any]]:
    """Load ALL real datasets from pickle file.

    Also extracts n_features_orig (channels) and T (length) from the 3D data.
    """
    real_data_dir = Path(__file__).resolve().parent.parent / '01_real_data'

    if str(real_data_dir) not in sys.path:
        sys.path.insert(0, str(real_data_dir))

    from src.data_loader import TimeSeriesDataset  # noqa: F401

    pkl_path = Path(__file__).parent / config.real_data_path
    if not pkl_path.exists():
        pkl_path = (Path(__file__).parent.parent
                    / '01_real_data' / 'AEON' / 'data'
                    / 'classification_datasets.pkl')

    if not pkl_path.exists():
        print(f"Warning: Real data not found at {pkl_path}")
        return []

    with open(pkl_path, 'rb') as f:
        datasets_list = pickle.load(f)

    valid = []
    for dataset in datasets_list:
        try:
            name = dataset.name
            X_train = dataset.X_train
            y_train = dataset.y_train
            X_test = dataset.X_test
            y_test = dataset.y_test

            if X_train is None or X_test is None:
                continue

            if X_train.ndim == 2:
                X_train = X_train[:, np.newaxis, :]
            if X_test.ndim == 2:
                X_test = X_test[:, np.newaxis, :]

            # Pickle is already in aeon format: (n, m_channels, T_length)
            # NO transpose needed
            n_samples = X_train.shape[0] + X_test.shape[0]
            n_channels = X_train.shape[1]  # m (original features/channels)
            T = X_train.shape[2]            # T (timesteps/length)

            if n_samples > config.max_samples:
                continue
            if T > config.max_T:
                continue
            if n_channels > config.max_m:
                continue
            if n_channels * T > config.max_m_times_T:
                continue

            n_classes = len(np.unique(y_train))
            if n_classes > config.max_classes:
                continue

            X_train_flat = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
            X_test_flat  = X_test.reshape(X_test.shape[0], -1).astype(np.float32)

            gs = config.group_size
            X_train_flat, T_padded = pad_to_group(X_train_flat, n_channels, T, group_size=gs)
            X_test_flat, _         = pad_to_group(X_test_flat, n_channels, T, group_size=gs)
            T = T_padded

            le = LabelEncoder()
            le.fit(y_train)
            y_train_enc = le.transform(y_train).astype(np.int64)
            y_test_enc  = le.transform(y_test).astype(np.int64)

            if np.any(np.isnan(X_train_flat)):
                col_means = np.nan_to_num(np.nanmean(X_train_flat, axis=0), nan=0.0)
                for i in range(X_train_flat.shape[1]):
                    X_train_flat[:, i] = np.where(
                        np.isnan(X_train_flat[:, i]), col_means[i], X_train_flat[:, i])
                    X_test_flat[:, i] = np.where(
                        np.isnan(X_test_flat[:, i]), col_means[i], X_test_flat[:, i])

            valid.append({
                'name': name,
                'X_train': X_train_flat,
                'X_test': X_test_flat,
                'y_train': y_train_enc,
                'y_test': y_test_enc,
                'n_classes': n_classes,
                'n_features': X_train_flat.shape[1],  # actual flat (after padding)
                'n_features_orig': n_channels,
                'T': T,
                'n_samples': n_samples,
            })
        except Exception:
            continue

    return valid


# ============================================================================
# TabPFN Temporal Fine-Tuner
# ============================================================================

class TabPFNTemporalFineTuner:
    """Fine-tunes TabPFN with temporal positional encoding."""

    def __init__(self, config: FinetuneConfig):
        self.config = config
        self.device = self._get_device()

        print(f"\nInitializing Temporal TabPFN fine-tuner...")
        print(f"  Device: {self.device}")
        print(f"  LR: {config.lr}  Batch: {config.batch_size}")

        # Build temporal model with fpg=8
        # fresh_params = encoder + emb_proj + first n_fresh_transformer_layers
        self.model, self.clf, fresh_params = build_temporal_tabpfn_fpg8(
            device=self.device,
            n_fresh_transformer_layers=config.n_fresh_transformer_layers)
        self.model.to(self.device)
        self.model.train()

        # Track fresh param ids for dual-LR optimizer
        fresh_param_ids = {id(p) for p in fresh_params}

        # Freeze first N transformer layers if requested
        n_total_layers = len(list(self.model.transformer_encoder.layers))
        n_freeze = min(config.freeze_layers, n_total_layers)

        for param in self.model.parameters():
            param.requires_grad = True

        if n_freeze > 0:
            for layer in list(self.model.transformer_encoder.layers)[:n_freeze]:
                for param in layer.parameters():
                    param.requires_grad = False

        n_params = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_frozen = n_params - n_trainable
        print(f"  Parameters: {n_params:,} total, {n_trainable:,} trainable, "
              f"{n_frozen:,} frozen")
        if n_freeze > 0:
            print(f"  Frozen: first {n_freeze}/{n_total_layers} transformer layers "
                  f"({100*n_frozen/n_params:.1f}%)")

        # Dual-LR optimizer: higher LR for fresh params, lower for pretrained
        pretrained_params = [p for p in self.model.parameters()
                             if p.requires_grad and id(p) not in fresh_param_ids]
        fresh_trainable = [p for p in fresh_params if p.requires_grad]

        fresh_lr = config.lr * config.encoder_lr_mult
        print(f"  Optimizer: pretrained LR={config.lr:.1e}, "
              f"fresh LR={fresh_lr:.1e} ({config.encoder_lr_mult}x)")

        self.optimizer = AdamW([
            {'params': pretrained_params, 'lr': config.lr},
            {'params': fresh_trainable,   'lr': fresh_lr},
        ], weight_decay=config.weight_decay)

        # LR schedule: linear warmup → cosine annealing to lr_min
        warmup = config.warmup_steps
        total  = config.n_steps
        lr_max = config.lr
        lr_min = config.lr_min

        def _lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)          # 0 → 1
            progress = (step - warmup) / max(1, total - warmup)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return lr_min / lr_max + (1.0 - lr_min / lr_max) * cosine

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, _lr_lambda)

        self.train_losses: List[float] = []
        self.train_accs: List[float] = []
        self.eval_results: List[List[Dict]] = []
        self.eval_steps: List[int] = []
        self.synth_eval_losses: List[float] = []
        self.synth_eval_accs: List[float] = []
        self.synth_eval_aucs: List[float] = []
        self.current_step = 0

    def _get_device(self) -> str:
        if self.config.device != "auto":
            return self.config.device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # ── Forward pass ──────────────────────────────────────────────────────

    def _make_dummy_ensemble_config(self) -> ClassifierEnsembleConfig:
        """Create a minimal ensemble config that does no preprocessing."""
        return ClassifierEnsembleConfig(
            preprocess_config=PreprocessorConfig("none", categorical_name="numeric"),
            feature_shift_count=0,
            class_permutation=None,
            add_fingerprint_feature=False,
            polynomial_features="no",
            feature_shift_decoder=None,
            subsample_ix=None,
            _model_index=0,
        )

    def forward_single_dataset(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray,
        n_classes: int,
        n_features_orig: int, T: int,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass bypassing sklearn preprocessing entirely.

        Data goes directly to the model. The model's internal encoder handles:
        - RemoveEmptyFeaturesEncoderStep (per group of 8)
        - NanHandling
        - InputNormalizationEncoderStep (z-score per group, train-only)
        - VariableNumFeaturesEncoderStep (pad + rescale per group)
        - LinearInputEncoderStep (projection to emsize=192)
        """
        try:
            y_test_t = torch.tensor(y_test, dtype=torch.long, device=self.device)

            set_temporal_info(self.model, n_features_orig, T,
                              group_size=self.config.group_size)

            # Convert to tensors — shape (1, n_samples, n_features)
            X_train_t = torch.as_tensor(
                X_train, dtype=torch.float32,
                device=self.device).unsqueeze(0)
            y_train_t = torch.as_tensor(
                y_train, dtype=torch.float32,
                device=self.device).unsqueeze(0)
            X_test_t = torch.as_tensor(
                X_test, dtype=torch.float32,
                device=self.device).unsqueeze(0)

            # Bypass sklearn: pass data directly via fit_from_preprocessed
            dummy_cfg = self._make_dummy_ensemble_config()
            self.clf.n_classes_ = n_classes
            self.clf.fit_from_preprocessed(
                [X_train_t], [y_train_t],
                cat_ix=[[[]]],          # no categorical features
                configs=[[dummy_cfg]],
            )
            logits = self.clf.forward([X_test_t], return_raw_logits=True)

            if logits.ndim == 2:
                logits_QL = logits
            elif logits.ndim == 3:
                logits_QL = logits.squeeze(1)
            elif logits.ndim == 4:
                logits_QL = logits.mean(dim=(1, 2))
            else:
                raise ValueError(f"Unexpected logits shape: {logits.shape}")

            if y_test_t.min() < 0 or y_test_t.max() >= logits_QL.shape[-1]:
                return self._skip_result()

            try:
                loss = F.cross_entropy(logits_QL, y_test_t)
            except RuntimeError:
                return self._skip_result()

            with torch.no_grad():
                acc = (logits_QL.argmax(dim=-1) == y_test_t).float().mean()

            return {'loss': loss, 'accuracy': acc, 'logits': logits_QL.detach()}

        except Exception as e:
            if getattr(self, '_n_fwd_warns', 0) < 5:
                print(f"[Skip] forward exception: {type(e).__name__}: {e}",
                      flush=True)
                self._n_fwd_warns = getattr(self, '_n_fwd_warns', 0) + 1
            return self._skip_result()

    def _skip_result(self):
        return {
            'loss': torch.tensor(0.0, device=self.device, requires_grad=False),
            'accuracy': torch.tensor(0.0),
            'logits': None,
        }

    # ── Training step ─────────────────────────────────────────────────────

    def train_step(self, batch: List[Dict]) -> Dict[str, float]:
        self.model.train()

        total_loss, total_acc, n_valid = 0.0, 0.0, 0

        # Pre-allocate gradient accumulator (per-dataset clipping)
        params = [p for p in self.model.parameters() if p.requires_grad]
        grad_accum = [torch.zeros_like(p) for p in params]

        for i, data in enumerate(batch):
            try:
                self.optimizer.zero_grad()

                out = self.forward_single_dataset(
                    data['X_train'], data['y_train'],
                    data['X_test'], data['y_test'],
                    data['n_classes'],
                    data['n_features_orig'], data['T'])

                if out['loss'].requires_grad:
                    out['loss'].backward()

                    # Per-dataset clip: no single dataset can dominate
                    if self.config.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            params, self.config.grad_clip)

                    # Accumulate clipped gradients
                    for j, p in enumerate(params):
                        if p.grad is not None:
                            grad_accum[j] += p.grad

                    total_loss += out['loss'].item()
                    total_acc += out['accuracy'].item()
                    n_valid += 1
                del out
            except Exception:
                continue

            # Free CUDA cache periodically to avoid OOM during accumulation
            if torch.cuda.is_available() and (i + 1) % 8 == 0:
                torch.cuda.empty_cache()

        if n_valid > 0:
            # Set averaged clipped gradients as final .grad
            for j, p in enumerate(params):
                p.grad = grad_accum[j] / n_valid
            self.optimizer.step()
            self.scheduler.step()

        # Free accumulator
        del grad_accum

        self.current_step += 1
        return {
            'loss': total_loss / max(1, n_valid),
            'accuracy': total_acc / max(1, n_valid),
            'n_valid': n_valid,
            'lr': self.scheduler.get_last_lr()[0],
        }

    # ── Evaluation ────────────────────────────────────────────────────────

    def evaluate_all(self, datasets: List[Dict]) -> List[Dict]:
        """Evaluate on real datasets, bypassing sklearn preprocessing.

        Reuses self.model directly (no fresh TabPFNClassifier per dataset)
        to avoid GPU memory fragmentation from repeated model loads.
        Calls the model forward pass manually instead of going through the
        classifier API.
        """
        self.model.eval()
        results = []

        for data in datasets:
            res = {'name': data['name'], 'n_classes': data['n_classes']}
            try:
                # Clear GPU cache before each dataset
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                with torch.no_grad():
                    set_temporal_info(self.model, data['n_features_orig'],
                                     data['T'],
                                     group_size=self.config.group_size)

                    # Build tensors on device
                    X_train_t = torch.as_tensor(
                        data['X_train'], dtype=torch.float32,
                        device=self.device)
                    y_train_t = torch.as_tensor(
                        data['y_train'], dtype=torch.float32,
                        device=self.device)
                    X_test_t = torch.as_tensor(
                        data['X_test'], dtype=torch.float32,
                        device=self.device)

                    # Concatenate train+test as (seq_len, 1, n_features)
                    # Model expects x=(seq_len, batch, features), y=(train_len, batch)
                    X_full = torch.cat([X_train_t, X_test_t], dim=0)  # (n_all, feat)
                    X_full = X_full.unsqueeze(1)  # (n_all, 1, feat)
                    y_in = y_train_t.unsqueeze(1)  # (n_train, 1)

                    # Direct model forward
                    output = self.model(
                        X_full, y_in,
                        only_return_standard_out=True,
                        categorical_inds=[[]],
                    )

                    # output shape: (n_test, 1, n_out) or (n_test, n_out)
                    if output.ndim == 3:
                        logits = output.squeeze(1)
                    else:
                        logits = output
                    logits = logits[:, :data['n_classes']]

                    proba = torch.softmax(logits / SOFTMAX_TEMPERATURE, dim=-1)
                    proba_np = proba.cpu().numpy()

                # Free GPU tensors
                del X_train_t, y_train_t, X_test_t, X_full, y_in
                del output, logits, proba

                preds = proba_np.argmax(axis=1)
                res['accuracy'] = float(
                    accuracy_score(data['y_test'], preds))
                try:
                    if data['n_classes'] == 2:
                        res['auc'] = float(
                            roc_auc_score(data['y_test'], proba_np[:, 1]))
                    else:
                        res['auc'] = float(
                            roc_auc_score(data['y_test'], proba_np,
                                          multi_class='ovr'))
                except Exception:
                    res['auc'] = None
                res['status'] = 'success'
            except Exception as e:
                res['accuracy'] = None
                res['auc'] = None
                res['status'] = 'failed'
                res['error'] = str(e)[:200]
                if getattr(self, '_n_eval_warns', 0) < 10:
                    print(f"  [Eval fail] {data['name']}: {type(e).__name__}: "
                          f"{str(e)[:150]}", flush=True)
                    self._n_eval_warns = getattr(self, '_n_eval_warns', 0) + 1
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            results.append(res)

        self.model.train()
        return results

    # ── Synthetic evaluation ─────────────────────────────────────────────

    def evaluate_synthetic(self, synth_batches: List[List[Dict]]) -> Dict[str, float]:
        """Evaluate on fixed synthetic batches (no_grad).

        Returns dict with mean loss, accuracy, auc across all datasets
        in all batches.
        """
        self.model.eval()
        total_loss, total_acc, n_valid = 0.0, 0.0, 0
        all_aucs = []

        for batch in synth_batches:
            for data in batch:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    with torch.no_grad():
                        out = self.forward_single_dataset(
                            data['X_train'], data['y_train'],
                            data['X_test'], data['y_test'],
                            data['n_classes'],
                            data['n_features_orig'], data['T'])

                        if out['logits'] is None:
                            del out
                            continue

                        total_loss += out['loss'].item()
                        total_acc += out['accuracy'].item()
                        n_valid += 1

                        # Compute AUC from logits
                        logits = out['logits']  # (n_test, n_classes)
                        proba = torch.softmax(
                            logits / SOFTMAX_TEMPERATURE, dim=-1
                        ).cpu().numpy()

                        y_test = data['y_test']
                        try:
                            if data['n_classes'] == 2:
                                auc = float(roc_auc_score(y_test, proba[:, 1]))
                            else:
                                auc = float(roc_auc_score(
                                    y_test, proba, multi_class='ovr'))
                            all_aucs.append(auc)
                        except Exception:
                            pass

                        del out
                except Exception:
                    continue

        self.model.train()

        n = max(1, n_valid)
        return {
            'loss': total_loss / n,
            'accuracy': total_acc / n,
            'auc': float(np.mean(all_aucs)) if all_aucs else 0.0,
            'n_valid': n_valid,
        }

    # ── Checkpoints ───────────────────────────────────────────────────────

    def save_checkpoint(self, path: Path, extra: Dict = None):
        path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            'step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'eval_results': self.eval_results,
            'eval_steps': self.eval_steps,
            'synth_eval_losses': self.synth_eval_losses,
            'synth_eval_accs': self.synth_eval_accs,
            'synth_eval_aucs': self.synth_eval_aucs,
            'config': self.config.__dict__,
            'version': 'v3_temporal_pe_fpg8',
        }
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)
        print(f"  Saved checkpoint: {path}", flush=True)

    def load_checkpoint(self, path: Path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.current_step = ckpt['step']
        self.train_losses = ckpt.get('train_losses', [])
        self.train_accs = ckpt.get('train_accs', [])
        self.eval_results = ckpt.get('eval_results', [])
        self.eval_steps = ckpt.get('eval_steps', [])
        self.synth_eval_losses = ckpt.get('synth_eval_losses', [])
        self.synth_eval_accs = ckpt.get('synth_eval_accs', [])
        self.synth_eval_aucs = ckpt.get('synth_eval_aucs', [])
        print(f"  Loaded checkpoint from step {self.current_step}", flush=True)


# ============================================================================
# Plotting / Logging
# ============================================================================

def plot_training_curves(trainer: TabPFNTemporalFineTuner, log_dir: Path):
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))

    # Row 0: Training loss & accuracy (per-step, on random synthetic batches)
    if trainer.train_losses:
        axes[0, 0].plot(trainer.train_losses, alpha=0.7)
        axes[0, 0].set(xlabel='Step', ylabel='Loss', title='Training Loss (random synth)')
        axes[0, 0].set_yscale('log')

    if trainer.train_accs:
        axes[0, 1].plot(trainer.train_accs, alpha=0.7)
        axes[0, 1].set(xlabel='Step', ylabel='Accuracy', title='Training Acc (random synth)')

    # Row 1: Fixed synthetic eval (loss, AUC, Acc)
    if trainer.synth_eval_losses:
        steps = list(range(len(trainer.synth_eval_losses)))
        axes[1, 0].plot(steps, trainer.synth_eval_losses, alpha=0.7, color='tab:blue')
        axes[1, 0].set(xlabel='Step', ylabel='Loss', title='Synth Eval Loss (fixed batches)')
        axes[1, 0].set_yscale('log')
        if trainer.synth_eval_losses:
            axes[1, 0].axhline(y=trainer.synth_eval_losses[0], color='r',
                               ls='--', alpha=0.5, label='Baseline')
            axes[1, 0].legend()

    if trainer.synth_eval_aucs:
        steps = list(range(len(trainer.synth_eval_aucs)))
        ax_auc = axes[1, 1]
        ax_auc.plot(steps, trainer.synth_eval_aucs, 'o-', color='tab:green',
                    markersize=2, label='AUC')
        if trainer.synth_eval_accs:
            ax_auc.plot(steps, trainer.synth_eval_accs, 'o-', color='tab:orange',
                        markersize=2, label='Acc')
        ax_auc.set(xlabel='Step', ylabel='Score',
                   title='Synth Eval AUC & Acc (fixed batches)')
        if trainer.synth_eval_aucs:
            ax_auc.axhline(y=trainer.synth_eval_aucs[0], color='r',
                           ls='--', alpha=0.3, label='AUC baseline')
        ax_auc.legend()

    # Row 2: Real dataset eval
    if trainer.eval_results:
        mean_aucs = [
            np.mean([r['auc'] for r in sr if r.get('auc') is not None]) or 0
            for sr in trainer.eval_results]
        axes[2, 0].plot(trainer.eval_steps, mean_aucs, 'o-')
        axes[2, 0].set(xlabel='Step', ylabel='Mean AUC', title='Real Dataset AUC')
        if mean_aucs:
            axes[2, 0].axhline(y=mean_aucs[0], color='r', ls='--', alpha=0.5,
                               label='Baseline')
            axes[2, 0].legend()

        mean_accs = [
            np.mean([r['accuracy'] for r in sr
                     if r.get('accuracy') is not None]) or 0
            for sr in trainer.eval_results]
        axes[2, 1].plot(trainer.eval_steps, mean_accs, 'o-')
        axes[2, 1].set(xlabel='Step', ylabel='Mean Acc', title='Real Dataset Acc')
        if mean_accs:
            axes[2, 1].axhline(y=mean_accs[0], color='r', ls='--', alpha=0.5,
                               label='Baseline')
            axes[2, 1].legend()

    plt.tight_layout()
    plt.savefig(log_dir / 'training_curves.png', dpi=150)
    plt.close()


def save_eval_json(trainer: TabPFNTemporalFineTuner, log_dir: Path):
    with open(log_dir / 'eval_results.json', 'w') as f:
        json.dump({'eval_steps': trainer.eval_steps,
                   'results_per_step': trainer.eval_results}, f, indent=2)


# ============================================================================
# Training Loop
# ============================================================================

def train(config: FinetuneConfig, resume_from: Optional[str] = None):
    global _trainer_instance

    script_dir = Path(__file__).parent
    ckpt_dir = script_dir / config.checkpoint_dir
    log_dir  = script_dir / config.log_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FINE-TUNING TABPFN V3  —  Temporal PE (fpg=8)")
    print(f"  features_per_group={config.group_size}")
    print(f"  groups = {config.group_size} consecutive timesteps from same feature")
    print(f"  fresh init: encoder + emb_proj + first "
          f"{config.n_fresh_transformer_layers} transformer layers (Xavier)")
    print(f"  pretrained: remaining transformer layers + decoder")
    print("  embeddings = feature_emb(j) + sinusoidal_PE(group_idx)")
    print("  feature shuffle: DISABLED")
    print("  sklearn preprocessing: BYPASSED (no RemoveConst/Scaler/SVD)")
    print("  normalization: model-internal per group (z-score)")
    print(f"  data limits: T<={config.max_T}, m<={config.max_m}, "
          f"m*T<={config.max_m_times_T}")
    print(f"  augmentation: N/4 orig + 3N/4 augmented (feat perm + "
          f"class perm + none/log/exp/squash/KDI/kuma per feature)")
    print(f"  LR schedule: warmup {config.warmup_steps} steps -> "
          f"{config.lr:.1e}, cosine -> {config.lr_min:.1e}")
    print(f"  fresh params LR: {config.lr * config.encoder_lr_mult:.1e} "
          f"({config.encoder_lr_mult}x)")
    print(f"  per-dataset gradient clipping: {config.grad_clip}")
    print(f"  eval temperature: T={SOFTMAX_TEMPERATURE}")
    print("=" * 60)

    # Synthetic generator
    print("\nInitializing DAG-based synthetic generator (folder 11)...")
    synth_gen = SyntheticDataGenerator(config, seed=config.seed)

    # ── Generate 2 fixed synthetic eval batches ──
    print("\nGenerating 2 fixed synthetic eval batches...")
    synth_eval_rng = np.random.RandomState(config.seed + 9999)
    synth_eval_gen = SyntheticDataGenerator(config, seed=config.seed + 5000)
    synth_eval_batches: List[List[Dict]] = []
    for batch_idx in range(2):
        n_orig = config.batch_size // 4     # 32 originals per eval batch
        originals_eval = []
        for _ in range(n_orig):
            try:
                originals_eval.append(synth_eval_gen.generate_one())
            except Exception:
                continue
        # Build batch: originals + 3 augmented copies each
        eval_batch = list(originals_eval)
        for orig in originals_eval:
            for _ in range(3):
                try:
                    eval_batch.append(augment_dataset(orig, synth_eval_rng))
                except Exception:
                    eval_batch.append(orig)
        synth_eval_batches.append(eval_batch)
        print(f"  Eval batch {batch_idx+1}: {len(eval_batch)} datasets "
              f"({len(originals_eval)} orig + {len(eval_batch)-len(originals_eval)} aug)")

    # Real datasets
    print("\nLoading real datasets for evaluation...")
    real_datasets = load_real_datasets(config)
    print(f"  {len(real_datasets)} real datasets meet constraints")
    for ds in real_datasets:
        print(f"    {ds['name']}: {ds['n_samples']} samp, "
              f"{ds['n_features']} feat ({ds['n_features_orig']}ch x {ds['T']}t), "
              f"{ds['n_classes']} cls")

    # Trainer
    trainer = TabPFNTemporalFineTuner(config)
    _trainer_instance = trainer

    if resume_from:
        trainer.load_checkpoint(Path(resume_from))

    # Best synth eval loss tracking
    best_synth_loss = None
    best_synth_loss_step = None

    # ── Baseline evaluations ──
    # Synthetic baseline
    print("\nSynthetic baseline evaluation...")
    synth_base = trainer.evaluate_synthetic(synth_eval_batches)
    trainer.synth_eval_losses.append(synth_base['loss'])
    trainer.synth_eval_accs.append(synth_base['accuracy'])
    trainer.synth_eval_aucs.append(synth_base['auc'])
    best_synth_loss = synth_base['loss']
    best_synth_loss_step = trainer.current_step
    print(f"  SYNTH BASELINE: Loss={synth_base['loss']:.4f}  "
          f"Acc={synth_base['accuracy']:.4f}  AUC={synth_base['auc']:.4f}  "
          f"({synth_base['n_valid']} valid)")
    trainer.save_checkpoint(
        ckpt_dir / "checkpoint_best_synth_loss.pt",
        extra={'best_synth_loss': best_synth_loss,
               'best_synth_loss_step': best_synth_loss_step})

    # Real baseline
    if real_datasets:
        print(f"\nReal baseline evaluation ({len(real_datasets)} datasets)...")
        eval_res = trainer.evaluate_all(real_datasets)
        aucs = [r['auc'] for r in eval_res if r.get('auc') is not None]
        accs = [r['accuracy'] for r in eval_res if r.get('accuracy') is not None]
        mean_auc = np.mean(aucs)
        mean_acc = np.mean(accs)
        print(f"  REAL BASELINE: AUC={mean_auc:.4f}  Acc={mean_acc:.4f}  "
              f"({len(aucs)}/{len(real_datasets)} OK)\n")
        trainer.eval_steps.append(trainer.current_step)
        trainer.eval_results.append(eval_res)
        trainer.baseline_auc = mean_auc
        save_eval_json(trainer, log_dir)

    # Training
    print(f"Training {config.n_steps} steps  batch={config.batch_size}  "
          f"lr={config.lr}  lr_min={config.lr_min}  eval_every={config.eval_every}")
    t0 = time.time()

    aug_rng = np.random.RandomState(config.seed + 7777)

    for step in range(trainer.current_step, config.n_steps):
        step_t = time.time()

        # Generate N/4 unique datasets + 3 augmented copies each = N total
        n_unique = config.batch_size // 4
        originals = []
        for _ in range(n_unique):
            try:
                originals.append(synth_gen.generate_one())
            except Exception as e:
                print(f"  Gen error: {e}", flush=True)

        if not originals:
            print(f"Step {step}: no valid datasets")
            continue

        # Build batch: originals + 3 augmented copies per original
        batch = list(originals)
        for orig in originals:
            for _ in range(3):
                try:
                    batch.append(augment_dataset(orig, aug_rng))
                except Exception:
                    batch.append(orig)  # fallback: use original if augment fails

        result = trainer.train_step(batch)
        trainer.train_losses.append(result['loss'])
        trainer.train_accs.append(result['accuracy'])

        # Synthetic eval on fixed batches (every step)
        synth_res = trainer.evaluate_synthetic(synth_eval_batches)
        trainer.synth_eval_losses.append(synth_res['loss'])
        trainer.synth_eval_accs.append(synth_res['accuracy'])
        trainer.synth_eval_aucs.append(synth_res['auc'])

        # Save checkpoint if best synth eval loss
        if best_synth_loss is None or synth_res['loss'] < best_synth_loss:
            best_synth_loss = synth_res['loss']
            best_synth_loss_step = step + 1
            trainer.save_checkpoint(
                ckpt_dir / "checkpoint_best_synth_loss.pt",
                extra={'best_synth_loss': best_synth_loss,
                       'best_synth_loss_step': best_synth_loss_step})
            print(f"  >> New best synth loss: {best_synth_loss:.4f} @ step {best_synth_loss_step}",
                  flush=True)

        dt = time.time() - step_t
        print(f"Step {step:5d} | Loss {result['loss']:.4f} | "
              f"Acc {result['accuracy']:.4f} | "
              f"SynthEval L={synth_res['loss']:.4f} A={synth_res['auc']:.4f} | "
              f"LR {result['lr']:.2e} | {result['n_valid']}/{len(batch)} valid | "
              f"{dt:.1f}s", flush=True)

        if (step + 1) % config.eval_every == 0 and real_datasets:
            eval_res = trainer.evaluate_all(real_datasets)
            aucs = [r['auc'] for r in eval_res if r.get('auc') is not None]
            accs = [r['accuracy'] for r in eval_res if r.get('accuracy') is not None]
            mean_acc = np.mean(accs)
            mean_auc = np.mean(aucs)
            print(f"  >> EVAL step {step+1}: Acc={mean_acc:.4f}  "
                  f"AUC={mean_auc:.4f}  ({len(aucs)}/{len(real_datasets)} OK)",
                  flush=True)
            trainer.eval_steps.append(step + 1)
            trainer.eval_results.append(eval_res)

            if (step + 1) % 50 == 0:
                trainer.save_checkpoint(
                    ckpt_dir / f"checkpoint_step{step+1}.pt",
                    extra={'step_eval_results': eval_res})
            save_eval_json(trainer, log_dir)

    # Final
    print("\n" + "=" * 50)
    print("FINAL EVALUATION")
    print("=" * 50)
    if real_datasets:
        final = trainer.evaluate_all(real_datasets)
        aucs = [r['auc'] for r in final if r.get('auc') is not None]
        accs = [r['accuracy'] for r in final if r.get('accuracy') is not None]
        delta = np.mean(aucs) - getattr(trainer, 'baseline_auc', np.mean(aucs))
        print(f"  FINAL: AUC={np.mean(aucs):.4f} (delta={delta:+.4f})  "
              f"Acc={np.mean(accs):.4f}")
    else:
        final = None

    trainer.save_checkpoint(
        ckpt_dir / "checkpoint_final.pt",
        extra={'final_results': final})

    json.dump({
        'train_losses': trainer.train_losses,
        'train_accs': trainer.train_accs,
        'synth_eval_losses': trainer.synth_eval_losses,
        'synth_eval_accs': trainer.synth_eval_accs,
        'synth_eval_aucs': trainer.synth_eval_aucs,
        'eval_steps': trainer.eval_steps,
        'config': config.__dict__,
        'total_time': time.time() - t0,
    }, open(log_dir / 'training_history.json', 'w'), indent=2)

    save_eval_json(trainer, log_dir)
    plot_training_curves(trainer, log_dir)

    print(f"\nDone! {(time.time()-t0)/60:.1f} min")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Logs: {log_dir}")


# ============================================================================
# Signal handling
# ============================================================================

def signal_handler(signum, frame):
    global _trainer_instance
    print("\n\nInterrupt — saving checkpoint...")
    if _trainer_instance is not None:
        d = Path(__file__).parent / _trainer_instance.config.checkpoint_dir
        _trainer_instance.save_checkpoint(
            d / f"checkpoint_interrupted_step{_trainer_instance.current_step}.pt")
    sys.exit(0)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune TabPFN V3 with temporal positional encoding')
    parser.add_argument('--n-steps', type=int, default=1500)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=7e-5)
    parser.add_argument('--lr-min', type=float, default=1e-7)
    parser.add_argument('--warmup-steps', type=int, default=10)
    parser.add_argument('--eval-every', type=int, default=1)
    parser.add_argument('--freeze-layers', type=int, default=0,
                        help='Freeze first N transformer layers (0=all trainable)')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run-name', type=str, default='default',
                        help='Run name (separates checkpoints/logs per experiment)')
    parser.add_argument('--encoder-lr-mult', type=float, default=10.0,
                        help='LR multiplier for fresh params (1.0 = same as backbone)')
    parser.add_argument('--n-fresh-transformer-layers', type=int, default=4,
                        help='First N transformer layers reinited from scratch (default 4)')
    args = parser.parse_args()
    print(f"[ARG] n_fresh_transformer_layers = {args.n_fresh_transformer_layers}")

    config = FinetuneConfig(
        n_steps=args.n_steps, batch_size=args.batch_size,
        lr=args.lr, lr_min=args.lr_min,
        warmup_steps=args.warmup_steps,
        freeze_layers=args.freeze_layers,
        eval_every=args.eval_every,
        device=args.device, seed=args.seed,
        run_name=args.run_name,
        encoder_lr_mult=args.encoder_lr_mult,
        n_fresh_transformer_layers=args.n_fresh_transformer_layers)

    if args.debug:
        print("DEBUG MODE")
        config.n_steps = 10
        config.batch_size = 8
        config.eval_every = 5

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    train(config, resume_from=args.resume)


if __name__ == "__main__":
    main()
