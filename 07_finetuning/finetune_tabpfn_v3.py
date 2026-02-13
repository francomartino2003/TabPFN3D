"""
Fine-tune TabPFN with Temporal Positional Encoding (V3).

Key changes from V2:
- Keeps features_per_group=3 (same architecture, same speed, same memory)
- Feature shuffle disabled (temporal order must be preserved)
- Pads T to multiple of 3 so groups align to consecutive timesteps
- Structured group embeddings: feature_emb(j) + sinusoidal_PE(group_idx)
- 100% pretrained weights — NO new parameters, pure fine-tuning
- Synthetic data from 11_final_generator (same as V2)
- BYPASSES sklearn preprocessing entirely (no RemoveConstantFeatures,
  no SquashingScaler, no SVD). Data goes directly to the model encoder
  which handles normalization per group internally.

Usage:
    python finetune_tabpfn_v3.py --n-steps 1000 --eval-every 1
    python finetune_tabpfn_v3.py --debug
"""

import argparse
import sys
import os
import time
import json
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
sys.path.insert(0, str(Path(__file__).parent.parent / '11_final_generator'))
sys.path.insert(0, str(Path(__file__).parent.parent / '00_TabPFN' / 'src'))

from tabpfn import TabPFNClassifier
from tabpfn.preprocessing.configs import ClassifierEnsembleConfig, PreprocessorConfig

# Import DAG-based generator (folder 11)
from generator import DatasetGenerator
from hyperparameters import GeneratorHyperparameters

# Import temporal TabPFN wrapper
from tabpfn_temporal import (
    build_temporal_tabpfn,
    set_temporal_info,
    clear_temporal_info,
    pad_to_group3,
)

# Global for signal handling
_trainer_instance = None


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class FinetuneConfig:
    """Configuration for fine-tuning temporal TabPFN."""

    # Training
    lr: float = 1e-5
    weight_decay: float = 0.01
    batch_size: int = 64        # Number of datasets per gradient update
    n_steps: int = 1000
    grad_clip: float = 1.0

    # Evaluation
    eval_every: int = 1         # Evaluate on real datasets every N steps

    # Data constraints
    max_samples: int = 1000
    max_flat_features: int = 500
    max_classes: int = 10

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
# Synthetic Data Generation (using 11_final_generator)
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

                # Flatten 3D → 2D: (n, m, T) → (n, m*T)
                n_train = X_train_3d.shape[0]
                n_test  = X_test_3d.shape[0]
                X_train = X_train_3d.reshape(n_train, -1).astype(np.float32)
                X_test  = X_test_3d.reshape(n_test, -1).astype(np.float32)

                # Pad T to multiple of 3 for fpg=3 grouping
                # NB: must use original T for both calls (don't overwrite before 2nd)
                X_train, T_padded = pad_to_group3(X_train, n_features_orig, T)
                X_test, _         = pad_to_group3(X_test, n_features_orig, T)
                T = T_padded

                # Constraint: flattened features
                if X_train.shape[1] > self.config.max_flat_features:
                    continue
                if n_classes > self.config.max_classes or n_classes < 2:
                    continue

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
                X_train = X_train[:, :, np.newaxis]
            if X_test.ndim == 2:
                X_test = X_test[:, :, np.newaxis]

            # (n, length, channels) → (n, channels, length)
            X_train = np.transpose(X_train, (0, 2, 1))
            X_test  = np.transpose(X_test,  (0, 2, 1))

            n_samples = X_train.shape[0] + X_test.shape[0]
            n_channels = X_train.shape[1]  # m (original features)
            T = X_train.shape[2]            # T (timesteps)
            flat_features = n_channels * T

            if n_samples > config.max_samples:
                continue
            if flat_features > config.max_flat_features:
                continue

            n_classes = len(np.unique(y_train))
            if n_classes > config.max_classes:
                continue

            X_train_flat = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
            X_test_flat  = X_test.reshape(X_test.shape[0], -1).astype(np.float32)

            # Pad T to multiple of 3 for fpg=3 grouping
            X_train_flat, T_padded = pad_to_group3(X_train_flat, n_channels, T)
            X_test_flat, _         = pad_to_group3(X_test_flat, n_channels, T)
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
                'n_features': flat_features,
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

        # Build temporal model
        self.model, self.clf = build_temporal_tabpfn(device=self.device)
        self.model.to(self.device)
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Parameters: {n_params:,} (all trainable)")

        self.optimizer = AdamW(
            self.model.parameters(), lr=config.lr,
            weight_decay=config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: 1.0)

        self.train_losses: List[float] = []
        self.train_accs: List[float] = []
        self.eval_results: List[List[Dict]] = []
        self.eval_steps: List[int] = []
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
        - RemoveEmptyFeaturesEncoderStep (per group of 3)
        - NanHandling
        - InputNormalizationEncoderStep (z-score per group, train-only)
        - VariableNumFeaturesEncoderStep (pad + rescale per group)
        - LinearInputEncoderStep (projection to emsize=192)
        """
        try:
            y_test_t = torch.tensor(y_test, dtype=torch.long, device=self.device)

            # Set temporal info so add_embeddings knows m and T
            set_temporal_info(self.model, n_features_orig, T)

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

            return {'loss': loss, 'accuracy': acc}

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
        }

    # ── Training step ─────────────────────────────────────────────────────

    def train_step(self, batch: List[Dict]) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()

        total_loss, total_acc, n_valid = 0.0, 0.0, 0

        for i, data in enumerate(batch):
            try:
                out = self.forward_single_dataset(
                    data['X_train'], data['y_train'],
                    data['X_test'], data['y_test'],
                    data['n_classes'],
                    data['n_features_orig'], data['T'])

                if out['loss'].requires_grad:
                    out['loss'].backward()
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
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad /= n_valid
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

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

        Uses the same bypass approach as training: data goes directly to the
        model encoder. We use a fresh TabPFNClassifier for each dataset, load
        fine-tuned weights, and patch add_embeddings.
        """
        self.model.eval()
        results = []
        with torch.no_grad():
            for data in datasets:
                res = {'name': data['name'], 'n_classes': data['n_classes']}
                try:
                    # Create fresh classifier (just to get the model)
                    eval_clf = TabPFNClassifier(
                        device=self.device,
                        n_estimators=1,
                        ignore_pretraining_limits=True,
                        fit_mode="batched",
                        inference_config={"FEATURE_SHIFT_METHOD": None},
                    )
                    eval_clf._initialize_model_variables()
                    eval_model = eval_clf.model_

                    # Patch add_embeddings and load fine-tuned weights
                    import types
                    from tabpfn_temporal import temporal_add_embeddings
                    eval_model.add_embeddings = types.MethodType(
                        temporal_add_embeddings, eval_model)
                    eval_model.load_state_dict(self.model.state_dict())
                    eval_model.eval()

                    # Set temporal info
                    set_temporal_info(eval_model, data['n_features_orig'], data['T'])

                    # Bypass sklearn: pass data directly
                    X_train_t = torch.as_tensor(
                        data['X_train'], dtype=torch.float32,
                        device=self.device).unsqueeze(0)
                    y_train_t = torch.as_tensor(
                        data['y_train'], dtype=torch.float32,
                        device=self.device).unsqueeze(0)
                    X_test_t = torch.as_tensor(
                        data['X_test'], dtype=torch.float32,
                        device=self.device).unsqueeze(0)

                    dummy_cfg = self._make_dummy_ensemble_config()
                    eval_clf.n_classes_ = data['n_classes']
                    eval_clf.fit_from_preprocessed(
                        [X_train_t], [y_train_t],
                        cat_ix=[[[]]],
                        configs=[[dummy_cfg]],
                    )
                    logits = eval_clf.forward(
                        [X_test_t], return_raw_logits=True)

                    # logits → probabilities
                    if logits.ndim == 4:
                        logits = logits.mean(dim=(1, 2))
                    elif logits.ndim == 3:
                        logits = logits.squeeze(1)
                    proba = torch.softmax(logits, dim=-1)
                    proba = proba[:, :data['n_classes']]
                    proba_np = proba.cpu().numpy()
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
                results.append(res)
        return results

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
            'config': self.config.__dict__,
            'version': 'v3_temporal_pe',
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
        print(f"  Loaded checkpoint from step {self.current_step}", flush=True)


# ============================================================================
# Plotting / Logging
# ============================================================================

def plot_training_curves(trainer: TabPFNTemporalFineTuner, log_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    if trainer.train_losses:
        axes[0, 0].plot(trainer.train_losses, alpha=0.7)
        axes[0, 0].set(xlabel='Step', ylabel='Loss', title='Training Loss')
        axes[0, 0].set_yscale('log')

    if trainer.train_accs:
        axes[0, 1].plot(trainer.train_accs, alpha=0.7)
        axes[0, 1].set(xlabel='Step', ylabel='Accuracy', title='Training Accuracy')

    if trainer.eval_results:
        mean_aucs = [
            np.mean([r['auc'] for r in sr if r.get('auc') is not None]) or 0
            for sr in trainer.eval_results]
        axes[1, 0].plot(trainer.eval_steps, mean_aucs, 'o-')
        axes[1, 0].set(xlabel='Step', ylabel='Mean AUC', title='Real Dataset AUC')
        if mean_aucs:
            axes[1, 0].axhline(y=mean_aucs[0], color='r', ls='--', alpha=0.5,
                               label='Baseline')
            axes[1, 0].legend()

        mean_accs = [
            np.mean([r['accuracy'] for r in sr
                     if r.get('accuracy') is not None]) or 0
            for sr in trainer.eval_results]
        axes[1, 1].plot(trainer.eval_steps, mean_accs, 'o-')
        axes[1, 1].set(xlabel='Step', ylabel='Mean Acc', title='Real Dataset Acc')
        if mean_accs:
            axes[1, 1].axhline(y=mean_accs[0], color='r', ls='--', alpha=0.5,
                               label='Baseline')
            axes[1, 1].legend()

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
    print("FINE-TUNING TABPFN V3  —  Temporal Positional Encoding")
    print("  features_per_group=3 (kept as pretrained)")
    print("  groups = 3 consecutive timesteps from same feature")
    print("  embeddings = feature_emb(j) + sinusoidal_PE(group_idx)")
    print("  feature shuffle: DISABLED, no new params")
    print("  sklearn preprocessing: BYPASSED (no RemoveConst/Scaler/SVD)")
    print("  normalization: model-internal per group (z-score)")
    print("=" * 60)

    # Synthetic generator
    print("\nInitializing DAG-based synthetic generator (folder 11)...")
    synth_gen = SyntheticDataGenerator(config, seed=config.seed)

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

    # Baseline evaluation
    if real_datasets:
        print(f"\nBaseline evaluation ({len(real_datasets)} datasets)...")
        eval_res = trainer.evaluate_all(real_datasets)
        aucs = [r['auc'] for r in eval_res if r.get('auc') is not None]
        accs = [r['accuracy'] for r in eval_res if r.get('accuracy') is not None]
        print(f"  BASELINE: AUC={np.mean(aucs):.4f}  Acc={np.mean(accs):.4f}  "
              f"({len(aucs)}/{len(real_datasets)} OK)\n")
        trainer.eval_steps.append(trainer.current_step)
        trainer.eval_results.append(eval_res)
        trainer.baseline_auc = np.mean(aucs)
        save_eval_json(trainer, log_dir)

    # Training
    print(f"Training {config.n_steps} steps  batch={config.batch_size}  "
          f"lr={config.lr}  eval_every={config.eval_every}")
    t0 = time.time()

    for step in range(trainer.current_step, config.n_steps):
        step_t = time.time()

        batch = []
        for _ in range(config.batch_size):
            try:
                batch.append(synth_gen.generate_one())
            except Exception as e:
                print(f"  Gen error: {e}", flush=True)

        if not batch:
            print(f"Step {step}: no valid datasets")
            continue

        result = trainer.train_step(batch)
        trainer.train_losses.append(result['loss'])
        trainer.train_accs.append(result['accuracy'])

        dt = time.time() - step_t
        print(f"Step {step:5d} | Loss {result['loss']:.4f} | "
              f"Acc {result['accuracy']:.4f} | "
              f"LR {result['lr']:.2e} | {result['n_valid']}/{len(batch)} valid | "
              f"{dt:.1f}s", flush=True)

        if (step + 1) % config.eval_every == 0 and real_datasets:
            eval_res = trainer.evaluate_all(real_datasets)
            aucs = [r['auc'] for r in eval_res if r.get('auc') is not None]
            accs = [r['accuracy'] for r in eval_res if r.get('accuracy') is not None]
            print(f"  >> EVAL step {step+1}: Acc={np.mean(accs):.4f}  "
                  f"AUC={np.mean(aucs):.4f}  ({len(aucs)}/{len(real_datasets)} OK)",
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
    parser.add_argument('--n-steps', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--eval-every', type=int, default=1)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run-name', type=str, default='default',
                        help='Run name (separates checkpoints/logs per experiment)')
    args = parser.parse_args()

    config = FinetuneConfig(
        n_steps=args.n_steps, batch_size=args.batch_size,
        lr=args.lr, eval_every=args.eval_every,
        device=args.device, seed=args.seed,
        run_name=args.run_name)

    if args.debug:
        print("DEBUG MODE")
        config.n_steps = 10
        config.batch_size = 4
        config.eval_every = 5

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    train(config, resume_from=args.resume)


if __name__ == "__main__":
    main()
