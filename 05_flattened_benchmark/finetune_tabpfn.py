"""
Fine-tune TabPFN on Flattened Synthetic 3D Datasets.

Training approach:
- Generate synthetic 3D datasets on-the-fly (never repeat)
- Flatten them to 2D (n_samples, n_features * length)
- Fine-tune TabPFN using its official finetuning API
- Batch size = 64 datasets (gradient accumulation)
- Evaluate on real datasets periodically

Constraints (for both synthetic and real):
- n_samples ≤ 1000
- n_features × length ≤ 500
- n_classes ≤ 10

Usage:
    python finetune_tabpfn.py --n-steps 1000 --eval-every 50
    python finetune_tabpfn.py --debug  # Quick test run
"""

import argparse
import sys
import os
import time
import json
import signal
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
import pickle
import copy
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / '03_synthetic_generator_3D'))
sys.path.insert(0, str(Path(__file__).parent.parent / '00_TabPFN' / 'src'))

from tabpfn import TabPFNClassifier
from tabpfn.finetuning.data_util import ClassifierBatch
from tabpfn.preprocessing import fit_preprocessing

# Global for signal handling
_trainer_instance = None


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class FinetuneConfig:
    """Configuration for fine-tuning TabPFN."""
    
    # Training
    lr: float = 1e-5  # Very low LR to not destroy pretrained weights
    weight_decay: float = 0.01
    batch_size: int = 64  # Number of datasets per gradient update
    n_steps: int = 1000  # Number of optimizer steps
    warmup_steps: int = 50
    grad_clip: float = 1.0
    
    # Evaluation
    eval_every: int = 50  # Evaluate on real datasets every N steps
    n_eval_real: int = 20  # Number of real datasets to evaluate on
    
    # Data constraints
    max_samples: int = 1000
    max_flat_features: int = 500
    max_classes: int = 10
    
    # Synthetic generator settings
    prob_iid_mode: float = 0.6  # Higher probability for i.i.d. mode
    prob_sliding_window: float = 0.3
    prob_mixed: float = 0.1
    
    # Paths (relative to project root or script directory)
    checkpoint_dir: str = "results/finetune_checkpoints"
    log_dir: str = "results/finetune_logs"
    real_data_path: str = "../01_real_data/AEON/data/classification_datasets.pkl"
    
    # Device - auto-detect
    device: str = "auto"  # Will be set to cuda/mps/cpu
    
    # Number of estimators for finetuning 
    # MUST be 1 for batched mode (finetuning with gradients)
    n_estimators_finetune: int = 1
    n_estimators_eval: int = 4
    
    # Random seed
    seed: int = 42


# ============================================================================
# Data Generation
# ============================================================================

def create_synthetic_generator_prior(config: FinetuneConfig):
    """Create PriorConfig3D that matches our constraints."""
    from config import PriorConfig3D
    
    prior = PriorConfig3D(
        # Size constraints matching real data
        max_samples=config.max_samples,
        max_features=10,
        max_t_subseq=500,
        max_classes=config.max_classes,
        
        # Sample ranges
        n_samples_range=(50, config.max_samples),
        
        # Feature count - mix of univariate and multivariate
        prob_univariate=0.65,
        n_features_beta_a=1.5,
        n_features_beta_b=4.0,
        n_features_range=(2, 8),
        
        # Temporal parameters
        T_total_range=(30, 400),
        t_subseq_range=(20, 300),
        
        # Higher i.i.d. probability as requested
        prob_iid_mode=config.prob_iid_mode,
        prob_sliding_window_mode=config.prob_sliding_window,
        prob_mixed_mode=config.prob_mixed,
        
        # Graph structure
        n_nodes_range=(8, 25),
        density_range=(0.1, 0.5),
        
        # Transformations - more tree for complexity
        prob_nn_transform=0.40,
        prob_tree_transform=0.45,
        prob_discretization=0.15,
        
        # Force classification
        force_classification=True,
        prob_classification=1.0,
        min_samples_per_class=25,
        
        # Low noise for cleaner signals
        prob_edge_noise=0.03,
        noise_scale_range=(0.001, 0.03),
        prob_low_noise_dataset=0.5,
    )
    
    return prior


class SyntheticDataGenerator:
    """Generates synthetic flattened datasets on the fly."""
    
    def __init__(self, config: FinetuneConfig, seed: int = 42):
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.dataset_count = 0
        
        # Import generator
        from generator import SyntheticDatasetGenerator3D
        
        self.prior = create_synthetic_generator_prior(config)
        self.generator = SyntheticDatasetGenerator3D(self.prior, seed=seed)
    
    def generate_one(self) -> Dict[str, Any]:
        """Generate one synthetic dataset (flattened, train/test split)."""
        # Update seed for each dataset to ensure uniqueness
        self.dataset_count += 1
        self.generator = self.generator.__class__(
            self.prior, 
            seed=self.seed + self.dataset_count
        )
        
        try:
            dataset = self.generator.generate()
            X_3d = dataset.X  # (n_samples, n_features, length)
            y = dataset.y
            
            # Flatten
            n_samples, n_features, length = X_3d.shape
            X_flat = X_3d.reshape(n_samples, -1)
            
            # Check constraints
            if X_flat.shape[1] > self.config.max_flat_features:
                # Skip this dataset
                return self.generate_one()
            
            if len(np.unique(y)) > self.config.max_classes:
                return self.generate_one()
            
            # Train/test split (70/30)
            n_train = int(0.7 * n_samples)
            indices = self.rng.permutation(n_samples)
            train_idx, test_idx = indices[:n_train], indices[n_train:]
            
            X_train = X_flat[train_idx].astype(np.float32)
            X_test = X_flat[test_idx].astype(np.float32)
            y_train = y[train_idx].astype(np.int64)
            y_test = y[test_idx].astype(np.int64)
            
            # Encode labels to be contiguous
            le = LabelEncoder()
            le.fit(y_train)
            y_train = le.transform(y_train)
            y_test = le.transform(y_test)
            
            # Handle missing values (use train mean)
            if np.any(np.isnan(X_train)):
                col_means = np.nanmean(X_train, axis=0)
                col_means = np.nan_to_num(col_means, nan=0.0)
                for i in range(X_train.shape[1]):
                    X_train[:, i] = np.where(np.isnan(X_train[:, i]), col_means[i], X_train[:, i])
                    X_test[:, i] = np.where(np.isnan(X_test[:, i]), col_means[i], X_test[:, i])
            
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'n_classes': len(le.classes_),
                'n_features': X_flat.shape[1],
                'n_samples': n_samples,
                'metadata': dataset.metadata,
            }
        except Exception as e:
            # If generation fails, try again
            print(f"  Generation warning: {e}")
            return self.generate_one()


# ============================================================================
# Real Data Loading
# ============================================================================

def load_real_datasets(config: FinetuneConfig) -> List[Dict[str, Any]]:
    """Load real datasets from pickle file."""
    # The pickle was saved with 'src.data_loader.TimeSeriesDataset'
    # We need '01_real_data' in path so that 'src' is importable as a submodule
    real_data_dir = Path(__file__).resolve().parent.parent / '01_real_data'
    
    if str(real_data_dir) not in sys.path:
        sys.path.insert(0, str(real_data_dir))
    
    # Now 'src' should be importable
    from src.data_loader import TimeSeriesDataset  # noqa: F401 - needed for pickle
    
    pkl_path = Path(__file__).parent / config.real_data_path
    if not pkl_path.exists():
        # Try alternative path
        pkl_path = Path(__file__).parent.parent / '01_real_data' / 'AEON' / 'data' / 'classification_datasets.pkl'
    
    if not pkl_path.exists():
        print(f"Warning: Real data not found at {pkl_path}")
        return []
    
    with open(pkl_path, 'rb') as f:
        datasets_list = pickle.load(f)
    
    valid_datasets = []
    
    for dataset in datasets_list:
        try:
            name = dataset.name
            # Get data
            X_train = dataset.X_train  # (n_samples, length, n_channels) or (n_samples, length)
            y_train = dataset.y_train
            X_test = dataset.X_test
            y_test = dataset.y_test
            
            if X_train is None or X_test is None:
                continue
            
            # Ensure 3D
            if X_train.ndim == 2:
                X_train = X_train[:, :, np.newaxis]
            if X_test.ndim == 2:
                X_test = X_test[:, :, np.newaxis]
            
            # Shape: (n_samples, length, n_channels) -> need (n_samples, n_channels, length)
            X_train = np.transpose(X_train, (0, 2, 1))
            X_test = np.transpose(X_test, (0, 2, 1))
            
            n_samples = X_train.shape[0] + X_test.shape[0]
            n_channels = X_train.shape[1]
            length = X_train.shape[2]
            flat_features = n_channels * length
            n_classes = len(np.unique(y_train))
            
            # Check constraints
            if n_samples > config.max_samples:
                continue
            if flat_features > config.max_flat_features:
                continue
            if n_classes > config.max_classes:
                continue
            
            # Flatten
            X_train_flat = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
            X_test_flat = X_test.reshape(X_test.shape[0], -1).astype(np.float32)
            
            # Encode labels
            le = LabelEncoder()
            le.fit(y_train)
            y_train_enc = le.transform(y_train).astype(np.int64)
            y_test_enc = le.transform(y_test).astype(np.int64)
            
            # Handle missing values
            if np.any(np.isnan(X_train_flat)):
                col_means = np.nanmean(X_train_flat, axis=0)
                col_means = np.nan_to_num(col_means, nan=0.0)
                for i in range(X_train_flat.shape[1]):
                    X_train_flat[:, i] = np.where(np.isnan(X_train_flat[:, i]), col_means[i], X_train_flat[:, i])
                    X_test_flat[:, i] = np.where(np.isnan(X_test_flat[:, i]), col_means[i], X_test_flat[:, i])
            
            valid_datasets.append({
                'name': name,
                'X_train': X_train_flat,
                'X_test': X_test_flat,
                'y_train': y_train_enc,
                'y_test': y_test_enc,
                'n_classes': n_classes,
                'n_features': flat_features,
                'n_samples': n_samples,
            })
        except Exception as e:
            continue
    
    return valid_datasets


# ============================================================================
# TabPFN Fine-Tuner using Official API
# ============================================================================

def _compute_classification_loss(
    logits_BLQ: torch.Tensor,
    targets_BQ: torch.Tensor,
) -> torch.Tensor:
    """Compute cross-entropy loss (same as TabPFN's finetuning)."""
    return F.cross_entropy(logits_BLQ, targets_BQ)


class TabPFNFineTuner:
    """Fine-tunes TabPFN using its official API."""
    
    def __init__(self, config: FinetuneConfig):
        self.config = config
        self.device = self._get_device()
        
        print(f"\nInitializing TabPFN fine-tuner...")
        print(f"  Device: {self.device}")
        print(f"  Learning rate: {config.lr}")
        print(f"  Batch size: {config.batch_size}")
        
        # Create TabPFNClassifier with batched mode for finetuning
        self.clf = TabPFNClassifier(
            device=self.device,
            n_estimators=config.n_estimators_finetune,
            ignore_pretraining_limits=True,
            fit_mode="batched",  # Required for finetuning
            differentiable_input=False,
        )
        
        # Initialize model
        self._initialize_model()
        
        # Create optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
        # Create scheduler
        self.scheduler = self._create_scheduler()
        
        # Tracking
        self.train_losses = []
        self.train_accs = []
        self.eval_aucs = []
        self.eval_accs = []
        self.eval_steps = []
        self.current_step = 0
    
    def _get_device(self) -> str:
        """Auto-detect best device."""
        if self.config.device != "auto":
            return self.config.device
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _initialize_model(self):
        """Initialize TabPFN model for training."""
        # Initialize model variables
        self.clf._initialize_model_variables()
        
        # Get the model
        self.model = self.clf.model_
        self.model.to(self.device)
        
        # Enable training mode
        self.model.train()
        
        # Enable gradients for all parameters
        for param in self.model.parameters():
            param.requires_grad = True
        
        n_params = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Model parameters: {n_params:,}")
        print(f"  Trainable: {n_trainable:,}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup + cosine decay."""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            progress = (step - self.config.warmup_steps) / max(1, self.config.n_steps - self.config.warmup_steps)
            return max(0.01, 0.5 * (1 + np.cos(np.pi * progress)))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    
    def forward_single_dataset(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_classes: int
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for a single dataset using TabPFN's finetuning API.
        
        ensemble_configs from _initialize_dataset_preprocessing is a list of 
        ClassifierEnsembleConfig objects (one per estimator).
        """
        try:
            y_test_t = torch.tensor(y_test, dtype=torch.long, device=self.device)
            
            # Get preprocessing config
            rng = np.random.default_rng(self.config.seed + self.current_step)
            
            # Use TabPFN's internal preprocessing
            # Returns: (list[ClassifierEnsembleConfig], X_processed, y_processed)
            ensemble_configs_list, X_train_proc, y_train_proc = self.clf._initialize_dataset_preprocessing(
                X_train, y_train, rng
            )
            
            # ensemble_configs_list is already a list of configs (one per estimator)
            n_estimators = len(ensemble_configs_list)
            
            # Prepare data for fit_from_preprocessed
            # Need list format: one tensor per estimator
            X_context_list = []
            y_context_list = []
            X_query_list = []
            cat_indices_list = []
            configs_list = []
            
            for config in ensemble_configs_list:
                X_train_tensor = torch.tensor(X_train_proc, dtype=torch.float32, device=self.device)
                y_train_tensor = torch.tensor(y_train_proc, dtype=torch.long, device=self.device)
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=self.device)
                
                X_context_list.append(X_train_tensor)
                y_context_list.append(y_train_tensor)
                X_query_list.append(X_test_tensor)
                cat_indices_list.append(self.clf.inferred_categorical_indices_)
                configs_list.append(config)
            
            # fit_from_preprocessed expects configs as list[list[EnsembleConfig]]
            # Wrap in another list for batch dimension
            configs_batched = [configs_list]
            cat_indices_batched = [cat_indices_list]
            
            # Fit the context (sets up internal state)
            self.clf.fit_from_preprocessed(
                X_context_list,
                y_context_list,
                cat_indices_batched,
                configs_batched,
            )
            
            # Forward pass to get logits
            # Shape: (Q, B, E, L) where Q=n_queries, B=batch(=1), E=n_estimators, L=n_classes
            logits_QBEL = self.clf.forward(
                X_query_list,
                return_raw_logits=True,
            )
            
            Q, B, E, L = logits_QBEL.shape
            
            # Reshape for loss: (B*E, L, Q)
            logits_BLQ = logits_QBEL.permute(1, 2, 3, 0).reshape(B * E, L, Q)
            
            # Targets: repeat for each estimator
            targets_BQ = y_test_t.unsqueeze(0).repeat(B * E, 1)
            
            # Compute loss
            loss = _compute_classification_loss(logits_BLQ, targets_BQ)
            
            # Compute accuracy (average over estimators)
            with torch.no_grad():
                avg_logits = logits_QBEL.mean(dim=(1, 2))  # (Q, L)
                preds = avg_logits.argmax(dim=-1)
                acc = (preds == y_test_t).float().mean()
            
            return {
                'loss': loss,
                'accuracy': acc,
            }
            
        except Exception as e:
            import traceback
            print(f"  Forward error: {e}")
            traceback.print_exc()
            return {
                'loss': torch.tensor(0.0, device=self.device, requires_grad=True),
                'accuracy': torch.tensor(0.0),
            }
    
    def train_step(self, batch: List[Dict]) -> Dict[str, float]:
        """Execute one training step over a batch of datasets."""
        self.model.train()
        self.optimizer.zero_grad()
        
        total_loss = 0.0
        total_acc = 0.0
        n_valid = 0
        
        for data in batch:
            try:
                output = self.forward_single_dataset(
                    data['X_train'],
                    data['y_train'],
                    data['X_test'],
                    data['y_test'],
                    data['n_classes']
                )
                
                if output['loss'].requires_grad:
                    # Scale loss by batch size for proper gradient accumulation
                    scaled_loss = output['loss'] / len(batch)
                    scaled_loss.backward()
                    
                    total_loss += output['loss'].item()
                    total_acc += output['accuracy'].item()
                    n_valid += 1
                    
            except Exception as e:
                print(f"  Batch item error: {e}")
                continue
        
        # Gradient clipping
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.grad_clip
            )
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        self.current_step += 1
        
        avg_loss = total_loss / max(1, n_valid)
        avg_acc = total_acc / max(1, n_valid)
        
        return {
            'loss': avg_loss,
            'accuracy': avg_acc,
            'n_valid': n_valid,
            'lr': self.scheduler.get_last_lr()[0],
        }
    
    def evaluate(self, datasets: List[Dict]) -> Dict[str, float]:
        """Evaluate on a list of datasets."""
        self.model.eval()
        
        all_aucs = []
        all_accs = []
        
        # Create evaluation classifier (fresh, shares weights)
        eval_clf = TabPFNClassifier(
            device=self.device,
            n_estimators=self.config.n_estimators_eval,
            ignore_pretraining_limits=True,
        )
        # Copy model weights
        eval_clf._initialize_model_variables()
        eval_clf.model_.load_state_dict(self.model.state_dict())
        eval_clf.model_.eval()
        
        with torch.no_grad():
            for data in datasets:
                try:
                    # Use standard fit/predict for evaluation
                    eval_clf.fit(data['X_train'], data['y_train'])
                    
                    proba = eval_clf.predict_proba(data['X_test'])
                    preds = proba.argmax(axis=1)
                    
                    # Accuracy
                    acc = accuracy_score(data['y_test'], preds)
                    all_accs.append(acc)
                    
                    # AUC
                    try:
                        if data['n_classes'] == 2:
                            auc = roc_auc_score(data['y_test'], proba[:, 1])
                        else:
                            auc = roc_auc_score(data['y_test'], proba, multi_class='ovr')
                        all_aucs.append(auc)
                    except Exception:
                        pass
                        
                except Exception as e:
                    print(f"  Eval error on {data.get('name', 'synthetic')}: {e}")
                    continue
        
        return {
            'mean_auc': np.mean(all_aucs) if all_aucs else 0.0,
            'mean_acc': np.mean(all_accs) if all_accs else 0.0,
            'n_evaluated': len(all_aucs),
        }
    
    def save_checkpoint(self, path: Path, extra: Dict = None):
        """Save model checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'eval_aucs': self.eval_aucs,
            'eval_accs': self.eval_accs,
            'eval_steps': self.eval_steps,
            'config': self.config.__dict__,
        }
        if extra:
            checkpoint.update(extra)
        
        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_step = checkpoint['step']
        self.train_losses = checkpoint.get('train_losses', [])
        self.train_accs = checkpoint.get('train_accs', [])
        self.eval_aucs = checkpoint.get('eval_aucs', [])
        self.eval_accs = checkpoint.get('eval_accs', [])
        self.eval_steps = checkpoint.get('eval_steps', [])
        
        print(f"  Loaded checkpoint from step {self.current_step}")


# ============================================================================
# Training Loop
# ============================================================================

def plot_training_curves(trainer: TabPFNFineTuner, log_dir: Path):
    """Plot and save training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training loss
    ax = axes[0, 0]
    if trainer.train_losses:
        ax.plot(trainer.train_losses, alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.set_yscale('log')
    
    # Training accuracy
    ax = axes[0, 1]
    if trainer.train_accs:
        ax.plot(trainer.train_accs, alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training Accuracy')
    
    # Eval AUC
    ax = axes[1, 0]
    if trainer.eval_aucs:
        ax.plot(trainer.eval_steps, trainer.eval_aucs, 'o-')
        ax.set_xlabel('Step')
        ax.set_ylabel('AUC ROC')
        ax.set_title('Real Dataset AUC')
        ax.axhline(y=trainer.eval_aucs[0], color='r', linestyle='--', alpha=0.5, label='Initial')
        ax.legend()
    
    # Eval Accuracy
    ax = axes[1, 1]
    if trainer.eval_accs:
        ax.plot(trainer.eval_steps, trainer.eval_accs, 'o-')
        ax.set_xlabel('Step')
        ax.set_ylabel('Accuracy')
        ax.set_title('Real Dataset Accuracy')
        ax.axhline(y=trainer.eval_accs[0], color='r', linestyle='--', alpha=0.5, label='Initial')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(log_dir / 'training_curves.png', dpi=150)
    plt.close()


def train(config: FinetuneConfig, resume_from: Optional[str] = None):
    """Main training loop."""
    global _trainer_instance
    
    # Setup directories
    script_dir = Path(__file__).parent
    checkpoint_dir = script_dir / config.checkpoint_dir
    log_dir = script_dir / config.log_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("FINE-TUNING TABPFN ON FLATTENED SYNTHETIC DATA")
    print("=" * 60)
    
    # Initialize synthetic generator
    print("\nInitializing synthetic data generator...")
    synth_gen = SyntheticDataGenerator(config, seed=config.seed)
    
    # Load real datasets
    print("\nLoading real datasets for evaluation...")
    real_datasets = load_real_datasets(config)
    print(f"Loaded {len(real_datasets)} real datasets meeting constraints")
    
    if not real_datasets:
        print("Warning: No real datasets found. Evaluation will be skipped.")
    
    # Initialize trainer
    trainer = TabPFNFineTuner(config)
    _trainer_instance = trainer
    
    # Resume from checkpoint
    if resume_from:
        trainer.load_checkpoint(Path(resume_from))
    
    # Initial evaluation
    print(f"\nInitial evaluation on real datasets...")
    if real_datasets:
        eval_subset = real_datasets[:config.n_eval_real]
        eval_result = trainer.evaluate(eval_subset)
        print(f"  Real AUC: {eval_result['mean_auc']:.4f}, Acc: {eval_result['mean_acc']:.4f}")
        
        trainer.eval_steps.append(trainer.current_step)
        trainer.eval_aucs.append(eval_result['mean_auc'])
        trainer.eval_accs.append(eval_result['mean_acc'])
    
    # Training loop
    print(f"\nStarting training for {config.n_steps} steps...")
    print(f"  Batch size: {config.batch_size} datasets")
    print(f"  Learning rate: {config.lr}")
    print(f"  Eval every: {config.eval_every} steps")
    print(f"  Device: {trainer.device}")
    
    start_time = time.time()
    
    for step in range(trainer.current_step, config.n_steps):
        step_start = time.time()
        
        # Generate batch of synthetic datasets
        batch = []
        for _ in range(config.batch_size):
            data = synth_gen.generate_one()
            batch.append(data)
        
        # Training step
        result = trainer.train_step(batch)
        
        trainer.train_losses.append(result['loss'])
        trainer.train_accs.append(result['accuracy'])
        
        step_time = time.time() - step_start
        
        # Log progress
        print(f"Step {step:5d} | Loss: {result['loss']:.4f} | "
              f"Acc: {result['accuracy']:.4f} | LR: {result['lr']:.2e} | "
              f"Time: {step_time:.2f}s")
        
        # Evaluation
        if (step + 1) % config.eval_every == 0 and real_datasets:
            print(f"\n{'='*40}")
            print(f"EVALUATION at step {step + 1}")
            
            eval_subset = real_datasets[:config.n_eval_real]
            eval_result = trainer.evaluate(eval_subset)
            
            print(f"  Real datasets: {eval_result['n_evaluated']}")
            print(f"  Mean AUC: {eval_result['mean_auc']:.4f}")
            print(f"  Mean Acc: {eval_result['mean_acc']:.4f}")
            print(f"{'='*40}\n")
            
            trainer.eval_steps.append(step + 1)
            trainer.eval_aucs.append(eval_result['mean_auc'])
            trainer.eval_accs.append(eval_result['mean_acc'])
            
            # Save checkpoint
            trainer.save_checkpoint(
                checkpoint_dir / f"checkpoint_step{step+1}.pt",
                extra={'eval_result': eval_result}
            )
            
            # Plot curves
            plot_training_curves(trainer, log_dir)
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    if real_datasets:
        final_result = trainer.evaluate(real_datasets)
        print(f"All {len(real_datasets)} real datasets:")
        print(f"  Mean AUC: {final_result['mean_auc']:.4f}")
        print(f"  Mean Acc: {final_result['mean_acc']:.4f}")
    
    # Save final checkpoint
    trainer.save_checkpoint(
        checkpoint_dir / "checkpoint_final.pt",
        extra={'final_result': final_result if real_datasets else None}
    )
    
    # Save training history
    history = {
        'train_losses': trainer.train_losses,
        'train_accs': trainer.train_accs,
        'eval_steps': trainer.eval_steps,
        'eval_aucs': trainer.eval_aucs,
        'eval_accs': trainer.eval_accs,
        'config': config.__dict__,
        'total_time': time.time() - start_time,
    }
    
    with open(log_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Final plots
    plot_training_curves(trainer, log_dir)
    
    print(f"\nTraining complete! Total time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Logs saved to: {log_dir}")


# ============================================================================
# Signal Handling
# ============================================================================

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    global _trainer_instance
    print("\n\nInterrupt received, saving checkpoint...")
    
    if _trainer_instance is not None:
        checkpoint_dir = Path(__file__).parent / _trainer_instance.config.checkpoint_dir
        _trainer_instance.save_checkpoint(
            checkpoint_dir / f"checkpoint_interrupted_step{_trainer_instance.current_step}.pt"
        )
    
    sys.exit(0)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Fine-tune TabPFN on flattened synthetic data')
    
    # Training args
    parser.add_argument('--n-steps', type=int, default=1000, help='Number of training steps')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size (datasets)')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--eval-every', type=int, default=50, help='Eval frequency')
    parser.add_argument('--n-eval-real', type=int, default=20, help='Number of real datasets to evaluate')
    
    # Device
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cuda/mps/cpu)')
    
    # Resume
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    # Debug mode
    parser.add_argument('--debug', action='store_true', help='Debug mode (quick run)')
    
    # Seed
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Create config
    config = FinetuneConfig(
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_every=args.eval_every,
        n_eval_real=args.n_eval_real,
        device=args.device,
        seed=args.seed,
    )
    
    # Debug mode overrides
    if args.debug:
        print("DEBUG MODE: Quick run with reduced settings")
        config.n_steps = 20
        config.batch_size = 4
        config.eval_every = 5
        config.n_eval_real = 5
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run training
    train(config, resume_from=args.resume)


if __name__ == "__main__":
    main()
