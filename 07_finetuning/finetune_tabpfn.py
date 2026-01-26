"""
Fine-tune TabPFN on Flattened Synthetic 3D Datasets.

Training approach:
- Generate synthetic 3D datasets on-the-fly using generator from 06_generator_experiments
- Flatten them to 2D (n_samples, n_features * length)
- Fine-tune TabPFN using its official finetuning API
- Batch size = 64 datasets (gradient accumulation)
- Evaluate on ALL real datasets at checkpoints

Constraints (for both synthetic and real):
- n_samples <= 1000
- n_features * length <= 500
- n_classes <= 10

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
sys.path.insert(0, str(Path(__file__).parent.parent / '06_generator_experiments'))
sys.path.insert(0, str(Path(__file__).parent.parent / '00_TabPFN' / 'src'))

from tabpfn import TabPFNClassifier
from tabpfn.finetuning.data_util import ClassifierBatch
from tabpfn.preprocessing import fit_preprocessing, EnsembleConfig

# Import generator from 06
from dataset_generator import (
    generate_dataset, GeneratedDataset,
    RandomNNConfig, DatasetConfig
)

# Global for signal handling
_trainer_instance = None


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class FinetuneConfig:
    """Configuration for fine-tuning TabPFN."""
    
    # Training
    lr: float = 1e-5  # Low LR to not destroy pretrained weights
    weight_decay: float = 0.01
    batch_size: int = 64  # Number of datasets per gradient update
    n_steps: int = 1000  # Number of optimizer steps
    grad_clip: float = 1.0
    
    # Evaluation
    eval_every: int = 50  # Evaluate on real datasets every N steps
    
    # Data constraints (matching our generator)
    max_samples: int = 1000
    max_flat_features: int = 500
    max_classes: int = 10
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    real_data_path: str = "../01_real_data/AEON/data/classification_datasets.pkl"
    
    # Device - auto-detect
    device: str = "auto"
    
    # Number of estimators
    n_estimators_finetune: int = 1  # Must be 1 for finetuning
    n_estimators_eval: int = 4
    
    # Random seed
    seed: int = 42


# ============================================================================
# Synthetic Data Generation (using 06_generator_experiments)
# ============================================================================

def create_nn_config():
    """Create config matching the random_all experiments."""
    return RandomNNConfig(
        # Memory - RANDOM 1-8 dimensions
        memory_dim_range=(1, 8),
        memory_init='uniform',
        
        # Stochastic inputs - none
        stochastic_input_dim_range=(0, 0),
        
        # Time transforms - RANDOM 1-5 (all linear)
        n_time_transforms_range=(1, 5),
        
        # Architecture - shallow networks (2-5 layers)
        n_hidden_layers_range=(2, 5),
        n_nodes_per_layer_range=(4, 16),
        
        # Activations - diverse per-node activations (12 types)
        activation_choices=(
            'identity', 'log', 'sigmoid', 'abs', 'sin', 'tanh',
            'rank', 'square', 'power', 'softplus', 'step', 'modulo',
        ),
        
        # Initialization
        weight_init_choices=('xavier_normal',),
        weight_scale_range=(0.9, 1.1),
        bias_std_range=(0.0, 0.1),
        
        # Per-node noise - 5-30% of nodes, higher std
        node_noise_prob_range=(0.05, 0.30),
        node_noise_std_range=(0.01, 0.1),
        noise_dist_choices=('normal',),
        
        per_layer_activation=True,
        
        # NO quantization nodes - will apply quantization to target later
        quantization_node_prob=0.0,
        quantization_n_classes_range=(2, 8),
        
        seq_length=200,  # Will be overwritten per dataset
    )


def create_dataset_config():
    """Create dataset config for finetuning."""
    return DatasetConfig(
        max_samples=1000,
        max_features=8,
        max_timesteps=1000,
        max_m_times_t=500,
        max_target_offset=15,
        prob_iid=1.0,  # All IID
        prob_sliding=0.0,
        prob_mixed=0.0,
        train_ratio=0.8,
    )


class SyntheticDataGenerator:
    """Generates synthetic flattened datasets using generator from 06."""
    
    def __init__(self, config: FinetuneConfig, seed: int = 42):
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.dataset_count = 0
        
        self.nn_config = create_nn_config()
        self.ds_config = create_dataset_config()
    
    def generate_one(self, max_retries: int = 10) -> Dict[str, Any]:
        """Generate one synthetic dataset (flattened, train/test split)."""
        
        for retry in range(max_retries):
            self.dataset_count += 1
            current_seed = self.seed + self.dataset_count * 100 + retry
            
            try:
                dataset = generate_dataset(
                    self.nn_config, 
                    self.ds_config, 
                    seed=current_seed
                )
                
                # Get data
                X_train_3d = dataset.X_train  # (n_samples, n_features, length)
                X_test_3d = dataset.X_test
                y_train = dataset.y_train
                y_test = dataset.y_test
                
                # Flatten
                n_train, n_features, length = X_train_3d.shape
                n_test = X_test_3d.shape[0]
                
                X_train = X_train_3d.reshape(n_train, -1).astype(np.float32)
                X_test = X_test_3d.reshape(n_test, -1).astype(np.float32)
                
                # Check constraints
                if X_train.shape[1] > self.config.max_flat_features:
                    continue
                
                n_classes = dataset.n_classes
                if n_classes > self.config.max_classes or n_classes < 2:
                    continue
                
                # Handle NaN/Inf
                X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
                X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
                
                # Check for constant features
                feature_std = np.std(X_train, axis=0)
                if np.all(feature_std < 1e-8):
                    continue
                
                return {
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train.astype(np.int64),
                    'y_test': y_test.astype(np.int64),
                    'n_classes': n_classes,
                    'n_features': X_train.shape[1],
                    'n_samples': n_train + n_test,
                    'sample_mode': dataset.sample_mode,
                }
                
            except Exception as e:
                if retry == max_retries - 1:
                    print(f"  Generation failed after {max_retries} retries: {e}")
                continue
        
        raise RuntimeError(f"Could not generate valid dataset after {max_retries} retries")


# ============================================================================
# Real Data Loading
# ============================================================================

def load_real_datasets(config: FinetuneConfig) -> List[Dict[str, Any]]:
    """Load ALL real datasets from pickle file."""
    real_data_dir = Path(__file__).resolve().parent.parent / '01_real_data'
    
    if str(real_data_dir) not in sys.path:
        sys.path.insert(0, str(real_data_dir))
    
    from src.data_loader import TimeSeriesDataset  # noqa: F401 - needed for pickle
    
    pkl_path = Path(__file__).parent / config.real_data_path
    if not pkl_path.exists():
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
            X_train = dataset.X_train
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
            
            # Shape: (n_samples, length, n_channels) -> (n_samples, n_channels, length)
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
# TabPFN Fine-Tuner
# ============================================================================

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
            fit_mode="batched",
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
        
        # Constant LR scheduler
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: 1.0)
        
        # Tracking
        self.train_losses = []
        self.train_accs = []
        self.eval_results = []  # Full results per step
        self.eval_steps = []
        self.current_step = 0
    
    def _get_device(self) -> str:
        if self.config.device != "auto":
            return self.config.device
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _initialize_model(self):
        self.clf._initialize_model_variables()
        self.model = self.clf.model_
        self.model.to(self.device)
        self.model.train()
        
        for param in self.model.parameters():
            param.requires_grad = True
        
        n_params = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Model parameters: {n_params:,}")
        print(f"  Trainable: {n_trainable:,}")
    
    def forward_single_dataset(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_classes: int
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for a single dataset."""
        try:
            y_test_t = torch.tensor(y_test, dtype=torch.long, device=self.device)
            
            rng = np.random.default_rng(self.config.seed + self.current_step)
            
            ensemble_configs_list, X_train_init, y_train_init = self.clf._initialize_dataset_preprocessing(
                X_train, y_train, rng
            )
            
            cat_ix = self.clf.inferred_categorical_indices_ or []
            
            preprocessing_results = list(fit_preprocessing(
                configs=ensemble_configs_list,
                X_train=X_train_init,
                y_train=y_train_init,
                random_state=rng,
                cat_ix=cat_ix,
                n_preprocessing_jobs=1,
                parallel_mode="block",
            ))
            
            configs_processed = []
            X_trains_preprocessed = []
            y_trains_preprocessed = []
            cat_ixs_processed = []
            preprocessors = []
            
            for config, preprocessor, X_train_pp, y_train_pp, cat_ix_pp in preprocessing_results:
                configs_processed.append(config)
                preprocessors.append(preprocessor)
                X_trains_preprocessed.append(X_train_pp)
                y_trains_preprocessed.append(y_train_pp)
                cat_ixs_processed.append(cat_ix_pp)
            
            X_tests_preprocessed = []
            for preprocessor in preprocessors:
                X_test_pp = preprocessor.transform(X_test).X
                X_tests_preprocessed.append(X_test_pp)
            
            X_context_list = []
            y_context_list = []
            X_query_list = []
            
            for i in range(len(configs_processed)):
                X_train_tensor = torch.as_tensor(
                    X_trains_preprocessed[i], dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                X_test_tensor = torch.as_tensor(
                    X_tests_preprocessed[i], dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                y_train_tensor = torch.as_tensor(
                    y_trains_preprocessed[i], dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                
                X_context_list.append(X_train_tensor)
                y_context_list.append(y_train_tensor)
                X_query_list.append(X_test_tensor)
            
            configs_batched = [configs_processed]
            cat_indices_batched = [cat_ixs_processed]
            
            self.clf.fit_from_preprocessed(
                X_context_list,
                y_context_list,
                cat_indices_batched,
                configs_batched,
            )
            
            logits = self.clf.forward(
                X_query_list,
                return_raw_logits=True,
            )
            
            if logits.ndim == 2:
                logits_QL = logits
            elif logits.ndim == 3:
                logits_QL = logits.squeeze(1)
            elif logits.ndim == 4:
                logits_QL = logits.mean(dim=(1, 2))
            else:
                raise ValueError(f"Unexpected logits shape: {logits.shape}")
            
            loss = F.cross_entropy(logits_QL, y_test_t)
            
            with torch.no_grad():
                preds = logits_QL.argmax(dim=-1)
                acc = (preds == y_test_t).float().mean()
                probs = F.softmax(logits_QL, dim=-1)
            
            return {
                'loss': loss,
                'accuracy': acc,
                'probs': probs.detach().cpu().numpy(),
                'y_true': y_test_t.detach().cpu().numpy(),
            }
            
        except Exception as e:
            return {
                'loss': torch.tensor(0.0, device=self.device, requires_grad=False),
                'accuracy': torch.tensor(0.0),
                'probs': None,
                'y_true': None,
            }
    
    def train_step(self, batch: List[Dict]) -> Dict[str, float]:
        """Execute one training step over a batch of datasets."""
        self.model.train()
        self.optimizer.zero_grad()
        
        total_loss = 0.0
        total_acc = 0.0
        n_valid = 0
        
        # Accumulate gradients without scaling
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
                    # Backward without scaling - we'll scale by n_valid at the end
                    output['loss'].backward()
                    
                    total_loss += output['loss'].item()
                    total_acc += output['accuracy'].item()
                    n_valid += 1
                
                del output
                    
            except Exception as e:
                continue
        
        # Scale gradients by n_valid (correct averaging over successful datasets)
        if n_valid > 0:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad /= n_valid
        
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.grad_clip
            )
        
        self.optimizer.step()
        self.scheduler.step()
        
        self.current_step += 1
        
        return {
            'loss': total_loss / max(1, n_valid),
            'accuracy': total_acc / max(1, n_valid),
            'n_valid': n_valid,
            'lr': self.scheduler.get_last_lr()[0],
        }
    
    def evaluate_all(self, datasets: List[Dict], verbose: bool = False) -> List[Dict]:
        """Evaluate on ALL datasets, return per-dataset results."""
        self.model.eval()
        
        results = []
        
        with torch.no_grad():
            for i, data in enumerate(datasets):
                result = {'name': data['name'], 'n_classes': data['n_classes']}
                try:
                    eval_clf = TabPFNClassifier(
                        device=self.device,
                        n_estimators=self.config.n_estimators_eval,
                        ignore_pretraining_limits=True,
                    )
                    
                    eval_clf.fit(data['X_train'], data['y_train'])
                    eval_clf.model_.load_state_dict(self.model.state_dict())
                    eval_clf.model_.eval()
                    
                    proba = eval_clf.predict_proba(data['X_test'])
                    preds = proba.argmax(axis=1)
                    
                    acc = accuracy_score(data['y_test'], preds)
                    result['accuracy'] = float(acc)
                    
                    try:
                        if data['n_classes'] == 2:
                            auc = roc_auc_score(data['y_test'], proba[:, 1])
                        else:
                            auc = roc_auc_score(data['y_test'], proba, multi_class='ovr')
                        result['auc'] = float(auc)
                    except Exception:
                        result['auc'] = None
                    
                    result['status'] = 'success'
                        
                except Exception as e:
                    result['accuracy'] = None
                    result['auc'] = None
                    result['status'] = 'failed'
                    result['error'] = str(e)[:100]
                
                results.append(result)
        
        return results
    
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
            'eval_results': self.eval_results,
            'eval_steps': self.eval_steps,
            'config': self.config.__dict__,
        }
        if extra:
            checkpoint.update(extra)
        
        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_step = checkpoint['step']
        self.train_losses = checkpoint.get('train_losses', [])
        self.train_accs = checkpoint.get('train_accs', [])
        self.eval_results = checkpoint.get('eval_results', [])
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
    
    # Eval AUC (mean across all datasets)
    ax = axes[1, 0]
    if trainer.eval_results:
        mean_aucs = []
        for step_results in trainer.eval_results:
            aucs = [r['auc'] for r in step_results if r.get('auc') is not None]
            mean_aucs.append(np.mean(aucs) if aucs else 0)
        ax.plot(trainer.eval_steps, mean_aucs, 'o-')
        ax.set_xlabel('Step')
        ax.set_ylabel('Mean AUC')
        ax.set_title('Real Dataset AUC')
        if mean_aucs:
            ax.axhline(y=mean_aucs[0], color='r', linestyle='--', alpha=0.5, label='Initial')
        ax.legend()
    
    # Eval Accuracy
    ax = axes[1, 1]
    if trainer.eval_results:
        mean_accs = []
        for step_results in trainer.eval_results:
            accs = [r['accuracy'] for r in step_results if r.get('accuracy') is not None]
            mean_accs.append(np.mean(accs) if accs else 0)
        ax.plot(trainer.eval_steps, mean_accs, 'o-')
        ax.set_xlabel('Step')
        ax.set_ylabel('Mean Accuracy')
        ax.set_title('Real Dataset Accuracy')
        if mean_accs:
            ax.axhline(y=mean_accs[0], color='r', linestyle='--', alpha=0.5, label='Initial')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(log_dir / 'training_curves.png', dpi=150)
    plt.close()


def save_eval_results_json(trainer: TabPFNFineTuner, log_dir: Path):
    """Save detailed evaluation results to JSON."""
    results = {
        'eval_steps': trainer.eval_steps,
        'results_per_step': trainer.eval_results,
    }
    
    with open(log_dir / 'eval_results.json', 'w') as f:
        json.dump(results, f, indent=2)


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
    print("Using generator from 06_generator_experiments")
    print("=" * 60)
    
    # Initialize synthetic generator
    print("\nInitializing synthetic data generator...")
    synth_gen = SyntheticDataGenerator(config, seed=config.seed)
    
    # Load ALL real datasets
    print("\nLoading ALL real datasets for evaluation...")
    real_datasets = load_real_datasets(config)
    print(f"Loaded {len(real_datasets)} real datasets meeting constraints")
    
    for ds in real_datasets:
        print(f"  - {ds['name']}: {ds['n_samples']} samples, {ds['n_features']} features, {ds['n_classes']} classes")
    
    if not real_datasets:
        print("Warning: No real datasets found. Evaluation will be skipped.")
    
    # Initialize trainer
    trainer = TabPFNFineTuner(config)
    _trainer_instance = trainer
    
    # Resume from checkpoint
    if resume_from:
        trainer.load_checkpoint(Path(resume_from))
    
    # Initial evaluation on ALL datasets
    print(f"\nInitial evaluation on {len(real_datasets)} real datasets...")
    if real_datasets:
        eval_results = trainer.evaluate_all(real_datasets, verbose=False)
        
        aucs = [r['auc'] for r in eval_results if r.get('auc') is not None]
        accs = [r['accuracy'] for r in eval_results if r.get('accuracy') is not None]
        
        print(f"  BASELINE: AUC={np.mean(aucs):.4f}, Acc={np.mean(accs):.4f} ({len(aucs)}/{len(real_datasets)} OK)\n")
        
        trainer.eval_steps.append(trainer.current_step)
        trainer.eval_results.append(eval_results)
        trainer.baseline_auc = np.mean(aucs)  # Store for comparison
        
        save_eval_results_json(trainer, log_dir)
    
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
            try:
                data = synth_gen.generate_one()
                batch.append(data)
            except Exception as e:
                print(f"  Generation error: {e}")
                continue
        
        if not batch:
            print(f"Step {step}: No valid datasets generated")
            continue
        
        # Training step
        result = trainer.train_step(batch)
        
        trainer.train_losses.append(result['loss'])
        trainer.train_accs.append(result['accuracy'])
        
        step_time = time.time() - step_start
        
        # Log progress
        print(f"Step {step:5d} | Loss: {result['loss']:.4f} | "
              f"Acc: {result['accuracy']:.4f} | "
              f"LR: {result['lr']:.2e} | Valid: {result['n_valid']}/{len(batch)} | "
              f"Time: {step_time:.2f}s")
        
        # Evaluation on ALL datasets (every step)
        if (step + 1) % config.eval_every == 0 and real_datasets:
            eval_results = trainer.evaluate_all(real_datasets, verbose=False)
            
            aucs = [r['auc'] for r in eval_results if r.get('auc') is not None]
            accs = [r['accuracy'] for r in eval_results if r.get('accuracy') is not None]
            mean_auc = np.mean(aucs) if aucs else 0
            mean_acc = np.mean(accs) if accs else 0
            
            # Compact output: just the mean
            print(f"  >> REAL EVAL: AUC={mean_auc:.4f}, Acc={mean_acc:.4f} ({len(aucs)}/{len(real_datasets)} OK)")
            
            trainer.eval_steps.append(step + 1)
            trainer.eval_results.append(eval_results)
            
            # Save checkpoint only every 50 steps to save disk
            if (step + 1) % 50 == 0:
                trainer.save_checkpoint(
                    checkpoint_dir / f"checkpoint_step{step+1}.pt",
                    extra={'step_eval_results': eval_results}
                )
            
            # Save eval results
            save_eval_results_json(trainer, log_dir)
    
    # Final evaluation
    print("\n" + "=" * 50)
    print("FINAL EVALUATION")
    print("=" * 50)
    
    if real_datasets:
        final_results = trainer.evaluate_all(real_datasets, verbose=False)
        
        aucs = [r['auc'] for r in final_results if r.get('auc') is not None]
        accs = [r['accuracy'] for r in final_results if r.get('accuracy') is not None]
        
        baseline_auc = getattr(trainer, 'baseline_auc', np.mean(aucs))
        delta = np.mean(aucs) - baseline_auc
        
        print(f"  FINAL: AUC={np.mean(aucs):.4f} (delta={delta:+.4f}), Acc={np.mean(accs):.4f}")
        print("=" * 50)
    
    # Save final checkpoint
    trainer.save_checkpoint(
        checkpoint_dir / "checkpoint_final.pt",
        extra={'final_results': final_results if real_datasets else None}
    )
    
    # Save training history
    history = {
        'train_losses': trainer.train_losses,
        'train_accs': trainer.train_accs,
        'eval_steps': trainer.eval_steps,
        'config': config.__dict__,
        'total_time': time.time() - start_time,
    }
    
    with open(log_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    save_eval_results_json(trainer, log_dir)
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
        device=args.device,
        seed=args.seed,
    )
    
    # Debug mode overrides
    if args.debug:
        print("DEBUG MODE: Quick run with reduced settings")
        config.n_steps = 10
        config.batch_size = 4
        config.eval_every = 5
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run training
    train(config, resume_from=args.resume)


if __name__ == "__main__":
    main()
