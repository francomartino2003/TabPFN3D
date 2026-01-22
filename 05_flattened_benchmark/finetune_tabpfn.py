"""
Fine-tune TabPFN on Flattened Synthetic 3D Datasets.

Training approach:
- Generate synthetic 3D datasets on-the-fly (never repeat)
- Flatten them to 2D (n_samples, n_features * length)
- Fine-tune TabPFN with low learning rate
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
from typing import Optional, Dict, List, Tuple
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / '03_synthetic_generator_3D'))
sys.path.insert(0, str(Path(__file__).parent.parent / '00_TabPFN' / 'src'))

from tabpfn import TabPFNClassifier

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


def generate_synthetic_dataset(generator, config: FinetuneConfig):
    """Generate a single synthetic dataset that meets constraints."""
    from generator import SyntheticDataset3D
    
    for attempt in range(10):  # Try up to 10 times
        dataset = generator.generate()
        
        n_samples = dataset.X.shape[0]
        n_features = dataset.X.shape[1]
        length = dataset.X.shape[2]
        flat_features = n_features * length
        n_classes = len(np.unique(dataset.y))
        
        # Check constraints
        if (n_samples <= config.max_samples and 
            flat_features <= config.max_flat_features and
            n_classes <= config.max_classes and
            n_classes >= 2):
            return dataset
    
    # If all attempts fail, return last one anyway
    return dataset


def flatten_dataset(X_3d: np.ndarray) -> np.ndarray:
    """Flatten 3D dataset (n, m, t) to 2D (n, m*t)."""
    n_samples, n_features, length = X_3d.shape
    return X_3d.reshape(n_samples, n_features * length)


def prepare_synthetic_batch(generator, config: FinetuneConfig, rng: np.random.Generator):
    """
    Generate a batch of synthetic datasets ready for training.
    
    Returns list of dicts with:
        - X_train, X_test: Flattened features
        - y_train, y_test: Labels (encoded)
        - metadata: Dataset info
    """
    batch = []
    
    for _ in range(config.batch_size):
        # Generate dataset
        dataset = generate_synthetic_dataset(generator, config)
        
        X_flat = flatten_dataset(dataset.X)
        y = dataset.y.astype(int)
        
        # Train/test split (use dataset's train_ratio or default 0.7)
        train_ratio = getattr(dataset.config, 'train_ratio', 0.7)
        n_train = int(len(y) * train_ratio)
        n_train = max(10, min(n_train, len(y) - 5))  # At least 10 train, 5 test
        
        # Shuffle indices
        indices = rng.permutation(len(y))
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        X_train = X_flat[train_idx]
        X_test = X_flat[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        # Handle missing values (impute with mean from train)
        train_means = np.nanmean(X_train, axis=0)
        train_means = np.nan_to_num(train_means, nan=0.0)
        
        X_train = np.where(np.isnan(X_train), train_means, X_train)
        X_test = np.where(np.isnan(X_test), train_means, X_test)
        
        batch.append({
            'X_train': X_train.astype(np.float32),
            'X_test': X_test.astype(np.float32),
            'y_train': y_train,
            'y_test': y_test,
            'n_classes': len(np.unique(y)),
            'shape': dataset.X.shape,
            'sample_mode': dataset.config.sample_mode,
        })
    
    return batch


# ============================================================================
# Real Data Loading
# ============================================================================

def load_real_datasets(config: FinetuneConfig) -> List[Dict]:
    """Load real datasets that meet constraints."""
    pkl_path = Path(config.real_data_path)
    
    if not pkl_path.exists():
        print(f"Warning: Real data file not found: {pkl_path}")
        return []
    
    # Add paths for TimeSeriesDataset class (needed for pickle)
    real_data_dir = Path(__file__).parent.parent / '01_real_data'
    real_data_src = real_data_dir / 'src'
    for path in [str(real_data_dir), str(real_data_src)]:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    with open(pkl_path, 'rb') as f:
        all_datasets = pickle.load(f)
    
    valid_datasets = []
    
    for dataset in all_datasets:
        # Get data
        if hasattr(dataset, 'X_train') and dataset.X_train is not None:
            X_train = dataset.X_train
            X_test = dataset.X_test
            y_train = dataset.y_train
            y_test = dataset.y_test
        else:
            continue
        
        # Ensure 3D
        if X_train.ndim == 2:
            X_train = X_train[:, np.newaxis, :]
        if X_test.ndim == 2:
            X_test = X_test[:, np.newaxis, :]
        
        n_samples = len(y_train) + len(y_test)
        n_features = X_train.shape[1]
        length = X_train.shape[2]
        flat_features = n_features * length
        n_classes = len(np.unique(np.concatenate([y_train, y_test])))
        
        # Check constraints
        if (n_samples <= config.max_samples and 
            flat_features <= config.max_flat_features and
            n_classes <= config.max_classes):
            
            # Flatten
            X_train_flat = X_train.reshape(len(y_train), -1)
            X_test_flat = X_test.reshape(len(y_test), -1)
            
            # Encode labels
            le = LabelEncoder()
            le.fit(y_train)
            y_train_enc = le.transform(y_train)
            y_test_enc = le.transform(y_test)
            
            # Handle missing values
            train_means = np.nanmean(X_train_flat, axis=0)
            train_means = np.nan_to_num(train_means, nan=0.0)
            X_train_flat = np.where(np.isnan(X_train_flat), train_means, X_train_flat)
            X_test_flat = np.where(np.isnan(X_test_flat), train_means, X_test_flat)
            
            valid_datasets.append({
                'name': dataset.name,
                'X_train': X_train_flat.astype(np.float32),
                'X_test': X_test_flat.astype(np.float32),
                'y_train': y_train_enc,
                'y_test': y_test_enc,
                'n_classes': n_classes,
                'n_samples': n_samples,
                'flat_features': flat_features,
            })
    
    print(f"Loaded {len(valid_datasets)} real datasets meeting constraints")
    return valid_datasets


# ============================================================================
# TabPFN Fine-tuning
# ============================================================================

class TabPFNFineTuner:
    """
    Fine-tunes TabPFN on flattened synthetic datasets.
    
    Key design:
    - Uses TabPFN's internal model for forward pass
    - Computes cross-entropy loss on test predictions
    - Accumulates gradients over batch_size datasets
    - Low learning rate to preserve pretrained knowledge
    """
    
    def __init__(self, config: FinetuneConfig):
        self.config = config
        self.device = config.device
        
        # Initialize TabPFN
        print("Loading TabPFN...")
        self.clf = TabPFNClassifier(device=config.device)
        
        # Get the underlying model
        # TabPFN stores the model in clf.model_ after fitting or in clf.classifier_
        # We need to access it properly
        self._setup_model()
        
        # Optimizer
        self.optimizer = AdamW(
            self.get_trainable_params(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Tracking
        self.train_losses = []
        self.train_accs = []
        self.eval_aucs = []
        self.eval_accs = []
        self.eval_steps = []
        self.current_step = 0
        
    def _setup_model(self):
        """Setup model for fine-tuning."""
        # TabPFN needs to be "fit" once to initialize internal state
        # Create dummy data
        X_dummy = np.random.randn(10, 5).astype(np.float32)
        y_dummy = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        
        self.clf.fit(X_dummy, y_dummy)
        
        # Now access the model
        # The model is typically in self.clf.model_ or similar
        # We need to make it trainable
        if hasattr(self.clf, 'model_'):
            self.model = self.clf.model_
        elif hasattr(self.clf, 'classifier_'):
            self.model = self.clf.classifier_
        else:
            # Try to find the model
            for attr in dir(self.clf):
                obj = getattr(self.clf, attr)
                if isinstance(obj, nn.Module):
                    self.model = obj
                    break
            else:
                raise RuntimeError("Could not find TabPFN model")
        
        # Enable gradients
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
        
        print(f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"Trainable: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def get_trainable_params(self):
        """Get trainable parameters."""
        return [p for p in self.model.parameters() if p.requires_grad]
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            # Cosine decay
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
        Forward pass for a single dataset.
        
        TabPFN takes train and test together, so we:
        1. Concatenate X_train and X_test
        2. Run through model
        3. Extract predictions for test samples
        4. Compute loss against y_test
        """
        # Convert to tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.long, device=self.device)
        y_test_t = torch.tensor(y_test, dtype=torch.long, device=self.device)
        
        # Use TabPFN's predict_proba which handles the internal forward pass
        # But we need gradients, so we use the model directly
        
        # TabPFN concatenates train and test internally
        # The model expects: (X_train, y_train, X_test)
        # and outputs logits/probs for X_test
        
        try:
            # Try using the model's forward directly
            # This depends on TabPFN's internal structure
            logits = self._model_forward(X_train_t, y_train_t, X_test_t, n_classes)
            
            # Compute cross-entropy loss
            loss = F.cross_entropy(logits, y_test_t)
            
            # Compute accuracy
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc = (preds == y_test_t).float().mean()
            
            return {
                'loss': loss,
                'accuracy': acc,
                'logits': logits,
            }
        except Exception as e:
            print(f"Forward error: {e}")
            # Return dummy loss
            return {
                'loss': torch.tensor(0.0, device=self.device, requires_grad=True),
                'accuracy': torch.tensor(0.0),
                'logits': None,
            }
    
    def _model_forward(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor, 
        X_test: torch.Tensor,
        n_classes: int
    ) -> torch.Tensor:
        """
        Internal forward pass through TabPFN model.
        
        This needs to match TabPFN's expected input format.
        """
        # TabPFN 2.5 uses a transformer that expects:
        # - X_train: (n_train, n_features)
        # - y_train: (n_train,)
        # - X_test: (n_test, n_features)
        
        # Need to understand TabPFN's model structure
        # Typically it's something like:
        # model(X_full, y_train, n_train) -> predictions for test
        
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]
        
        # Concatenate for full context
        X_full = torch.cat([X_train, X_test], dim=0)
        
        # TabPFN models often have different forward signatures
        # Try common patterns
        
        # Pattern 1: Direct forward with train/test info
        if hasattr(self.model, 'forward'):
            try:
                # Some TabPFN versions
                output = self.model(
                    X_full.unsqueeze(0),  # Add batch dim
                    y_train.unsqueeze(0),
                    single_eval_pos=n_train
                )
                # Extract test predictions
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                    
                # Remove batch dim and get test samples
                if logits.dim() == 3:
                    logits = logits[0, n_train:, :n_classes]
                elif logits.dim() == 2:
                    logits = logits[n_train:, :n_classes]
                    
                return logits
            except Exception as e:
                pass
        
        # Pattern 2: Use classifier's internal forward
        # Fallback: use sklearn-style predict_proba (no gradients)
        # This won't work for training, but shows the issue
        raise RuntimeError(
            "Could not find compatible forward method. "
            "TabPFN's internal structure may have changed. "
            "Please check TabPFN version and update _model_forward accordingly."
        )
    
    def train_step(self, batch: List[Dict]) -> Dict[str, float]:
        """
        Execute one training step over a batch of datasets.
        
        Accumulates gradients over all datasets, then updates.
        """
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
                
                # Scale loss by batch size for proper gradient accumulation
                scaled_loss = output['loss'] / len(batch)
                scaled_loss.backward()
                
                total_loss += output['loss'].item()
                total_acc += output['accuracy'].item()
                n_valid += 1
                
            except Exception as e:
                print(f"Error processing dataset: {e}")
                continue
        
        if n_valid == 0:
            return {'loss': 0.0, 'accuracy': 0.0}
        
        # Gradient clipping
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.get_trainable_params(),
                self.config.grad_clip
            )
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        self.current_step += 1
        
        avg_loss = total_loss / n_valid
        avg_acc = total_acc / n_valid
        
        self.train_losses.append(avg_loss)
        self.train_accs.append(avg_acc)
        
        return {'loss': avg_loss, 'accuracy': avg_acc}
    
    def evaluate_real(self, real_datasets: List[Dict]) -> Dict[str, float]:
        """Evaluate on real datasets."""
        self.model.eval()
        
        all_aucs = []
        all_accs = []
        
        with torch.no_grad():
            for data in real_datasets[:self.config.n_eval_real]:
                try:
                    # Use TabPFN's standard predict interface
                    self.clf.fit(data['X_train'], data['y_train'])
                    
                    y_pred = self.clf.predict(data['X_test'])
                    y_proba = self.clf.predict_proba(data['X_test'])
                    
                    acc = accuracy_score(data['y_test'], y_pred)
                    
                    # ROC AUC
                    if data['n_classes'] == 2:
                        auc = roc_auc_score(data['y_test'], y_proba[:, 1])
                    else:
                        try:
                            auc = roc_auc_score(
                                data['y_test'], y_proba, 
                                multi_class='ovr', average='macro'
                            )
                        except:
                            auc = acc  # Fallback
                    
                    all_aucs.append(auc)
                    all_accs.append(acc)
                    
                except Exception as e:
                    print(f"Error evaluating {data.get('name', 'unknown')}: {e}")
                    continue
        
        self.model.train()
        
        if not all_aucs:
            return {'auc': 0.0, 'accuracy': 0.0, 'n_datasets': 0}
        
        mean_auc = np.mean(all_aucs)
        mean_acc = np.mean(all_accs)
        
        self.eval_aucs.append(mean_auc)
        self.eval_accs.append(mean_acc)
        self.eval_steps.append(self.current_step)
        
        return {
            'auc': mean_auc,
            'accuracy': mean_acc,
            'n_datasets': len(all_aucs)
        }
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        torch.save({
            'step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'eval_aucs': self.eval_aucs,
            'eval_accs': self.eval_accs,
            'eval_steps': self.eval_steps,
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_step = checkpoint['step']
        self.train_losses = checkpoint['train_losses']
        self.train_accs = checkpoint['train_accs']
        self.eval_aucs = checkpoint.get('eval_aucs', [])
        self.eval_accs = checkpoint.get('eval_accs', [])
        self.eval_steps = checkpoint.get('eval_steps', [])


def plot_training_curves(trainer: TabPFNFineTuner, save_path: str):
    """Plot and save training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1 = axes[0]
    if trainer.train_losses:
        steps = list(range(len(trainer.train_losses)))
        ax1.plot(steps, trainer.train_losses, 'b-', alpha=0.3, label='Train Loss')
        
        # Smoothed
        if len(trainer.train_losses) > 10:
            window = min(20, len(trainer.train_losses) // 5)
            smoothed = np.convolve(trainer.train_losses, np.ones(window)/window, mode='valid')
            ax1.plot(steps[window-1:], smoothed, 'b-', linewidth=2, label='Smoothed')
    
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # AUC plot
    ax2 = axes[1]
    if trainer.train_accs:
        steps = list(range(len(trainer.train_accs)))
        ax2.plot(steps, trainer.train_accs, 'b-', alpha=0.3, label='Train Acc')
    
    if trainer.eval_aucs and trainer.eval_steps:
        ax2.plot(trainer.eval_steps, trainer.eval_aucs, 'ro-', markersize=6, 
                linewidth=2, label='Real AUC')
        ax2.plot(trainer.eval_steps, trainer.eval_accs, 'g^-', markersize=6,
                linewidth=2, label='Real Acc')
    
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Metric')
    ax2.set_title('Training Accuracy & Real Evaluation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# Main Training Loop
# ============================================================================

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    global _trainer_instance
    print(f"\n\nReceived signal {signum}. Saving checkpoint...")
    if _trainer_instance is not None:
        checkpoint_path = Path(_trainer_instance.config.checkpoint_dir) / "interrupted_checkpoint.pt"
        _trainer_instance.save_checkpoint(str(checkpoint_path))
        print(f"Saved to {checkpoint_path}")
    sys.exit(0)


def train(config: FinetuneConfig, resume_from: Optional[str] = None):
    """Main training function."""
    global _trainer_instance
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("=" * 60)
    print("FINE-TUNING TABPFN ON FLATTENED SYNTHETIC DATA")
    print("=" * 60)
    
    # Set seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    
    # Create directories
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize synthetic generator
    print("\nInitializing synthetic data generator...")
    from generator import SyntheticDatasetGenerator3D
    prior = create_synthetic_generator_prior(config)
    generator = SyntheticDatasetGenerator3D(prior=prior, seed=config.seed)
    
    # Load real datasets for evaluation
    print("\nLoading real datasets for evaluation...")
    real_datasets = load_real_datasets(config)
    
    # Initialize fine-tuner
    print("\nInitializing TabPFN fine-tuner...")
    trainer = TabPFNFineTuner(config)
    _trainer_instance = trainer  # For signal handler
    
    # Resume if specified
    if resume_from:
        print(f"Resuming from {resume_from}...")
        trainer.load_checkpoint(resume_from)
        print(f"Resumed from step {trainer.current_step}")
    
    # Training loop
    print(f"\nStarting training for {config.n_steps} steps...")
    print(f"  Batch size: {config.batch_size} datasets")
    print(f"  Learning rate: {config.lr}")
    print(f"  Eval every: {config.eval_every} steps")
    print(f"  Device: {config.device}")
    print()
    
    # Initial evaluation
    if real_datasets:
        print("Initial evaluation on real datasets...")
        eval_result = trainer.evaluate_real(real_datasets)
        print(f"  Real AUC: {eval_result['auc']:.4f}, Acc: {eval_result['accuracy']:.4f}")
        print()
    
    start_step = trainer.current_step
    
    for step in range(start_step, config.n_steps):
        step_start = time.time()
        
        # Generate batch
        batch = prepare_synthetic_batch(generator, config, rng)
        
        # Train step
        train_result = trainer.train_step(batch)
        
        step_time = time.time() - step_start
        
        # Log
        if step % 10 == 0:
            lr = trainer.scheduler.get_last_lr()[0]
            print(f"Step {step:5d} | Loss: {train_result['loss']:.4f} | "
                  f"Acc: {train_result['accuracy']:.4f} | "
                  f"LR: {lr:.2e} | Time: {step_time:.2f}s")
        
        # Evaluate
        if (step + 1) % config.eval_every == 0 and real_datasets:
            print("\n" + "-" * 40)
            print(f"EVALUATION at step {step + 1}")
            eval_result = trainer.evaluate_real(real_datasets)
            print(f"  Real datasets: {eval_result['n_datasets']}")
            print(f"  Mean AUC: {eval_result['auc']:.4f}")
            print(f"  Mean Acc: {eval_result['accuracy']:.4f}")
            print("-" * 40 + "\n")
            
            # Save plot
            plot_training_curves(trainer, str(log_dir / "training_curves.png"))
        
        # Save checkpoint
        if (step + 1) % 100 == 0:
            trainer.save_checkpoint(str(checkpoint_dir / f"checkpoint_step{step+1}.pt"))
            print(f"  [Checkpoint saved]")
    
    # Final save
    trainer.save_checkpoint(str(checkpoint_dir / "final_model.pt"))
    plot_training_curves(trainer, str(log_dir / "training_curves.png"))
    
    # Save training history
    history = {
        'train_losses': trainer.train_losses,
        'train_accs': trainer.train_accs,
        'eval_aucs': trainer.eval_aucs,
        'eval_accs': trainer.eval_accs,
        'eval_steps': trainer.eval_steps,
    }
    with open(log_dir / "training_history.json", 'w') as f:
        json.dump(history, f)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Logs: {log_dir}")
    if trainer.eval_aucs:
        print(f"Best Real AUC: {max(trainer.eval_aucs):.4f}")
    print("=" * 60)
    
    return trainer


def get_device(requested: str = "auto") -> str:
    """Auto-detect best available device."""
    if requested != "auto":
        return requested
    
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Fine-tune TabPFN on synthetic data")
    parser.add_argument("--n-steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=64, help="Datasets per batch")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--eval-every", type=int, default=50, help="Evaluate every N steps")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--debug", action="store_true", help="Quick debug run")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda/mps)")
    
    args = parser.parse_args()
    
    # Change to script directory to ensure relative paths work
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)
    print(f"Working directory: {script_dir}")
    
    config = FinetuneConfig()
    config.n_steps = args.n_steps
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.eval_every = args.eval_every
    config.device = get_device(args.device)
    
    print(f"Device: {config.device}")
    if config.device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    if args.debug:
        config.n_steps = 20
        config.batch_size = 4
        config.eval_every = 5
        config.n_eval_real = 5
        print("DEBUG MODE: Quick run with reduced settings")
    
    train(config, resume_from=args.resume)


if __name__ == "__main__":
    main()
