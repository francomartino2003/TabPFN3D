"""
Evaluation utilities for Temporal Encoder + TabPFN.

This module provides:
1. Evaluation on synthetic validation set (fixed seed)
2. Evaluation on real datasets
3. Metrics computation (accuracy, CE loss, macro-F1)
4. Plotting utilities for training curves
"""
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json

from sklearn.metrics import accuracy_score, f1_score, log_loss

try:
    from .training_config import FullConfig
    from .model import TemporalTabPFN, LossComputer
    from .data_loader import DatasetSample, SyntheticDataLoader, RealDataLoader
    from .preprocessing_3d import Preprocessor3D, numpy_to_torch
except ImportError:
    from training_config import FullConfig
    from model import TemporalTabPFN, LossComputer
    from data_loader import DatasetSample, SyntheticDataLoader, RealDataLoader
    from preprocessing_3d import Preprocessor3D, numpy_to_torch


@dataclass
class EvaluationResult:
    """Results from evaluating on a dataset."""
    accuracy: float
    ce_loss: float
    macro_f1: float
    n_samples: int
    n_classes: int
    metadata: Dict[str, Any]


@dataclass
class BatchEvaluationResult:
    """Results from evaluating on multiple datasets."""
    mean_accuracy: float
    mean_ce_loss: float
    mean_macro_f1: float
    std_accuracy: float
    std_ce_loss: float
    std_macro_f1: float
    n_datasets: int
    per_dataset_results: List[EvaluationResult]


class Evaluator:
    """
    Evaluator for temporal encoder model.
    
    Handles evaluation on both synthetic and real datasets,
    computing relevant metrics.
    """
    
    def __init__(self, model: TemporalTabPFN, config: FullConfig):
        self.model = model
        self.config = config
        self.device = config.device
    
    @torch.no_grad()
    def evaluate_dataset(self, sample: DatasetSample) -> EvaluationResult:
        """
        Evaluate model on a single dataset.
        
        Args:
            sample: DatasetSample with train/test split
        
        Returns:
            EvaluationResult with metrics
        """
        self.model.eval()
        
        # Preprocess
        X_full = sample.X_full
        preprocessor = Preprocessor3D(self.config.preprocessing)
        X_proc = preprocessor.fit_transform(X_full)
        
        # Convert to tensors
        X_tensor = numpy_to_torch(X_proc, self.device)
        y_train = torch.from_numpy(sample.y_train).to(self.device)
        y_test = torch.from_numpy(sample.y_test).to(self.device)
        
        # Forward pass
        output = self.model.forward(X_tensor, y_train, sample.n_train)
        
        # Get predictions
        probs = output["probs"].cpu().numpy()
        logits = output["logits"].cpu().numpy()
        preds = probs.argmax(axis=-1)
        
        # Compute metrics
        y_test_np = sample.y_test
        
        accuracy = accuracy_score(y_test_np, preds)
        
        # Cross-entropy loss
        # Ensure proper format for log_loss
        if probs.shape[1] < sample.n_classes:
            # Pad probabilities if needed
            probs_padded = np.zeros((len(probs), sample.n_classes))
            probs_padded[:, :probs.shape[1]] = probs
            probs = probs_padded
        
        try:
            ce_loss = log_loss(y_test_np, probs, labels=range(sample.n_classes))
        except Exception:
            ce_loss = float('nan')
        
        # Macro F1
        try:
            macro_f1 = f1_score(y_test_np, preds, average='macro', zero_division=0)
        except Exception:
            macro_f1 = float('nan')
        
        return EvaluationResult(
            accuracy=accuracy,
            ce_loss=ce_loss,
            macro_f1=macro_f1,
            n_samples=sample.n_samples,
            n_classes=sample.n_classes,
            metadata=sample.metadata
        )
    
    def evaluate_batch(self, samples: List[DatasetSample]) -> BatchEvaluationResult:
        """
        Evaluate model on a batch of datasets.
        
        Args:
            samples: List of DatasetSample objects
        
        Returns:
            BatchEvaluationResult with aggregated metrics
        """
        results = []
        for sample in samples:
            try:
                result = self.evaluate_dataset(sample)
                results.append(result)
            except Exception as e:
                print(f"Warning: Failed to evaluate dataset: {e}")
                continue
        
        if not results:
            return BatchEvaluationResult(
                mean_accuracy=0.0,
                mean_ce_loss=float('inf'),
                mean_macro_f1=0.0,
                std_accuracy=0.0,
                std_ce_loss=0.0,
                std_macro_f1=0.0,
                n_datasets=0,
                per_dataset_results=[]
            )
        
        accuracies = [r.accuracy for r in results]
        ce_losses = [r.ce_loss for r in results if not np.isnan(r.ce_loss)]
        macro_f1s = [r.macro_f1 for r in results if not np.isnan(r.macro_f1)]
        
        return BatchEvaluationResult(
            mean_accuracy=np.mean(accuracies),
            mean_ce_loss=np.mean(ce_losses) if ce_losses else float('nan'),
            mean_macro_f1=np.mean(macro_f1s) if macro_f1s else float('nan'),
            std_accuracy=np.std(accuracies),
            std_ce_loss=np.std(ce_losses) if ce_losses else float('nan'),
            std_macro_f1=np.std(macro_f1s) if macro_f1s else float('nan'),
            n_datasets=len(results),
            per_dataset_results=results
        )


def evaluate_on_synthetic(
    model: TemporalTabPFN,
    config: FullConfig,
    n_datasets: int = 100,
    seed: int = 42
) -> BatchEvaluationResult:
    """
    Evaluate model on synthetic validation set.
    
    Uses a fixed seed for reproducibility across training runs.
    """
    # Create synthetic loader and generate fixed validation set
    data_config = config.data
    loader = SyntheticDataLoader(data_config, seed=seed)
    val_samples = loader.generate_fixed_validation_set(n_datasets, seed)
    
    # Evaluate
    evaluator = Evaluator(model, config)
    return evaluator.evaluate_batch(val_samples)


def evaluate_on_real(
    model: TemporalTabPFN,
    config: FullConfig
) -> BatchEvaluationResult:
    """
    Evaluate model on real datasets.
    """
    # Load real datasets
    loader = RealDataLoader(config.data.real_data_path, seed=config.training.seed)
    real_samples = loader.load_all()
    
    if not real_samples:
        print("Warning: No real datasets found")
        return BatchEvaluationResult(
            mean_accuracy=0.0,
            mean_ce_loss=float('nan'),
            mean_macro_f1=0.0,
            std_accuracy=0.0,
            std_ce_loss=0.0,
            std_macro_f1=0.0,
            n_datasets=0,
            per_dataset_results=[]
        )
    
    # Evaluate
    evaluator = Evaluator(model, config)
    return evaluator.evaluate_batch(real_samples)


@dataclass
class TrainingMetrics:
    """Container for training metrics history."""
    steps: List[int]
    train_losses: List[float]
    train_accuracies: List[float]
    val_synth_accuracies: List[float]
    val_synth_ce_losses: List[float]
    val_synth_f1s: List[float]
    val_real_accuracies: List[float]
    val_real_ce_losses: List[float]
    val_real_f1s: List[float]
    
    def __init__(self):
        self.steps = []
        self.train_losses = []
        self.train_accuracies = []
        self.val_synth_accuracies = []
        self.val_synth_ce_losses = []
        self.val_synth_f1s = []
        self.val_real_accuracies = []
        self.val_real_ce_losses = []
        self.val_real_f1s = []
    
    def add_train_metrics(self, step: int, loss: float, accuracy: float):
        """Add training metrics."""
        self.steps.append(step)
        self.train_losses.append(loss)
        self.train_accuracies.append(accuracy)
    
    def add_val_metrics(
        self,
        synth_result: BatchEvaluationResult,
        real_result: BatchEvaluationResult
    ):
        """Add validation metrics."""
        self.val_synth_accuracies.append(synth_result.mean_accuracy)
        self.val_synth_ce_losses.append(synth_result.mean_ce_loss)
        self.val_synth_f1s.append(synth_result.mean_macro_f1)
        
        self.val_real_accuracies.append(real_result.mean_accuracy)
        self.val_real_ce_losses.append(real_result.mean_ce_loss)
        self.val_real_f1s.append(real_result.mean_macro_f1)
    
    def save(self, path: str):
        """Save metrics to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump({
                "steps": self.steps,
                "train_losses": self.train_losses,
                "train_accuracies": self.train_accuracies,
                "val_synth_accuracies": self.val_synth_accuracies,
                "val_synth_ce_losses": self.val_synth_ce_losses,
                "val_synth_f1s": self.val_synth_f1s,
                "val_real_accuracies": self.val_real_accuracies,
                "val_real_ce_losses": self.val_real_ce_losses,
                "val_real_f1s": self.val_real_f1s,
            }, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingMetrics':
        """Load metrics from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        metrics = cls()
        metrics.steps = data["steps"]
        metrics.train_losses = data["train_losses"]
        metrics.train_accuracies = data["train_accuracies"]
        metrics.val_synth_accuracies = data["val_synth_accuracies"]
        metrics.val_synth_ce_losses = data["val_synth_ce_losses"]
        metrics.val_synth_f1s = data["val_synth_f1s"]
        metrics.val_real_accuracies = data["val_real_accuracies"]
        metrics.val_real_ce_losses = data["val_real_ce_losses"]
        metrics.val_real_f1s = data["val_real_f1s"]
        
        return metrics


def plot_training_curves(
    metrics: TrainingMetrics,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot training curves showing synthetic vs real performance.
    
    Creates a figure with:
    - Training loss
    - Validation accuracy (synthetic vs real)
    - Validation CE loss (synthetic vs real)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Get eval steps (assuming eval_every divides evenly)
    eval_steps = metrics.steps[::len(metrics.steps) // len(metrics.val_synth_accuracies)] \
        if metrics.val_synth_accuracies else []
    
    # Training loss
    ax = axes[0, 0]
    ax.plot(metrics.steps, metrics.train_losses, label='Train Loss', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Training accuracy
    ax = axes[0, 1]
    ax.plot(metrics.steps, metrics.train_accuracies, label='Train Acc', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Validation accuracy
    ax = axes[1, 0]
    if eval_steps and metrics.val_synth_accuracies:
        ax.plot(eval_steps, metrics.val_synth_accuracies, 
                label='Synthetic Val', marker='o', markersize=4)
        ax.plot(eval_steps, metrics.val_real_accuracies, 
                label='Real Val', marker='s', markersize=4)
    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation Accuracy (Synthetic vs Real)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Validation CE loss
    ax = axes[1, 1]
    if eval_steps and metrics.val_synth_ce_losses:
        ax.plot(eval_steps, metrics.val_synth_ce_losses, 
                label='Synthetic Val', marker='o', markersize=4)
        valid_real_losses = [l for l in metrics.val_real_ce_losses if not np.isnan(l)]
        if valid_real_losses:
            ax.plot(eval_steps[:len(valid_real_losses)], valid_real_losses, 
                    label='Real Val', marker='s', markersize=4)
    ax.set_xlabel('Step')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('Validation Loss (Synthetic vs Real)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    
    if show:
        plt.show()
    
    plt.close()

