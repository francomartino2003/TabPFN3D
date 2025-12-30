"""
Training Loop for Temporal Encoder + TabPFN.

This script implements TabPFN-style training where:
- Each batch contains multiple complete datasets
- Each dataset has its own train/test split
- Loss is computed only on test samples
- Only the temporal encoder is trained (TabPFN is frozen)

Usage:
    python train.py --config config.json
    python train.py --debug  # Quick debug run
"""
import argparse
import sys
from pathlib import Path
from typing import Optional
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

try:
    from .training_config import FullConfig, get_default_config, get_debug_config
    from .model import TemporalTabPFN, LossComputer
    from .data_loader import SyntheticDataLoader, DatasetSample
    from .evaluate import (
        Evaluator, 
        evaluate_on_synthetic, 
        evaluate_on_real,
        TrainingMetrics,
        plot_training_curves
    )
    from .preprocessing_3d import Preprocessor3D, numpy_to_torch
except ImportError:
    from training_config import FullConfig, get_default_config, get_debug_config
    from model import TemporalTabPFN, LossComputer
    from data_loader import SyntheticDataLoader, DatasetSample
    from evaluate import (
        Evaluator, 
        evaluate_on_synthetic, 
        evaluate_on_real,
        TrainingMetrics,
        plot_training_curves
    )
    from preprocessing_3d import Preprocessor3D, numpy_to_torch


def get_linear_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int
) -> LambdaLR:
    """Create linear warmup scheduler."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))
    
    return LambdaLR(optimizer, lr_lambda)


def train_step(
    model: TemporalTabPFN,
    optimizer: torch.optim.Optimizer,
    samples: list,
    config: FullConfig
) -> dict:
    """
    Execute one training step on a batch of datasets.
    
    Args:
        model: The TemporalTabPFN model
        optimizer: The optimizer
        samples: List of DatasetSample objects
        config: Configuration
    
    Returns:
        dict with "loss", "accuracy"
    """
    model.train()
    optimizer.zero_grad()
    
    total_loss = 0.0
    total_accuracy = 0.0
    n_datasets = len(samples)
    
    for sample in samples:
        # Preprocess this dataset
        X_full = sample.X_full
        preprocessor = Preprocessor3D(config.preprocessing)
        X_proc = preprocessor.fit_transform(X_full)
        
        # Convert to tensors
        X_tensor = numpy_to_torch(X_proc, config.device)
        y_train = torch.from_numpy(sample.y_train).to(config.device)
        y_test = torch.from_numpy(sample.y_test).to(config.device)
        
        # Forward and compute loss
        output = model.compute_loss(
            X_tensor,
            y_train,
            y_test,
            sample.n_train
        )
        
        # Accumulate loss (will backward later)
        total_loss += output["loss"]
        total_accuracy += output["accuracy"].item()
    
    # Average loss and backward
    avg_loss = total_loss / n_datasets
    avg_loss.backward()
    
    # Gradient clipping
    if config.training.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(
            model.encoder.parameters(),
            config.training.grad_clip
        )
    
    optimizer.step()
    
    return {
        "loss": avg_loss.item(),
        "accuracy": total_accuracy / n_datasets,
    }


def train(
    config: FullConfig,
    resume_from: Optional[str] = None
) -> TemporalTabPFN:
    """
    Main training function.
    
    Args:
        config: Full configuration
        resume_from: Path to checkpoint to resume from
    
    Returns:
        Trained model
    """
    print("=" * 60)
    print("TEMPORAL ENCODER + TABPFN TRAINING")
    print("=" * 60)
    
    # Set seed
    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)
    
    # Create directories
    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config.training.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config.save(checkpoint_dir / "config.json")
    
    # Initialize model
    print("\nInitializing model...")
    model = TemporalTabPFN(config)
    model = model.to(config.device)
    
    print(f"  Trainable parameters (encoder): {model.get_num_trainable_params():,}")
    print(f"  Frozen parameters (TabPFN): {model.get_num_frozen_params():,}")
    
    # Optimizer (only encoder parameters)
    optimizer = AdamW(
        model.get_trainable_parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay
    )
    
    # Scheduler
    scheduler = get_linear_warmup_scheduler(
        optimizer,
        config.training.warmup_steps,
        config.training.n_steps
    )
    
    # Resume if specified
    start_step = 0
    if resume_from:
        checkpoint = torch.load(resume_from)
        model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_step = checkpoint["step"]
        print(f"Resumed from step {start_step}")
    
    # Data loader
    print("\nInitializing data loader...")
    data_loader = SyntheticDataLoader(config.data, seed=config.training.seed)
    
    # Metrics tracker
    metrics = TrainingMetrics()
    
    # Wandb (optional)
    if config.training.wandb_project:
        try:
            import wandb
            wandb.init(
                project=config.training.wandb_project,
                name=config.training.wandb_run_name or config.experiment_name,
                config=config.__dict__
            )
        except ImportError:
            print("Warning: wandb not installed, skipping logging")
    
    # Training loop
    print(f"\nStarting training for {config.training.n_steps} steps...")
    print(f"  Batch size: {config.training.batch_datasets} datasets")
    print(f"  Eval every: {config.training.eval_every} steps")
    print()
    
    best_val_accuracy = 0.0
    
    for step in range(start_step, config.training.n_steps):
        step_start = time.time()
        
        # Generate batch of datasets
        samples = data_loader.generate_batch(config.training.batch_datasets)
        
        # Training step
        train_output = train_step(model, optimizer, samples, config)
        scheduler.step()
        
        # Log
        metrics.add_train_metrics(step, train_output["loss"], train_output["accuracy"])
        
        step_time = time.time() - step_start
        
        # Print progress
        if step % 10 == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"Step {step:5d} | Loss: {train_output['loss']:.4f} | "
                  f"Acc: {train_output['accuracy']:.4f} | "
                  f"LR: {lr:.2e} | Time: {step_time:.2f}s")
        
        # Wandb logging
        if config.training.wandb_project:
            try:
                wandb.log({
                    "train/loss": train_output["loss"],
                    "train/accuracy": train_output["accuracy"],
                    "train/lr": scheduler.get_last_lr()[0],
                }, step=step)
            except:
                pass
        
        # Evaluation
        if (step + 1) % config.training.eval_every == 0:
            print("\n" + "-" * 40)
            print(f"EVALUATION at step {step + 1}")
            print("-" * 40)
            
            # Evaluate on synthetic
            synth_result = evaluate_on_synthetic(
                model, config,
                n_datasets=config.training.val_synth_size,
                seed=config.training.val_synth_seed
            )
            print(f"Synthetic Val: Acc={synth_result.mean_accuracy:.4f} | "
                  f"CE={synth_result.mean_ce_loss:.4f} | "
                  f"F1={synth_result.mean_macro_f1:.4f}")
            
            # Evaluate on real
            real_result = evaluate_on_real(model, config)
            if real_result.n_datasets > 0:
                print(f"Real Val ({real_result.n_datasets} datasets): "
                      f"Acc={real_result.mean_accuracy:.4f} | "
                      f"F1={real_result.mean_macro_f1:.4f}")
            else:
                print("Real Val: No datasets available")
            
            # Add to metrics
            metrics.add_val_metrics(synth_result, real_result)
            
            # Wandb logging
            if config.training.wandb_project:
                try:
                    wandb.log({
                        "val_synth/accuracy": synth_result.mean_accuracy,
                        "val_synth/ce_loss": synth_result.mean_ce_loss,
                        "val_synth/macro_f1": synth_result.mean_macro_f1,
                        "val_real/accuracy": real_result.mean_accuracy,
                        "val_real/macro_f1": real_result.mean_macro_f1,
                    }, step=step)
                except:
                    pass
            
            # Save best model
            if synth_result.mean_accuracy > best_val_accuracy:
                best_val_accuracy = synth_result.mean_accuracy
                torch.save({
                    "step": step + 1,
                    "encoder_state_dict": model.encoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_accuracy": best_val_accuracy,
                }, checkpoint_dir / "best_model.pt")
                print(f"  -> New best model saved (acc={best_val_accuracy:.4f})")
            
            print("-" * 40 + "\n")
        
        # Periodic checkpoint
        if (step + 1) % config.training.save_every == 0:
            torch.save({
                "step": step + 1,
                "encoder_state_dict": model.encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }, checkpoint_dir / f"checkpoint_step{step+1}.pt")
            
            # Save metrics
            metrics.save(log_dir / "metrics.json")
    
    # Final save
    torch.save({
        "step": config.training.n_steps,
        "encoder_state_dict": model.encoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, checkpoint_dir / "final_model.pt")
    
    metrics.save(log_dir / "metrics.json")
    
    # Plot training curves
    try:
        plot_training_curves(
            metrics,
            save_path=str(log_dir / "training_curves.png"),
            show=False
        )
    except Exception as e:
        print(f"Warning: Failed to plot training curves: {e}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Logs saved to: {log_dir}")
    print("=" * 60)
    
    return model


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train Temporal Encoder + TabPFN")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--debug", action="store_true", help="Quick debug run")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        config = FullConfig.load(args.config)
    elif args.debug:
        config = get_debug_config()
        print("Using DEBUG configuration (quick run)")
    else:
        config = get_default_config()
    
    # Override device if specified
    if args.device:
        config.device = args.device
    
    # Check device availability
    if config.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        config.device = "cpu"
    
    # Train
    model = train(config, resume_from=args.resume)
    
    return model


if __name__ == "__main__":
    main()

