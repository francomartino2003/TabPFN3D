"""
Training Loop for Temporal Encoder + TabPFN.

This script implements TabPFN-style training where:
- Each batch contains multiple complete datasets (default: 64)
- Each dataset has its own train/test split
- Loss is computed only on test samples
- Only the temporal encoder is trained (TabPFN is frozen)
- CUDA is used by default if available
- Training can be interrupted (Ctrl+C) and resumed later

Usage:
    python train.py --config config.json
    python train.py --debug  # Quick debug run
    python train.py --debug --multi-gpu  # Debug with multiple GPUs
    python train.py --resume checkpoints/checkpoint_step1000.pt
    python train.py --multi-gpu  # Full training with multiple GPUs
"""
import argparse
import signal
import sys
from pathlib import Path
from typing import Optional, List
import time
import json
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import math
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
        plot_training_curves,
        BatchEvaluationResult
    )
    from .preprocessing_3d import Preprocessor3D, numpy_to_torch
    from .multi_gpu import MultiGPUTrainer, get_available_gpus
except ImportError:
    from training_config import FullConfig, get_default_config, get_debug_config
    from model import TemporalTabPFN, LossComputer
    from data_loader import SyntheticDataLoader, DatasetSample
    from evaluate import (
        Evaluator, 
        evaluate_on_synthetic, 
        evaluate_on_real,
        TrainingMetrics,
        plot_training_curves,
        BatchEvaluationResult
    )
    from preprocessing_3d import Preprocessor3D, numpy_to_torch
    from multi_gpu import MultiGPUTrainer, get_available_gpus


def check_cuda_info() -> dict:
    """Check CUDA availability and return GPU info."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": 0,
        "device_name": None,
        "memory_total_gb": 0,
        "memory_free_gb": 0,
    }
    
    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        info["device_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["memory_total_gb"] = props.total_memory / 1e9
        try:
            free, total = torch.cuda.mem_get_info(0)
            info["memory_free_gb"] = free / 1e9
        except:
            pass
    
    return info


def plot_training_progress(
    train_losses: list,
    train_accs: list,
    val_synth_losses: list,
    val_synth_accs: list,
    val_real_accs: list,
    steps: list,
    val_steps: list,
    save_path: str
):
    """
    Plot and save training curves during training.
    
    Updates the plot file on each call.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1 = axes[0]
    if train_losses:
        ax1.plot(steps, train_losses, 'b-', alpha=0.3, label='Train Loss')
        # Smoothed version
        if len(train_losses) > 10:
            window = min(50, len(train_losses) // 5)
            smoothed = np.convolve(train_losses, np.ones(window)/window, mode='valid')
            ax1.plot(steps[window-1:], smoothed, 'b-', linewidth=2, label='Train Loss (smoothed)')
    
    if val_synth_losses:
        ax1.plot(val_steps, val_synth_losses, 'go-', markersize=4, label='Val Synth Loss')
    
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2 = axes[1]
    if train_accs:
        ax2.plot(steps, train_accs, 'b-', alpha=0.3, label='Train Acc')
        if len(train_accs) > 10:
            window = min(50, len(train_accs) // 5)
            smoothed = np.convolve(train_accs, np.ones(window)/window, mode='valid')
            ax2.plot(steps[window-1:], smoothed, 'b-', linewidth=2, label='Train Acc (smoothed)')
    
    if val_synth_accs:
        ax2.plot(val_steps, val_synth_accs, 'go-', markersize=4, label='Val Synth Acc')
    
    if val_real_accs:
        ax2.plot(val_steps, val_real_accs, 'r^-', markersize=4, label='Val Real Acc')
    
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def get_linear_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.01
) -> LambdaLR:
    """
    Create linear warmup + cosine annealing scheduler.
    
    This is the standard scheduler used in transformer training:
    1. Linear warmup from 0 to lr over warmup_steps
    2. Cosine decay from lr to min_lr over remaining steps
    
    Args:
        optimizer: The optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr_ratio: Minimum LR as fraction of initial LR (default 1%)
    
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup: 0 -> 1
            return step / max(1, warmup_steps)
        else:
            # Cosine annealing: 1 -> min_lr_ratio
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            # Scale between min_lr_ratio and 1.0
            return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay
    
    return LambdaLR(optimizer, lr_lambda)


def get_linear_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int
) -> LambdaLR:
    """Create linear warmup + linear decay scheduler (legacy)."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))
    
    return LambdaLR(optimizer, lr_lambda)


def forward_backward_step(
    model: TemporalTabPFN,
    sample,
    config: FullConfig,
    accumulation_steps: int = 1
) -> dict:
    """
    Execute forward and backward pass for a single dataset.
    Does NOT call optimizer.step() - gradients are accumulated.
    
    Args:
        model: The TemporalTabPFN model
        sample: A DatasetSample object
        config: Configuration
        accumulation_steps: Number of steps to accumulate (for loss scaling)
    
    Returns:
        dict with "loss", "accuracy"
    """
    model.train()
    
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
    
    # Scale loss by accumulation steps and backward
    # This ensures the total gradient is correctly averaged
    scaled_loss = output["loss"] / accumulation_steps
    scaled_loss.backward()
    
    return {
        "loss": output["loss"].item(),
        "accuracy": output["accuracy"].item(),
    }


def optimizer_step(
    model: TemporalTabPFN,
    optimizer: torch.optim.Optimizer,
    config: FullConfig
):
    """
    Perform optimizer step with gradient clipping.
    Call this after accumulating gradients.
    """
    # Gradient clipping
    if config.training.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(
            model.encoder.parameters(),
            config.training.grad_clip
        )
    
    optimizer.step()
    optimizer.zero_grad()


def train(
    config: FullConfig,
    resume_from: Optional[str] = None,
    multi_gpu: bool = False
) -> TemporalTabPFN:
    """
    Main training function.
    
    Args:
        config: Full configuration
        resume_from: Path to checkpoint to resume from
        multi_gpu: Whether to use multiple GPUs (if available)
    
    Returns:
        Trained model
    """
    print("=" * 60)
    print("TEMPORAL ENCODER + TABPFN TRAINING")
    print("=" * 60)
    
    # Check CUDA
    cuda_info = check_cuda_info()
    print(f"\n[GPU Info]")
    if cuda_info["cuda_available"]:
        print(f"  CUDA available: Yes")
        print(f"  Device: {cuda_info['device_name']}")
        print(f"  Memory: {cuda_info['memory_total_gb']:.1f} GB total, "
              f"{cuda_info['memory_free_gb']:.1f} GB free")
        
        # Show all available GPUs
        all_gpus = get_available_gpus()
        print(f"  Total GPUs: {len(all_gpus)}")
        if multi_gpu and len(all_gpus) > 1:
            print(f"  Multi-GPU: ENABLED")
            for gpu in all_gpus:
                print(f"    GPU {gpu['id']}: {gpu['name']} ({gpu['total_memory_gb']:.1f} GB)")
        elif multi_gpu:
            print(f"  Multi-GPU: requested but only 1 GPU available")
            multi_gpu = False
        
        if config.device == "cpu":
            print(f"  WARNING: Using CPU despite CUDA being available!")
    else:
        print(f"  CUDA available: No")
        print(f"  Using CPU")
        config.device = "cpu"
        multi_gpu = False
    
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
    
    # Initialize multi-GPU trainer if requested
    multi_gpu_trainer = None
    if multi_gpu:
        multi_gpu_trainer = MultiGPUTrainer(model, config)
        if multi_gpu_trainer.n_gpus < 2:
            multi_gpu_trainer = None
            multi_gpu = False
    
    # Optimizer (only encoder parameters)
    optimizer = AdamW(
        model.get_trainable_parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay
    )
    
    # Scheduler: Linear warmup + Cosine annealing
    scheduler = get_linear_warmup_cosine_scheduler(
        optimizer,
        config.training.warmup_steps,
        config.training.n_steps,
        min_lr_ratio=0.01  # Decay to 1% of initial LR
    )
    
    # Tracking lists for plotting
    train_losses = []
    train_accs = []
    train_steps = []
    val_synth_losses = []
    val_synth_accs = []
    val_real_accs = []
    val_steps = []
    
    best_val_accuracy = 0.0
    start_step = 0
    
    # Resume if specified
    if resume_from:
        print(f"\nResuming from {resume_from}...")
        checkpoint = torch.load(resume_from, map_location=config.device, weights_only=False)
        model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_step = checkpoint["step"]
        best_val_accuracy = checkpoint.get("best_val_accuracy", 0.0)
        
        # Load existing metrics if available
        metrics_path = log_dir / "training_history.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                history = json.load(f)
                train_losses = history.get("train_losses", [])
                train_accs = history.get("train_accs", [])
                train_steps = history.get("train_steps", [])
                val_synth_losses = history.get("val_synth_losses", [])
                val_synth_accs = history.get("val_synth_accs", [])
                val_real_accs = history.get("val_real_accs", [])
                val_steps = history.get("val_steps", [])
        
        print(f"  Resumed from step {start_step}")
        print(f"  Best val accuracy so far: {best_val_accuracy:.4f}")
    
    # Data loader
    print("\nInitializing data loader...")
    data_loader = SyntheticDataLoader(config.data, seed=config.training.seed)
    
    # Prefetching executor for parallel data generation
    prefetch_executor = ThreadPoolExecutor(max_workers=1)
    
    # Metrics tracker
    metrics = TrainingMetrics()
    
    # Wandb (optional)
    if config.training.wandb_project:
        try:
            import wandb
            wandb.init(
                project=config.training.wandb_project,
                name=config.training.wandb_run_name or config.experiment_name,
                config=config.__dict__,
                resume="allow" if resume_from else None
            )
        except ImportError:
            print("Warning: wandb not installed, skipping logging")
    
    # Interrupt handler for graceful shutdown
    interrupted = [False]  # Use list to allow modification in nested function
    
    def save_checkpoint_on_interrupt():
        """Save checkpoint when training is interrupted."""
        print("\n\n" + "=" * 60)
        print("TRAINING INTERRUPTED - Saving checkpoint...")
        
        # Save current state
        torch.save({
            "step": current_step + 1,
            "encoder_state_dict": model.encoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_accuracy": best_val_accuracy,
        }, checkpoint_dir / "interrupted_checkpoint.pt")
        
        # Save training history
        history = {
            "train_losses": train_losses,
            "train_accs": train_accs,
            "train_steps": train_steps,
            "val_synth_losses": val_synth_losses,
            "val_synth_accs": val_synth_accs,
            "val_real_accs": val_real_accs,
            "val_steps": val_steps,
        }
        with open(log_dir / "training_history.json", 'w') as f:
            json.dump(history, f)
        
        # Save final plot
        if train_losses:
            plot_training_progress(
                train_losses, train_accs,
                val_synth_losses, val_synth_accs, val_real_accs,
                train_steps, val_steps,
                str(log_dir / "training_curves.png")
            )
        
        print(f"Checkpoint saved to: {checkpoint_dir / 'interrupted_checkpoint.pt'}")
        print(f"To resume: python train.py --resume {checkpoint_dir / 'interrupted_checkpoint.pt'}")
        print("=" * 60)
    
    def signal_handler(sig, frame):
        interrupted[0] = True
        save_checkpoint_on_interrupt()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Training loop with gradient accumulation
    accumulation_steps = getattr(config.training, 'accumulation_steps', 1)
    effective_batch = config.training.batch_datasets * accumulation_steps
    
    print(f"\nStarting training for {config.training.n_steps} steps...")
    print(f"  Datasets per forward: {config.training.batch_datasets}")
    print(f"  Accumulation steps: {accumulation_steps}")
    print(f"  Effective batch size: {effective_batch} datasets")
    print(f"  Eval every: {config.training.eval_every} steps")
    print(f"  Device: {config.device}")
    print(f"  Prefetching: enabled (parallel data generation)")
    if multi_gpu_trainer:
        print(f"  Multi-GPU: {multi_gpu_trainer.n_gpus} GPUs")
    else:
        print(f"  Multi-GPU: disabled")
    print(f"  Press Ctrl+C to save and exit")
    print()
    
    current_step = start_step
    
    # Pre-generate first batch(es) for prefetching
    def generate_step_batches() -> List[List]:
        """Generate all batches needed for one training step."""
        return [
            data_loader.generate_batch(config.training.batch_datasets)
            for _ in range(accumulation_steps)
        ]
    
    # Start prefetching first step's data
    prefetch_future = prefetch_executor.submit(generate_step_batches)
    
    try:
        for step in range(start_step, config.training.n_steps):
            current_step = step
            step_start = time.time()
            
            # Get pre-generated batches (blocks if not ready yet)
            step_batches = prefetch_future.result()
            
            # Immediately start generating next step's batches in background
            prefetch_future = prefetch_executor.submit(generate_step_batches)
            
            # Zero gradients at start of accumulation
            if multi_gpu_trainer:
                multi_gpu_trainer.zero_grad()
            else:
            optimizer.zero_grad()
            
            # Accumulate gradients over multiple forward passes
            step_loss = 0.0
            step_acc = 0.0
            n_processed = 0
            
            if multi_gpu_trainer:
                # Multi-GPU: process all samples in parallel across GPUs
                all_samples = [s for batch in step_batches for s in batch]
                result = multi_gpu_trainer.forward_backward_batch(all_samples, effective_batch)
                step_loss = result["loss"]
                step_acc = result["accuracy"]
                n_processed = result["n_processed"]
            else:
                # Single GPU: process samples sequentially
                for accum_idx, samples in enumerate(step_batches):
                
                # Forward and backward for each sample in batch
                for sample in samples:
                    try:
                        output = forward_backward_step(
                            model, sample, config, 
                            accumulation_steps=effective_batch
                        )
                        step_loss += output["loss"]
                        step_acc += output["accuracy"]
                        n_processed += 1
                        except (RuntimeError, ValueError) as e:
                            error_msg = str(e).lower()
                            if "out of memory" in error_msg:
                            # Clear CUDA cache and skip this sample
                            torch.cuda.empty_cache()
                            print(f"  [OOM] Skipped dataset with shape {sample.X_full.shape}")
                            continue
                            elif "cannot reshape" in error_msg or "size 0" in error_msg:
                                # Skip empty or invalid datasets
                                print(f"  [Invalid] Skipped dataset with shape {sample.X_full.shape}: {str(e)[:100]}")
                                continue
                        else:
                            raise e
            
            # Optimizer step after accumulation
            optimizer_step(model, optimizer, config)
            scheduler.step()
            
            # Sync weights to replicas after optimizer step (multi-GPU only)
            if multi_gpu_trainer:
                multi_gpu_trainer.sync_weights()
            
            # Average metrics (use n_processed to handle skipped OOM samples)
            if n_processed == 0:
                n_processed = 1  # Avoid division by zero
            train_output = {
                "loss": step_loss / n_processed,
                "accuracy": step_acc / n_processed,
            }
            
            # Track metrics
            train_losses.append(train_output["loss"])
            train_accs.append(train_output["accuracy"])
            train_steps.append(step)
            
            # Log
            metrics.add_train_metrics(step, train_output["loss"], train_output["accuracy"])
            
            step_time = time.time() - step_start
            
            # Print progress
            if step % 10 == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"Step {step:5d} | Loss: {train_output['loss']:.4f} | "
                      f"Acc: {train_output['accuracy']:.4f} | "
                      f"LR: {lr:.2e} | Time: {step_time:.2f}s")
            
            # Update plot periodically
            update_plot_every = getattr(config.training, 'update_plot_every', 100)
            if (step + 1) % update_plot_every == 0 and train_losses:
                plot_training_progress(
                    train_losses, train_accs,
                    val_synth_losses, val_synth_accs, val_real_accs,
                    train_steps, val_steps,
                    str(log_dir / "training_curves.png")
                )
            
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
                
                # Track validation metrics
                val_synth_losses.append(synth_result.mean_ce_loss)
                val_synth_accs.append(synth_result.mean_accuracy)
                val_steps.append(step + 1)
                
                # Evaluate on real (only if enabled)
                if config.training.eval_real_datasets:
                val_real_size = getattr(config.training, 'val_real_size', 20)
                real_result = evaluate_on_real(model, config, n_datasets=val_real_size)
                if real_result.n_datasets > 0:
                    print(f"Real Val ({real_result.n_datasets} datasets): "
                          f"Acc={real_result.mean_accuracy:.4f} | "
                          f"F1={real_result.mean_macro_f1:.4f}")
                    val_real_accs.append(real_result.mean_accuracy)
                else:
                    print("Real Val: No datasets available")
                    val_real_accs.append(0.0)
                
                # Add to metrics
                metrics.add_val_metrics(synth_result, real_result)
                else:
                    # Skip real dataset evaluation
                    real_result = BatchEvaluationResult(
                        mean_accuracy=0.0,
                        mean_ce_loss=float('nan'),
                        mean_macro_f1=0.0,
                        std_accuracy=0.0,
                        std_ce_loss=0.0,
                        std_macro_f1=0.0,
                        n_datasets=0,
                        per_dataset_results=[]
                    )
                    metrics.add_val_metrics(synth_result, real_result)
                
                # Update plot after evaluation
                plot_training_progress(
                    train_losses, train_accs,
                    val_synth_losses, val_synth_accs, val_real_accs,
                    train_steps, val_steps,
                    str(log_dir / "training_curves.png")
                )
                
                # Wandb logging
                if config.training.wandb_project:
                    try:
                        log_dict = {
                            "val_synth/accuracy": synth_result.mean_accuracy,
                            "val_synth/ce_loss": synth_result.mean_ce_loss,
                            "val_synth/macro_f1": synth_result.mean_macro_f1,
                        }
                        if config.training.eval_real_datasets:
                            log_dict["val_real/accuracy"] = real_result.mean_accuracy
                            log_dict["val_real/macro_f1"] = real_result.mean_macro_f1
                        wandb.log(log_dict, step=step)
                    except:
                        pass
                
                # Save best model (based on synthetic validation)
                if synth_result.mean_accuracy > best_val_accuracy:
                    best_val_accuracy = synth_result.mean_accuracy
                    torch.save({
                        "step": step + 1,
                        "encoder_state_dict": model.encoder.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_val_accuracy": best_val_accuracy,
                    }, checkpoint_dir / "best_model.pt")
                    print(f"  -> NEW BEST MODEL saved (acc={best_val_accuracy:.4f})")
                
                print("-" * 40 + "\n")
            
            # Periodic checkpoint
            if (step + 1) % config.training.save_every == 0:
                torch.save({
                    "step": step + 1,
                    "encoder_state_dict": model.encoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_accuracy": best_val_accuracy,
                }, checkpoint_dir / f"checkpoint_step{step+1}.pt")
                
                # Save training history
                history = {
                    "train_losses": train_losses,
                    "train_accs": train_accs,
                    "train_steps": train_steps,
                    "val_synth_losses": val_synth_losses,
                    "val_synth_accs": val_synth_accs,
                    "val_real_accs": val_real_accs,
                    "val_steps": val_steps,
                }
                with open(log_dir / "training_history.json", 'w') as f:
                    json.dump(history, f)
                
                # Save metrics
                metrics.save(log_dir / "metrics.json")
                
                print(f"  [Checkpoint saved: step {step+1}]")
    
    except KeyboardInterrupt:
        # This shouldn't happen due to signal handler, but just in case
        prefetch_executor.shutdown(wait=False)
        if multi_gpu_trainer:
            multi_gpu_trainer.shutdown()
        save_checkpoint_on_interrupt()
        return model
    
    # Cleanup prefetch executor and multi-GPU trainer
    prefetch_executor.shutdown(wait=False)
    if multi_gpu_trainer:
        multi_gpu_trainer.shutdown()
    
    # Final save
    torch.save({
        "step": config.training.n_steps,
        "encoder_state_dict": model.encoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_accuracy": best_val_accuracy,
    }, checkpoint_dir / "final_model.pt")
    
    # Save final training history
    history = {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "train_steps": train_steps,
        "val_synth_losses": val_synth_losses,
        "val_synth_accs": val_synth_accs,
        "val_real_accs": val_real_accs,
        "val_steps": val_steps,
    }
    with open(log_dir / "training_history.json", 'w') as f:
        json.dump(history, f)
    
    metrics.save(log_dir / "metrics.json")
    
    # Final plot
    plot_training_progress(
        train_losses, train_accs,
        val_synth_losses, val_synth_accs, val_real_accs,
        train_steps, val_steps,
        str(log_dir / "training_curves.png")
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Logs saved to: {log_dir}")
    print(f"Training curves: {log_dir / 'training_curves.png'}")
    print("=" * 60)
    
    return model


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train Temporal Encoder + TabPFN")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--debug", action="store_true", help="Quick debug run")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--multi-gpu", action="store_true", help="Use multiple GPUs if available")
    
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
    multi_gpu = getattr(args, 'multi_gpu', False)
    model = train(config, resume_from=args.resume, multi_gpu=multi_gpu)
    
    return model


if __name__ == "__main__":
    main()

