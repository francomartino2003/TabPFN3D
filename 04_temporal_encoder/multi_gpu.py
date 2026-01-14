"""
Multi-GPU training utilities.

This module provides utilities for distributing training across multiple GPUs.
Each GPU processes a subset of datasets in parallel, then gradients are synchronized.
"""
import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import threading

try:
    from .model import TemporalTabPFN
    from .training_config import FullConfig
    from .preprocessing_3d import Preprocessor3D, numpy_to_torch
except ImportError:
    from model import TemporalTabPFN
    from training_config import FullConfig
    from preprocessing_3d import Preprocessor3D, numpy_to_torch


class MultiGPUTrainer:
    """
    Trainer that distributes dataset processing across multiple GPUs.
    
    Strategy:
    1. Create a model replica on each GPU
    2. Distribute datasets among GPUs (e.g., 64 datasets / 4 GPUs = 16 per GPU)
    3. Each GPU processes its datasets in parallel
    4. Aggregate gradients from all replicas to the main model
    5. Optimizer step on main model
    6. Sync weights back to replicas
    """
    
    def __init__(
        self,
        model: TemporalTabPFN,
        config: FullConfig,
        gpu_ids: Optional[List[int]] = None
    ):
        """
        Initialize multi-GPU trainer.
        
        Args:
            model: The main model (on GPU 0)
            config: Training configuration
            gpu_ids: List of GPU IDs to use. If None, use all available.
        """
        self.config = config
        self.main_device = config.device
        
        # Detect available GPUs
        if gpu_ids is None:
            self.n_gpus = torch.cuda.device_count()
            self.gpu_ids = list(range(self.n_gpus))
        else:
            self.gpu_ids = gpu_ids
            self.n_gpus = len(gpu_ids)
        
        if self.n_gpus < 2:
            print(f"[MultiGPU] Only {self.n_gpus} GPU(s) available, using single GPU mode")
            self.models = [model]
            self.devices = [config.device]
            return
        
        print(f"[MultiGPU] Initializing with {self.n_gpus} GPUs: {self.gpu_ids}")
        
        # Main model is on first GPU
        self.models = [model]
        self.devices = [f"cuda:{self.gpu_ids[0]}"]
        
        # Create replicas on other GPUs
        for gpu_id in self.gpu_ids[1:]:
            device = f"cuda:{gpu_id}"
            print(f"  Creating replica on {device}...")
            
            # Create new config for this device
            replica_config = FullConfig()
            replica_config.__dict__.update(config.__dict__)
            replica_config.device = device
            
            # Create replica model
            replica = TemporalTabPFN(replica_config)
            
            # Copy encoder weights from main model
            replica.encoder.load_state_dict(model.encoder.state_dict())
            
            self.models.append(replica)
            self.devices.append(device)
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=self.n_gpus)
        
        # Lock for thread-safe gradient accumulation
        self.grad_lock = threading.Lock()
        
        print(f"[MultiGPU] Ready with {self.n_gpus} GPUs")
    
    def _process_sample_on_gpu(
        self,
        gpu_idx: int,
        sample,
        accumulation_steps: int
    ) -> Dict[str, float]:
        """Process a single sample on a specific GPU."""
        model = self.models[gpu_idx]
        device = self.devices[gpu_idx]
        
        model.train()
        
        # Preprocess
        preprocessor = Preprocessor3D(self.config.preprocessing)
        X_proc = preprocessor.fit_transform(sample.X_full)
        
        # Convert to tensors on this GPU's device
        X_tensor = numpy_to_torch(X_proc, device)
        y_train = torch.from_numpy(sample.y_train).to(device)
        y_test = torch.from_numpy(sample.y_test).to(device)
        
        # Forward and compute loss
        output = model.compute_loss(
            X_tensor,
            y_train,
            y_test,
            sample.n_train
        )
        
        # Scale loss and backward
        scaled_loss = output["loss"] / accumulation_steps
        scaled_loss.backward()
        
        return {
            "loss": output["loss"].item(),
            "accuracy": output["accuracy"].item(),
        }
    
    def _process_batch_on_gpu(
        self,
        gpu_idx: int,
        samples: List,
        accumulation_steps: int
    ) -> Dict[str, float]:
        """Process multiple samples on a specific GPU."""
        total_loss = 0.0
        total_acc = 0.0
        n_processed = 0
        
        for sample in samples:
            try:
                result = self._process_sample_on_gpu(gpu_idx, sample, accumulation_steps)
                total_loss += result["loss"]
                total_acc += result["accuracy"]
                n_processed += 1
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    print(f"  [GPU{gpu_idx} OOM] Skipped shape {sample.X_full.shape}")
                else:
                    raise
        
        return {
            "loss": total_loss,
            "accuracy": total_acc,
            "n_processed": n_processed
        }
    
    def forward_backward_batch(
        self,
        samples: List,
        accumulation_steps: int
    ) -> Dict[str, float]:
        """
        Process a batch of samples across all GPUs in parallel.
        
        Args:
            samples: List of DatasetSample objects
            accumulation_steps: Total accumulation steps (for loss scaling)
        
        Returns:
            Dict with aggregated loss and accuracy
        """
        if self.n_gpus == 1:
            # Single GPU fallback
            return self._process_batch_on_gpu(0, samples, accumulation_steps)
        
        # Distribute samples across GPUs
        samples_per_gpu = len(samples) // self.n_gpus
        gpu_batches = []
        
        for i in range(self.n_gpus):
            start_idx = i * samples_per_gpu
            if i == self.n_gpus - 1:
                # Last GPU gets any remainder
                gpu_batches.append(samples[start_idx:])
            else:
                gpu_batches.append(samples[start_idx:start_idx + samples_per_gpu])
        
        # Process in parallel
        futures = []
        for gpu_idx, batch in enumerate(gpu_batches):
            if batch:  # Only submit if there are samples
                future = self.executor.submit(
                    self._process_batch_on_gpu,
                    gpu_idx, batch, accumulation_steps
                )
                futures.append(future)
        
        # Collect results
        total_loss = 0.0
        total_acc = 0.0
        total_processed = 0
        
        for future in futures:
            result = future.result()
            total_loss += result["loss"]
            total_acc += result["accuracy"]
            total_processed += result["n_processed"]
        
        # Aggregate gradients from replicas to main model
        self._aggregate_gradients()
        
        return {
            "loss": total_loss,
            "accuracy": total_acc,
            "n_processed": total_processed
        }
    
    def _aggregate_gradients(self):
        """Aggregate gradients from all replicas to the main model."""
        if self.n_gpus == 1:
            return
        
        main_model = self.models[0]
        main_device = self.devices[0]
        
        # For each parameter in the main encoder
        for name, param in main_model.encoder.named_parameters():
            if param.grad is None:
                continue
            
            # Add gradients from replicas
            for replica in self.models[1:]:
                replica_param = dict(replica.encoder.named_parameters())[name]
                if replica_param.grad is not None:
                    # Move gradient to main device and add
                    param.grad += replica_param.grad.to(main_device)
    
    def sync_weights(self):
        """Sync encoder weights from main model to all replicas."""
        if self.n_gpus == 1:
            return
        
        main_state = self.models[0].encoder.state_dict()
        
        for replica in self.models[1:]:
            # Copy weights (they're on different devices, state_dict handles this)
            replica.encoder.load_state_dict(main_state)
    
    def zero_grad(self):
        """Zero gradients on all models."""
        for model in self.models:
            model.encoder.zero_grad()
    
    def shutdown(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


def get_available_gpus() -> List[Dict[str, Any]]:
    """Get info about available GPUs."""
    gpus = []
    n_gpus = torch.cuda.device_count()
    
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        try:
            free, total = torch.cuda.mem_get_info(i)
            free_gb = free / 1e9
        except:
            free_gb = 0
        
        gpus.append({
            "id": i,
            "name": props.name,
            "total_memory_gb": props.total_memory / 1e9,
            "free_memory_gb": free_gb,
        })
    
    return gpus
