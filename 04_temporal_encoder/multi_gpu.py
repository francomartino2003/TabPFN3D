"""
Multi-GPU training utilities.

This module provides utilities for distributing training across multiple GPUs.
Uses torch.multiprocessing for true parallelization (avoids Python GIL).
"""
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from typing import List, Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import os

try:
    from .model import TemporalTabPFN
    from .training_config import FullConfig
    from .preprocessing_3d import Preprocessor3D, numpy_to_torch
except ImportError:
    from model import TemporalTabPFN
    from training_config import FullConfig
    from preprocessing_3d import Preprocessor3D, numpy_to_torch


def _worker_process(
    gpu_id: int,
    config_dict: dict,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    encoder_state_dict: dict,
    ready_event: mp.Event,
    weight_queue: mp.Queue
):
    """
    Worker process that runs on a specific GPU.
    Processes samples from task_queue and puts results in result_queue.
    Receives weight updates from weight_queue.
    """
    try:
        # Set CUDA device for this process
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"
        
        # Reconstruct config
        config = FullConfig()
        config.__dict__.update(config_dict)
        config.device = device
        
        # Create model on this GPU
        model = TemporalTabPFN(config)
        model.encoder.load_state_dict(encoder_state_dict)
        model = model.to(device)
        
        # Signal that we're ready
        ready_event.set()
        
        # Process loop
        while True:
            # Check for weight updates first (non-blocking)
            try:
                new_weights = weight_queue.get_nowait()
                if new_weights is not None:
                    model.encoder.load_state_dict(new_weights)
            except queue.Empty:
                pass
            
            try:
                task = task_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            if task is None:  # Shutdown signal
                break
            
            task_id, sample_data, accumulation_steps = task
            
            # Debug timing
            import time as _time
            proc_start = _time.time()
            
            try:
                # Unpack sample data
                X_full, y_train, y_test, n_train = sample_data
                
                model.train()
                
                # Preprocess
                preprocessor = Preprocessor3D(config.preprocessing)
                X_proc = preprocessor.fit_transform(X_full)
                
                # Convert to tensors
                X_tensor = numpy_to_torch(X_proc, device)
                y_train_t = torch.from_numpy(y_train).to(device)
                y_test_t = torch.from_numpy(y_test).to(device)
                
                # Forward and compute loss
                output = model.compute_loss(X_tensor, y_train_t, y_test_t, n_train)
                
                # Backward
                scaled_loss = output["loss"] / accumulation_steps
                scaled_loss.backward()
                
                # Collect gradients
                grads = {}
                for name, param in model.encoder.named_parameters():
                    if param.grad is not None:
                        grads[name] = param.grad.cpu().clone()
                
                # Zero grads for next sample
                model.encoder.zero_grad()
                
                result_queue.put((task_id, {
                    "loss": output["loss"].item(),
                    "accuracy": output["accuracy"].item(),
                    "grads": grads,
                    "error": None
                }))
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    result_queue.put((task_id, {
                        "loss": 0,
                        "accuracy": 0,
                        "grads": {},
                        "error": f"OOM: {X_full.shape}"
                    }))
                else:
                    result_queue.put((task_id, {
                        "loss": 0,
                        "accuracy": 0,
                        "grads": {},
                        "error": str(e)
                    }))
            except Exception as e:
                result_queue.put((task_id, {
                    "loss": 0,
                    "accuracy": 0,
                    "grads": {},
                    "error": str(e)
                }))
    
    except Exception as e:
        print(f"[GPU {gpu_id}] Worker crashed: {e}")
        ready_event.set()  # Unblock main process


class MultiGPUTrainer:
    """
    Trainer that distributes dataset processing across multiple GPUs.
    Uses multiprocessing for true parallelization (bypasses Python GIL).
    
    Strategy:
    1. Spawn a worker process per GPU
    2. Each worker has its own model replica
    3. Distribute samples round-robin to workers
    4. Collect gradients and aggregate on main process
    5. Update main model and sync weights to workers
    """
    
    def __init__(
        self,
        model: TemporalTabPFN,
        config: FullConfig,
        gpu_ids: Optional[List[int]] = None
    ):
        """
        Initialize multi-GPU trainer with worker processes.
        """
        self.config = config
        self.model = model
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
            self.workers = []
            return
        
        print(f"[MultiGPU] Initializing with {self.n_gpus} GPUs using multiprocessing")
        
        # Set multiprocessing start method
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
        
        # Prepare config dict for workers (can't pickle FullConfig directly)
        config_dict = {}
        for key, value in config.__dict__.items():
            try:
                # Test if serializable
                import pickle
                pickle.dumps(value)
                config_dict[key] = value
            except:
                pass  # Skip non-serializable
        
        # Get encoder state dict
        encoder_state = {k: v.cpu() for k, v in model.encoder.state_dict().items()}
        
        # Create queues and events
        self.task_queues = []
        self.weight_queues = []
        self.result_queue = mp.Queue()
        self.ready_events = []
        self.workers = []
        
        for gpu_id in self.gpu_ids:
            task_queue = mp.Queue()
            weight_queue = mp.Queue()
            ready_event = mp.Event()
            
            worker = mp.Process(
                target=_worker_process,
                args=(gpu_id, config_dict, task_queue, self.result_queue, 
                      encoder_state, ready_event, weight_queue)
            )
            worker.start()
            
            self.task_queues.append(task_queue)
            self.weight_queues.append(weight_queue)
            self.ready_events.append(ready_event)
            self.workers.append(worker)
            
            print(f"  Started worker on GPU {gpu_id}")
        
        # Wait for all workers to be ready
        print("  Waiting for workers to initialize...")
        for i, event in enumerate(self.ready_events):
            event.wait(timeout=120)  # 2 min timeout
            print(f"    GPU {self.gpu_ids[i]} ready")
        
        print(f"[MultiGPU] Ready with {self.n_gpus} GPUs (multiprocessing)")
        
        self._task_counter = 0
    
    def forward_backward_batch(
        self,
        samples: List,
        accumulation_steps: int
    ) -> Dict[str, float]:
        """
        Process a batch of samples across all GPUs in TRUE parallel.
        """
        if not self.workers:
            # Fallback to single GPU
            return self._process_single_gpu(samples, accumulation_steps)
        
        # Submit tasks round-robin to workers
        n_samples = len(samples)
        task_ids = []
        
        import time as _time
        submit_start = _time.time()
        
        for i, sample in enumerate(samples):
            gpu_idx = i % self.n_gpus
            task_id = self._task_counter
            self._task_counter += 1
            
            # Prepare sample data (numpy arrays are picklable)
            sample_data = (
                sample.X_full,
                sample.y_train,
                sample.y_test,
                sample.n_train
            )
            
            self.task_queues[gpu_idx].put((task_id, sample_data, accumulation_steps))
            task_ids.append(task_id)
        
        submit_time = _time.time() - submit_start
        if submit_time > 1.0:
            print(f"  [MultiGPU] Task submission took {submit_time:.1f}s")
        
        # Collect results
        results = {}
        collect_start = _time.time()
        while len(results) < n_samples:
            try:
                task_id, result = self.result_queue.get(timeout=300)  # 5 min timeout
                results[task_id] = result
            except queue.Empty:
                print("[MultiGPU] Timeout waiting for results!")
                break
        
        collect_time = _time.time() - collect_start
        # Debug: show which GPUs processed how many tasks
        gpu_counts = [0] * self.n_gpus
        for i, tid in enumerate(task_ids):
            if tid in results:
                gpu_counts[i % self.n_gpus] += 1
        
        # Aggregate results
        total_loss = 0.0
        total_acc = 0.0
        n_processed = 0
        all_grads = {}
        
        for task_id in task_ids:
            if task_id not in results:
                continue
            result = results[task_id]
            
            if result["error"]:
                if "OOM" in result["error"]:
                    print(f"  [OOM] Skipped dataset with shape {result['error'].split(': ')[1]}")
                continue
            
            total_loss += result["loss"]
            total_acc += result["accuracy"]
            n_processed += 1
            
            # Aggregate gradients
            for name, grad in result["grads"].items():
                if name not in all_grads:
                    all_grads[name] = grad.to(self.main_device)
                else:
                    all_grads[name] += grad.to(self.main_device)
        
        # Apply aggregated gradients to main model
        for name, param in self.model.encoder.named_parameters():
            if name in all_grads:
                if param.grad is None:
                    param.grad = all_grads[name]
                else:
                    param.grad += all_grads[name]
        
        return {
            "loss": total_loss,
            "accuracy": total_acc,
            "n_processed": n_processed
        }
    
    def _process_single_gpu(
        self,
        samples: List,
        accumulation_steps: int
    ) -> Dict[str, float]:
        """Fallback for single GPU processing."""
        total_loss = 0.0
        total_acc = 0.0
        n_processed = 0
        
        for sample in samples:
            try:
                self.model.train()
                preprocessor = Preprocessor3D(self.config.preprocessing)
                X_proc = preprocessor.fit_transform(sample.X_full)
                X_tensor = numpy_to_torch(X_proc, self.config.device)
                y_train = torch.from_numpy(sample.y_train).to(self.config.device)
                y_test = torch.from_numpy(sample.y_test).to(self.config.device)
                
                output = self.model.compute_loss(X_tensor, y_train, y_test, sample.n_train)
                scaled_loss = output["loss"] / accumulation_steps
                scaled_loss.backward()
                
                total_loss += output["loss"].item()
                total_acc += output["accuracy"].item()
                n_processed += 1
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    print(f"  [OOM] Skipped dataset with shape {sample.X_full.shape}")
        
        return {
            "loss": total_loss,
            "accuracy": total_acc,
            "n_processed": n_processed
        }
    
    def sync_weights(self):
        """
        Sync encoder weights from main model to all workers.
        Sends updated state dict through weight queues.
        """
        if not self.workers:
            return
        
        # Get current weights from main model (on CPU for transfer)
        state_dict = {k: v.cpu() for k, v in self.model.encoder.state_dict().items()}
        
        # Send to all workers
        for weight_queue in self.weight_queues:
            try:
                weight_queue.put(state_dict)
            except:
                pass
    
    def zero_grad(self):
        """Zero gradients on main model."""
        self.model.encoder.zero_grad()
    
    def shutdown(self):
        """Cleanup worker processes."""
        for task_queue in self.task_queues:
            try:
                task_queue.put(None)  # Shutdown signal
            except:
                pass
        
        for worker in self.workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()
        
        print("[MultiGPU] Workers shut down")


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
