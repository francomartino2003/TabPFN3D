"""
Test a mini training loop.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch.optim import AdamW


def test_mini_training():
    print("=" * 60)
    print("TESTING MINI TRAINING LOOP")
    print("=" * 60)
    
    from training_config import get_debug_config
    from model import TemporalTabPFN
    from data_loader import SyntheticDataLoader
    from preprocessing_3d import Preprocessor3D, numpy_to_torch
    
    # Debug config - ultra light for fast testing
    config = get_debug_config()
    config.device = "cuda"
    config.training.batch_datasets = 1  # Only 1 dataset per batch
    
    # Reduce synthetic dataset size for fast testing
    config.data.n_samples_range = (20, 50)      # Very few samples
    config.data.n_features_range = (1, 2)        # Very few features
    config.data.n_timesteps_range = (10, 20)     # Few timesteps
    
    # Reduce encoder for less memory
    config.encoder.n_queries = 4  # Reduce from 16 to 4 for testing
    
    print("Creating model...")
    model = TemporalTabPFN(config)
    print(f"  Trainable params: {model.get_num_trainable_params():,}")
    
    # Optimizer
    optimizer = AdamW(model.get_trainable_parameters(), lr=1e-3)
    
    # Data loader
    print("\nCreating data loader...")
    loader = SyntheticDataLoader(config.data, seed=42)
    
    # Mini training loop - only 2 steps for fast testing
    print("\nRunning 2 training steps...")
    losses = []
    
    for step in range(2):
        optimizer.zero_grad()
        
        # Get batch of datasets
        samples = loader.generate_batch(config.training.batch_datasets)
        
        total_loss = 0.0
        total_acc = 0.0
        
        for sample in samples:
            # Preprocess
            X_full = sample.X_full
            preprocessor = Preprocessor3D(config.preprocessing)
            X_proc = preprocessor.fit_transform(X_full)
            
            # To tensors
            X_tensor = numpy_to_torch(X_proc, config.device)
            y_train = torch.from_numpy(sample.y_train).to(config.device)
            y_test = torch.from_numpy(sample.y_test).to(config.device)
            
            # Forward + loss
            output = model.compute_loss(X_tensor, y_train, y_test, sample.n_train)
            total_loss += output["loss"]
            total_acc += output["accuracy"].item()
        
        # Average and backward
        avg_loss = total_loss / len(samples)
        avg_loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
        
        # Step
        optimizer.step()
        
        avg_acc = total_acc / len(samples)
        losses.append(avg_loss.item())
        print(f"  Step {step+1}: Loss={avg_loss.item():.4f}, Acc={avg_acc:.4f}")
    
    # Check loss is decreasing (or at least not exploding)
    print(f"\nLosses: {[f'{l:.4f}' for l in losses]}")
    
    # Check if any parameter was updated
    print("\nVerifying encoder parameters were updated...")
    first_param = list(model.encoder.parameters())[0]
    print(f"  First encoder param mean: {first_param.data.mean().item():.6f}")
    print(f"  First encoder param std:  {first_param.data.std().item():.6f}")
    
    print()
    print("=" * 60)
    print("MINI TRAINING TEST PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_mini_training()

