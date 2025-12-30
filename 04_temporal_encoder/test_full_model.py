"""
Test the complete TemporalTabPFN model.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import numpy as np


def test_full_model():
    print("=" * 60)
    print("TESTING COMPLETE MODEL: TemporalTabPFN")
    print("=" * 60)
    
    from training_config import FullConfig, get_debug_config
    from model import TemporalTabPFN
    from preprocessing_3d import Preprocessor3D
    
    # Use debug config with CPU
    config = get_debug_config()
    config.device = "cpu"
    
    print("Creating model...")
    model = TemporalTabPFN(config)
    print(f"  Trainable params (encoder): {model.get_num_trainable_params():,}")
    print(f"  Frozen params (TabPFN): {model.get_num_frozen_params():,}")
    
    # Create dummy data
    print()
    print("Testing forward pass...")
    n_samples = 50
    n_features = 3
    n_timesteps = 30
    n_train = 35
    n_classes = 5
    
    # Dummy 3D data
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features, n_timesteps).astype(np.float32)
    y_train = np.random.randint(0, n_classes, n_train)
    y_test = np.random.randint(0, n_classes, n_samples - n_train)
    
    # Preprocess
    preprocessor = Preprocessor3D(config.preprocessing)
    X_proc = preprocessor.fit_transform(X)
    print(f"  After preprocessing: {X_proc.shape}")
    
    # Convert to tensors
    X_tensor = torch.from_numpy(X_proc).float()
    y_train_tensor = torch.from_numpy(y_train)
    y_test_tensor = torch.from_numpy(y_test)
    
    # Forward pass
    print("  Running forward...")
    output = model.compute_loss(X_tensor, y_train_tensor, y_test_tensor, n_train)
    
    print(f"  Loss: {output['loss'].item():.4f}")
    print(f"  Accuracy: {output['accuracy'].item():.4f}")
    print(f"  Logits shape: {output['logits'].shape}")
    print()
    
    # Test backward
    print("Testing backward pass...")
    output["loss"].backward()
    encoder_grads = sum(1 for p in model.encoder.parameters() if p.grad is not None)
    tabpfn_grads = sum(1 for p in model.tabpfn.parameters() if p.grad is not None)
    print(f"  Encoder params with gradients: {encoder_grads}")
    print(f"  TabPFN params with gradients: {tabpfn_grads} (should be 0)")
    
    print()
    print("=" * 60)
    print("COMPLETE MODEL TEST PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_full_model()

