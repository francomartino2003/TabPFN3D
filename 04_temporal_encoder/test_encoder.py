"""
Quick test script for the temporal encoder.
Run from the project root: python -m 04_temporal_encoder.test_encoder
Or from this directory with proper path setup.
"""
import sys
from pathlib import Path

# Setup path for direct execution
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    # Change to absolute imports for direct execution
    from training_config import EncoderConfig
    from encoder import (
        TemporalEncoder, 
        SinusoidalPositionalEncoding,
        LearnedPositionalEncoding,
        TemporalCrossAttention
    )
else:
    from .training_config import EncoderConfig
    from .encoder import (
        TemporalEncoder, 
        SinusoidalPositionalEncoding,
        LearnedPositionalEncoding,
        TemporalCrossAttention
    )

import torch


def test_positional_encoding():
    """Test positional encodings."""
    print("Testing Positional Encodings...")
    
    d_model = 128
    max_len = 100
    batch_size = 4
    seq_len = 50
    
    # Test sinusoidal
    sin_pe = SinusoidalPositionalEncoding(d_model, max_len)
    x = torch.randn(batch_size, seq_len, d_model)
    out = sin_pe(x)
    assert out.shape == x.shape, f"Sinusoidal PE shape mismatch: {out.shape} vs {x.shape}"
    print(f"  Sinusoidal PE: OK (shape {tuple(out.shape)})")
    
    # Test learned
    learn_pe = LearnedPositionalEncoding(d_model, max_len)
    out = learn_pe(x)
    assert out.shape == x.shape, f"Learned PE shape mismatch: {out.shape} vs {x.shape}"
    print(f"  Learned PE: OK (shape {tuple(out.shape)})")
    
    print("  All positional encoding tests passed!")


def test_cross_attention():
    """Test cross attention module."""
    print("\nTesting Cross Attention...")
    
    d_model = 128
    n_queries = 16
    n_heads = 8
    batch_size = 4
    seq_len = 50
    
    cross_attn = TemporalCrossAttention(d_model, n_queries, n_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    out = cross_attn(x)
    
    expected_shape = (batch_size, n_queries, d_model)
    assert out.shape == expected_shape, f"Cross attention shape mismatch: {out.shape} vs {expected_shape}"
    print(f"  Input:  {tuple(x.shape)}")
    print(f"  Output: {tuple(out.shape)}")
    print("  Cross attention test passed!")


def test_temporal_encoder():
    """Test full temporal encoder."""
    print("\nTesting Temporal Encoder...")
    
    config = EncoderConfig()
    encoder = TemporalEncoder(config)
    
    batch_size = 4
    n_features = 3
    n_timesteps = 50
    
    x = torch.randn(batch_size, n_features, n_timesteps)
    out = encoder(x)
    
    expected_shape = (batch_size, n_features * config.n_queries, config.d_model)
    assert out.shape == expected_shape, f"Encoder shape mismatch: {out.shape} vs {expected_shape}"
    
    print(f"  Config: d_model={config.d_model}, n_queries={config.n_queries}")
    print(f"  Input:  {tuple(x.shape)} (batch, features, timesteps)")
    print(f"  Output: {tuple(out.shape)} (batch, features*queries, d_model)")
    
    # Count parameters
    n_params = sum(p.numel() for p in encoder.parameters())
    n_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,} ({n_trainable:,} trainable)")
    
    print("  Temporal encoder test passed!")


def test_with_mask():
    """Test encoder with missing value mask."""
    print("\nTesting with Mask...")
    
    config = EncoderConfig()
    encoder = TemporalEncoder(config)
    
    batch_size = 4
    n_features = 3
    n_timesteps = 50
    
    x = torch.randn(batch_size, n_features, n_timesteps)
    mask = torch.ones(batch_size, n_features, n_timesteps)
    mask[:, :, 10:15] = 0  # Simulate some missing values
    
    out = encoder(x, mask=mask)
    
    expected_shape = (batch_size, n_features * config.n_queries, config.d_model)
    assert out.shape == expected_shape
    print(f"  With mask: OK (shape {tuple(out.shape)})")


def test_gradient_flow():
    """Test that gradients flow correctly."""
    print("\nTesting Gradient Flow...")
    
    config = EncoderConfig()
    encoder = TemporalEncoder(config)
    
    batch_size = 2
    n_features = 2
    n_timesteps = 20
    
    x = torch.randn(batch_size, n_features, n_timesteps, requires_grad=True)
    out = encoder(x)
    
    # Compute dummy loss and backward
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None, "Gradients did not flow to input"
    
    # Check encoder parameters have gradients
    has_grads = all(p.grad is not None for p in encoder.parameters() if p.requires_grad)
    assert has_grads, "Some encoder parameters have no gradients"
    
    print("  Gradients flow correctly!")


def main():
    print("=" * 60)
    print("TEMPORAL ENCODER TESTS")
    print("=" * 60)
    
    test_positional_encoding()
    test_cross_attention()
    test_temporal_encoder()
    test_with_mask()
    test_gradient_flow()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()

