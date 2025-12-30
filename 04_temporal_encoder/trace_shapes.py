"""
Trace shapes through the Temporal Encoder and TabPFN pipeline.

Similar to 00_TabPFN/trace_shapes.py but for our 3D extension.
This helps understand how data flows through the model.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import numpy as np


def trace_encoder_shapes():
    """Trace shapes through the temporal encoder."""
    print("=" * 70)
    print("TEMPORAL ENCODER SHAPE TRACE")
    print("=" * 70)
    
    from training_config import EncoderConfig
    from encoder import TemporalEncoder
    
    config = EncoderConfig()
    print(f"\nEncoder Config:")
    print(f"  d_model (embedding dim): {config.d_model}")
    print(f"  n_queries: {config.n_queries}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  n_heads: {config.n_heads}")
    
    encoder = TemporalEncoder(config)
    
    # Example input
    batch_size = 4
    n_features = 3
    n_timesteps = 50
    
    print(f"\nInput shape: (batch={batch_size}, features={n_features}, timesteps={n_timesteps})")
    x = torch.randn(batch_size, n_features, n_timesteps)
    
    print("\n--- Step by step through encoder ---")
    
    # Step 1: Reshape for per-feature processing
    # (batch, features, timesteps) -> (batch * features, timesteps, 1)
    x_flat = x.permute(0, 2, 1).contiguous()  # (batch, timesteps, features)
    x_flat = x_flat.reshape(batch_size * n_features, n_timesteps, 1)
    print(f"1. Reshape for per-feature: {tuple(x.shape)} -> {tuple(x_flat.shape)}")
    print(f"   Each feature's time series is processed independently")
    
    # Step 2: Input projection
    # (batch*features, timesteps, 1) -> (batch*features, timesteps, d_model)
    x_proj = encoder.input_proj(x_flat)
    print(f"2. Input projection: {tuple(x_flat.shape)} -> {tuple(x_proj.shape)}")
    print(f"   Project scalar values to d_model={config.d_model}")
    
    # Step 3: Positional encoding
    x_pos = encoder.pos_enc(x_proj)
    print(f"3. Positional encoding: {tuple(x_proj.shape)} -> {tuple(x_pos.shape)}")
    print(f"   Add temporal position information")
    
    # Step 4: Cross-attention (compress timesteps to n_queries)
    x_queries = encoder.cross_attn(x_pos)
    print(f"4. Cross-attention: {tuple(x_pos.shape)} -> {tuple(x_queries.shape)}")
    print(f"   Compress {n_timesteps} timesteps -> {config.n_queries} queries")
    print(f"   This is the KEY step: learns to summarize temporal patterns")
    
    # Step 5: Self-attention layers
    x_self = x_queries
    for i, layer in enumerate(encoder.self_attn_layers):
        x_self = layer(x_self)
    print(f"5. Self-attention layers ({len(encoder.self_attn_layers)}x): {tuple(x_queries.shape)} -> {tuple(x_self.shape)}")
    
    # Step 6: Reshape back to batch
    # (batch*features, n_queries, d_model) -> (batch, features*n_queries, d_model)
    x_out = x_self.view(batch_size, n_features * config.n_queries, config.d_model)
    print(f"6. Reshape to batch: {tuple(x_self.shape)} -> {tuple(x_out.shape)}")
    print(f"   features * n_queries = {n_features} * {config.n_queries} = {n_features * config.n_queries}")
    
    # Full forward
    print(f"\n--- Full encoder forward ---")
    output = encoder(x)
    print(f"Input:  {tuple(x.shape)} = (batch, features, timesteps)")
    print(f"Output: {tuple(output.shape)} = (batch, features*n_queries, d_model)")
    
    return config


def trace_full_pipeline():
    """Trace shapes through the complete pipeline."""
    print("\n" + "=" * 70)
    print("FULL PIPELINE SHAPE TRACE")
    print("=" * 70)
    
    from training_config import get_debug_config
    from encoder import TemporalEncoder
    from preprocessing_3d import Preprocessor3D
    
    config = get_debug_config()
    
    # Example dataset
    n_samples = 30
    n_features = 3
    n_timesteps = 20
    n_train = 20
    n_test = n_samples - n_train
    n_classes = 5
    
    print(f"\nDataset:")
    print(f"  n_samples: {n_samples} (n_train={n_train}, n_test={n_test})")
    print(f"  n_features: {n_features}")
    print(f"  n_timesteps: {n_timesteps}")
    print(f"  n_classes: {n_classes}")
    
    # Raw data
    X = np.random.randn(n_samples, n_features, n_timesteps).astype(np.float32)
    y_train = np.random.randint(0, n_classes, n_train)
    
    print(f"\n1. Raw input X: {X.shape} = (samples, features, timesteps)")
    
    # Preprocessing
    preprocessor = Preprocessor3D(config.preprocessing)
    X_proc = preprocessor.fit_transform(X)
    print(f"2. After preprocessing: {X_proc.shape}")
    if X_proc.shape[1] != n_features:
        print(f"   Note: added {X_proc.shape[1] - n_features} missing indicator features")
    
    # To tensor
    X_tensor = torch.from_numpy(X_proc).float()
    
    # Encoder
    encoder = TemporalEncoder(config.encoder)
    embeddings = encoder(X_tensor)
    n_encoded_features = embeddings.shape[1]
    print(f"3. After temporal encoder: {tuple(embeddings.shape)}")
    print(f"   = (samples, features*n_queries, d_model)")
    print(f"   = ({n_samples}, {X_proc.shape[1]}*{config.encoder.n_queries}, {config.encoder.d_model})")
    
    # NO FLATTENING - inject directly into TabPFN
    print(f"4. Ready for TabPFN injection: {tuple(embeddings.shape)}")
    print(f"   = (samples, n_encoded_features, emsize)")
    print(f"   This goes directly into TabPFN, replacing its X encoder output!")
    
    # This is what TabPFN receives - NO FLATTENING!
    print(f"\n--- What TabPFN sees (NO FLATTEN) ---")
    print(f"Embedded X injected into TabPFN: {tuple(embeddings.shape)}")
    print(f"  = (n_samples={n_samples}, n_encoded_features={n_encoded_features}, emsize=128)")
    print(f"This REPLACES TabPFN's X encoder output!")
    print(f"TabPFN will output: {n_test} predictions x {n_classes} classes")
    
    # Memory estimate - now based on n_encoded_features, not flattened
    print(f"\n--- Memory consideration ---")
    print(f"Number of 'virtual features' for TabPFN: {n_encoded_features}")
    print(f"TabPFN handles up to ~100-500 features well")
    if n_encoded_features > 500:
        print(f"WARNING: {n_encoded_features} features may be high!")
        print(f"Consider reducing n_queries from {config.encoder.n_queries}")


def trace_with_different_configs():
    """Show how different configs affect output shapes."""
    print("\n" + "=" * 70)
    print("CONFIG COMPARISON")
    print("=" * 70)
    
    from training_config import EncoderConfig
    from encoder import TemporalEncoder
    
    configs = [
        ("Default", EncoderConfig()),
        ("n_queries=4", EncoderConfig(n_queries=4)),
        ("n_queries=8", EncoderConfig(n_queries=8)),
        ("n_queries=1", EncoderConfig(n_queries=1)),
    ]
    
    n_features = 5
    n_timesteps = 50
    
    print(f"\nInput: (batch=1, features={n_features}, timesteps={n_timesteps})")
    print("-" * 60)
    print(f"{'Config':<20} {'n_queries':<12} {'Output Shape':<25} {'Virtual Features'}")
    print("-" * 60)
    
    x = torch.randn(1, n_features, n_timesteps)
    
    for name, cfg in configs:
        encoder = TemporalEncoder(cfg)
        out = encoder(x)
        virtual_features = out.shape[1]  # n_features * n_queries
        print(f"{name:<20} {cfg.n_queries:<12} {str(tuple(out.shape)):<25} {virtual_features}")


if __name__ == "__main__":
    enc_config = trace_encoder_shapes()
    trace_full_pipeline()
    trace_with_different_configs()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
The temporal encoder transforms:
  (n_samples, n_features, n_timesteps) -> (n_samples, n_features * n_queries, d_model=128)

This output is INJECTED directly into TabPFN, replacing TabPFN's X encoder:

  TabPFN flow with our encoder:
  
    Our Temporal Encoder
         |
         v
    embeddings: (n_samples, n_features * n_queries, 128)
         |  <-- INJECTION POINT (replaces TabPFN's X encoder output)
         v
    TabPFN.y_encoder -> add_embeddings -> transformer_encoder -> decoder
         |
         v
    predictions: (n_test, n_classes)

Key parameters:
  - n_queries: How many tokens to compress each time series into
    This determines the number of "virtual features" TabPFN sees
    
  - d_model: MUST be 128 to match TabPFN's emsize

Memory usage depends on n_features * n_queries (number of feature embeddings):
  - n_features=5, n_queries=16 -> 80 virtual features OK
  - n_features=10, n_queries=16 -> 160 virtual features OK
  - n_features=50, n_queries=4 -> 200 virtual features OK
""")

