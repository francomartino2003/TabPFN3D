"""
04_temporal_encoder - Temporal Encoder for TabPFN 3D

This module implements a temporal encoder that transforms 3D time series data
(n_samples, n_features, n_timesteps) into embeddings compatible with TabPFN,
enabling in-context learning for time series classification.

Main components:
- TemporalEncoder: Perceiver-style encoder for time series
- FrozenTabPFN: Wrapper for frozen TabPFN model
- TemporalTabPFN: Combined model (encoder + TabPFN)
- Preprocessor3D: 3D-adapted preprocessing pipeline
"""

from .training_config import (
    EncoderConfig,
    TrainingConfig, 
    DataConfig,
    Preprocessing3DConfig,
    FullConfig,
)

__all__ = [
    "EncoderConfig",
    "TrainingConfig", 
    "DataConfig",
    "Preprocessing3DConfig",
    "FullConfig",
]

