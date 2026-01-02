"""
3D Synthetic Dataset Generator with Temporal Dependencies.

Generates classification/regression datasets with shape (n_samples, n_features, t_timesteps).

Key features:
- Temporal dependencies through state (memory) inputs
- Time-dependent inputs with various activation functions
- Three sampling modes: IID, sliding window, mixed
- Target at different time offsets (within sequence, future prediction, etc.)

Usage:
    from generator import SyntheticDatasetGenerator3D, generate_3d_dataset
    
    # Quick generation
    dataset = generate_3d_dataset(seed=42)
    X, y = dataset.X, dataset.y  # (n, m, t), (n,)
    
    # Custom configuration
    from config import PriorConfig3D
    prior = PriorConfig3D(max_features=10, prob_classification=1.0)
    generator = SyntheticDatasetGenerator3D(prior=prior, seed=42)
    dataset = generator.generate()
"""

from .config import PriorConfig3D, DatasetConfig3D
from .generator import SyntheticDatasetGenerator3D, SyntheticDataset3D, generate_3d_dataset

__all__ = [
    'PriorConfig3D',
    'DatasetConfig3D', 
    'SyntheticDatasetGenerator3D',
    'SyntheticDataset3D',
    'generate_3d_dataset'
]


