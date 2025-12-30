"""
Synthetic Generator 3D - Time Series Classification Datasets

This module generates synthetic time series classification datasets
by extending the 2D TabPFN methodology to temporal data.

Key differences from 2D:
- Input: (n_samples, n_features, n_timesteps) instead of (n_samples, n_features)
- DAG is "unrolled" across time with temporal connections
- Features are extracted as contiguous time windows
- Target is from a specific timestep (before/within/after feature window)

Main components:
- config: PriorConfig3D, DatasetConfig3D
- temporal_dag_builder: Constructs unrolled temporal DAG
- row_generator_3d: Propagates noise through temporal DAG
- feature_selector_3d: Selects features and target across time
- generator: Main SyntheticDatasetGenerator3D class
"""

try:
    from .config import PriorConfig3D, DatasetConfig3D, TemporalConnectionConfig
    from .temporal_dag_builder import TemporalDAGBuilder, TemporalDAG, TemporalNode
    from .row_generator_3d import RowGenerator3D, TemporalPropagatedValues
    from .feature_selector_3d import FeatureSelector3D, TemporalFeatureSelection, TableBuilder3D
    from .generator import SyntheticDatasetGenerator3D, SyntheticDataset3D
except ImportError:
    from config import PriorConfig3D, DatasetConfig3D, TemporalConnectionConfig
    from temporal_dag_builder import TemporalDAGBuilder, TemporalDAG, TemporalNode
    from row_generator_3d import RowGenerator3D, TemporalPropagatedValues
    from feature_selector_3d import FeatureSelector3D, TemporalFeatureSelection, TableBuilder3D
    from generator import SyntheticDatasetGenerator3D, SyntheticDataset3D

__all__ = [
    'PriorConfig3D',
    'DatasetConfig3D',
    'TemporalConnectionConfig',
    'TemporalDAGBuilder',
    'TemporalDAG',
    'TemporalNode',
    'RowGenerator3D',
    'TemporalPropagatedValues',
    'FeatureSelector3D',
    'TemporalFeatureSelection',
    'TableBuilder3D',
    'SyntheticDatasetGenerator3D',
    'SyntheticDataset3D',
]

__version__ = '0.1.0'
