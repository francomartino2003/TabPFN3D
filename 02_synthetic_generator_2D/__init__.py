"""
Synthetic Dataset Generator for TabPFN-style training.

This module implements a generator that creates tabular datasets from random
causal graphs (DAGs), following the methodology described in the TabPFN paper.

The key idea is to generate many different datasets, each from a different
generative process, to train a model that learns to solve tabular problems
in general.

Main components:
- DAG Builder: Constructs random directed acyclic graphs
- Transformations: Defines edge operations (NN, discretization, trees, noise)
- Row Generator: Propagates noise through the DAG to generate rows
- Feature Selector: Chooses which nodes become observed features/target
- Post-Processing: Applies realistic transformations (warping, missing values, etc.)
"""

# Support both package import and direct execution
try:
    from .config import PriorConfig, DatasetConfig
    from .dag_builder import DAGBuilder
    from .transformations import TransformationFactory
    from .row_generator import RowGenerator
    from .feature_selector import FeatureSelector
    from .post_processing import PostProcessor
    from .generator import SyntheticDatasetGenerator
except ImportError:
    from config import PriorConfig, DatasetConfig
    from dag_builder import DAGBuilder
    from transformations import TransformationFactory
    from row_generator import RowGenerator
    from feature_selector import FeatureSelector
    from post_processing import PostProcessor
    from generator import SyntheticDatasetGenerator

__all__ = [
    'PriorConfig',
    'DatasetConfig', 
    'DAGBuilder',
    'TransformationFactory',
    'RowGenerator',
    'FeatureSelector',
    'PostProcessor',
    'SyntheticDatasetGenerator',
]

__version__ = '0.1.0'
