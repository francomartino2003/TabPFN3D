"""
Main 3D Synthetic Dataset Generator for Time Series Classification.

This module provides the main interface for generating synthetic time series
classification datasets.

Usage:
    from generator import SyntheticDatasetGenerator3D
    from config import PriorConfig3D
    
    # Create generator with default prior
    generator = SyntheticDatasetGenerator3D()
    
    # Generate a single dataset
    dataset = generator.generate()
    X, y = dataset.X, dataset.y  # X: (n_samples, n_features, n_timesteps)
    
    # Generate many datasets for training
    for dataset in generator.generate_many(1000):
        pass
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Iterator, List, Tuple
import numpy as np
import json
import os
import sys

# Import from 2D generator - need careful path management
_2d_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '02_synthetic_generator_2D')
_3d_path = os.path.dirname(os.path.abspath(__file__))

# Save current state
_original_path = sys.path.copy()
_saved_modules = {}

# Temporarily save and remove conflicting modules
for mod_name in ['config', 'post_processing']:
    if mod_name in sys.modules:
        _saved_modules[mod_name] = sys.modules.pop(mod_name)

# Add 2D path, remove 3D path temporarily  
sys.path = [p for p in sys.path if os.path.normpath(p) != os.path.normpath(_3d_path)]
sys.path.insert(0, _2d_path)

try:
    from post_processing import Warper, Quantizer, MissingValueInjector
finally:
    # Restore path
    sys.path = _original_path
    
    # Remove 2D modules from cache
    for mod_name in ['config', 'post_processing']:
        if mod_name in sys.modules:
            sys.modules.pop(mod_name)
    
    # Restore saved 3D modules
    for mod_name, mod in _saved_modules.items():
        sys.modules[mod_name] = mod

# Now import 3D modules
from config import PriorConfig3D, DatasetConfig3D
from temporal_dag_builder import TemporalDAGBuilder, TemporalDAG
from row_generator_3d import RowGenerator3D, create_transformations_3d
from feature_selector_3d import FeatureSelector3D, TableBuilder3D


@dataclass
class SyntheticDataset3D:
    """
    A synthetic 3D time series dataset with all metadata.
    
    Attributes:
        X: Feature tensor (n_samples, n_features, n_timesteps)
        y: Target vector (n_samples,)
        config: The configuration used for generation
        metadata: Additional metadata
        feature_names: Names of features
        n_classes: Number of classes
    """
    X: np.ndarray
    y: np.ndarray
    config: DatasetConfig3D
    metadata: Dict[str, Any]
    feature_names: List[str]
    n_classes: int
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return shape (n_samples, n_features, n_timesteps)."""
        return self.X.shape
    
    @property
    def n_samples(self) -> int:
        return self.X.shape[0]
    
    @property
    def n_features(self) -> int:
        return self.X.shape[1]
    
    @property
    def n_timesteps(self) -> int:
        return self.X.shape[2]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'X': self.X.tolist(),
            'y': self.y.tolist(),
            'config': self.config.to_dict(),
            'metadata': self.metadata,
            'feature_names': self.feature_names,
            'n_classes': self.n_classes,
            'shape': self.shape
        }
    
    def save(self, path: str) -> None:
        """Save dataset to file."""
        np.savez_compressed(
            path,
            X=self.X,
            y=self.y,
            config=self.config.to_dict(),
            metadata=self.metadata,
            feature_names=self.feature_names,
            n_classes=self.n_classes
        )
    
    @classmethod
    def load(cls, path: str) -> 'SyntheticDataset3D':
        """Load dataset from file."""
        data = np.load(path, allow_pickle=True)
        return cls(
            X=data['X'],
            y=data['y'],
            config=data['config'].item(),
            metadata=data['metadata'].item(),
            feature_names=list(data['feature_names']),
            n_classes=int(data['n_classes'])
        )
    
    def to_2d(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Flatten to 2D format (n_samples, n_features * n_timesteps).
        
        Useful for compatibility with standard ML models.
        """
        n_samples = self.X.shape[0]
        X_2d = self.X.reshape(n_samples, -1)
        return X_2d, self.y


class PostProcessor3D:
    """
    Post-processor for 3D time series data.
    
    Applies transformations that respect temporal structure.
    """
    
    def __init__(self, config: DatasetConfig3D, rng: Optional[np.random.Generator] = None):
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng(config.seed)
        
        self.warper = Warper(config.warping_intensity, self.rng)
        min_bins = 2
        max_bins = max(min_bins + 1, config.n_quantization_bins)
        self.quantizer = Quantizer((min_bins, max_bins), self.rng)
        self.missing_injector = MissingValueInjector(config.missing_rate, self.rng)
    
    def process(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Apply post-processing to 3D data.
        
        Args:
            X: (n_samples, n_features, n_timesteps)
            y: (n_samples,)
            
        Returns:
            Tuple of (X_processed, y, applied_transforms dict)
        """
        X_proc = X.copy()
        applied = {}
        
        n_samples, n_features, n_timesteps = X.shape
        
        # Warping: apply per-feature across time
        if self.config.apply_warping:
            for f in range(n_features):
                if self.rng.random() < 0.5:  # Warp some features
                    for s in range(n_samples):
                        X_proc[s, f, :] = self.warper.warp_column(X_proc[s, f, :])
            applied['warping'] = True
        else:
            applied['warping'] = False
        
        # Quantization: apply per-feature
        if self.config.apply_quantization:
            for f in range(n_features):
                if self.rng.random() < 0.3:  # Quantize some features
                    for s in range(n_samples):
                        X_proc[s, f, :] = self.quantizer.quantize_column(X_proc[s, f, :])
            applied['quantization'] = True
        else:
            applied['quantization'] = False
        
        # Missing values: apply randomly across all dimensions
        if self.config.apply_missing:
            mask = self.rng.random(X.shape) < self.config.missing_rate
            X_proc[mask] = np.nan
            applied['missing_values'] = True
            applied['missing_rate'] = mask.mean()
        else:
            applied['missing_values'] = False
        
        return X_proc, y, applied


class SyntheticDatasetGenerator3D:
    """
    Main generator class for 3D synthetic time series classification datasets.
    
    Generation process:
    1. Sample hyperparameters from prior
    2. Build base DAG
    3. Unroll temporally with temporal connections
    4. Create transformations for spatial and temporal edges
    5. Generate observations by propagating noise
    6. Select features (time window) and target (timestep)
    7. Apply post-processing
    """
    
    def __init__(
        self,
        prior: Optional[PriorConfig3D] = None,
        seed: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Initialize the generator.
        
        Args:
            prior: Prior configuration (uses default if None)
            seed: Random seed
            verbose: Print progress
        """
        self.prior = prior if prior is not None else PriorConfig3D()
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        self.n_generated = 0
    
    def generate(
        self,
        config: Optional[DatasetConfig3D] = None,
        return_dag: bool = False
    ) -> SyntheticDataset3D:
        """
        Generate a single synthetic 3D dataset.
        
        Args:
            config: Specific configuration (samples from prior if None)
            return_dag: Include DAG in metadata
            
        Returns:
            SyntheticDataset3D
        """
        # Step 1: Sample or use provided config
        if config is None:
            config = self.prior.sample_hyperparams(self.rng)
        
        if self.verbose:
            print(f"Generating 3D dataset:")
            print(f"  Shape: ({config.n_samples}, {config.n_features}, {config.n_timesteps})")
            print(f"  Base DAG: {config.n_nodes} nodes, {config.n_temporal_connections} temporal patterns")
            print(f"  Classes: {config.n_classes}")
        
        dataset_rng = np.random.default_rng(config.seed)
        
        # Step 2: Build temporal DAG
        dag_builder = TemporalDAGBuilder(config, dataset_rng)
        dag = dag_builder.build()
        
        if self.verbose:
            print(f"  Built temporal DAG: {len(dag.nodes)} total nodes, "
                  f"{len(dag.temporal_edges)} temporal edges")
        
        # Step 3: Create transformations
        spatial_transforms, temporal_transforms = create_transformations_3d(
            dag, config, dataset_rng
        )
        
        if self.verbose:
            print(f"  Created {len(spatial_transforms)} spatial, "
                  f"{len(temporal_transforms)} temporal transforms")
        
        # Step 4: Generate observations
        row_generator = RowGenerator3D(
            config, dag, spatial_transforms, temporal_transforms, dataset_rng
        )
        propagated = row_generator.generate()
        
        if self.verbose:
            print(f"  Generated {config.n_samples} observations")
        
        # Step 5: Select features and target
        selector = FeatureSelector3D(config, dag, dataset_rng)
        selection = selector.select()
        
        if self.verbose:
            print(f"  Features: {selection.n_features} nodes, window [{selection.feature_window_start}:{selection.feature_window_end}]")
            print(f"  Target: node {selection.target_base_node} at t={selection.target_timestep} ({selection.target_position})")
        
        # Step 6: Build table
        table_builder = TableBuilder3D(selection, config, dataset_rng)
        X, y = table_builder.build(propagated)
        
        # Step 7: Post-processing
        post_processor = PostProcessor3D(config, dataset_rng)
        X_final, y_final, post_applied = post_processor.process(X, y)
        
        if self.verbose:
            print(f"  Post-processing: {post_applied}")
            print(f"  Final shape: {X_final.shape}")
        
        # Analyze temporal connection types
        connection_type_counts = {}
        total_skip = 0
        n_connections = len(config.temporal_connections)
        for conn in config.temporal_connections:
            conn_type = conn.connection_type
            connection_type_counts[conn_type] = connection_type_counts.get(conn_type, 0) + 1
            total_skip += conn.skip
        avg_skip = total_skip / max(1, n_connections)
        
        # Build metadata with process characteristics
        metadata = {
            'n_base_nodes': dag.n_base_nodes,
            'n_total_nodes': len(dag.nodes),
            'n_spatial_edges': len(dag.spatial_edges),
            'n_temporal_edges': len(dag.temporal_edges),
            'n_temporal_patterns': len(config.temporal_connections),
            'feature_selection': table_builder.get_metadata(),
            'post_processing': post_applied,
            'has_correlated_noise': config.has_correlated_noise,
            # Process characteristics for analysis
            'process_info': {
                'n_nodes': config.n_nodes,
                'density': config.density,
                'connection_types': connection_type_counts,
                'avg_skip': avg_skip,
                'n_disconnected_subgraphs': config.n_disconnected_subgraphs,
                'noise_scale': config.noise_scale,
                'noise_type': config.noise_type,
            }
        }
        
        if return_dag:
            metadata['dag'] = dag
        
        self.n_generated += 1
        
        return SyntheticDataset3D(
            X=X_final,
            y=y_final,
            config=config,
            metadata=metadata,
            feature_names=table_builder.get_feature_names(),
            n_classes=config.n_classes
        )
    
    def generate_many(self, n: int, yield_every: int = 10) -> Iterator[SyntheticDataset3D]:
        """Generate multiple datasets."""
        for i in range(n):
            if self.verbose and (i + 1) % yield_every == 0:
                print(f"Generating dataset {i + 1}/{n}...")
            yield self.generate()
    
    def generate_batch(self, n: int) -> List[SyntheticDataset3D]:
        """Generate a batch of datasets."""
        return [self.generate() for _ in range(n)]
    
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the generator."""
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.n_generated = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics."""
        return {
            'n_generated': self.n_generated,
            'seed': self.seed,
            'prior_config': {
                'n_samples_range': self.prior.n_samples_range,
                'n_features_range': self.prior.n_features_range,
                'n_timesteps_range': self.prior.n_timesteps_range,
                'prob_univariate': self.prior.prob_univariate,
            }
        }


def generate_training_data_3d(
    n_datasets: int,
    output_dir: str,
    prior: Optional[PriorConfig3D] = None,
    seed: int = 42,
    verbose: bool = True
) -> None:
    """
    Generate 3D datasets for training and save to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    generator = SyntheticDatasetGenerator3D(prior=prior, seed=seed, verbose=False)
    
    for i, dataset in enumerate(generator.generate_many(n_datasets)):
        if verbose and (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{n_datasets} datasets")
        
        path = os.path.join(output_dir, f"dataset_3d_{i:06d}.npz")
        dataset.save(path)
    
    # Save metadata
    meta_path = os.path.join(output_dir, "generation_metadata.json")
    with open(meta_path, 'w') as f:
        json.dump({
            'n_datasets': n_datasets,
            'seed': seed,
            'type': '3D_time_series_classification',
            'prior': {
                'n_samples_range': list(prior.n_samples_range) if prior else list(PriorConfig3D().n_samples_range),
                'n_features_range': list(prior.n_features_range) if prior else list(PriorConfig3D().n_features_range),
                'n_timesteps_range': list(prior.n_timesteps_range) if prior else list(PriorConfig3D().n_timesteps_range),
            }
        }, f, indent=2)
    
    if verbose:
        print(f"Saved {n_datasets} 3D datasets to {output_dir}")

