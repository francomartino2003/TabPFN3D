"""
Main 3D Synthetic Dataset Generator.

Generates temporal tabular datasets with shape (n_samples, n_features, t_subseq).

v2 CHANGES:
- Only time and state inputs (no direct noise roots)
- State inputs reference specific nodes at t-k
- No passthrough transformation
- Simplified feature/target selection
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Iterator
import numpy as np

# Local 3D modules
from config import PriorConfig3D, DatasetConfig3D
from temporal_inputs import TemporalInputManager
from temporal_propagator import TemporalPropagator, BatchTemporalPropagator
from sequence_sampler import (
    SequenceSampler, FeatureTargetSelection, Sample3D,
    discretize_targets, samples_to_arrays
)
from feature_selector import FeatureSelector3D

# 2D components via wrapper
from dag_utils import (
    DAG, DAGBuilder, 
    TransformationFactory, EdgeTransformation,
    Warper, MissingValueInjector
)


@dataclass
class SyntheticDataset3D:
    """
    A synthetic 3D temporal dataset.
    
    Attributes:
        X: Feature tensor of shape (n_samples, n_features, t_subseq)
        y: Target vector of shape (n_samples,)
        config: Configuration used to generate this dataset
        metadata: Additional generation metadata
        is_classification: Whether this is a classification task
        n_classes: Number of classes (for classification)
    """
    X: np.ndarray
    y: np.ndarray
    config: DatasetConfig3D
    metadata: Dict[str, Any]
    is_classification: bool
    n_classes: int
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return (n_samples, n_features, t_subseq)."""
        return self.X.shape
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'X': self.X.tolist(),
            'y': self.y.tolist(),
            'shape': self.shape,
            'config': self.config.to_dict(),
            'metadata': self.metadata,
            'is_classification': self.is_classification,
            'n_classes': self.n_classes
        }


class SyntheticDatasetGenerator3D:
    """
    Main generator for 3D temporal synthetic datasets.
    
    Usage:
        generator = SyntheticDatasetGenerator3D(seed=42)
        dataset = generator.generate()
        X, y = dataset.X, dataset.y  # Shape: (n, m, t), (n,)
    """
    
    def __init__(
        self,
        prior: Optional[PriorConfig3D] = None,
        seed: Optional[int] = None
    ):
        self.prior = prior if prior is not None else PriorConfig3D()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
    
    def generate(self, config: Optional[DatasetConfig3D] = None) -> SyntheticDataset3D:
        """
        Generate a single 3D temporal dataset.
        
        Args:
            config: Specific configuration. If None, samples from prior.
            
        Returns:
            SyntheticDataset3D with X of shape (n, m, t) and y of shape (n,)
        """
        # Sample or use provided config
        if config is None:
            config = DatasetConfig3D.sample_from_prior(self.prior, self.rng)
        
        # Create fresh RNG for this dataset
        dataset_rng = np.random.default_rng(config.seed)
        
        # Step 1: Build DAG
        dag, transformations = self._build_dag(config, dataset_rng)
        
        # Step 2: Select TARGET first (before configuring state inputs)
        # This allows state inputs to prefer nodes close to the target
        selector = FeatureSelector3D(
            dag, transformations, config, None, dataset_rng  # No input_manager yet
        )
        target_node = selector.select_target_only()
        
        # Step 3: Create temporal input manager
        input_manager = TemporalInputManager.from_config(config, dataset_rng)
        
        # Step 4: Create propagator (state inputs configured with target info)
        propagator = BatchTemporalPropagator(
            config, dag, transformations, input_manager, dataset_rng,
            target_node=target_node  # Pass target for state source selection
        )
        
        # Step 5: Complete feature selection (target already selected)
        selector.input_manager = input_manager  # Update selector
        selection = selector.select_with_target(target_node)
        
        # Step 5: Create sequence sampler
        sampler = SequenceSampler(config, selection, dataset_rng)
        
        # Step 6: Generate samples based on mode
        samples = self._generate_samples(
            config, propagator, sampler, dataset_rng
        )
        
        # Step 7: Discretize targets if classification
        if config.is_classification:
            samples = discretize_targets(samples, config.n_classes, dataset_rng)
        
        # Step 8: Convert to arrays
        X, y = samples_to_arrays(samples)
        
        # Step 9: Post-processing
        X, y = self._apply_post_processing(X, y, config, dataset_rng)
        
        # Build metadata
        # Derive offset type from offset value
        if config.target_offset == 0:
            offset_type = 'within'
        elif config.target_offset > 0:
            offset_type = 'future'
        else:
            offset_type = 'past'
        
        metadata = {
            'n_samples': len(samples),
            'n_features': X.shape[1] if X.ndim > 1 else 0,
            't_subseq': X.shape[2] if X.ndim > 2 else 0,
            'T_total': config.T_total,
            'sample_mode': config.sample_mode,
            'target_offset_type': offset_type,  # Derived from offset value
            'target_offset': config.target_offset,
            'n_time_inputs': config.n_time_inputs,
            'n_state_inputs': config.n_state_inputs,
            'time_activations': config.time_activations,
            'feature_nodes': selection.feature_nodes,
            'target_node': selection.target_node,
            'spatial_distance_alpha': config.spatial_distance_alpha
        }
        
        return SyntheticDataset3D(
            X=X,
            y=y,
            config=config,
            metadata=metadata,
            is_classification=config.is_classification,
            n_classes=config.n_classes if config.is_classification else 0
        )
    
    def generate_many(self, n_datasets: int) -> Iterator[SyntheticDataset3D]:
        """Generate multiple datasets."""
        for i in range(n_datasets):
            yield self.generate()
    
    def _build_dag(
        self, 
        config: DatasetConfig3D, 
        rng: np.random.Generator
    ) -> Tuple[DAG, Dict[int, EdgeTransformation]]:
        """Build the causal DAG and transformations."""
        
        # Build DAG
        dag_builder = DAGBuilder(config, rng)
        dag = dag_builder.build()
        
        # Create transformations - ONE per non-root node
        transform_factory = TransformationFactory(config, rng)
        transformations = {}
        
        for node_id, node in dag.nodes.items():
            if node.parents:
                transform = transform_factory.create(n_parents=len(node.parents))
                transformations[node_id] = transform
        
        return dag, transformations
    
    def _generate_samples(
        self,
        config: DatasetConfig3D,
        propagator: BatchTemporalPropagator,
        sampler: SequenceSampler,
        rng: np.random.Generator
    ) -> List[Sample3D]:
        """Generate samples based on sampling mode."""
        n_samples = config.n_samples
        T = config.T_total
        stride = config.window_stride
        
        if config.sample_mode == 'iid':
            propagated = propagator.generate_iid_sequences(n_samples, T)
            samples = sampler.sample_iid_batch(propagated, n_samples)
            
        elif config.sample_mode == 'sliding_window':
            batch_size = min(10, n_samples // 10 + 1)
            propagated = propagator.generate_single_long_sequence(batch_size, T)
            samples = sampler.sample_sliding_window(propagated, n_samples, stride)
            
        else:  # mixed
            n_sequences = config.n_sequences
            samples_per_seq = max(1, n_samples // n_sequences)
            propagated_list = propagator.generate_mixed_sequences(
                n_sequences, samples_per_seq, T
            )
            samples = sampler.sample_mixed(propagated_list, n_samples, stride)
        
        return samples
    
    def _apply_post_processing(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: DatasetConfig3D,
        rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply post-processing to the generated data."""
        if X.size == 0:
            return X, y
        
        n_samples, n_features, t_subseq = X.shape
        X_processed = X.copy()
        
        # Apply warping
        if config.apply_warping:
            warper = Warper(config.warping_intensity, rng)
            for f in range(n_features):
                feature_flat = X_processed[:, f, :].flatten()
                warped = warper.warp_column(feature_flat)
                X_processed[:, f, :] = warped.reshape(n_samples, t_subseq)
        
        # Apply quantization
        if config.apply_quantization:
            for f in range(n_features):
                feature_flat = X_processed[:, f, :].flatten()
                min_val, max_val = feature_flat.min(), feature_flat.max()
                if max_val > min_val:
                    bins = np.linspace(min_val, max_val, config.n_quantization_bins + 1)
                    digitized = np.digitize(feature_flat, bins[1:-1])
                    quantized = bins[:-1][digitized.clip(0, len(bins)-2)]
                    X_processed[:, f, :] = quantized.reshape(n_samples, t_subseq)
        
        # Apply missing values
        if config.apply_missing:
            mask = rng.random(X_processed.shape) < config.missing_rate
            X_processed[mask] = np.nan
        
        return X_processed, y


# Convenience function
def generate_3d_dataset(
    seed: Optional[int] = None,
    **kwargs
) -> SyntheticDataset3D:
    """
    Quick function to generate a single 3D dataset.
    
    Args:
        seed: Random seed
        **kwargs: Override prior config parameters
        
    Returns:
        SyntheticDataset3D
    """
    prior = PriorConfig3D(**kwargs) if kwargs else PriorConfig3D()
    generator = SyntheticDatasetGenerator3D(prior=prior, seed=seed)
    return generator.generate()


if __name__ == "__main__":
    # Quick test
    print("Generating 3D synthetic dataset (v3)...")
    dataset = generate_3d_dataset(seed=42)
    print(f"Shape: {dataset.shape}")
    print(f"Classification: {dataset.is_classification}")
    print(f"Classes: {dataset.n_classes}")
    print(f"Sample mode: {dataset.config.sample_mode}")
    print(f"Time inputs: {dataset.config.n_time_inputs}")
    print(f"State inputs: {dataset.config.n_state_inputs}")
    print(f"Target offset: {dataset.config.target_offset}")
    print(f"Spatial distance alpha: {dataset.config.spatial_distance_alpha}")
    print(f"\nX stats: mean={dataset.X[~np.isnan(dataset.X)].mean():.3f}, "
          f"std={dataset.X[~np.isnan(dataset.X)].std():.3f}")
    print(f"y unique: {np.unique(dataset.y)}")
