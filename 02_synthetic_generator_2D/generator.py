"""
Main synthetic dataset generator class.

This module provides the main interface for generating synthetic tabular
datasets following the TabPFN methodology.

Usage:
    from generator import SyntheticDatasetGenerator
    from config import PriorConfig
    
    # Create generator with default prior
    generator = SyntheticDatasetGenerator()
    
    # Generate a single dataset
    dataset = generator.generate()
    X, y = dataset['X'], dataset['y']
    
    # Generate many datasets for training
    for dataset in generator.generate_many(1000):
        # Train on each dataset
        pass
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Iterator, List, Tuple
import numpy as np
import json
import os

try:
    from .config import PriorConfig, DatasetConfig
    from .dag_builder import DAGBuilder, DAG
    from .transformations import TransformationFactory
    from .row_generator import RowGenerator
    from .feature_selector import FeatureSelector, TableBuilder
    from .post_processing import PostProcessor
except ImportError:
    from config import PriorConfig, DatasetConfig
    from dag_builder import DAGBuilder, DAG
    from transformations import TransformationFactory
    from row_generator import RowGenerator
    from feature_selector import FeatureSelector, TableBuilder
    from post_processing import PostProcessor


@dataclass
class SyntheticDataset:
    """
    A synthetic dataset with all its metadata.
    
    Attributes:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        config: The configuration used to generate this dataset
        metadata: Additional metadata about the generation process
        feature_names: Names of features
        is_classification: Whether this is a classification task
        n_classes: Number of classes (for classification)
    """
    X: np.ndarray
    y: np.ndarray
    config: DatasetConfig
    metadata: Dict[str, Any]
    feature_names: List[str]
    is_classification: bool
    n_classes: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'X': self.X.tolist(),
            'y': self.y.tolist(),
            'config': self.config.to_dict(),
            'metadata': self.metadata,
            'feature_names': self.feature_names,
            'is_classification': self.is_classification,
            'n_classes': self.n_classes
        }
    
    def save(self, path: str) -> None:
        """Save dataset to file."""
        data = {
            'X': self.X,
            'y': self.y,
            'config': self.config.to_dict(),
            'metadata': self.metadata,
            'feature_names': self.feature_names,
            'is_classification': self.is_classification,
            'n_classes': self.n_classes
        }
        np.savez_compressed(path, **data)
    
    @classmethod
    def load(cls, path: str) -> 'SyntheticDataset':
        """Load dataset from file."""
        data = np.load(path, allow_pickle=True)
        return cls(
            X=data['X'],
            y=data['y'],
            config=data['config'].item(),  # type: ignore
            metadata=data['metadata'].item(),  # type: ignore
            feature_names=list(data['feature_names']),
            is_classification=bool(data['is_classification']),
            n_classes=int(data['n_classes'])
        )


class SyntheticDatasetGenerator:
    """
    Main generator class for synthetic tabular datasets.
    
    This generator creates datasets by:
    1. Sampling hyperparameters from a prior
    2. Building a random causal DAG
    3. Defining transformations on edges
    4. Propagating noise through the DAG
    5. Selecting nodes as features/target
    6. Applying post-processing
    
    Each generated dataset comes from a different generative process,
    covering a huge family of possible data distributions.
    """
    
    def __init__(
        self, 
        prior: Optional[PriorConfig] = None,
        seed: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Initialize the generator.
        
        Args:
            prior: Prior configuration (uses default if None)
            seed: Random seed for reproducibility
            verbose: Whether to print progress information
        """
        self.prior = prior if prior is not None else PriorConfig()
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        
        # Track statistics
        self.n_generated = 0
    
    def generate(
        self, 
        config: Optional[DatasetConfig] = None,
        return_dag: bool = False
    ) -> SyntheticDataset:
        """
        Generate a single synthetic dataset.
        
        Args:
            config: Specific configuration (samples from prior if None)
            return_dag: Whether to include the DAG in metadata
            
        Returns:
            SyntheticDataset with features, target, and metadata
        """
        # Step 1: Sample or use provided config
        if config is None:
            config = self.prior.sample_hyperparams(self.rng)
        
        if self.verbose:
            print(f"Generating dataset with {config.n_rows} rows, {config.n_features} features")
            print(f"  DAG: {config.n_nodes} nodes, density={config.density:.2f}")
            print(f"  Task: {'classification' if config.is_classification else 'regression'}")
        
        # Create local RNG for this dataset
        dataset_rng = np.random.default_rng(config.seed)
        
        # Step 2: Build DAG
        dag_builder = DAGBuilder(config, dataset_rng)
        dag = dag_builder.build()
        
        if self.verbose:
            print(f"  Built DAG with {len(dag.nodes)} nodes, {len(dag.edges)} edges")
            print(f"  {dag.n_subgraphs} subgraphs, {len(dag.root_nodes)} root nodes")
        
        # Step 3: Create transformations
        transform_factory = TransformationFactory(config, dataset_rng)
        transformations = RowGenerator.create_transformations(dag, config, dataset_rng)
        
        if self.verbose:
            print(f"  Created {len(transformations)} transformations")
        
        # Step 4: Generate rows
        row_generator = RowGenerator(config, dag, transformations, dataset_rng)
        propagated = row_generator.generate()
        
        if self.verbose:
            print(f"  Generated {config.n_rows} rows")
        
        # Step 5: Select features and target
        selector = FeatureSelector(config, dag, transformations, dataset_rng)
        selection = selector.select()
        
        if self.verbose:
            print(f"  Selected {len(selection.feature_nodes)} features, target node {selection.target_node}")
            print(f"  Relevant: {len(selection.relevant_features)}, Irrelevant: {len(selection.irrelevant_features)}")
        
        # Step 6: Build table
        table_builder = TableBuilder(selection, transformations, config, dataset_rng)
        X, y = table_builder.build(propagated)
        
        # Step 7: Post-processing
        post_processor = PostProcessor(config, dataset_rng)
        
        # Convert feature_types to column indices
        feature_types_indexed = {
            i: selection.feature_types[nid]
            for i, nid in enumerate(selection.feature_nodes)
        }
        
        result = post_processor.process(X, y, feature_types_indexed)
        X_final, y_final = result.X, result.y
        
        if self.verbose:
            print(f"  Post-processing: warping={result.applied_transforms.get('warping', False)}, "
                  f"quantization={result.applied_transforms.get('quantization', False)}, "
                  f"missing={result.applied_transforms.get('missing_values', False)}")
        
        # Build metadata
        metadata = {
            'n_nodes': len(dag.nodes),
            'n_edges': len(dag.edges),
            'n_subgraphs': dag.n_subgraphs,
            'n_relevant_features': len(selection.relevant_features),
            'n_irrelevant_features': len(selection.irrelevant_features),
            'target_node': selection.target_node,
            'feature_nodes': selection.feature_nodes,
            'feature_types': feature_types_indexed,
            'post_processing': result.applied_transforms,
            'column_info': result.column_info,
            'generation_type': propagated.metadata.get('generation_type', 'independent'),
        }
        
        if return_dag:
            metadata['dag'] = dag
        
        self.n_generated += 1
        
        return SyntheticDataset(
            X=X_final,
            y=y_final,
            config=config,
            metadata=metadata,
            feature_names=table_builder.get_feature_names(),
            is_classification=selection.is_classification,
            n_classes=selection.n_classes
        )
    
    def generate_many(
        self, 
        n: int,
        yield_every: int = 1
    ) -> Iterator[SyntheticDataset]:
        """
        Generate multiple datasets.
        
        Args:
            n: Number of datasets to generate
            yield_every: Print progress every N datasets
            
        Yields:
            SyntheticDataset instances
        """
        for i in range(n):
            if self.verbose and (i + 1) % yield_every == 0:
                print(f"Generating dataset {i + 1}/{n}...")
            
            yield self.generate()
    
    def generate_batch(
        self, 
        n: int,
        parallel: bool = False
    ) -> List[SyntheticDataset]:
        """
        Generate a batch of datasets.
        
        Args:
            n: Number of datasets
            parallel: Whether to use parallel generation (TODO)
            
        Returns:
            List of SyntheticDataset instances
        """
        # TODO: Implement parallel generation
        return [self.generate() for _ in range(n)]
    
    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset the generator.
        
        Args:
            seed: New seed (uses original if None)
        """
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
                'n_rows_range': self.prior.n_rows_range,
                'n_features_range': self.prior.n_features_range,
                'n_nodes_range': self.prior.n_nodes_range,
                'prob_classification': self.prior.prob_classification,
            }
        }


def generate_training_data(
    n_datasets: int,
    output_dir: str,
    prior: Optional[PriorConfig] = None,
    seed: int = 42,
    verbose: bool = True
) -> None:
    """
    Generate datasets for training and save to disk.
    
    Args:
        n_datasets: Number of datasets to generate
        output_dir: Directory to save datasets
        prior: Prior configuration
        seed: Random seed
        verbose: Whether to print progress
    """
    os.makedirs(output_dir, exist_ok=True)
    
    generator = SyntheticDatasetGenerator(prior=prior, seed=seed, verbose=False)
    
    for i, dataset in enumerate(generator.generate_many(n_datasets)):
        if verbose and (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{n_datasets} datasets")
        
        # Save dataset
        path = os.path.join(output_dir, f"dataset_{i:06d}.npz")
        dataset.save(path)
    
    # Save metadata
    meta_path = os.path.join(output_dir, "generation_metadata.json")
    with open(meta_path, 'w') as f:
        json.dump({
            'n_datasets': n_datasets,
            'seed': seed,
            'prior': {
                'n_rows_range': list(prior.n_rows_range) if prior else list(PriorConfig().n_rows_range),
                'n_features_range': list(prior.n_features_range) if prior else list(PriorConfig().n_features_range),
            }
        }, f, indent=2)
    
    if verbose:
        print(f"Saved {n_datasets} datasets to {output_dir}")


class DatasetLoader:
    """
    Utility for loading saved datasets.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize loader.
        
        Args:
            data_dir: Directory containing saved datasets
        """
        self.data_dir = data_dir
        self.dataset_files = sorted([
            f for f in os.listdir(data_dir) 
            if f.startswith('dataset_') and f.endswith('.npz')
        ])
    
    def __len__(self) -> int:
        return len(self.dataset_files)
    
    def __getitem__(self, idx: int) -> SyntheticDataset:
        path = os.path.join(self.data_dir, self.dataset_files[idx])
        return SyntheticDataset.load(path)
    
    def __iter__(self) -> Iterator[SyntheticDataset]:
        for i in range(len(self)):
            yield self[i]
    
    def load_batch(
        self, 
        start: int, 
        end: int
    ) -> List[SyntheticDataset]:
        """Load a batch of datasets."""
        return [self[i] for i in range(start, min(end, len(self)))]

