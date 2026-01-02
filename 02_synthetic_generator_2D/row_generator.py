"""
Row generator for synthetic datasets.

This module handles:
1. Injecting noise into root nodes
2. Propagating values through the DAG following topological order
3. Optionally creating row dependencies (prototype-based generation)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    from .config import DatasetConfig
    from .dag_builder import DAG
    from .transformations import EdgeTransformation, TransformationFactory, RootNoiseGenerator
except ImportError:
    from config import DatasetConfig
    from dag_builder import DAG
    from transformations import EdgeTransformation, TransformationFactory, RootNoiseGenerator


@dataclass
class PropagatedValues:
    """
    Container for values propagated through the DAG.
    
    Attributes:
        values: Dictionary mapping node ID to array of values (n_samples,)
        metadata: Additional metadata about the propagation
    """
    values: Dict[int, np.ndarray]
    metadata: Dict[str, any]
    
    def get_node_value(self, node_id: int) -> np.ndarray:
        """Get values for a specific node."""
        return self.values[node_id]
    
    def get_multiple_nodes(self, node_ids: List[int]) -> np.ndarray:
        """Get values for multiple nodes as a 2D array."""
        return np.column_stack([self.values[nid] for nid in node_ids])


class RowGenerator:
    """
    Generates rows by propagating noise through a DAG.
    
    The generation process:
    1. Inject noise into root nodes
    2. For each node in topological order:
       - Collect parent values
       - Apply edge transformation
       - Store result
    3. Return values for all nodes
    """
    
    def __init__(
        self, 
        config: DatasetConfig, 
        dag: DAG,
        transformations: Dict[Tuple[int, int], EdgeTransformation],
        rng: Optional[np.random.Generator] = None
    ):
        """
        Initialize the row generator.
        
        Args:
            config: Dataset configuration
            dag: The causal DAG
            transformations: Dictionary mapping (parent_id, child_id) to transformation
            rng: Random number generator
        """
        self.config = config
        self.dag = dag
        self.transformations = transformations
        self.rng = rng if rng is not None else np.random.default_rng(config.seed)
        self.noise_generator = RootNoiseGenerator(config, self.rng)
    
    def generate(self, n_samples: Optional[int] = None) -> PropagatedValues:
        """
        Generate samples by propagating through the DAG.
        
        Args:
            n_samples: Number of samples to generate (uses config if not provided)
            
        Returns:
            PropagatedValues containing node values
        """
        if n_samples is None:
            n_samples = self.config.n_rows
        
        # Check if we should use prototype-based generation
        if self.config.has_row_dependency and self.config.n_prototypes > 0:
            return self._generate_with_prototypes(n_samples)
        else:
            return self._generate_independent(n_samples)
    
    def _generate_independent(self, n_samples: int) -> PropagatedValues:
        """Generate samples independently (no row dependencies)."""
        values: Dict[int, np.ndarray] = {}
        
        # Process nodes in topological order
        for node_id in self.dag.topological_order:
            node = self.dag.nodes[node_id]
            
            if node.is_root:
                # Root node: inject noise
                values[node_id] = self.noise_generator.generate(n_samples)
            else:
                # Non-root: combine parent values through transformations
                values[node_id] = self._compute_node_value(node_id, values, n_samples)
        
        return PropagatedValues(
            values=values,
            metadata={
                'n_samples': n_samples,
                'generation_type': 'independent'
            }
        )
    
    def _generate_with_prototypes(self, n_samples: int) -> PropagatedValues:
        """
        Generate samples using prototypes (row dependencies).
        
        Per paper: "for each input vector xi to be sampled, we assign weights αij 
        to the prototypes and linearly mix the final input as xi = Σj αij * xj*, 
        where Σj αij = 1. The weights αij are sampled from a multinomial distribution, 
        αi ~ Multinomial(β), where β is a temperature hyperparameter controlling the 
        degree of non-independence: larger β yields more uniform weights, whereas 
        smaller β concentrates the weights on fewer prototypes per sample."
        """
        n_prototypes = self.config.n_prototypes
        
        # Generate prototype values
        prototype_values: Dict[int, np.ndarray] = {}
        
        for node_id in self.dag.topological_order:
            node = self.dag.nodes[node_id]
            
            if node.is_root:
                prototype_values[node_id] = self.noise_generator.generate(n_prototypes)
            else:
                prototype_values[node_id] = self._compute_node_value(
                    node_id, prototype_values, n_prototypes
                )
        
        # Sample mixing weights using Dirichlet (continuous relaxation of Multinomial)
        # Temperature β controls concentration: 
        # - Small β (< 1): weights concentrate on few prototypes
        # - Large β (> 1): weights more uniform across prototypes
        temperature = self.rng.uniform(0.1, 2.0)  # Sample temperature
        
        # Dirichlet with uniform concentration parameter β
        alpha = np.ones(n_prototypes) * temperature
        weights = self.rng.dirichlet(alpha, size=n_samples)  # (n_samples, n_prototypes)
        
        # Generate samples as weighted mixtures of prototypes
        values: Dict[int, np.ndarray] = {}
        noise_scale = self.config.prototype_noise_scale
        
        for node_id in self.dag.nodes:
            # Linear combination: xi = Σj αij * xj*
            proto_vals = prototype_values[node_id]  # (n_prototypes,)
            mixed_values = weights @ proto_vals  # (n_samples,)
            
            # Add small noise for variation
            noise = self.rng.normal(0, noise_scale, size=n_samples)
            values[node_id] = mixed_values + noise
        
        return PropagatedValues(
            values=values,
            metadata={
                'n_samples': n_samples,
                'generation_type': 'prototype_mixture',
                'n_prototypes': n_prototypes,
                'temperature': temperature
            }
        )
    
    def _compute_node_value(
        self, 
        node_id: int, 
        current_values: Dict[int, np.ndarray],
        n_samples: int
    ) -> np.ndarray:
        """
        Compute the value of a non-root node from its parents.
        
        Args:
            node_id: ID of the node to compute
            current_values: Dictionary of already computed node values
            n_samples: Number of samples
            
        Returns:
            Array of computed values
        """
        node = self.dag.nodes[node_id]
        parents = node.parents
        
        if not parents:
            # This shouldn't happen for non-root nodes, but handle gracefully
            return self.noise_generator.generate(n_samples)
        
        # Collect parent values
        parent_values = np.column_stack([current_values[pid] for pid in parents])
        
        # Apply transformations and combine
        # For now, we use a single transformation for all incoming edges combined
        # (The paper isn't explicit about whether each edge has its own transform)
        
        # Look for a transformation for this node
        # Try to find any transformation that outputs to this node
        transform = None
        for (parent_id, child_id), t in self.transformations.items():
            if child_id == node_id:
                transform = t
                break
        
        if transform is None:
            # Default: weighted sum with noise
            weights = self.rng.normal(0, 1, size=len(parents))
            weights = weights / np.linalg.norm(weights)
            result = parent_values @ weights
            result = result + self.rng.normal(0, 0.1, size=n_samples)
            return result
        
        return transform.forward(parent_values)
    
    @staticmethod
    def create_transformations(
        dag: DAG, 
        config: DatasetConfig,
        rng: Optional[np.random.Generator] = None
    ) -> Dict[Tuple[int, int], EdgeTransformation]:
        """
        Create transformations for all nodes in the DAG.
        
        Note: We create one transformation per child node, not per edge.
        This transformation combines all parent inputs.
        
        Args:
            dag: The causal DAG
            config: Dataset configuration
            rng: Random number generator
            
        Returns:
            Dictionary mapping (first_parent_id, child_id) to transformation
        """
        if rng is None:
            rng = np.random.default_rng(config.seed)
        
        factory = TransformationFactory(config, rng)
        transformations: Dict[Tuple[int, int], EdgeTransformation] = {}
        
        for node_id in dag.topological_order:
            node = dag.nodes[node_id]
            
            if not node.is_root and node.parents:
                # Create transformation for this node
                n_parents = len(node.parents)
                transform = factory.create(n_parents)
                
                # Store with first parent as key (arbitrary but consistent)
                key = (node.parents[0], node_id)
                transformations[key] = transform
        
        return transformations


class BatchRowGenerator:
    """
    Generator that yields batches of rows for memory efficiency.
    
    Useful when generating very large datasets.
    """
    
    def __init__(
        self,
        config: DatasetConfig,
        dag: DAG,
        transformations: Dict[Tuple[int, int], EdgeTransformation],
        batch_size: int = 1000,
        rng: Optional[np.random.Generator] = None
    ):
        self.config = config
        self.dag = dag
        self.transformations = transformations
        self.batch_size = batch_size
        self.rng = rng if rng is not None else np.random.default_rng(config.seed)
        self.row_generator = RowGenerator(config, dag, transformations, self.rng)
    
    def __iter__(self):
        """Iterate over batches."""
        total_samples = self.config.n_rows
        generated = 0
        
        while generated < total_samples:
            batch_size = min(self.batch_size, total_samples - generated)
            yield self.row_generator.generate(batch_size)
            generated += batch_size
    
    def generate_all(self) -> PropagatedValues:
        """Generate all samples at once."""
        return self.row_generator.generate()

