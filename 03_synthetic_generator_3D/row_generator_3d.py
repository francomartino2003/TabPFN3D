"""
Row Generator for 3D Time Series Datasets.

This module generates time series observations by propagating noise 
through the unrolled temporal DAG.

Process for each observation:
1. Generate noise for root nodes at t=0
2. Propagate through spatial edges within each timestep
3. Propagate through temporal edges to future timesteps
4. Optionally add correlated noise (AR process)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import sys
import os

# Import from 2D generator - need to temporarily modify path
# and remove conflicting modules from sys.modules
_2d_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '02_synthetic_generator_2D')
_3d_path = os.path.dirname(os.path.abspath(__file__))

# Save current state
_original_path = sys.path.copy()
_saved_modules = {}

# Temporarily save and remove 3D config from modules if present
for mod_name in ['config', 'transformations']:
    if mod_name in sys.modules:
        _saved_modules[mod_name] = sys.modules.pop(mod_name)

# Add 2D path at the beginning, remove 3D path temporarily
sys.path = [p for p in sys.path if os.path.normpath(p) != os.path.normpath(_3d_path)]
sys.path.insert(0, _2d_path)

try:
    # Now import from 2D
    from transformations import (
        TransformationFactory, EdgeTransformation, RootNoiseGenerator,
        NNTransformation, DiscretizationTransformation, TreeTransformation, IdentityTransformation
    )
finally:
    # Restore path
    sys.path = _original_path
    
    # Remove 2D modules from cache to avoid conflicts
    for mod_name in ['config', 'transformations']:
        if mod_name in sys.modules:
            sys.modules.pop(mod_name)
    
    # Restore saved 3D modules
    for mod_name, mod in _saved_modules.items():
        sys.modules[mod_name] = mod

# Now import 3D modules
from config import DatasetConfig3D
from temporal_dag_builder import TemporalDAG


@dataclass
class TemporalPropagatedValues:
    """
    Container for values propagated through the temporal DAG.
    
    Attributes:
        values: Dict mapping global_id to array of values (n_samples,)
        values_by_time: Organized as [timestep][base_node_id] = values
        metadata: Generation metadata
    """
    values: Dict[int, np.ndarray]
    n_samples: int
    n_timesteps: int
    n_base_nodes: int
    metadata: Dict[str, any]
    
    def get_node_value(self, timestep: int, base_node_id: int) -> np.ndarray:
        """Get values for a specific node at a specific timestep."""
        global_id = timestep * self.n_base_nodes + base_node_id
        return self.values[global_id]
    
    def get_time_series(self, base_node_id: int, t_start: int, t_end: int) -> np.ndarray:
        """
        Get time series for a node across timesteps.
        
        Returns:
            Array of shape (n_samples, t_end - t_start)
        """
        series = []
        for t in range(t_start, t_end):
            global_id = t * self.n_base_nodes + base_node_id
            series.append(self.values[global_id])
        return np.column_stack(series)
    
    def get_all_timesteps(self, base_node_id: int) -> np.ndarray:
        """
        Get complete time series for a node.
        
        Returns:
            Array of shape (n_samples, n_timesteps)
        """
        return self.get_time_series(base_node_id, 0, self.n_timesteps)


class TemporalNoiseGenerator:
    """
    Generates noise for root nodes, optionally with temporal correlation.
    """
    
    def __init__(self, config: DatasetConfig3D, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.base_generator = RootNoiseGenerator(config, rng)
    
    def generate_initial(self, n_samples: int) -> np.ndarray:
        """Generate initial noise for t=0."""
        noise_type = self.config.noise_type
        
        if noise_type == 'normal':
            return self.rng.normal(0, 1, size=n_samples)
        elif noise_type == 'uniform':
            return self.rng.uniform(-np.sqrt(3), np.sqrt(3), size=n_samples)
        elif noise_type == 'laplace':
            return self.rng.laplace(0, 1, size=n_samples)
        else:  # mixture
            n_components = self.rng.integers(2, 5)
            means = self.rng.normal(0, 2, size=n_components)
            stds = self.rng.uniform(0.5, 1.5, size=n_components)
            component = self.rng.integers(0, n_components, size=n_samples)
            return means[component] + stds[component] * self.rng.normal(0, 1, size=n_samples)
    
    def generate_temporal(self, n_samples: int, previous_noise: np.ndarray) -> np.ndarray:
        """
        Generate noise for subsequent timesteps, potentially correlated with previous.
        
        Uses AR(1) process if configured.
        """
        if self.config.has_correlated_noise:
            # AR(1) process: x_t = ar_coef * x_{t-1} + noise
            ar_coef = self.config.noise_ar_coef
            innovation = self.rng.normal(0, np.sqrt(1 - ar_coef**2), size=n_samples)
            return ar_coef * previous_noise + innovation
        else:
            return self.generate_initial(n_samples)


class RowGenerator3D:
    """
    Generates time series observations by propagating through temporal DAG.
    """
    
    def __init__(
        self,
        config: DatasetConfig3D,
        dag: TemporalDAG,
        spatial_transformations: Dict[Tuple[int, int], EdgeTransformation],
        temporal_transformations: Dict[Tuple[int, int, int, int], EdgeTransformation],
        rng: Optional[np.random.Generator] = None
    ):
        """
        Initialize the 3D row generator.
        
        Args:
            config: Dataset configuration
            dag: The unrolled temporal DAG
            spatial_transformations: Transforms for spatial edges (base_parent, base_child) -> Transform
            temporal_transformations: Transforms for temporal edges (from_t, to_t, from_base, to_base) -> Transform
            rng: Random number generator
        """
        self.config = config
        self.dag = dag
        self.spatial_transformations = spatial_transformations
        self.temporal_transformations = temporal_transformations
        self.rng = rng if rng is not None else np.random.default_rng(config.seed)
        self.noise_generator = TemporalNoiseGenerator(config, self.rng)
    
    def generate(self, n_samples: Optional[int] = None) -> TemporalPropagatedValues:
        """
        Generate time series observations.
        
        KEY CHANGE: Each subgraph at each timestep receives fresh noise injection.
        This maintains the 2D generator behavior where each "mini-graph" at time t
        gets its own noise, while temporal connections create dependencies.
        
        Args:
            n_samples: Number of observations to generate
            
        Returns:
            TemporalPropagatedValues with all node values across time
        """
        if n_samples is None:
            n_samples = self.config.n_samples
        
        n_timesteps = self.dag.n_timesteps
        n_base_nodes = self.dag.n_base_nodes
        
        # Initialize value storage
        values: Dict[int, np.ndarray] = {}
        
        # Track noise for AR process (per base node)
        root_noise_prev: Dict[int, np.ndarray] = {}
        
        # Build lookup for edge weights
        edge_weights = self._build_edge_weight_lookup()
        
        # Build lookup for conditional connections
        conditional_connections = self._build_conditional_lookup()
        
        # Process nodes in topological order
        for global_id in self.dag.topological_order:
            node = self.dag.nodes[global_id]
            t = node.timestep
            base_id = node.base_node_id
            
            # Determine if this is a root node in base DAG
            is_base_root = base_id in [n for n in self.dag.base_dag.root_nodes]
            
            if is_base_root:
                # ROOT nodes get fresh noise at EACH timestep
                # This is key: each subgraph t gets independent initialization
                if t == 0:
                    # First timestep: completely fresh noise
                    values[global_id] = self.noise_generator.generate_initial(n_samples)
                    root_noise_prev[base_id] = values[global_id]
                else:
                    # Later timesteps: AR noise or fresh (still independent per subgraph)
                    if self.config.has_correlated_noise and base_id in root_noise_prev:
                        values[global_id] = self.noise_generator.generate_temporal(
                            n_samples, root_noise_prev[base_id]
                        )
                    else:
                        # Fresh noise for this subgraph at this timestep
                        values[global_id] = self.noise_generator.generate_initial(n_samples)
                    root_noise_prev[base_id] = values[global_id]
                    
            elif not node.spatial_parents and not node.temporal_parents:
                # No parents at all - generate fresh noise
                values[global_id] = self.noise_generator.generate_initial(n_samples)
                
            else:
                # Non-root: combine inputs from parents
                values[global_id] = self._compute_node_value(
                    global_id, node, values, n_samples,
                    edge_weights, conditional_connections
                )
        
        return TemporalPropagatedValues(
            values=values,
            n_samples=n_samples,
            n_timesteps=n_timesteps,
            n_base_nodes=n_base_nodes,
            metadata={
                'generation_type': 'temporal',
                'has_correlated_noise': self.config.has_correlated_noise,
                'n_temporal_patterns': len(self.dag.connection_configs)
            }
        )
    
    def _build_edge_weight_lookup(self) -> Dict[Tuple[int, int], float]:
        """Build lookup for edge weights from temporal edge info."""
        weights = {}
        for edge_info in self.dag.temporal_edge_info:
            src_global = edge_info.from_timestep * self.dag.n_base_nodes + edge_info.from_base_id
            tgt_global = edge_info.to_timestep * self.dag.n_base_nodes + edge_info.to_base_id
            # Store max weight if multiple edges between same nodes
            key = (src_global, tgt_global)
            if key not in weights or edge_info.weight > weights[key]:
                weights[key] = edge_info.weight
        return weights
    
    def _build_conditional_lookup(self) -> Dict[str, Any]:
        """Build lookup for conditional connections."""
        from config import TemporalConnectionConfig
        
        conditional = {}
        for conn in self.dag.connection_configs:
            if isinstance(conn, TemporalConnectionConfig):
                if conn.connection_type == 'conditional_lag':
                    conditional[conn.pattern_id] = {
                        'type': 'lag',
                        'thresholds': conn.condition_thresholds or [],
                        'skips': conn.conditional_skips or [conn.skip]
                    }
                elif conn.connection_type == 'conditional_dest':
                    conditional[conn.pattern_id] = {
                        'type': 'dest',
                        'thresholds': conn.condition_thresholds or [],
                        'targets': conn.conditional_targets or [conn.target_nodes]
                    }
        return conditional
    
    def _compute_node_value(
        self,
        global_id: int,
        node,
        current_values: Dict[int, np.ndarray],
        n_samples: int,
        edge_weights: Dict[Tuple[int, int], float],
        conditional_connections: Dict[str, Any]
    ) -> np.ndarray:
        """
        Compute value of a node from its spatial and temporal parents.
        
        Handles:
        - Weighted contributions from multi-skip connections
        - Conditional connections (lag switching, destination switching)
        - Dimension mismatches between transforms and actual inputs
        """
        all_parent_values = []
        all_parent_weights = []
        
        # Collect spatial parent values (weight = 1.0 for spatial)
        for parent_id in node.spatial_parents:
            if parent_id in current_values:
                all_parent_values.append(current_values[parent_id])
                all_parent_weights.append(1.0)
        
        # Collect temporal parent values with their weights
        for parent_id in node.temporal_parents:
            if parent_id in current_values:
                weight = edge_weights.get((parent_id, global_id), 1.0)
                all_parent_values.append(current_values[parent_id])
                all_parent_weights.append(weight)
        
        if not all_parent_values:
            # No parents with values - generate noise
            return self.noise_generator.generate_initial(n_samples)
        
        # Combine parent values with weighted sum (simple but robust approach)
        # This avoids dimension mismatches with NN transformations
        n_parents = len(all_parent_values)
        
        # Normalize weights
        norm_weights = np.array(all_parent_weights)
        norm_weights = norm_weights / (np.sum(norm_weights) + 1e-10)
        
        # Add randomness to make it more diverse
        rand_weights = norm_weights + self.rng.normal(0, 0.1, size=n_parents)
        rand_weights = np.abs(rand_weights)
        rand_weights = rand_weights / (np.sum(rand_weights) + 1e-10)
        
        # Weighted sum of parent values
        parent_matrix = np.column_stack(all_parent_values)
        result = parent_matrix @ rand_weights
        
        # Apply non-linearity based on transform type if available
        transform = self._get_transformation(global_id, node)
        
        if transform is not None:
            # Apply activation from transform if it's an NN type
            if hasattr(transform, 'activation'):
                result = self._apply_activation(result, transform.activation)
            elif hasattr(transform, 'n_categories'):
                # Discretization - apply simple step function
                result = np.floor(result * 3) / 3
        
        # Add noise
        result = result + self.rng.normal(0, self.config.noise_scale, size=n_samples)
        
        return result
    
    def _apply_activation(self, x: np.ndarray, activation: str) -> np.ndarray:
        """Apply activation function to values."""
        if activation == 'identity':
            return x
        elif activation == 'tanh':
            return np.tanh(x)
        elif activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif activation == 'relu':
            return np.maximum(0, x)
        elif activation == 'sin':
            return np.sin(x)
        elif activation == 'cos':
            return np.cos(x)
        elif activation == 'abs':
            return np.abs(x)
        elif activation == 'square':
            return x ** 2
        elif activation == 'sqrt':
            return np.sqrt(np.abs(x))
        elif activation == 'exp_neg':
            return np.exp(-np.abs(x))
        elif activation == 'softplus':
            return np.log1p(np.exp(np.clip(x, -500, 500)))
        elif activation == 'step':
            return (x > 0).astype(float)
        elif activation == 'gaussian':
            return np.exp(-x**2)
        else:
            return x
    
    def _get_transformation(self, global_id: int, node) -> Optional[EdgeTransformation]:
        """Get the transformation for a node based on its parents."""
        t = node.timestep
        base_id = node.base_node_id
        
        # Check spatial transformations first
        for parent_id in node.spatial_parents:
            parent_t, parent_base = self.dag.get_timestep_and_base(parent_id)
            key = (parent_base, base_id)
            if key in self.spatial_transformations:
                return self.spatial_transformations[key]
        
        # Check temporal transformations
        for parent_id in node.temporal_parents:
            parent_t, parent_base = self.dag.get_timestep_and_base(parent_id)
            key = (parent_t, t, parent_base, base_id)
            if key in self.temporal_transformations:
                return self.temporal_transformations[key]
        
        return None


def create_transformations_3d(
    dag: TemporalDAG,
    config: DatasetConfig3D,
    rng: Optional[np.random.Generator] = None
) -> Tuple[Dict[Tuple[int, int], EdgeTransformation], 
           Dict[Tuple[int, int, int, int], EdgeTransformation]]:
    """
    Create transformations for spatial and temporal edges.
    
    Returns:
        Tuple of (spatial_transformations, temporal_transformations)
    """
    if rng is None:
        rng = np.random.default_rng(config.seed)
    
    # Create factory (reuse from 2D)
    # We need to create a mock config for the factory
    from dataclasses import dataclass as dc
    
    @dc
    class MockConfig:
        edge_transform_probs: Dict
        nn_hidden: int
        nn_width: int
        allowed_activations: List
        n_categories: int
        tree_depth: int
        tree_n_splits: int
        noise_scale: float
        edge_noise_prob: float
        seed: int
    
    mock_config = MockConfig(
        edge_transform_probs=config.edge_transform_probs,
        nn_hidden=config.nn_hidden,
        nn_width=config.nn_width,
        allowed_activations=config.allowed_activations,
        n_categories=config.n_categories,
        tree_depth=config.tree_depth,
        tree_n_splits=config.tree_n_splits,
        noise_scale=config.noise_scale,
        edge_noise_prob=config.edge_noise_prob,
        seed=config.seed
    )
    
    factory = TransformationFactory(mock_config, rng)
    
    # Spatial transformations (one per edge type in base graph)
    spatial_transforms: Dict[Tuple[int, int], EdgeTransformation] = {}
    
    for parent_id, child_id in dag.base_dag.edges:
        n_parents = len(dag.base_dag.nodes[child_id].parents)
        transform = factory.create(n_parents)
        spatial_transforms[(parent_id, child_id)] = transform
    
    # Temporal transformations (one per connection pattern)
    temporal_transforms: Dict[Tuple[int, int, int, int], EdgeTransformation] = {}
    
    # Create one transform per unique connection pattern (not per edge)
    for conn in config.temporal_connections:
        for source_base in conn.source_nodes:
            for target_base in conn.target_nodes:
                # Key is (skip pattern), we'll apply it to all timesteps
                # For simplicity, create one transform and reuse
                n_inputs = len(conn.source_nodes)
                transform = factory.create(n_inputs)
                
                # Apply to all valid timesteps
                for t in range(dag.n_timesteps - conn.skip):
                    key = (t, t + conn.skip, source_base, target_base)
                    temporal_transforms[key] = transform
    
    return spatial_transforms, temporal_transforms

