"""
Temporal Propagator for 3D Synthetic Data.

Propagates values through the DAG for T timesteps, handling:
- Noise inputs (fresh each timestep)
- Time inputs (deterministic based on t)
- State inputs (memory from previous timestep)

This module extends the 2D propagation to handle temporal dependencies.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Local 3D modules
from config import DatasetConfig3D
from temporal_inputs import TemporalInputManager

# 2D components via wrapper
from dag_utils import DAG, DAGNode, EdgeTransformation


@dataclass
class TemporalPropagatedValues:
    """
    Container for values propagated through the DAG across all timesteps.
    
    Attributes:
        values: Dict mapping (timestep, node_id) -> values array of shape (n_samples,)
        T: Total number of timesteps
        n_samples: Number of samples
    """
    values: Dict[Tuple[int, int], np.ndarray]
    T: int
    n_samples: int
    
    def get_value(self, t: int, node_id: int) -> np.ndarray:
        """Get values for a specific timestep and node."""
        return self.values.get((t, node_id), np.zeros(self.n_samples))
    
    def get_node_timeseries(self, node_id: int) -> np.ndarray:
        """
        Get the full timeseries for a node.
        
        Returns:
            Array of shape (n_samples, T)
        """
        timeseries = np.zeros((self.n_samples, self.T))
        for t in range(self.T):
            if (t, node_id) in self.values:
                timeseries[:, t] = self.values[(t, node_id)]
        return timeseries
    
    def get_timestep_values(self, t: int) -> Dict[int, np.ndarray]:
        """
        Get all node values for a specific timestep.
        
        Returns:
            Dict mapping node_id -> values array
        """
        return {
            node_id: values 
            for (ts, node_id), values in self.values.items() 
            if ts == t
        }


class TemporalPropagator:
    """
    Propagates values through DAG for T timesteps with temporal dependencies.
    
    The propagation works as follows:
    1. For each timestep t from 0 to T-1:
       a. Generate root node inputs (noise, time, state)
       b. Propagate through DAG in topological order
       c. Store all node values for this timestep
       d. Extract state values for next timestep
    """
    
    def __init__(
        self,
        config: DatasetConfig3D,
        dag: DAG,
        transformations: Dict[Tuple[int, int], EdgeTransformation],
        input_manager: TemporalInputManager,
        rng: np.random.Generator
    ):
        """
        Initialize the temporal propagator.
        
        Args:
            config: Dataset configuration
            dag: The causal DAG
            transformations: Edge transformations
            input_manager: Manager for temporal inputs
            rng: Random number generator
        """
        self.config = config
        self.dag = dag
        self.transformations = transformations
        self.input_manager = input_manager
        self.rng = rng
        
        # Compute topological order once
        self.topo_order = self._compute_topological_order()
        
        # Identify root nodes
        self.root_nodes = [nid for nid, node in dag.nodes.items() if not node.parents]
        
        # Assign root nodes to input types
        self.input_manager.assign_root_nodes(self.root_nodes)
    
    def propagate(self, n_samples: int, T: Optional[int] = None) -> TemporalPropagatedValues:
        """
        Propagate values through the DAG for T timesteps.
        
        Args:
            n_samples: Number of samples to generate
            T: Number of timesteps (defaults to config.T_total)
            
        Returns:
            TemporalPropagatedValues containing all node values across time
        """
        if T is None:
            T = self.config.T_total
        
        all_values: Dict[Tuple[int, int], np.ndarray] = {}
        previous_state: Optional[np.ndarray] = None
        
        for t in range(T):
            # Generate inputs for this timestep
            root_inputs = self.input_manager.generate_inputs_for_timestep(
                t=t, T=T, n_samples=n_samples, previous_state=previous_state
            )
            
            # Propagate through DAG
            timestep_values = self._propagate_single_timestep(root_inputs, n_samples)
            
            # Store values
            for node_id, values in timestep_values.items():
                all_values[(t, node_id)] = values
            
            # Extract state for next timestep
            if self.input_manager.state_node_ids:
                previous_state = self.input_manager.state_generator.extract_state_values(
                    timestep_values
                )
        
        return TemporalPropagatedValues(
            values=all_values,
            T=T,
            n_samples=n_samples
        )
    
    def _propagate_single_timestep(
        self, 
        root_inputs: Dict[int, np.ndarray],
        n_samples: int
    ) -> Dict[int, np.ndarray]:
        """
        Propagate values through DAG for a single timestep.
        
        Args:
            root_inputs: Dict mapping root node_id -> input values
            n_samples: Number of samples
            
        Returns:
            Dict mapping node_id -> propagated values
        """
        values: Dict[int, np.ndarray] = {}
        
        for node_id in self.topo_order:
            node = self.dag.nodes[node_id]
            
            if not node.parents:
                # Root node - use provided inputs or generate noise
                if node_id in root_inputs:
                    values[node_id] = root_inputs[node_id]
                else:
                    # Fallback: generate noise
                    values[node_id] = self.rng.normal(0, 1, size=n_samples)
            else:
                # Non-root node - compute from parents
                values[node_id] = self._compute_node_value(node_id, node, values)
        
        return values
    
    def _compute_node_value(
        self, 
        node_id: int, 
        node: DAGNode, 
        values: Dict[int, np.ndarray]
    ) -> np.ndarray:
        """
        Compute value for a non-root node from its parents.
        
        Args:
            node_id: ID of the node
            node: The node object
            values: Current values dict
            
        Returns:
            Computed values for this node
        """
        n_samples = len(list(values.values())[0])
        
        # Collect parent values
        parent_values = []
        for parent_id in node.parents:
            if parent_id in values:
                parent_values.append(values[parent_id])
            else:
                # Shouldn't happen with correct topological order
                parent_values.append(np.zeros(n_samples))
        
        if not parent_values:
            return np.zeros(n_samples)
        
        # Stack parent values
        parent_array = np.column_stack(parent_values)
        
        # Apply transformation from first parent (simplified - could combine)
        first_parent = node.parents[0]
        transform_key = (first_parent, node_id)
        
        if transform_key in self.transformations:
            transform = self.transformations[transform_key]
            result = transform.forward(parent_array)
            
            # Handle multi-dimensional output
            if result.ndim > 1:
                result = result[:, 0] if result.shape[1] > 0 else result.flatten()
            
            return result
        else:
            # No transformation - weighted sum with noise
            weights = self.rng.normal(0, 1, size=len(parent_values))
            weights = weights / (np.linalg.norm(weights) + 1e-8)
            
            combined = sum(w * v for w, v in zip(weights, parent_values))
            noise = self.rng.normal(0, self.config.noise_scale, size=n_samples)
            
            return combined + noise
    
    def _compute_topological_order(self) -> List[int]:
        """Compute topological ordering of DAG nodes."""
        visited = set()
        order = []
        
        def dfs(node_id: int):
            if node_id in visited:
                return
            visited.add(node_id)
            
            # Visit parents first
            node = self.dag.nodes[node_id]
            for parent_id in node.parents:
                dfs(parent_id)
            
            order.append(node_id)
        
        for node_id in self.dag.nodes:
            dfs(node_id)
        
        return order


class BatchTemporalPropagator:
    """
    Efficient batch propagation for generating multiple sequences.
    
    Used for:
    - IID mode: Generate many independent sequences
    - Sliding window: Generate one long sequence, extract windows
    - Mixed mode: Generate several long sequences
    """
    
    def __init__(
        self,
        config: DatasetConfig3D,
        dag: DAG,
        transformations: Dict[Tuple[int, int], EdgeTransformation],
        input_manager: TemporalInputManager,
        rng: np.random.Generator
    ):
        self.config = config
        self.propagator = TemporalPropagator(
            config, dag, transformations, input_manager, rng
        )
        self.rng = rng
    
    def generate_iid_sequences(
        self, 
        n_sequences: int, 
        T: int
    ) -> TemporalPropagatedValues:
        """
        Generate independent sequences (IID mode).
        
        Each sequence (row) has different noise but same structure.
        All sequences are generated in parallel as a batch.
        
        Args:
            n_sequences: Number of sequences (batch size)
            T: Timesteps per sequence
            
        Returns:
            Single TemporalPropagatedValues with n_sequences samples
        """
        # Generate all sequences in parallel (much more efficient)
        return self.propagator.propagate(n_samples=n_sequences, T=T)
    
    def generate_single_long_sequence(
        self, 
        n_samples: int,
        T: int
    ) -> TemporalPropagatedValues:
        """
        Generate a single long sequence for sliding window extraction.
        
        Args:
            n_samples: Batch size for parallel generation
            T: Total timesteps
            
        Returns:
            Propagated values for the long sequence
        """
        return self.propagator.propagate(n_samples=n_samples, T=T)
    
    def generate_mixed_sequences(
        self, 
        n_long_sequences: int,
        samples_per_sequence: int,
        T: int
    ) -> List[TemporalPropagatedValues]:
        """
        Generate multiple long sequences (mixed mode).
        
        Args:
            n_long_sequences: Number of long sequences
            samples_per_sequence: Samples per long sequence
            T: Timesteps per sequence
            
        Returns:
            List of propagated values
        """
        sequences = []
        for _ in range(n_long_sequences):
            seq = self.propagator.propagate(n_samples=samples_per_sequence, T=T)
            sequences.append(seq)
        return sequences

