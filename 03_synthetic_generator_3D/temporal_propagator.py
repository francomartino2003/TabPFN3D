"""
Temporal Propagator for 3D Synthetic Data.

Propagates values through the DAG for T timesteps.

v4 Design:
- TIME input: u = t/T (same for all samples at t)
- MEMORY inputs: fixed per sequence (sampled once)
- No state inputs (no t-k lookups)
- No numerical clipping
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from config import DatasetConfig3D
from temporal_inputs import TemporalInputManager

from dag_utils import DAG, DAGNode, EdgeTransformation


class TemporalPropagatedValues:
    """
    Container for values propagated through the DAG across all timesteps.
    
    Attributes:
        values: Dict mapping (timestep, node_id) -> values array of shape (n_samples,)
        T: Total number of timesteps
        n_samples: Number of samples
    """
    
    def __init__(self, values: Dict[Tuple[int, int], np.ndarray], T: int, n_samples: int):
        self.values = values
        self.T = T
        self.n_samples = n_samples
        self._timeseries_cache: Dict[int, np.ndarray] = {}
    
    def get_value(self, t: int, node_id: int) -> np.ndarray:
        """Get values for a specific timestep and node."""
        return self.values.get((t, node_id), np.zeros(self.n_samples))
    
    def get_node_timeseries(self, node_id: int) -> np.ndarray:
        """
        Get the full timeseries for a node (cached).
        
        Returns:
            Array of shape (n_samples, T)
        """
        if node_id in self._timeseries_cache:
            return self._timeseries_cache[node_id]
        
        timeseries = np.zeros((self.n_samples, self.T), dtype=np.float32)
        for t in range(self.T):
            key = (t, node_id)
            if key in self.values:
                timeseries[:, t] = self.values[key]
        
        self._timeseries_cache[node_id] = timeseries
        return timeseries
    
    def get_timestep_values(self, t: int) -> Dict[int, np.ndarray]:
        """Get all node values for a specific timestep."""
        return {
            node_id: values 
            for (ts, node_id), values in self.values.items() 
            if ts == t
        }


class BatchTemporalPropagator:
    """
    Efficient batch propagation for generating multiple sequences.
    
    v4 Design:
    - No state lookups (no t-k)
    - No numerical clipping
    - MEMORY is sampled once per sequence/batch
    """
    
    def __init__(
        self,
        config: DatasetConfig3D,
        dag: DAG,
        transformations: Dict[int, EdgeTransformation],
        input_manager: TemporalInputManager,
        rng: np.random.Generator
    ):
        self.config = config
        self.dag = dag
        self.transformations = transformations
        self.input_manager = input_manager
        self.rng = rng
        
        # Pre-compute node info
        self.topo_order = self._compute_topological_order()
        self.n_nodes = len(dag.nodes)
        
        # Node mappings
        self.node_to_idx = {nid: i for i, nid in enumerate(self.topo_order)}
        self.idx_to_node = {i: nid for i, nid in enumerate(self.topo_order)}
        
        # Pre-compute parent indices
        self.parent_indices = {}
        for node_id in self.topo_order:
            node = dag.nodes[node_id]
            if node.parents:
                self.parent_indices[node_id] = [self.node_to_idx[p] for p in node.parents]
        
        # Root nodes
        self.root_node_ids = set(nid for nid, n in dag.nodes.items() if not n.parents)
        
        # Assign root nodes to input types
        self.input_manager.assign_root_nodes(list(self.root_node_ids))
    
    def _compute_topological_order(self) -> List[int]:
        """Compute topological ordering."""
        visited = set()
        order = []
        
        def dfs(node_id: int):
            if node_id in visited:
                return
            visited.add(node_id)
            node = self.dag.nodes[node_id]
            for parent_id in node.parents:
                dfs(parent_id)
            order.append(node_id)
        
        for node_id in self.dag.nodes:
            dfs(node_id)
        return order
    
    def _fast_propagate(self, n_samples: int, T: int, reset_memory: bool = True) -> np.ndarray:
        """
        Fast propagation using pre-allocated arrays.
        
        Args:
            n_samples: Number of samples
            T: Number of timesteps
            reset_memory: If True, sample new MEMORY vectors
        
        Returns:
            Array of shape (n_nodes, n_samples, T)
        """
        # Sample MEMORY vectors for this batch
        if reset_memory:
            self.input_manager.sample_memory_for_sequences(n_samples)
        
        # Pre-allocate output array
        values = np.zeros((self.n_nodes, n_samples, T), dtype=np.float32)
        
        for t in range(T):
            # Generate root inputs (TIME + MEMORY)
            root_inputs = self.input_manager.generate_inputs_for_timestep(
                t=t, T=T, n_samples=n_samples
            )
            
            # Process each node in topological order
            for node_id in self.topo_order:
                idx = self.node_to_idx[node_id]
                
                if node_id in self.root_node_ids:
                    # Root node: use input values
                    if node_id in root_inputs:
                        values[idx, :, t] = root_inputs[node_id]
                    else:
                        # Fallback (shouldn't happen)
                        values[idx, :, t] = self.rng.normal(0, 1, size=n_samples)
                else:
                    # Non-root: compute from parents
                    parent_idxs = self.parent_indices[node_id]
                    parent_vals = values[parent_idxs, :, t].T  # (n_samples, n_parents)
                    
                    if node_id in self.transformations:
                        result = self.transformations[node_id].forward(parent_vals)
                        if result.ndim > 1:
                            result = result[:, 0] if result.shape[1] > 0 else result.flatten()
                        values[idx, :, t] = result
                    else:
                        # Fallback: weighted sum
                        weights = self.rng.normal(0, 1, size=len(parent_idxs))
                        weights = weights / (np.linalg.norm(weights) + 1e-8)
                        values[idx, :, t] = parent_vals @ weights
            
                # NO CLIPPING - let values be what they are
                # Only handle NaN/Inf for numerical stability
                node_values = values[idx, :, t]
                node_values = np.nan_to_num(node_values, nan=0.0, posinf=1e10, neginf=-1e10)
                values[idx, :, t] = node_values
        
        return values
    
    def _array_to_propagated_values(self, arr: np.ndarray) -> TemporalPropagatedValues:
        """Convert numpy array to TemporalPropagatedValues."""
        T = arr.shape[2]
        n_samples = arr.shape[1]
        values_dict = {}
        for idx, node_id in self.idx_to_node.items():
            for t in range(T):
                values_dict[(t, node_id)] = arr[idx, :, t]
        return TemporalPropagatedValues(values=values_dict, T=T, n_samples=n_samples)
    
    def generate_iid_sequences(
        self, 
        n_sequences: int, 
        T: int
    ) -> TemporalPropagatedValues:
        """
        Generate independent sequences (IID mode).
        
        Each sequence gets its own MEMORY vector (sampled fresh).
        """
        arr = self._fast_propagate(n_samples=n_sequences, T=T, reset_memory=True)
        return self._array_to_propagated_values(arr)
    
    def generate_single_long_sequence(
        self, 
        n_samples: int,
        T: int
    ) -> TemporalPropagatedValues:
        """
        Generate a single long sequence for sliding window extraction.
        
        All samples share the same MEMORY (sampled once).
        """
        arr = self._fast_propagate(n_samples=n_samples, T=T, reset_memory=True)
        return self._array_to_propagated_values(arr)
    
    def generate_mixed_sequences(
        self, 
        n_long_sequences: int,
        samples_per_sequence: int,
        T: int
    ) -> List[TemporalPropagatedValues]:
        """
        Generate multiple long sequences (mixed mode).
        
        Each long sequence gets its own MEMORY.
        """
        sequences = []
        for _ in range(n_long_sequences):
            arr = self._fast_propagate(n_samples=samples_per_sequence, T=T, reset_memory=True)
            sequences.append(self._array_to_propagated_values(arr))
        return sequences
