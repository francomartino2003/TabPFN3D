"""
Temporal Propagator for 3D Synthetic Data.

Propagates values through the DAG for T timesteps.

NEW DESIGN (v2):
- State inputs look up specific nodes at t-k in history
- History is maintained as (t, node_id) -> values dict
- No direct noise inputs, only time and state
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


class TemporalPropagator:
    """
    Propagates values through DAG for T timesteps with temporal dependencies.
    
    NEW DESIGN: Uses history dict for state lookups at t-k.
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
        
        # Compute topological order once
        self.topo_order = self._compute_topological_order()
        
        # Identify root and non-root nodes
        self.root_nodes = [nid for nid, node in dag.nodes.items() if not node.parents]
        self.non_root_nodes = [nid for nid, node in dag.nodes.items() if node.parents]
        
        # Assign root nodes to input types
        # Note: TemporalPropagator doesn't have target info, uses uniform selection
        self.input_manager.assign_root_nodes(self.root_nodes, self.non_root_nodes, dag=dag)
    
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
        
        # History dict for state lookups
        history: Dict[Tuple[int, int], np.ndarray] = {}
        
        for t in range(T):
            # Generate inputs for this timestep (uses history for state lookups)
            root_inputs = self.input_manager.generate_inputs_for_timestep(
                t=t, T=T, n_samples=n_samples, history=history
            )
            
            # Propagate through DAG
            timestep_values = self._propagate_single_timestep(root_inputs, n_samples)
            
            # Store values in history with safety clipping
            # This prevents explosion accumulation across timesteps
            for node_id, values in timestep_values.items():
                # Clip to reasonable range and handle NaN/Inf
                clipped = np.clip(values, -100, 100)
                clipped = np.nan_to_num(clipped, nan=0.0, posinf=100, neginf=-100)
                history[(t, node_id)] = clipped
        
        return TemporalPropagatedValues(
            values=history,
            T=T,
            n_samples=n_samples
        )
    
    def _propagate_single_timestep(
        self, 
        root_inputs: Dict[int, np.ndarray],
        n_samples: int
    ) -> Dict[int, np.ndarray]:
        """Propagate values through DAG for a single timestep."""
        values: Dict[int, np.ndarray] = {}
        
        for node_id in self.topo_order:
            node = self.dag.nodes[node_id]
            
            if not node.parents:
                # Root node
                if node_id in root_inputs:
                    values[node_id] = root_inputs[node_id]
                else:
                    # Fallback: generate noise
                    values[node_id] = self.rng.normal(0, 1, size=n_samples)
            else:
                # Non-root node
                values[node_id] = self._compute_node_value(node_id, node, values)
        
        return values
    
    def _compute_node_value(
        self, 
        node_id: int, 
        node: DAGNode, 
        values: Dict[int, np.ndarray]
    ) -> np.ndarray:
        """Compute value for a non-root node from its parents."""
        n_samples = len(list(values.values())[0])
        
        # Collect parent values
        parent_values = []
        for parent_id in node.parents:
            if parent_id in values:
                parent_values.append(values[parent_id])
            else:
                parent_values.append(np.zeros(n_samples))
        
        if not parent_values:
            return np.zeros(n_samples)
        
        parent_array = np.column_stack(parent_values)
        
        # Apply transformation
        if node_id in self.transformations:
            transform = self.transformations[node_id]
            result = transform.forward(parent_array)
            
            if result.ndim > 1:
                result = result[:, 0] if result.shape[1] > 0 else result.flatten()
            
            # Add very small edge noise (if configured)
            if self.rng.random() < self.config.edge_noise_prob:
                result = result + self.rng.normal(0, self.config.noise_scale, size=n_samples)
            
            return result
        else:
            # Fallback: weighted sum
            weights = self.rng.normal(0, 1, size=len(parent_values))
            weights = weights / (np.linalg.norm(weights) + 1e-8)
            return sum(w * v for w, v in zip(weights, parent_values))
    
    def _compute_topological_order(self) -> List[int]:
        """Compute topological ordering of DAG nodes."""
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
        transformations: Dict[int, EdgeTransformation],
        input_manager: TemporalInputManager,
        rng: np.random.Generator,
        target_node: Optional[int] = None
    ):
        self.config = config
        self.dag = dag
        self.transformations = transformations
        self.input_manager = input_manager
        self.rng = rng
        self.target_node = target_node
        
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
        self.non_root_node_ids = [nid for nid, n in dag.nodes.items() if n.parents]
        
        # Assign root nodes (pass DAG and target for distance-based state source selection)
        self.input_manager.assign_root_nodes(
            list(self.root_node_ids), 
            self.non_root_node_ids,
            dag=dag,
            target_node=target_node
        )
    
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
    
    def _fast_propagate(self, n_samples: int, T: int, reset_state: bool = True) -> np.ndarray:
        """
        Fast propagation using pre-allocated arrays.
        
        Args:
            n_samples: Number of samples
            T: Number of timesteps
            reset_state: If True, reset state cache for new independent sequence
            
        Returns:
            Array of shape (n_nodes, n_samples, T)
        """
        if reset_state:
            self.input_manager.reset_for_new_sequence()
        
        # Pre-allocate output array
        values = np.zeros((self.n_nodes, n_samples, T), dtype=np.float32)
        
        # History dict for state lookups
        history: Dict[Tuple[int, int], np.ndarray] = {}
        
        for t in range(T):
            # Generate root inputs
            root_inputs = self.input_manager.generate_inputs_for_timestep(
                t=t, T=T, n_samples=n_samples, history=history
            )
            
            # Process each node in topological order
            for node_id in self.topo_order:
                idx = self.node_to_idx[node_id]
                
                if node_id in self.root_node_ids:
                    if node_id in root_inputs:
                        values[idx, :, t] = root_inputs[node_id]
                    else:
                        values[idx, :, t] = self.rng.normal(0, 1, size=n_samples)
                else:
                    # Non-root: compute from parents
                    parent_idxs = self.parent_indices[node_id]
                    parent_vals = values[parent_idxs, :, t].T  # (n_samples, n_parents)
                    
                    if node_id in self.transformations:
                        result = self.transformations[node_id].forward(parent_vals)
                        if result.ndim > 1:
                            result = result[:, 0] if result.shape[1] > 0 else result.flatten()
                        
                        # Small edge noise
                        if self.rng.random() < self.config.edge_noise_prob:
                            result = result + self.rng.normal(0, self.config.noise_scale, size=n_samples)
                        
                        values[idx, :, t] = result
                    else:
                        # Fallback
                        weights = self.rng.normal(0, 1, size=len(parent_idxs))
                        weights = weights / (np.linalg.norm(weights) + 1e-8)
                        values[idx, :, t] = parent_vals @ weights
                
                # Safety clip to prevent explosion across timesteps
                # Clip values before storing in array and history
                node_values = values[idx, :, t]
                node_values = np.clip(node_values, -100, 100)
                node_values = np.nan_to_num(node_values, nan=0.0, posinf=100, neginf=-100)
                values[idx, :, t] = node_values
                
                # Store in history for state lookups
                history[(t, node_id)] = node_values.copy()
        
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
        """Generate independent sequences (IID mode)."""
        arr = self._fast_propagate(n_samples=n_sequences, T=T, reset_state=True)
        return self._array_to_propagated_values(arr)
    
    def generate_single_long_sequence(
        self, 
        n_samples: int,
        T: int
    ) -> TemporalPropagatedValues:
        """Generate a single long sequence for sliding window extraction."""
        arr = self._fast_propagate(n_samples=n_samples, T=T, reset_state=True)
        return self._array_to_propagated_values(arr)
    
    def generate_mixed_sequences(
        self, 
        n_long_sequences: int,
        samples_per_sequence: int,
        T: int
    ) -> List[TemporalPropagatedValues]:
        """Generate multiple long sequences (mixed mode)."""
        sequences = []
        for _ in range(n_long_sequences):
            arr = self._fast_propagate(n_samples=samples_per_sequence, T=T, reset_state=True)
            sequences.append(self._array_to_propagated_values(arr))
        return sequences
