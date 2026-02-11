"""
Feature and Target Selector for 3D Synthetic Data.

v4 Design:
- Target: MUST be from a node with DiscretizationTransformation (classification)
- Features: 
  - At least 1 RELEVANT (from main connected graph)
  - At least 1 CONTINUOUS (not from discretization)
- Distance-weighted selection for remaining features
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
import numpy as np

from config import DatasetConfig3D
from sequence_sampler import FeatureTargetSelection
from temporal_inputs import TemporalInputManager

from dag_utils import DAG, EdgeTransformation, DiscretizationTransformation


class FeatureSelector3D:
    """
    Selects features and target for 3D temporal datasets.
    
    v4 Design:
    - Target: from discretization node (categorical = classification)
    - Features: at least 1 relevant + 1 continuous
    """
    
    def __init__(
        self,
        dag: DAG,
        transformations: Dict[int, EdgeTransformation],
        config: DatasetConfig3D,
        input_manager: Optional[TemporalInputManager],
        rng: np.random.Generator
    ):
        self.dag = dag
        self.transformations = transformations
        self.config = config
        self.input_manager = input_manager
        self.rng = rng
        
        # Identify root nodes
        self.root_nodes = set(nid for nid, node in dag.nodes.items() if not node.parents)
        
        # Identify node types
        self._categorize_nodes()
        
        # Build adjacency for shortest path
        self._build_adjacency()
    
    def _categorize_nodes(self):
        """Categorize nodes by type and connectivity."""
        # Discretization nodes (categorical outputs)
        self.discretization_nodes = set()
        # Continuous nodes (NN or Tree outputs)
        self.continuous_nodes = set()
        
        for node_id, transform in self.transformations.items():
            if isinstance(transform, DiscretizationTransformation):
                self.discretization_nodes.add(node_id)
        else:
                self.continuous_nodes.add(node_id)
        
        # Main subgraph (subgraph_id == 0) nodes
        self.main_subgraph_nodes = set()
        # Disconnected subgraph nodes
        self.disconnected_nodes = set()
        
        for node_id, node in self.dag.nodes.items():
            if node_id in self.root_nodes:
                continue
            if node.subgraph_id == 0:
                self.main_subgraph_nodes.add(node_id)
            else:
                self.disconnected_nodes.add(node_id)
        
        # Valid nodes for features (non-root)
        self.valid_feature_nodes = (self.main_subgraph_nodes | self.disconnected_nodes)
        
        # Relevant continuous nodes (main subgraph + continuous)
        self.relevant_continuous_nodes = self.main_subgraph_nodes & self.continuous_nodes
        
        # Relevant discretization nodes (main subgraph + discretization)
        self.relevant_discretization_nodes = self.main_subgraph_nodes & self.discretization_nodes
    
    def _build_adjacency(self):
        """Build undirected adjacency list for shortest path computation."""
        self.adjacency = {node_id: set() for node_id in self.dag.nodes}
        for node_id, node in self.dag.nodes.items():
            for parent_id in node.parents:
                self.adjacency[node_id].add(parent_id)
                self.adjacency[parent_id].add(node_id)
    
    def _shortest_path_length(self, start: int, end: int) -> int:
        """Compute shortest path length between two nodes (BFS, undirected)."""
        if start == end:
            return 0
        
        visited = {start}
        queue = [(start, 0)]
        
        while queue:
            current, dist = queue.pop(0)
            
            for neighbor in self.adjacency.get(current, []):
                if neighbor == end:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        
        return -1  # No path found (disconnected)
    
    def _compute_distance_probs(self, candidates: List[int], target_node: int) -> np.ndarray:
        """Compute selection probabilities based on distance to target."""
        alpha = self.config.spatial_distance_alpha
        
        distances = []
        for node in candidates:
            dist = self._shortest_path_length(node, target_node)
            if dist < 0:
                dist = len(self.dag.nodes)
            distances.append(dist)
        
        distances = np.array(distances, dtype=float)
        probs = 1.0 / (1.0 + np.power(distances, alpha))
        probs = probs / probs.sum()
        
        return probs
    
    def select_target(self) -> int:
        """
        Select target node.
        
        v4: Target MUST be from a discretization node (for classification).
        Prefers nodes in the main subgraph.
        """
        # First try: discretization nodes in main subgraph
        if self.relevant_discretization_nodes:
            candidates = list(self.relevant_discretization_nodes)
            return int(self.rng.choice(candidates))
            
        # Fallback: any discretization node
        if self.discretization_nodes:
            candidates = list(self.discretization_nodes)
            return int(self.rng.choice(candidates))
        
        # No discretization nodes exist - this is a problem
        # We need to ensure at least one exists during DAG construction
        raise ValueError("No discretization nodes available for target selection. "
                        "Ensure prob_discretization > 0 in config.")
    
    def select_features(self, target_node: int) -> List[int]:
        """
        Select feature nodes.
        
        v4 Constraints:
        - At least 1 RELEVANT feature (from main subgraph)
        - At least 1 CONTINUOUS feature (not discretization)
        - If target_offset == 0: exclude target from features
        """
        n_features = self.config.n_features
        
        # Determine which nodes can be features
        if self.config.target_offset == 0:
            available = self.valid_feature_nodes - {target_node}
        else:
            available = self.valid_feature_nodes.copy()
        
        if len(available) == 0:
            return []
        
        selected = set()
        
        # Constraint 1: At least 1 RELEVANT feature (main subgraph)
        relevant_available = available & self.main_subgraph_nodes
        if relevant_available:
            # Prefer continuous if available
            relevant_continuous = relevant_available & self.continuous_nodes
            if relevant_continuous:
                first_relevant = int(self.rng.choice(list(relevant_continuous)))
            else:
                first_relevant = int(self.rng.choice(list(relevant_available)))
            selected.add(first_relevant)
        
        # Constraint 2: At least 1 CONTINUOUS feature (not discretization)
        continuous_available = available & self.continuous_nodes
        continuous_not_selected = continuous_available - selected
        if continuous_not_selected and len(selected) < n_features:
            # Prefer relevant continuous
            relevant_continuous = continuous_not_selected & self.main_subgraph_nodes
            if relevant_continuous:
                first_continuous = int(self.rng.choice(list(relevant_continuous)))
            else:
                first_continuous = int(self.rng.choice(list(continuous_not_selected)))
            selected.add(first_continuous)
        
        # Fill remaining slots with distance-weighted selection
        remaining = available - selected
        n_remaining_needed = min(n_features - len(selected), len(remaining))
        
        if n_remaining_needed > 0 and remaining:
            remaining_list = list(remaining)
            probs = self._compute_distance_probs(remaining_list, target_node)
            additional = self.rng.choice(
                remaining_list, 
                size=n_remaining_needed, 
                replace=False, 
                p=probs
            )
            selected.update(int(x) for x in additional)
        
        # Convert to list and shuffle
        feature_list = list(selected)
        self.rng.shuffle(feature_list)
        
        return feature_list
    
    def select(self) -> FeatureTargetSelection:
        """
        Select features and target.
            
        Returns:
            FeatureTargetSelection with feature_nodes, target_node, etc.
        """
        target_node = self.select_target()
        feature_nodes = self.select_features(target_node)
        
        return FeatureTargetSelection(
            feature_nodes=feature_nodes,
            target_node=target_node,
            target_offset=self.config.target_offset,
            is_classification=True,  # Always classification in v4
            n_classes=self.config.n_classes
        )


def determine_node_types(
    dag: DAG,
    transformations: Dict[int, EdgeTransformation]
) -> Dict[int, str]:
    """
    Determine whether each node produces continuous or categorical values.
    """
    node_types = {}
    
    categorical_nodes = set()
    for node_id, transform in transformations.items():
        if isinstance(transform, DiscretizationTransformation):
            categorical_nodes.add(node_id)
    
    for node_id in dag.nodes:
        if node_id in categorical_nodes:
            node_types[node_id] = 'categorical'
        else:
            node_types[node_id] = 'continuous'
    
    return node_types
