"""
Feature and Target Selector for 3D Synthetic Data.

DESIGN (v3):
- Features: N nodos del DAG, sus valores en ventana temporal [t_start, t_end)
- Target: 1 nodo del DAG, su valor en un timestep específico

RESTRICCIONES:
- Si target_offset == 0: target_node ≠ feature_nodes (mismo t)
- Si target_offset != 0: target_node PUEDE ser feature (forecasting)

PREFERENCIA DE DISTANCIA ESPACIAL:
- Features se seleccionan con probabilidad inversamente proporcional
  a su distancia en el DAG respecto al target
- prob(node) ∝ 1 / (1 + distance^alpha)
- Esto favorece features cercanas al target sin forzar relación causal
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
    
    Strategy (v3):
    1. Target: random non-root node
    2. Features: N nodes selected with distance-weighted probabilities
       - prob(node) ∝ 1 / (1 + distance_to_target^alpha)
    3. If target_offset == 0: exclude target from features
    4. If target_offset != 0: target CAN be a feature (forecasting)
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
        
        # Pre-compute adjacency for shortest path calculation
        self._build_adjacency()
        
        # If input_manager not yet available, use all non-root nodes as valid
        # (will be refined when input_manager is set)
        if input_manager is None:
            # Initial mode: just identify roots by having no parents
            self.root_nodes = set(
                nid for nid, node in dag.nodes.items() if not node.parents
            )
            self.time_only_nodes = set()  # Can't determine without input_manager
            self.valid_nodes = [
                n for n in dag.nodes.keys() if n not in self.root_nodes
            ]
        else:
            self._setup_valid_nodes()
    
    def _setup_valid_nodes(self):
        """Setup valid nodes after input_manager is available."""
        # Nodos excluidos: raíces (time + state inputs)
        self.root_nodes = set(self.input_manager.get_all_root_node_ids())
        
        # También excluir nodos time-only (sin variabilidad entre muestras)
        self.time_only_nodes = self._identify_time_only_nodes()
        
        # Todos los nodos válidos (no raíz, no time-only)
        self.valid_nodes = [
            n for n in self.dag.nodes.keys() 
            if n not in self.root_nodes and n not in self.time_only_nodes
        ]
    
    def _build_adjacency(self):
        """Build undirected adjacency list for shortest path computation."""
        self.adjacency = {node_id: set() for node_id in self.dag.nodes}
        for node_id, node in self.dag.nodes.items():
            for parent_id in node.parents:
                self.adjacency[node_id].add(parent_id)
                self.adjacency[parent_id].add(node_id)
    
    def _shortest_path_length(self, start: int, end: int) -> int:
        """
        Compute shortest path length between two nodes (BFS).
        Treats edges as undirected (can go parent→child or child→parent).
        
        Returns:
            Path length, or -1 if no path exists
        """
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
        """
        Compute selection probabilities based on distance to target.
        
        prob(node) ∝ 1 / (1 + distance^alpha)
        
        Nodes closer to target have higher probability.
        Disconnected nodes get a small but non-zero probability.
        """
        alpha = self.config.spatial_distance_alpha
        
        distances = []
        for node in candidates:
            dist = self._shortest_path_length(node, target_node)
            if dist < 0:
                # Disconnected - assign large distance but not infinite
                dist = len(self.dag.nodes)  # Max possible distance
            distances.append(dist)
        
        distances = np.array(distances, dtype=float)
        
        # prob ∝ 1 / (1 + distance^alpha)
        probs = 1.0 / (1.0 + np.power(distances, alpha))
        
        # Normalize
        probs = probs / probs.sum()
        
        return probs
    
    def select_target_only(self) -> int:
        """
        Select only the target node (first step of two-step selection).
        
        This is called BEFORE state inputs are configured, so that
        state inputs can prefer nodes close to the target.
        
        Returns:
            target_node ID
        """
        if not self.valid_nodes:
            raise ValueError("No valid nodes available for target selection")
        
        # Seleccionar TARGET: cualquier nodo válido (uniform)
        target_node = int(self.rng.choice(self.valid_nodes))
        return target_node
    
    def select_with_target(self, target_node: int) -> FeatureTargetSelection:
        """
        Complete feature selection given an already-selected target.
        
        This is called AFTER state inputs are configured.
        
        Args:
            target_node: Already selected target node
            
        Returns:
            FeatureTargetSelection with feature_nodes, target_node, etc.
        """
        # Refresh valid nodes now that input_manager is set
        if self.input_manager is not None:
            self._setup_valid_nodes()
        
        # Determinar candidatos para features
        if self.config.target_offset == 0:
            # "within": target está en el mismo t que features
            # → target_node NO puede ser feature
            feature_candidates = [n for n in self.valid_nodes if n != target_node]
        else:
            # target_offset != 0: target PUEDE ser feature (forecasting)
            feature_candidates = self.valid_nodes.copy()
        
        # Seleccionar FEATURES con probabilidades basadas en distancia
        n_features = min(self.config.n_features, len(feature_candidates))
        n_features = max(1, n_features)
        
        if len(feature_candidates) >= n_features:
            probs = self._compute_distance_probs(feature_candidates, target_node)
            feature_nodes = list(self.rng.choice(
                feature_candidates, size=n_features, replace=False, p=probs
            ))
        else:
            feature_nodes = feature_candidates.copy()
        
        self.rng.shuffle(feature_nodes)
        
        return FeatureTargetSelection(
            feature_nodes=[int(n) for n in feature_nodes],
            target_node=target_node,
            target_offset=self.config.target_offset,
            is_classification=self.config.is_classification,
            n_classes=self.config.n_classes if self.config.is_classification else 0
        )
    
    def select(self) -> FeatureTargetSelection:
        """
        Select features and target with distance-weighted probabilities.
        (One-step selection - for backwards compatibility)
        
        Returns:
            FeatureTargetSelection with feature_nodes, target_node, etc.
        """
        target_node = self.select_target_only()
        return self.select_with_target(target_node)
    
    def _identify_time_only_nodes(self) -> Set[int]:
        """
        Identify nodes that depend ONLY on time inputs.
        
        These are deterministic (same value for all samples at each t)
        and useless as features or target.
        
        A node is time-only if ALL its root ancestors are TIME inputs
        (no STATE inputs in ancestry).
        """
        time_input_ids = set(self.input_manager.time_node_ids or [])
        state_input_ids = set(self.input_manager.state_node_ids or [])
        all_root_ids = time_input_ids | state_input_ids
        
        # State inputs provide variability (initialized with noise)
        variable_inputs = state_input_ids
        
        time_only_nodes = set()
        
        for node_id in self.dag.nodes:
            # Skip roots themselves
            if node_id in all_root_ids:
                continue
            
            # Get all ancestors
            ancestors = self._get_ancestors(node_id)
            root_ancestors = ancestors & all_root_ids
            
            if not root_ancestors:
                # No root ancestors (shouldn't happen in valid DAG)
                continue
            
            # Check if there's at least one STATE input in ancestry
            has_variable_ancestor = bool(root_ancestors & variable_inputs)
            
            if not has_variable_ancestor:
                # Only TIME inputs in ancestry → deterministic
                time_only_nodes.add(node_id)
        
        return time_only_nodes
    
    def _get_ancestors(self, node_id: int) -> Set[int]:
        """Get all ancestors of a node (BFS)."""
        ancestors = set()
        queue = list(self.dag.nodes[node_id].parents)
        
        while queue:
            current = queue.pop(0)
            if current not in ancestors:
                ancestors.add(current)
                queue.extend(self.dag.nodes[current].parents)
        
        return ancestors


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
