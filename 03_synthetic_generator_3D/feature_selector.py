"""
Feature and Target Selector for 3D Synthetic Data.

Selects which nodes become features and which becomes the target.
Adapted from 2D version with temporal considerations.

Key differences from 2D:
- Target can be at different time offsets (future prediction, within-sequence, etc.)
- Must exclude state nodes from being targets (they're memory, not observations)
- Feature nodes are observed across the subsequence window
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
import numpy as np

# Local 3D modules
from config import DatasetConfig3D
from sequence_sampler import FeatureTargetSelection
from temporal_inputs import TemporalInputManager

# 2D components via wrapper
from dag_utils import DAG, EdgeTransformation, DiscretizationTransformation


class FeatureSelector3D:
    """
    Selects features and target for 3D temporal datasets.
    
    Strategy:
    1. Exclude root nodes (inputs) and state nodes from being features/target
    2. Select target node that has sufficient ancestors (not trivial)
    3. Select feature nodes that are ancestors of target (relevant) 
       or from disconnected subgraphs (irrelevant)
    4. Ensure no data leakage (target's direct parents excluded)
    """
    
    def __init__(
        self,
        dag: DAG,
        transformations: Dict[Tuple[int, int], EdgeTransformation],
        config: DatasetConfig3D,
        input_manager: TemporalInputManager,
        rng: np.random.Generator
    ):
        """
        Initialize the feature selector.
        
        Args:
            dag: The causal DAG
            transformations: Edge transformations
            config: Dataset configuration
            input_manager: Temporal input manager (to know which nodes are inputs)
            rng: Random number generator
        """
        self.dag = dag
        self.transformations = transformations
        self.config = config
        self.input_manager = input_manager
        self.rng = rng
        
        # Get excluded nodes (root inputs)
        self.excluded_nodes = set(input_manager.get_all_root_node_ids())
        
        # Identify nodes that depend ONLY on time inputs (deterministic)
        # These lack sample-to-sample variability and can't be features/targets
        self.time_only_nodes = self._identify_time_only_nodes()
        self.excluded_nodes.update(self.time_only_nodes)
        
        # Identify equivalent nodes (e.g., before/after discretization)
        self.equivalence_groups = self._identify_equivalent_nodes()
    
    def select(self) -> FeatureTargetSelection:
        """
        Select features and target.
        
        Returns:
            FeatureTargetSelection with all selection info
        """
        n_features = min(self.config.n_features, len(self.dag.nodes) - len(self.excluded_nodes) - 1)
        n_features = max(1, n_features)
        
        # Select target
        target_node = self._select_target()
        
        # Only exclude target itself from features
        # Parents of target are valid features (they help predict target!)
        feature_excluded = self.excluded_nodes | {target_node}
        
        # Get candidate nodes
        relevant_candidates = self._get_relevant_candidates(target_node, feature_excluded)
        irrelevant_candidates = self._get_irrelevant_candidates(target_node, feature_excluded)
        
        # Select features
        feature_nodes = self._select_features(
            n_features, relevant_candidates, irrelevant_candidates, feature_excluded
        )
        
        # Determine number of classes
        if self.config.is_classification:
            n_classes = self.config.n_classes
        else:
            n_classes = 0
        
        return FeatureTargetSelection(
            feature_nodes=feature_nodes,
            target_node=target_node,
            target_offset=self.config.target_offset,
            is_classification=self.config.is_classification,
            n_classes=n_classes
        )
    
    def _select_target(self) -> int:
        """
        Select the target node.
        
        Prefers nodes with many ancestors (complex dependencies).
        Excludes:
        - Input nodes (noise, time, state)
        - Time-only nodes (constant across samples)
        """
        candidates = []
        
        for node_id, node in self.dag.nodes.items():
            # Skip excluded nodes (includes time-only nodes)
            if node_id in self.excluded_nodes:
                continue
            
            # Skip nodes with no parents (roots)
            if not node.parents:
                continue
            
            # Count ancestors
            n_ancestors = len(self._get_ancestors(node_id))
            
            # Prefer nodes with more ancestors
            if n_ancestors >= 2:
                candidates.append((node_id, n_ancestors))
        
        if not candidates:
            # Fallback: any non-excluded node with parents
            for node_id, node in self.dag.nodes.items():
                if node_id not in self.excluded_nodes and node.parents:
                    candidates.append((node_id, 1))
        
        if not candidates:
            # Last resort: first non-excluded node
            for node_id in self.dag.nodes:
                if node_id not in self.excluded_nodes:
                    return node_id
            # Really last resort
            return list(self.dag.nodes.keys())[0]
        
        # Weight by number of ancestors
        candidates.sort(key=lambda x: -x[1])
        top_k = min(5, len(candidates))
        top_candidates = [c[0] for c in candidates[:top_k]]
        
        return self.rng.choice(top_candidates)
    
    def _get_ancestors(self, node_id: int) -> Set[int]:
        """Get all ancestors of a node."""
        ancestors = set()
        queue = list(self.dag.nodes[node_id].parents)
        
        while queue:
            current = queue.pop(0)
            if current not in ancestors:
                ancestors.add(current)
                queue.extend(self.dag.nodes[current].parents)
        
        return ancestors
    
    def _identify_time_only_nodes(self) -> Set[int]:
        """
        Identify nodes that depend ONLY on time inputs.
        
        These nodes are deterministic (same value for all samples at each timestep)
        and should NEVER be selected as features or targets.
        
        Nodes that depend on noise OR state inputs are OK because:
        - Noise inputs: fresh random values each timestep, vary between samples
        - State inputs: initialized with noise at t=0, so have sample variability
        
        Only nodes with EXCLUSIVELY time input ancestors are excluded.
        """
        time_input_ids = set(self.input_manager.time_node_ids or [])
        noise_input_ids = set(self.input_manager.noise_node_ids or [])
        state_input_ids = set(self.input_manager.state_node_ids or [])
        all_root_ids = time_input_ids | noise_input_ids | state_input_ids
        
        # Inputs that provide sample variability
        variable_inputs = noise_input_ids | state_input_ids
        
        time_only_nodes = set()
        
        for node_id in self.dag.nodes:
            # Skip root nodes themselves
            if node_id in all_root_ids:
                continue
            
            # Get all ancestors
            ancestors = self._get_ancestors(node_id)
            
            # Get which root types are in ancestry
            root_ancestors = ancestors & all_root_ids
            
            if not root_ancestors:
                # No root ancestors (shouldn't happen in valid DAG)
                continue
            
            # Check if there's at least one variable input (noise or state) in ancestry
            has_variable_ancestor = bool(root_ancestors & variable_inputs)
            
            if not has_variable_ancestor:
                # Only time inputs in ancestry - deterministic node
                time_only_nodes.add(node_id)
        
        return time_only_nodes
    
    def _identify_equivalent_nodes(self) -> List[Set[int]]:
        """
        Identify groups of equivalent nodes.
        
        Two nodes are equivalent if:
        1. One is the input to a discretization transformation and the other is output
           (they carry the same information, just different representations)
        2. One is a copy of another (identity transformation with same input)
        
        Returns:
            List of sets, each set contains equivalent node IDs
        """
        equivalence = []
        seen = set()
        
        # transformations is now keyed by child_id, not (parent_id, child_id)
        for child_id, transform in self.transformations.items():
            # Discretization: child node is a categorical version of its parents
            if isinstance(transform, DiscretizationTransformation):
                node = self.dag.nodes[child_id]
                for parent_id in node.parents:
                    if parent_id not in seen and child_id not in seen:
                        equivalence.append({parent_id, child_id})
                        seen.add(parent_id)
                        seen.add(child_id)
                    elif parent_id in seen:
                        # Add child to existing group
                        for group in equivalence:
                            if parent_id in group:
                                group.add(child_id)
                                seen.add(child_id)
                                break
                    elif child_id in seen:
                        for group in equivalence:
                            if child_id in group:
                                group.add(parent_id)
                                seen.add(parent_id)
                                break
        
        return equivalence
    
    def _get_equivalent_representative(self, selected: List[int]) -> List[int]:
        """
        Given selected nodes, remove duplicates from equivalence groups.
        
        Keeps only one representative from each equivalence group.
        """
        result = []
        groups_used = set()
        
        for node_id in selected:
            # Check if this node is in an equivalence group
            group_idx = None
            for i, group in enumerate(self.equivalence_groups):
                if node_id in group:
                    group_idx = i
                    break
            
            if group_idx is None:
                # Not in any equivalence group, keep it
                result.append(node_id)
            elif group_idx not in groups_used:
                # First node from this group, keep it
                result.append(node_id)
                groups_used.add(group_idx)
            # else: skip, already have a node from this group
        
        return result
    
    def _get_relevant_candidates(
        self, 
        target_node: int, 
        excluded: Set[int]
    ) -> List[int]:
        """
        Get nodes that are relevant for predicting the target.
        
        These are ancestors of the target (including direct parents).
        """
        ancestors = self._get_ancestors(target_node)
        
        # Include all ancestors as relevant (parents are great features!)
        relevant = ancestors - excluded
        return list(relevant)
    
    def _get_irrelevant_candidates(
        self, 
        target_node: int,
        excluded: Set[int]
    ) -> List[int]:
        """
        Get nodes that are irrelevant for predicting the target.
        
        These are nodes from disconnected subgraphs.
        """
        target_subgraph = self.dag.nodes[target_node].subgraph_id
        
        irrelevant = []
        for node_id, node in self.dag.nodes.items():
            if node_id in excluded:
                continue
            if node.subgraph_id != target_subgraph:
                irrelevant.append(node_id)
        
        return irrelevant
    
    def _select_features(
        self,
        n_features: int,
        relevant_candidates: List[int],
        irrelevant_candidates: List[int],
        excluded: Set[int]
    ) -> List[int]:
        """
        Select feature nodes from candidates.
        
        Args:
            n_features: Number of features to select
            relevant_candidates: Nodes that influence target
            irrelevant_candidates: Nodes from other subgraphs
            excluded: Nodes to never select
            
        Returns:
            List of selected feature node IDs
        """
        selected = []
        
        # Determine split between relevant and irrelevant
        n_irrelevant = min(
            len(irrelevant_candidates),
            int(n_features * self.rng.uniform(0, 0.3))  # Up to 30% irrelevant
        )
        n_relevant = n_features - n_irrelevant
        
        # Select relevant features
        if relevant_candidates and n_relevant > 0:
            n_select = min(n_relevant, len(relevant_candidates))
            selected_relevant = self.rng.choice(
                relevant_candidates, size=n_select, replace=False
            )
            selected.extend(selected_relevant)
        
        # Select irrelevant features
        if irrelevant_candidates and n_irrelevant > 0:
            n_select = min(n_irrelevant, len(irrelevant_candidates))
            selected_irrelevant = self.rng.choice(
                irrelevant_candidates, size=n_select, replace=False
            )
            selected.extend(selected_irrelevant)
        
        # If we still need more, select from remaining nodes
        remaining = n_features - len(selected)
        if remaining > 0:
            available = [
                n for n in self.dag.nodes.keys()
                if n not in excluded and n not in selected
            ]
            if available:
                n_select = min(remaining, len(available))
                additional = self.rng.choice(available, size=n_select, replace=False)
                selected.extend(additional)
        
        # Remove duplicates from equivalence groups
        selected = self._get_equivalent_representative(selected)
        
        # Shuffle
        self.rng.shuffle(selected)
        
        return list(selected)


def determine_node_types(
    dag: DAG,
    transformations: Dict[Tuple[int, int], EdgeTransformation]
) -> Dict[int, str]:
    """
    Determine whether each node produces continuous or categorical values.
    
    Args:
        dag: The DAG
        transformations: Edge transformations
        
    Returns:
        Dict mapping node_id -> 'continuous' or 'categorical'
    """
    node_types = {}
    
    # Find nodes with discretization transformations
    # transformations is now keyed by node_id, not (parent_id, child_id)
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

