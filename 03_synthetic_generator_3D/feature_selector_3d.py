"""
Feature and Target Selection for 3D Time Series Datasets.

This module handles selecting:
- Which nodes become observed features (same nodes across all timesteps)
- The temporal window for extracting feature time series
- The target node and timestep

Key decisions:
- Feature nodes: Selected from base graph, applied to all timesteps
- Feature window: Contiguous subseries of the full time range
- Target: One node at one timestep (converted to classification)
"""

from dataclasses import dataclass
from typing import List, Set, Optional, Dict, Tuple
import numpy as np

from config import DatasetConfig3D
from temporal_dag_builder import TemporalDAG
from row_generator_3d import TemporalPropagatedValues


@dataclass
class TemporalFeatureSelection:
    """
    Result of feature and target selection for 3D datasets.
    
    Attributes:
        feature_base_nodes: Base node IDs selected as features
        target_base_node: Base node ID for target
        feature_window_start: First timestep of feature window
        feature_window_end: Last timestep of feature window (exclusive)
        target_timestep: Timestep for target extraction
        target_position: 'before', 'within', or 'after' feature window
        n_features: Number of features
        n_timesteps_output: Length of output time series
    """
    feature_base_nodes: List[int]
    target_base_node: int
    feature_window_start: int
    feature_window_end: int
    target_timestep: int
    target_position: str
    n_features: int
    n_timesteps_output: int
    
    # Feature relevance
    relevant_features: Set[int]
    irrelevant_features: Set[int]


class FeatureSelector3D:
    """
    Selects features and target for 3D time series datasets.
    
    Strategy:
    1. Select target node from main subgraph (has many temporal influences)
    2. Select feature nodes from ancestors and some irrelevant nodes
    3. Use configured feature window and target position
    """
    
    def __init__(
        self,
        config: DatasetConfig3D,
        dag: TemporalDAG,
        rng: Optional[np.random.Generator] = None
    ):
        self.config = config
        self.dag = dag
        self.rng = rng if rng is not None else np.random.default_rng(config.seed)
    
    def select(self) -> TemporalFeatureSelection:
        """Select features and target."""
        
        # Get the number of features to select
        n_features = min(self.config.n_features, self.dag.n_base_nodes - 1)
        
        # Select target node
        target_base_node = self._select_target_node()
        
        # Select feature nodes
        feature_nodes, relevant, irrelevant = self._select_feature_nodes(
            n_features, target_base_node
        )
        
        # Use configured window and target position
        feature_window_start = self.config.feature_window_start
        feature_window_end = self.config.feature_window_end
        target_timestep = self.config.target_timestep
        target_position = self.config.target_position
        
        # Calculate output time series length
        n_timesteps_output = feature_window_end - feature_window_start
        
        return TemporalFeatureSelection(
            feature_base_nodes=feature_nodes,
            target_base_node=target_base_node,
            feature_window_start=feature_window_start,
            feature_window_end=feature_window_end,
            target_timestep=target_timestep,
            target_position=target_position,
            n_features=len(feature_nodes),
            n_timesteps_output=n_timesteps_output,
            relevant_features=relevant,
            irrelevant_features=irrelevant
        )
    
    def _select_target_node(self) -> int:
        """
        Select target node from the base graph.
        
        Prefers nodes with many ancestors (capture complex dependencies).
        """
        base_dag = self.dag.base_dag
        
        # Get nodes from main subgraph
        main_subgraph_nodes = base_dag.get_subgraph_nodes(0)
        
        if not main_subgraph_nodes:
            main_subgraph_nodes = list(base_dag.nodes.keys())
        
        # Score by number of ancestors
        scored_nodes = []
        for node_id in main_subgraph_nodes:
            n_ancestors = len(base_dag.get_ancestors(node_id))
            # Also consider temporal connectivity
            temporal_ancestors = 0
            for conn in self.config.temporal_connections:
                if node_id in conn.target_nodes:
                    temporal_ancestors += len(conn.source_nodes)
            
            score = n_ancestors + temporal_ancestors * 0.5
            scored_nodes.append((node_id, score))
        
        # Sort by score descending
        scored_nodes.sort(key=lambda x: -x[1])
        
        # Sample from top candidates
        n_candidates = min(5, len(scored_nodes))
        top_candidates = [n[0] for n in scored_nodes[:n_candidates]]
        weights = np.array([scored_nodes[i][1] + 1 for i in range(n_candidates)])
        weights = weights / weights.sum()
        
        return int(self.rng.choice(top_candidates, p=weights))
    
    def _select_feature_nodes(
        self, 
        n_features: int,
        target_node: int
    ) -> Tuple[List[int], Set[int], Set[int]]:
        """
        Select feature nodes from the base graph.
        """
        base_dag = self.dag.base_dag
        
        # Find ancestors of target (relevant nodes)
        relevant_candidates = list(base_dag.get_ancestors(target_node))
        
        # Find nodes from disconnected subgraphs (irrelevant)
        target_subgraph = base_dag.nodes[target_node].subgraph_id
        irrelevant_candidates = [
            n for n in base_dag.nodes.keys()
            if base_dag.nodes[n].subgraph_id != target_subgraph
        ]
        
        selected = []
        relevant_selected = set()
        irrelevant_selected = set()
        
        # Exclude target from selection
        relevant_candidates = [n for n in relevant_candidates if n != target_node]
        irrelevant_candidates = [n for n in irrelevant_candidates if n != target_node]
        
        # Determine split
        n_irrelevant = min(
            len(irrelevant_candidates),
            int(n_features * self.rng.uniform(0, 0.3))  # Up to 30% irrelevant
        )
        n_relevant = n_features - n_irrelevant
        
        # Select relevant
        if relevant_candidates and n_relevant > 0:
            n_to_select = min(n_relevant, len(relevant_candidates))
            chosen = self.rng.choice(relevant_candidates, size=n_to_select, replace=False)
            selected.extend(chosen)
            relevant_selected.update(chosen)
        
        # Select irrelevant
        if irrelevant_candidates and n_irrelevant > 0:
            n_to_select = min(n_irrelevant, len(irrelevant_candidates))
            chosen = self.rng.choice(irrelevant_candidates, size=n_to_select, replace=False)
            selected.extend(chosen)
            irrelevant_selected.update(chosen)
        
        # Fill remaining from any available nodes
        remaining = n_features - len(selected)
        if remaining > 0:
            available = [
                n for n in base_dag.nodes.keys()
                if n != target_node and n not in selected
            ]
            if available:
                n_to_select = min(remaining, len(available))
                additional = self.rng.choice(available, size=n_to_select, replace=False)
                for node in additional:
                    selected.append(node)
                    if base_dag.nodes[node].subgraph_id == target_subgraph:
                        relevant_selected.add(node)
                    else:
                        irrelevant_selected.add(node)
        
        # Shuffle
        self.rng.shuffle(selected)
        
        return list(selected), relevant_selected, irrelevant_selected


class TableBuilder3D:
    """
    Builds the final 3D dataset from propagated values and feature selection.
    
    Output:
    - X: (n_samples, n_features, n_timesteps) time series features
    - y: (n_samples,) classification labels
    """
    
    def __init__(
        self,
        selection: TemporalFeatureSelection,
        config: DatasetConfig3D,
        rng: Optional[np.random.Generator] = None
    ):
        self.selection = selection
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng(config.seed)
    
    def build(self, propagated: TemporalPropagatedValues) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build feature tensor X and target vector y.
        
        Args:
            propagated: Values from temporal DAG propagation
            
        Returns:
            Tuple of (X, y) where:
            - X has shape (n_samples, n_features, n_timesteps_output)
            - y has shape (n_samples,)
        """
        n_samples = propagated.n_samples
        n_features = len(self.selection.feature_base_nodes)
        t_start = self.selection.feature_window_start
        t_end = self.selection.feature_window_end
        n_timesteps = t_end - t_start
        
        # Build X: (n_samples, n_features, n_timesteps)
        X = np.zeros((n_samples, n_features, n_timesteps))
        
        for i, base_node_id in enumerate(self.selection.feature_base_nodes):
            # Get time series for this feature
            series = propagated.get_time_series(base_node_id, t_start, t_end)
            X[:, i, :] = series
        
        # Build y from target node at target timestep
        target_values = propagated.get_node_value(
            self.selection.target_timestep,
            self.selection.target_base_node
        )
        
        # Discretize target into classes
        y = self._discretize_target(target_values)
        
        return X, y
    
    def _discretize_target(self, values: np.ndarray) -> np.ndarray:
        """Convert continuous target to classification labels."""
        n_classes = self.config.n_classes
        
        # Quantile-based discretization
        quantiles = np.linspace(0, 100, n_classes + 1)
        thresholds = np.percentile(values, quantiles[1:-1])
        
        y = np.digitize(values, thresholds)
        
        return y
    
    def get_feature_names(self) -> List[str]:
        """Generate feature names."""
        names = []
        for i, node_id in enumerate(self.selection.feature_base_nodes):
            relevance = 'rel' if node_id in self.selection.relevant_features else 'irrel'
            names.append(f"feat_{i}_node{node_id}_{relevance}")
        return names
    
    def get_metadata(self) -> Dict:
        """Get metadata about the dataset."""
        return {
            'n_features': self.selection.n_features,
            'n_timesteps': self.selection.n_timesteps_output,
            'n_relevant': len(self.selection.relevant_features),
            'n_irrelevant': len(self.selection.irrelevant_features),
            'target_base_node': self.selection.target_base_node,
            'target_timestep': self.selection.target_timestep,
            'target_position': self.selection.target_position,
            'feature_window': (self.selection.feature_window_start, self.selection.feature_window_end),
            'feature_base_nodes': self.selection.feature_base_nodes
        }

