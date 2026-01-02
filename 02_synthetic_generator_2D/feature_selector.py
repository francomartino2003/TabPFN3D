"""
Feature and target selection for synthetic datasets.

This module handles selecting which nodes become:
- Observed features (columns in the final table)
- Target variable (what we're predicting)

Key insight from the paper:
- Not all nodes are observed
- Some features may be from disconnected subgraphs (irrelevant)
- The target is usually from the main connected component
"""

from dataclasses import dataclass
from typing import List, Set, Optional, Dict, Tuple
import numpy as np

try:
    from .config import DatasetConfig
    from .dag_builder import DAG
    from .row_generator import PropagatedValues
    from .transformations import DiscretizationTransformation, EdgeTransformation
except ImportError:
    from config import DatasetConfig
    from dag_builder import DAG
    from row_generator import PropagatedValues
    from transformations import DiscretizationTransformation, EdgeTransformation


@dataclass
class FeatureSelection:
    """
    Result of feature/target selection.
    
    Attributes:
        feature_nodes: List of node IDs selected as features
        target_node: Node ID selected as target
        feature_types: Dict mapping node ID to 'continuous' or 'categorical'
        is_classification: Whether this is a classification task
        n_classes: Number of classes (for classification)
        relevant_features: Set of feature node IDs that influence the target
        irrelevant_features: Set of feature node IDs that don't influence target
    """
    feature_nodes: List[int]
    target_node: int
    feature_types: Dict[int, str]
    is_classification: bool
    n_classes: int
    relevant_features: Set[int]
    irrelevant_features: Set[int]
    
    def get_feature_mask(self, node_id: int) -> bool:
        """Check if a node is a selected feature."""
        return node_id in self.feature_nodes
    
    def get_relevant_mask(self) -> np.ndarray:
        """Get boolean mask of which features are relevant."""
        return np.array([nid in self.relevant_features for nid in self.feature_nodes])


class FeatureSelector:
    """
    Selects which nodes become features and which becomes the target.
    
    Strategy:
    1. Select target from the main subgraph (subgraph 0)
    2. Select some features from nodes that influence target (relevant)
    3. Select some features from disconnected subgraphs (irrelevant)
    4. Ensure we have the requested number of features
    """
    
    def __init__(
        self, 
        config: DatasetConfig, 
        dag: DAG,
        transformations: Dict[Tuple[int, int], EdgeTransformation],
        rng: Optional[np.random.Generator] = None
    ):
        """
        Initialize the feature selector.
        
        Args:
            config: Dataset configuration
            dag: The causal DAG
            transformations: Edge transformations (to detect categorical nodes)
            rng: Random number generator
        """
        self.config = config
        self.dag = dag
        self.transformations = transformations
        self.rng = rng if rng is not None else np.random.default_rng(config.seed)
    
    def select(self) -> FeatureSelection:
        """
        Select features and target.
        
        Returns:
            FeatureSelection with all selection information
        """
        n_features = min(self.config.n_features, len(self.dag.nodes) - 1)
        
        # Select target node
        target_node = self._select_target()
        
        # Find nodes that influence the target (ancestors + connected via paths)
        relevant_candidates = self._get_relevant_candidates(target_node)
        
        # Find nodes from disconnected subgraphs (irrelevant)
        irrelevant_candidates = self._get_irrelevant_candidates(target_node)
        
        # Select features
        feature_nodes, relevant_set, irrelevant_set = self._select_features(
            n_features, target_node, relevant_candidates, irrelevant_candidates
        )
        
        # Determine feature types (continuous vs categorical)
        feature_types = self._determine_feature_types(feature_nodes)
        
        # Determine if classification and number of classes
        is_classification = self.config.is_classification
        
        # For classification, get n_classes from the actual categorical node
        if is_classification:
            n_classes = self._get_n_classes_for_target(target_node)
        else:
            n_classes = 0
        
        return FeatureSelection(
            feature_nodes=feature_nodes,
            target_node=target_node,
            feature_types=feature_types,
            is_classification=is_classification,
            n_classes=n_classes,
            relevant_features=relevant_set,
            irrelevant_features=irrelevant_set
        )
    
    def _select_target(self) -> int:
        """
        Select the target node.
        
        Per paper:
        - "For classification labels, we select a random categorical feature 
           that contains up to 10 classes"
        - "To generate target labels for regression tasks, we select a randomly 
           chosen continuous feature without post-processing"
        
        Target should be from the main subgraph (subgraph 0).
        """
        # Get nodes from main subgraph
        main_subgraph_nodes = self.dag.get_subgraph_nodes(0)
        
        if not main_subgraph_nodes:
            # Fallback: use any node
            main_subgraph_nodes = list(self.dag.nodes.keys())
        
        # Find categorical nodes (nodes with discretization transformation)
        categorical_nodes = set()
        for (parent_id, child_id), transform in self.transformations.items():
            if isinstance(transform, DiscretizationTransformation):
                if child_id in main_subgraph_nodes:
                    categorical_nodes.add(child_id)
        
        # Find continuous nodes (nodes without discretization)
        continuous_nodes = [nid for nid in main_subgraph_nodes 
                           if nid not in categorical_nodes]
        
        if self.config.is_classification:
            # For classification: select a categorical node (per paper)
            # Prefer categorical nodes with many ancestors for complexity
            if categorical_nodes:
                candidates = list(categorical_nodes)
                # Weight by number of ancestors (more ancestors = more complex)
                weights = []
                for nid in candidates:
                    n_ancestors = len(self.dag.get_ancestors(nid))
                    weights.append(n_ancestors + 1)  # +1 to avoid zero weights
                weights = np.array(weights, dtype=float)
                weights = weights / weights.sum()
                return self.rng.choice(candidates, p=weights)
            else:
                # Fallback: will need to discretize a continuous node
                # This shouldn't happen often if we have discretization transforms
                pass
        
        # For regression OR fallback: prefer nodes with many ancestors
        candidates = continuous_nodes if continuous_nodes else main_subgraph_nodes
        
        nodes_with_ancestors = []
        for node_id in candidates:
            n_ancestors = len(self.dag.get_ancestors(node_id))
            nodes_with_ancestors.append((node_id, n_ancestors))
        
        # Sort by number of ancestors (descending)
        nodes_with_ancestors.sort(key=lambda x: -x[1])
        
        # Sample from top candidates (with some randomness)
        n_candidates = min(5, len(nodes_with_ancestors))
        top_candidates = [n[0] for n in nodes_with_ancestors[:n_candidates]]
        
        if not top_candidates:
            return main_subgraph_nodes[0]
        
        # Weight by ancestor count
        weights = np.array([nodes_with_ancestors[i][1] + 1 for i in range(n_candidates)])
        weights = weights / weights.sum()
        
        return self.rng.choice(top_candidates, p=weights)
    
    def _get_relevant_candidates(self, target_node: int) -> List[int]:
        """
        Get nodes that are relevant to predicting the target.
        
        These are ancestors of the target in the DAG.
        """
        ancestors = self.dag.get_ancestors(target_node)
        # Exclude target itself
        return list(ancestors - {target_node})
    
    def _get_irrelevant_candidates(self, target_node: int) -> List[int]:
        """
        Get nodes that are irrelevant for predicting the target.
        
        These are nodes from disconnected subgraphs.
        """
        target_subgraph = self.dag.nodes[target_node].subgraph_id
        
        irrelevant = []
        for node_id, node in self.dag.nodes.items():
            if node.subgraph_id != target_subgraph:
                irrelevant.append(node_id)
        
        return irrelevant
    
    def _select_features(
        self,
        n_features: int,
        target_node: int,
        relevant_candidates: List[int],
        irrelevant_candidates: List[int]
    ) -> Tuple[List[int], Set[int], Set[int]]:
        """
        Select feature nodes from candidates.
        
        Returns:
            Tuple of (feature_nodes, relevant_set, irrelevant_set)
        """
        selected_features = []
        relevant_selected = set()
        irrelevant_selected = set()
        
        # Exclude target AND its direct parents from selection
        # (direct parents can leak target info via transformations)
        target_parents = set(self.dag.nodes[target_node].parents)
        excluded_nodes = {target_node} | target_parents
        
        relevant_candidates = [n for n in relevant_candidates if n not in excluded_nodes]
        irrelevant_candidates = [n for n in irrelevant_candidates if n not in excluded_nodes]
        
        # Determine how many from each category
        n_irrelevant = min(
            len(irrelevant_candidates),
            int(n_features * self.rng.uniform(0, 0.4))  # Up to 40% irrelevant
        )
        n_relevant = min(len(relevant_candidates), n_features - n_irrelevant)
        
        # If we don't have enough relevant, fill with other nodes from same subgraph
        if n_relevant < n_features - n_irrelevant:
            target_subgraph = self.dag.nodes[target_node].subgraph_id
            same_subgraph = [
                n for n in self.dag.nodes.keys()
                if self.dag.nodes[n].subgraph_id == target_subgraph
                and n not in excluded_nodes
                and n not in relevant_candidates
            ]
            relevant_candidates = relevant_candidates + same_subgraph
            n_relevant = min(len(relevant_candidates), n_features - n_irrelevant)
        
        # Select relevant features
        if relevant_candidates and n_relevant > 0:
            selected_relevant = self.rng.choice(
                relevant_candidates, 
                size=min(n_relevant, len(relevant_candidates)),
                replace=False
            )
            selected_features.extend(selected_relevant)
            relevant_selected.update(selected_relevant)
        
        # Select irrelevant features
        if irrelevant_candidates and n_irrelevant > 0:
            selected_irrelevant = self.rng.choice(
                irrelevant_candidates,
                size=min(n_irrelevant, len(irrelevant_candidates)),
                replace=False
            )
            selected_features.extend(selected_irrelevant)
            irrelevant_selected.update(selected_irrelevant)
        
        # If we still need more features, sample from all remaining nodes
        remaining_needed = n_features - len(selected_features)
        if remaining_needed > 0:
            available = [
                n for n in self.dag.nodes.keys()
                if n not in excluded_nodes and n not in selected_features
            ]
            if available:
                additional = self.rng.choice(
                    available,
                    size=min(remaining_needed, len(available)),
                    replace=False
                )
                for node_id in additional:
                    selected_features.append(node_id)
                    # Classify as relevant or irrelevant
                    if self.dag.nodes[node_id].subgraph_id == self.dag.nodes[target_node].subgraph_id:
                        relevant_selected.add(node_id)
                    else:
                        irrelevant_selected.add(node_id)
        
        # Shuffle to mix relevant and irrelevant
        self.rng.shuffle(selected_features)
        
        return list(selected_features), relevant_selected, irrelevant_selected
    
    def _determine_feature_types(self, feature_nodes: List[int]) -> Dict[int, str]:
        """
        Determine if each feature is continuous or categorical.
        
        A feature is categorical if it went through a discretization transformation.
        """
        feature_types = {}
        
        # Check which nodes have discretization transformations
        nodes_with_discretization = set()
        for (parent_id, child_id), transform in self.transformations.items():
            if isinstance(transform, DiscretizationTransformation):
                nodes_with_discretization.add(child_id)
        
        for node_id in feature_nodes:
            if node_id in nodes_with_discretization:
                feature_types[node_id] = 'categorical'
            else:
                feature_types[node_id] = 'continuous'
        
        return feature_types
    
    def _get_n_classes_for_target(self, target_node: int) -> int:
        """
        Get the number of classes for a classification target.
        
        If the target is a categorical node (has discretization transform),
        return the number of categories. Otherwise, use config.n_classes.
        """
        # Find if target has a discretization transformation
        for (parent_id, child_id), transform in self.transformations.items():
            if child_id == target_node and isinstance(transform, DiscretizationTransformation):
                # Return the number of prototypes (= number of categories)
                return len(transform.prototypes)
        
        # Fallback to config (will need to discretize later)
        return self.config.n_classes


class TableBuilder:
    """
    Builds the final tabular dataset from propagated values and feature selection.
    """
    
    def __init__(
        self,
        selection: FeatureSelection,
        transformations: Dict[Tuple[int, int], EdgeTransformation],
        config: DatasetConfig,
        rng: Optional[np.random.Generator] = None
    ):
        """
        Initialize the table builder.
        
        Args:
            selection: Feature selection result
            transformations: Edge transformations
            config: Dataset configuration
            rng: Random number generator
        """
        self.selection = selection
        self.transformations = transformations
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng(config.seed)
    
    def build(self, propagated: PropagatedValues) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build feature matrix X and target vector y.
        
        Args:
            propagated: Propagated values from row generator
            
        Returns:
            Tuple of (X, y) where X is (n_samples, n_features) and y is (n_samples,)
        """
        n_samples = len(list(propagated.values.values())[0])
        n_features = len(self.selection.feature_nodes)
        
        # Build X
        X = np.zeros((n_samples, n_features))
        
        for i, node_id in enumerate(self.selection.feature_nodes):
            values = propagated.get_node_value(node_id)
            
            # For categorical features, get discrete indices
            if self.selection.feature_types[node_id] == 'categorical':
                # Find the discretization transformation
                for (parent_id, child_id), transform in self.transformations.items():
                    if child_id == node_id and isinstance(transform, DiscretizationTransformation):
                        # Get parent values
                        parent_node = self.selection.feature_nodes[0]  # Approximate
                        if parent_id in propagated.values:
                            parent_vals = propagated.values[parent_id].reshape(-1, 1)
                            values = transform.get_category_indices(parent_vals).astype(float)
                        break
            
            X[:, i] = values
        
        # Build y (target)
        target_node = self.selection.target_node
        
        # For classification, try to use categorical indices from discretization
        if self.selection.is_classification:
            y = self._get_classification_target(propagated, target_node)
        else:
            # For regression, use continuous values
            y = propagated.get_node_value(target_node)
        
        return X, y
    
    def _get_classification_target(
        self, 
        propagated: PropagatedValues, 
        target_node: int
    ) -> np.ndarray:
        """
        Get classification target labels.
        
        Per paper: "For classification labels, we select a random categorical 
        feature that contains up to 10 classes"
        
        If the target is a categorical node, use its category indices.
        Otherwise, fall back to quantile-based discretization.
        
        Always validates that classes are balanced enough for train/test split.
        """
        y = propagated.get_node_value(target_node)
        n_samples = len(y)
        min_samples_per_class = 2  # Minimum for stratified split
        
        # Check if target has a discretization transformation
        for (parent_id, child_id), transform in self.transformations.items():
            if child_id == target_node and isinstance(transform, DiscretizationTransformation):
                # Get parent values and compute category indices
                if parent_id in propagated.values:
                    parent_vals = propagated.values[parent_id].reshape(-1, 1)
                    y_cat = transform.get_category_indices(parent_vals).astype(float)
                    
                    # Verify all classes have enough samples
                    unique, counts = np.unique(y_cat, return_counts=True)
                    if len(unique) >= 2 and np.all(counts >= min_samples_per_class):
                        return y_cat
                    # Otherwise fall through to quantile-based
        
        # Fallback: quantile-based discretization (always balanced)
        return self._discretize_target(y)
    
    def _discretize_target(self, y: np.ndarray) -> np.ndarray:
        """
        Convert continuous target to discrete classes (fallback method).
        
        Uses quantile-based discretization for balanced classes.
        """
        n_classes = self.selection.n_classes
        n_samples = len(y)
        
        # Ensure we don't have more classes than we can support (min 2 samples per class for stratification)
        n_classes = min(n_classes, n_samples // 2)
        n_classes = max(2, n_classes)
        
        # Quantile-based discretization for balanced classes
        quantiles = np.linspace(0, 100, n_classes + 1)
        thresholds = np.percentile(y, quantiles[1:-1])
        
        # Remove duplicate thresholds (can happen with repeated values)
        thresholds = np.unique(thresholds)
        
        y_discrete = np.digitize(y, thresholds)
        
        # If we ended up with fewer classes than desired, that's OK
        # But ensure we have at least 2 classes
        unique_classes = np.unique(y_discrete)
        if len(unique_classes) < 2:
            # Fallback: just split in half
            median = np.median(y)
            y_discrete = (y > median).astype(int)
        
        return y_discrete
    
    def get_feature_names(self) -> List[str]:
        """Generate feature names."""
        names = []
        for i, node_id in enumerate(self.selection.feature_nodes):
            feat_type = self.selection.feature_types[node_id]
            relevance = 'rel' if node_id in self.selection.relevant_features else 'irrel'
            names.append(f"feat_{i}_{feat_type}_{relevance}")
        return names
    
    def get_metadata(self) -> Dict:
        """Get metadata about the table."""
        return {
            'n_features': len(self.selection.feature_nodes),
            'n_relevant': len(self.selection.relevant_features),
            'n_irrelevant': len(self.selection.irrelevant_features),
            'is_classification': self.selection.is_classification,
            'n_classes': self.selection.n_classes,
            'feature_types': {
                f"feat_{i}": self.selection.feature_types[nid]
                for i, nid in enumerate(self.selection.feature_nodes)
            }
        }

