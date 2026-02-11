"""
Wrapper for importing 2D generator components.

This module handles the import of DAG and transformation components from
02_synthetic_generator_2D without conflicting with the 3D config.

Also provides a modified DAGBuilder that creates multiple root nodes
(required for 3D temporal generator).

v4 CHANGES:
- Roots = 1 (base TIME) + n_extra_time_inputs + memory_dim (MEMORY)
- Extra TIME inputs have activations (sin, cos, tanh, etc.)
- NN 70%, Tree 15%, Discretization 15%
- Noise only at end of transformations
"""

import sys
import os
from typing import Dict, List, Tuple, Optional
import numpy as np

# Path to 2D generator
_2d_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                        '..', '02_synthetic_generator_2D')
_2d_path = os.path.abspath(_2d_path)

# Save original path and modules
_original_path = sys.path.copy()
_original_modules = {k: v for k, v in sys.modules.items() if 'config' in k.lower()}

# Remove any conflicting config modules
for key in list(sys.modules.keys()):
    if key == 'config' or key.startswith('config.'):
        del sys.modules[key]

# Set path to only include 2D directory for imports
sys.path = [_2d_path] + [p for p in sys.path if '03_synthetic_generator_3D' not in p]

try:
    # Import 2D modules
    from dag_builder import DAG, Node, DAGBuilder as DAGBuilder2D
    from transformations import (
        EdgeTransformation, 
        TransformationFactory as TransformationFactory2D, 
        DiscretizationTransformation,
        NNTransformation,
        TreeTransformation,
    )
    from post_processing import Warper, MissingValueInjector
    
    # Alias for compatibility
    DAGNode = Node
finally:
    # Restore original path
    sys.path = _original_path
    
    # Remove the 2D config from modules to avoid conflicts
    if 'config' in sys.modules:
        del sys.modules['config']

# Export DAGNode alias
DAGNode = Node


class TransformationFactory:
    """
    Transformation factory for 3D generator.
    
    v4: All NN with same activation per DAG
    Noise is applied at the end of each transformation.
    """
    
    def __init__(self, config, rng: Optional[np.random.Generator] = None, shared_activation: Optional[str] = None):
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()
        
        # Select ONE activation for ALL NN transformations in this DAG
        if shared_activation is None:
            prob_identity = getattr(config, 'prob_identity_activation', 0.5)
            if self.rng.random() < prob_identity and 'identity' in config.allowed_activations:
                self.shared_activation = 'identity'
            else:
                other_activations = [a for a in config.allowed_activations if a != 'identity']
                if other_activations:
                    self.shared_activation = self.rng.choice(other_activations)
                else:
                    self.shared_activation = 'identity'
        else:
            self.shared_activation = shared_activation
    
    def create(self, n_parents: int, force_discretization: bool = False) -> EdgeTransformation:
        """
        Create a random transformation.
        
        Args:
            n_parents: Number of parent nodes
            force_discretization: If True, always create discretization (for target node)
        """
        if force_discretization:
            return self._create_discretization(n_parents)
        
        # Get probabilities
        probs = self.config.transform_probs.copy()
        
        # Normalize
        types = list(probs.keys())
        type_probs = np.array([probs[t] for t in types])
        type_probs = type_probs / type_probs.sum()
        
        transform_type = self.rng.choice(types, p=type_probs)
        
        if transform_type == 'nn':
            return self._create_nn(n_parents)
        elif transform_type == 'tree':
            return self._create_tree(n_parents)
        else:  # discretization
            return self._create_discretization(n_parents)
    
    def _create_nn(self, n_parents: int) -> NNTransformation:
        """Create a neural network transformation with shared activation."""
        input_dim = max(1, n_parents)
        noise_scale = self.config.noise_scale
        
        # More conservative Xavier initialization (reduced from 2.0 to 1.0)
        xavier_std = np.sqrt(1.0 / (input_dim + 1))
        weights = self.rng.normal(0, xavier_std, size=(input_dim,))
        # Reduced bias range for more stability
        bias = self.rng.normal(0, 0.05)
        
        # Use the shared activation for all NN transformations in this DAG
        activation = self.shared_activation
        
        return NNTransformation(
            weights=weights,
            bias=bias,
            activation=activation,
            noise_scale=noise_scale,
            rng=self.rng
        )
    
    def _create_tree(self, n_parents: int) -> TreeTransformation:
        """Create a decision tree transformation."""
        depth = self.config.tree_depth
        n_internal = 2 ** depth - 1
        n_leaves = 2 ** depth
        total_nodes = n_internal + n_leaves
        noise_scale = self.config.noise_scale
        
        input_dim = max(1, n_parents)
        
        # Select subset of features
        max_features = max(1, int(input_dim * self.config.tree_max_features_fraction))
        n_features_to_use = self.rng.integers(1, max_features + 1)
        available_features = self.rng.choice(
            input_dim, 
            size=min(n_features_to_use, input_dim), 
            replace=False
        )
        
        # Initialize arrays
        feature_indices = np.zeros(total_nodes, dtype=int)
        thresholds = np.zeros(total_nodes)
        left_children = np.full(total_nodes, -1, dtype=int)
        right_children = np.full(total_nodes, -1, dtype=int)
        leaf_values = self.rng.normal(0, 1, size=(total_nodes,))
        
        # Build tree structure
        for node in range(n_internal):
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            
            if left_child < total_nodes:
                left_children[node] = left_child
            if right_child < total_nodes:
                right_children[node] = right_child
            
            feature_indices[node] = self.rng.choice(available_features)
            thresholds[node] = self.rng.normal(0, 1)
        
        return TreeTransformation(
            feature_indices=feature_indices,
            thresholds=thresholds,
            left_children=left_children,
            right_children=right_children,
            leaf_values=leaf_values,
            noise_scale=noise_scale,
            rng=self.rng
        )
    
    def _create_discretization(self, n_parents: int) -> DiscretizationTransformation:
        """Create a discretization transformation."""
        n_categories = self.config.n_categories
        input_dim = max(1, n_parents)
        noise_scale = self.config.noise_scale
        
        prototypes = self.rng.normal(0, 1, size=(n_categories, input_dim))
        
        return DiscretizationTransformation(
            prototypes=prototypes,
            n_categories=n_categories,
            noise_scale=noise_scale,
            rng=self.rng
        )


class DAGBuilder:
    """
    Modified DAG Builder for 3D temporal generator.
    
    v4: Creates TIME roots (1 base + n_extra) + MEMORY roots
    """
    
    def __init__(self, config, rng: Optional[np.random.Generator] = None):
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()
        
        # Total roots = 1 (base TIME) + n_extra_time_inputs + memory_dim (MEMORY)
        n_extra_time = getattr(config, 'n_extra_time_inputs', 0)
        memory_dim = getattr(config, 'memory_dim', 4)
        self.n_roots = 1 + n_extra_time + memory_dim
    
    def build(self) -> DAG:
        """Build a random DAG with TIME + MEMORY roots."""
        n_nodes = self.config.n_nodes
        n_disconnected = getattr(self.config, 'n_disconnected_subgraphs', 0)
        density = getattr(self.config, 'density', 0.3)
        
        # Ensure enough nodes for roots
        n_roots = self.n_roots
        n_nodes = max(n_nodes, n_roots + 3)
        
        # Calculate subgraph allocation
        min_disconnected_size = 2
        max_disconnected_size = max(3, n_nodes // 10)
        min_main_size = max(n_roots + 2, int(n_nodes * 0.6))
        available_for_disconnected = n_nodes - min_main_size
        
        if available_for_disconnected < n_disconnected * min_disconnected_size:
            n_disconnected = available_for_disconnected // min_disconnected_size
        
        # Allocate nodes to subgraphs
        if n_disconnected > 0:
            subgraph_sizes = []
            remaining = available_for_disconnected
            
            for i in range(n_disconnected):
                if i == n_disconnected - 1:
                    size = remaining
                else:
                    max_size = min(max_disconnected_size, 
                                   remaining - (n_disconnected - i - 1) * min_disconnected_size)
                    size = self.rng.integers(min_disconnected_size, max_size + 1)
                    remaining -= size
                subgraph_sizes.append(size)
            
            n_main_subgraph = n_nodes - sum(subgraph_sizes)
            subgraph_sizes = [n_main_subgraph] + subgraph_sizes
        else:
            subgraph_sizes = [n_nodes]
        
        # Roots per subgraph (all in main subgraph)
        roots_per_subgraph = [n_roots]
        for i in range(1, len(subgraph_sizes)):
            n_subgraph_roots = min(2, max(1, subgraph_sizes[i] // 3))
            roots_per_subgraph.append(n_subgraph_roots)
        
        # Build subgraphs
        all_nodes: Dict[int, Node] = {}
        all_edges: List[Tuple[int, int]] = []
        all_roots: List[int] = []
        
        node_id_offset = 0
        
        for subgraph_id, (size, n_subgraph_roots) in enumerate(zip(subgraph_sizes, roots_per_subgraph)):
            n_subgraph_roots = min(n_subgraph_roots, size)
            
            nodes, edges, roots = self._build_subgraph_multi_root(
                size, n_subgraph_roots, density, subgraph_id, node_id_offset
            )
            all_nodes.update(nodes)
            all_edges.extend(edges)
            all_roots.extend(roots)
            node_id_offset += size
        
        # Compute topological order
        topological_order = self._topological_sort(all_nodes, all_edges)
        
        for order, node_id in enumerate(topological_order):
            all_nodes[node_id].topological_order = order
        
        return DAG(
            nodes=all_nodes,
            edges=all_edges,
            root_nodes=all_roots,
            n_subgraphs=len(subgraph_sizes),
            topological_order=topological_order
        )
    
    def _build_subgraph_multi_root(
        self, 
        n_nodes: int,
        n_roots: int,
        density: float, 
        subgraph_id: int,
        node_id_offset: int
    ) -> Tuple[Dict[int, Node], List[Tuple[int, int]], List[int]]:
        """Build a subgraph with multiple root nodes."""
        if n_nodes == 0:
            return {}, [], []
        
        n_roots = min(n_roots, n_nodes)
        n_non_roots = n_nodes - n_roots
        
        nodes: Dict[int, Node] = {}
        edges: List[Tuple[int, int]] = []
        root_ids: List[int] = []
        
        # Assign topological order
        topo_order = list(range(n_roots))
        non_root_orders = list(range(n_roots, n_nodes))
        self.rng.shuffle(non_root_orders)
        topo_order.extend(non_root_orders)
        
        order_to_id = {}
        for order in range(n_nodes):
            node_id = node_id_offset + order
            is_root = order < n_roots
            
            nodes[node_id] = Node(
                id=node_id,
                parents=[],
                children=[],
                is_root=is_root,
                subgraph_id=subgraph_id,
                topological_order=topo_order[order]
            )
            order_to_id[topo_order[order]] = node_id
            
            if is_root:
                root_ids.append(node_id)
        
        # Calculate target edges
        min_edges = n_non_roots
        max_possible = sum(range(n_roots, n_nodes))
        target_edges = int(min_edges + density * (max_possible - min_edges))
        target_edges = max(min_edges, min(max_possible, target_edges))
        
        # Ensure connectivity
        for order in range(n_roots, n_nodes):
            node_id = order_to_id[order]
            parent_order = self.rng.integers(0, order)
            parent_id = order_to_id[parent_order]
            
            edges.append((parent_id, node_id))
            nodes[node_id].parents.append(parent_id)
            nodes[parent_id].children.append(node_id)
        
        # Add additional edges
        current_edges = len(edges)
        edge_set = set(edges)
        
        attempts = 0
        max_attempts = target_edges * 10
        
        while current_edges < target_edges and attempts < max_attempts:
            attempts += 1
            
            child_order = self.rng.integers(n_roots, n_nodes)
            child_id = order_to_id[child_order]
            
            parent_order = self.rng.integers(0, child_order)
            parent_id = order_to_id[parent_order]
            
            edge = (parent_id, child_id)
            if edge not in edge_set:
                edge_set.add(edge)
                edges.append(edge)
                nodes[child_id].parents.append(parent_id)
                nodes[parent_id].children.append(child_id)
                current_edges += 1
        
        return nodes, edges, root_ids
    
    def _topological_sort(
        self, 
        nodes: Dict[int, Node], 
        edges: List[Tuple[int, int]]
    ) -> List[int]:
        """Compute topological ordering."""
        children = {nid: [] for nid in nodes}
        in_degree = {nid: 0 for nid in nodes}
        
        for parent, child in edges:
            children[parent].append(child)
            in_degree[child] += 1
        
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order = []
        
        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in children[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        
        return order
