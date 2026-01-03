"""
Wrapper for importing 2D generator components.

This module handles the import of DAG and transformation components from
02_synthetic_generator_2D without conflicting with the 3D config.

Also provides a modified DAGBuilder that creates multiple root nodes
(required for 3D temporal generator to have noise/time/state inputs).
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
    from transformations import EdgeTransformation, TransformationFactory, DiscretizationTransformation, PassthroughTransformation
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


class DAGBuilder:
    """
    Modified DAG Builder for 3D temporal generator.
    
    Key difference from 2D: Creates MULTIPLE root nodes to accommodate
    noise, time, and state inputs (minimum 3 roots needed).
    
    Algorithm (using topological ordering for acyclicity):
    1. Create n_roots initial root nodes (not connected to each other)
    2. Assign topological order to all nodes
    3. Add edges from lower to higher order (guarantees acyclicity)
    4. Control edge count via density parameter
    """
    
    def __init__(self, config, rng: Optional[np.random.Generator] = None):
        """
        Initialize the DAG builder.
        
        Args:
            config: Dataset configuration with graph parameters
            rng: Random number generator
        """
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()
        
        # Minimum roots needed = noise + time + state inputs
        self.min_roots = (
            getattr(config, 'n_noise_inputs', 1) +
            getattr(config, 'n_time_inputs', 1) +
            getattr(config, 'n_state_inputs', 1)
        )
    
    def build(self) -> DAG:
        """
        Build a random DAG with multiple root nodes.
        
        Returns:
            DAG object with nodes and edges
        """
        n_nodes = self.config.n_nodes
        n_disconnected = getattr(self.config, 'n_disconnected_subgraphs', 0)
        density = getattr(self.config, 'density', 0.3)
        
        # Ensure we have enough nodes for roots
        n_roots = max(self.min_roots, 3)  # At least 3 roots
        n_nodes = max(n_nodes, n_roots + 5)  # At least 5 non-root nodes
        
        # Minimum size for a disconnected subgraph to be meaningful
        min_disconnected_size = 3
        max_disconnected_size = max(5, n_nodes // 10)
        
        # Main subgraph needs at least 60% of nodes
        min_main_size = max(n_roots + 2, int(n_nodes * 0.6))
        available_for_disconnected = n_nodes - min_main_size
        
        # Can we fit the requested disconnected subgraphs?
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
                    size = self.rng.integers(min_disconnected_size, 
                                            min(max_disconnected_size, remaining - (n_disconnected - i - 1) * min_disconnected_size) + 1)
                    remaining -= size
                subgraph_sizes.append(size)
            
            n_main_subgraph = n_nodes - sum(subgraph_sizes)
            subgraph_sizes = [n_main_subgraph] + subgraph_sizes
        else:
            subgraph_sizes = [n_nodes]
        
        # Allocate roots per subgraph
        roots_per_subgraph = [n_roots]  # Main subgraph gets all required roots
        for i in range(1, len(subgraph_sizes)):
            # Disconnected subgraphs get 1-3 roots depending on size
            n_subgraph_roots = min(3, max(1, subgraph_sizes[i] // 3))
            roots_per_subgraph.append(n_subgraph_roots)
        
        # Build each subgraph
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
        
        # Update topological order in nodes
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
        """
        Build a subgraph with MULTIPLE root nodes using topological ordering.
        
        Args:
            n_nodes: Number of nodes in this subgraph
            n_roots: Number of root nodes to create
            density: Edge density (0 = tree, 1 = complete DAG)
            subgraph_id: ID for this subgraph
            node_id_offset: Offset for node IDs
            
        Returns:
            Tuple of (nodes dict, edges list, root node IDs)
        """
        if n_nodes == 0:
            return {}, [], []
        
        n_roots = min(n_roots, n_nodes)
        n_non_roots = n_nodes - n_roots
        
        # Create all nodes with assigned topological order
        # Roots get orders 0 to n_roots-1, then non-roots get random permutation
        nodes: Dict[int, Node] = {}
        edges: List[Tuple[int, int]] = []
        root_ids: List[int] = []
        
        # Assign topological order: roots first, then shuffled non-roots
        topo_order = list(range(n_roots))  # Roots get first positions
        non_root_orders = list(range(n_roots, n_nodes))
        self.rng.shuffle(non_root_orders)
        topo_order.extend(non_root_orders)
        
        # Create mapping from order to node_id
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
        
        # Calculate target number of edges based on density
        # Minimum: n_non_roots edges (each non-root needs at least 1 parent)
        # Maximum: sum from i=n_roots to n_nodes-1 of i = all possible edges
        min_edges = n_non_roots
        max_possible = sum(range(n_roots, n_nodes))  # Each non-root can connect to all lower-order nodes
        target_edges = int(min_edges + density * (max_possible - min_edges))
        target_edges = max(min_edges, min(max_possible, target_edges))
        
        # First, ensure connectivity: each non-root must have at least one parent
        for order in range(n_roots, n_nodes):
            node_id = order_to_id[order]
            # Pick random parent from lower-order nodes
            parent_order = self.rng.integers(0, order)
            parent_id = order_to_id[parent_order]
            
            edges.append((parent_id, node_id))
            nodes[node_id].parents.append(parent_id)
            nodes[parent_id].children.append(node_id)
        
        # Add additional edges to reach target density
        current_edges = len(edges)
        edge_set = set(edges)
        
        attempts = 0
        max_attempts = target_edges * 10
        
        while current_edges < target_edges and attempts < max_attempts:
            attempts += 1
            
            # Pick a random non-root node
            child_order = self.rng.integers(n_roots, n_nodes)
            child_id = order_to_id[child_order]
            
            # Pick a random lower-order node as parent
            parent_order = self.rng.integers(0, child_order)
            parent_id = order_to_id[parent_order]
            
            # Check if edge already exists
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
        """Compute topological ordering of nodes."""
        # Build adjacency list
        children = {nid: [] for nid in nodes}
        in_degree = {nid: 0 for nid in nodes}
        
        for parent, child in edges:
            children[parent].append(child)
            in_degree[child] += 1
        
        # Kahn's algorithm
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

