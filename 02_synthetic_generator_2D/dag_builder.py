"""
DAG (Directed Acyclic Graph) builder for synthetic dataset generation.

This module constructs random DAGs that define the causal structure
of the generative process. The key features are:

1. Uses topological ordering to guarantee acyclicity
2. Controls edge density (from tree to dense DAG)
3. Supports multiple root/input nodes
4. Can create disconnected subgraphs (for irrelevant features)
"""

from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional
import numpy as np

try:
    from .config import DatasetConfig
except ImportError:
    from config import DatasetConfig


@dataclass
class Node:
    """Represents a node in the causal DAG."""
    id: int
    parents: List[int]  # IDs of parent nodes
    children: List[int]  # IDs of child nodes
    is_root: bool  # True if no parents (input node)
    subgraph_id: int  # Which subgraph this node belongs to
    topological_order: int  # Position in topological sort
    
    def __repr__(self) -> str:
        return f"Node({self.id}, parents={self.parents}, children={self.children}, root={self.is_root})"


@dataclass
class DAG:
    """
    Directed Acyclic Graph representing the causal structure.
    
    Attributes:
        nodes: Dictionary mapping node ID to Node object
        edges: List of (parent_id, child_id) tuples
        root_nodes: List of node IDs with no parents (input nodes)
        n_subgraphs: Number of disconnected subgraphs
        topological_order: List of node IDs in topological order
    """
    nodes: Dict[int, Node]
    edges: List[Tuple[int, int]]
    root_nodes: List[int]
    n_subgraphs: int
    topological_order: List[int]
    
    def get_ancestors(self, node_id: int) -> Set[int]:
        """Get all ancestors of a node (nodes that can reach this node)."""
        ancestors = set()
        to_visit = list(self.nodes[node_id].parents)
        while to_visit:
            parent = to_visit.pop()
            if parent not in ancestors:
                ancestors.add(parent)
                to_visit.extend(self.nodes[parent].parents)
        return ancestors
    
    def get_descendants(self, node_id: int) -> Set[int]:
        """Get all descendants of a node (nodes reachable from this node)."""
        descendants = set()
        to_visit = list(self.nodes[node_id].children)
        while to_visit:
            child = to_visit.pop()
            if child not in descendants:
                descendants.add(child)
                to_visit.extend(self.nodes[child].children)
        return descendants
    
    def get_subgraph_nodes(self, subgraph_id: int) -> List[int]:
        """Get all nodes belonging to a specific subgraph."""
        return [nid for nid, node in self.nodes.items() if node.subgraph_id == subgraph_id]
    
    def is_connected_to(self, node_a: int, node_b: int) -> bool:
        """Check if two nodes are in the same connected component."""
        return self.nodes[node_a].subgraph_id == self.nodes[node_b].subgraph_id


class DAGBuilder:
    """
    Builds random DAGs for synthetic dataset generation.
    
    Uses topological ordering to guarantee acyclicity and controls
    edge density from sparse trees to dense DAGs.
    """
    
    def __init__(self, config: DatasetConfig, rng: Optional[np.random.Generator] = None):
        """
        Initialize the DAG builder.
        
        Args:
            config: Dataset configuration with graph parameters
            rng: Random number generator
        """
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng(config.seed)
        
    def build(self) -> DAG:
        """
        Build a random DAG according to the configuration.
        
        Returns:
            DAG object with nodes and edges
        """
        n_nodes = self.config.n_nodes
        n_disconnected = self.config.n_disconnected_subgraphs
        density = self.config.density
        n_roots = self.config.n_roots
        
        # Minimum size for a disconnected subgraph to be meaningful
        min_disconnected_size = 3
        max_disconnected_size = max(5, n_nodes // 10)  # Up to 10% of total nodes per subgraph
        
        # Check if we have enough nodes for disconnected subgraphs
        # Main subgraph needs at least 60% of nodes
        min_main_size = max(n_roots + 2, int(n_nodes * 0.6))
        available_for_disconnected = n_nodes - min_main_size
        
        # Can we fit the requested disconnected subgraphs?
        if available_for_disconnected < n_disconnected * min_disconnected_size:
            # Not enough space - reduce or eliminate disconnected subgraphs
            n_disconnected = available_for_disconnected // min_disconnected_size
        
        # Allocate nodes to subgraphs
        if n_disconnected > 0:
            # Distribute remaining nodes among disconnected subgraphs
            subgraph_sizes = []
            remaining = available_for_disconnected
            
            for i in range(n_disconnected):
                if i == n_disconnected - 1:
                    size = remaining
                else:
                    # Random size within bounds
                    size = self.rng.integers(min_disconnected_size, 
                                            min(max_disconnected_size, remaining - (n_disconnected - i - 1) * min_disconnected_size) + 1)
                    remaining -= size
                subgraph_sizes.append(size)
            
            # Main subgraph gets the rest
            n_main_subgraph = n_nodes - sum(subgraph_sizes)
            subgraph_sizes = [n_main_subgraph] + subgraph_sizes
        else:
            # No disconnected subgraphs
            subgraph_sizes = [n_nodes]
        
        # Allocate roots per subgraph
        # Main subgraph gets most roots
        roots_per_subgraph = [n_roots]
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
            n_subgraph_roots = min(n_subgraph_roots, size)  # Can't have more roots than nodes
            nodes, edges, roots = self._build_subgraph(
                n_nodes=size,
                n_roots=n_subgraph_roots,
                density=density,
                subgraph_id=subgraph_id,
                node_id_offset=node_id_offset
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
    
    def _build_subgraph(
        self, 
        n_nodes: int, 
        n_roots: int,
        density: float,
        subgraph_id: int,
        node_id_offset: int
    ) -> Tuple[Dict[int, Node], List[Tuple[int, int]], List[int]]:
        """
        Build a single connected subgraph using topological ordering.
        
        Algorithm:
        1. Assign topological order to nodes (random permutation)
        2. First n_roots nodes become root/input nodes
        3. For each non-root node, connect to at least one earlier node (ensures connectivity)
        4. Add additional edges based on density (always from lower to higher order = acyclic)
        
        Args:
            n_nodes: Number of nodes in this subgraph
            n_roots: Number of root/input nodes
            density: Desired edge density (0.0 = tree, 1.0 = complete DAG)
            subgraph_id: ID for this subgraph
            node_id_offset: Offset for node IDs
            
        Returns:
            Tuple of (nodes dict, edges list, root node IDs)
        """
        if n_nodes == 0:
            return {}, [], []
        
        if n_nodes == 1:
            # Single node is always a root
            node_id = node_id_offset
            nodes = {
                node_id: Node(
                    id=node_id,
                    parents=[],
                    children=[],
                    is_root=True,
                    subgraph_id=subgraph_id,
                    topological_order=0
                )
            }
            return nodes, [], [node_id]
        
        nodes: Dict[int, Node] = {}
        edges: List[Tuple[int, int]] = []
        edge_set: Set[Tuple[int, int]] = set()  # For fast lookup
        
        # Step 1: Create nodes with random topological ordering
        # The order within node_ids will be the topological order
        node_ids = list(range(node_id_offset, node_id_offset + n_nodes))
        self.rng.shuffle(node_ids)  # Random permutation = random topological order
        
        # Step 2: First n_roots nodes are roots (no parents)
        n_roots = min(n_roots, n_nodes)
        n_roots = max(1, n_roots)  # At least 1 root
        
        for order, node_id in enumerate(node_ids):
            is_root = order < n_roots
            nodes[node_id] = Node(
                id=node_id,
                parents=[],
                children=[],
                is_root=is_root,
                subgraph_id=subgraph_id,
                topological_order=order
            )
        
        # Step 3: Connect each non-root node to at least one parent (ensures connectivity)
        for order in range(n_roots, n_nodes):
            child_id = node_ids[order]
            # Select a random parent from nodes with lower topological order
            possible_parents = node_ids[:order]
            parent_id = self.rng.choice(possible_parents)
            
            # Add edge
            edge = (parent_id, child_id)
            edges.append(edge)
            edge_set.add(edge)
            nodes[parent_id].children.append(child_id)
            nodes[child_id].parents.append(parent_id)
        
        # Step 4: Add additional edges based on density
        # Number of edges so far: n_nodes - n_roots (one per non-root)
        current_edges = n_nodes - n_roots
        
        # Maximum possible edges: sum of possible parents for each non-root
        # For node at order i (where i >= n_roots), it can have parents from orders 0 to i-1
        # So max edges = sum from i=n_roots to n_nodes-1 of i = (n_nodes-1)*n_nodes/2 - (n_roots-1)*n_roots/2
        max_edges = 0
        for order in range(n_roots, n_nodes):
            max_edges += order  # Can connect to any of the 'order' previous nodes
        
        # Target number of additional edges
        min_edges = current_edges  # Already have tree connectivity
        target_edges = int(min_edges + density * (max_edges - min_edges))
        target_edges = min(target_edges, max_edges)
        
        # Add additional edges
        attempts = 0
        max_attempts = max_edges * 10
        
        while len(edges) < target_edges and attempts < max_attempts:
            attempts += 1
            
            # Select a random non-root node
            child_order = self.rng.integers(n_roots, n_nodes)
            child_id = node_ids[child_order]
            
            # Select a random potential parent (any node with lower order)
            parent_order = self.rng.integers(0, child_order)
            parent_id = node_ids[parent_order]
            
            # Check if edge already exists
            edge = (parent_id, child_id)
            if edge not in edge_set:
                edges.append(edge)
                edge_set.add(edge)
                nodes[parent_id].children.append(child_id)
                nodes[child_id].parents.append(parent_id)
        
        # Get root node IDs
        roots = [node_ids[i] for i in range(n_roots)]
        
        return nodes, edges, roots
    
    def _topological_sort(
        self, 
        nodes: Dict[int, Node], 
        edges: List[Tuple[int, int]]
    ) -> List[int]:
        """
        Compute topological ordering of nodes.
        
        Uses Kahn's algorithm.
        
        Args:
            nodes: Dictionary of nodes
            edges: List of edges
            
        Returns:
            List of node IDs in topological order
        """
        if not nodes:
            return []
        
        # Build in-degree count
        in_degree = {nid: 0 for nid in nodes}
        for parent, child in edges:
            in_degree[child] += 1
        
        # Start with nodes that have no incoming edges (roots)
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        result = []
        
        while queue:
            # Sort queue for deterministic order
            queue.sort()
            node = queue.pop(0)
            result.append(node)
            
            # Remove this node's outgoing edges
            for child in nodes[node].children:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        
        if len(result) != len(nodes):
            raise ValueError("Graph has cycles - this should not happen with our construction")
        
        return result
    
    @staticmethod
    def visualize(dag: DAG, filename: Optional[str] = None) -> None:
        """
        Visualize the DAG using matplotlib.
        
        Args:
            dag: The DAG to visualize
            filename: If provided, save to file instead of showing
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError:
            print("matplotlib and networkx required for visualization")
            return
        
        # Create networkx graph
        G = nx.DiGraph()
        G.add_nodes_from(dag.nodes.keys())
        G.add_edges_from(dag.edges)
        
        # Color nodes: roots in green, others by subgraph
        colors = []
        for nid in G.nodes():
            if dag.nodes[nid].is_root:
                colors.append('lightgreen')
            else:
                # Use different shades for different subgraphs
                subgraph_colors = ['lightblue', 'lightyellow', 'lightcoral', 'lightgray', 'plum']
                sg_id = dag.nodes[nid].subgraph_id % len(subgraph_colors)
                colors.append(subgraph_colors[sg_id])
        
        # Layout - use topological layout for DAG
        try:
            pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
        except:
            # Fallback to spring layout if graphviz not available
            pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw
        plt.figure(figsize=(14, 10))
        nx.draw(G, pos, 
                node_color=colors,
                with_labels=True,
                node_size=500,
                font_size=8,
                arrows=True,
                arrowsize=15,
                edge_color='gray',
                alpha=0.8)
        
        n_roots = len(dag.root_nodes)
        plt.title(f"DAG: {len(dag.nodes)} nodes, {len(dag.edges)} edges, "
                  f"{n_roots} roots, {dag.n_subgraphs} subgraphs")
        
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
