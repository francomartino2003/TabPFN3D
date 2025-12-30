"""
DAG (Directed Acyclic Graph) builder for synthetic dataset generation.

This module constructs random DAGs that define the causal structure
of the generative process. The key features are:

1. Uses preferential attachment for realistic graph structure
2. Can create disconnected subgraphs (for irrelevant features)
3. Guarantees acyclicity (topological ordering)
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
    is_root: bool  # True if no parents
    subgraph_id: int  # Which subgraph this node belongs to
    topological_order: int  # Position in topological sort
    
    def __repr__(self) -> str:
        return f"Node({self.id}, parents={self.parents}, children={self.children})"


@dataclass
class DAG:
    """
    Directed Acyclic Graph representing the causal structure.
    
    Attributes:
        nodes: Dictionary mapping node ID to Node object
        edges: List of (parent_id, child_id) tuples
        root_nodes: List of node IDs with no parents
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
    
    Uses a modified preferential attachment algorithm to create
    graphs with realistic structure (some nodes are hubs, others are leaves).
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
        
        # Allocate nodes to subgraphs
        # Main subgraph gets most nodes, disconnected subgraphs share the rest
        n_main_subgraph = n_nodes - n_disconnected * 2  # At least 2 nodes per disconnected subgraph
        n_main_subgraph = max(n_main_subgraph, n_nodes // 2)  # At least half the nodes
        
        subgraph_sizes = [n_main_subgraph]
        remaining = n_nodes - n_main_subgraph
        
        for i in range(n_disconnected):
            if i == n_disconnected - 1:
                size = remaining
            else:
                size = max(2, remaining // (n_disconnected - i))
                remaining -= size
            if size > 0:
                subgraph_sizes.append(size)
        
        # Build each subgraph
        all_nodes: Dict[int, Node] = {}
        all_edges: List[Tuple[int, int]] = []
        all_roots: List[int] = []
        node_id_offset = 0
        
        for subgraph_id, size in enumerate(subgraph_sizes):
            nodes, edges, roots = self._build_subgraph(
                size, density, subgraph_id, node_id_offset
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
        density: float, 
        subgraph_id: int,
        node_id_offset: int
    ) -> Tuple[Dict[int, Node], List[Tuple[int, int]], List[int]]:
        """
        Build a single connected subgraph using preferential attachment.
        
        Args:
            n_nodes: Number of nodes in this subgraph
            density: Controls edge density (higher = more edges)
            subgraph_id: ID for this subgraph
            node_id_offset: Offset for node IDs
            
        Returns:
            Tuple of (nodes dict, edges list, root node IDs)
        """
        if n_nodes == 0:
            return {}, [], []
        
        nodes: Dict[int, Node] = {}
        edges: List[Tuple[int, int]] = []
        
        # Create first node (always a root)
        first_id = node_id_offset
        nodes[first_id] = Node(
            id=first_id,
            parents=[],
            children=[],
            is_root=True,
            subgraph_id=subgraph_id,
            topological_order=-1
        )
        
        # Track node "attractiveness" for preferential attachment
        # Attractiveness increases with number of connections
        attractiveness = {first_id: 1.0}
        
        # Add remaining nodes one by one
        for i in range(1, n_nodes):
            new_id = node_id_offset + i
            
            # Determine number of parents for this node
            # Use density parameter to control expected number of edges
            existing_nodes = list(nodes.keys())
            max_parents = min(len(existing_nodes), 5)  # Cap at 5 parents
            
            # Sample number of parents based on density
            n_parents = self.rng.poisson(density)
            n_parents = np.clip(n_parents, 1, max_parents)
            
            # Select parents using preferential attachment
            probs = np.array([attractiveness[nid] for nid in existing_nodes])
            probs = probs / probs.sum()
            
            parent_indices = self.rng.choice(
                len(existing_nodes), 
                size=min(n_parents, len(existing_nodes)),
                replace=False,
                p=probs
            )
            parents = [existing_nodes[idx] for idx in parent_indices]
            
            # Create new node
            nodes[new_id] = Node(
                id=new_id,
                parents=parents,
                children=[],
                is_root=len(parents) == 0,
                subgraph_id=subgraph_id,
                topological_order=-1
            )
            
            # Add edges and update children lists
            for parent_id in parents:
                edges.append((parent_id, new_id))
                nodes[parent_id].children.append(new_id)
                # Increase attractiveness of parent
                attractiveness[parent_id] += 0.5
            
            # Initialize attractiveness of new node
            attractiveness[new_id] = 1.0 + len(parents) * 0.5
        
        # Additional random edges (to increase connectivity if density is high)
        n_additional = int(density * n_nodes * 0.3)
        for _ in range(n_additional):
            if len(nodes) < 2:
                break
                
            # Sample two nodes
            node_ids = list(nodes.keys())
            idx1, idx2 = self.rng.choice(len(node_ids), size=2, replace=False)
            n1, n2 = node_ids[idx1], node_ids[idx2]
            
            # Ensure n1 comes before n2 (for acyclicity)
            # Use current position in nodes dict as proxy for order
            if idx1 > idx2:
                n1, n2 = n2, n1
            
            # Add edge if it doesn't exist
            if n2 not in nodes[n1].children and n1 not in nodes[n2].parents:
                edges.append((n1, n2))
                nodes[n1].children.append(n2)
                nodes[n2].parents.append(n1)
                if nodes[n2].is_root:
                    nodes[n2].is_root = False
        
        # Find root nodes
        roots = [nid for nid, node in nodes.items() if node.is_root]
        
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
        
        # Build adjacency list and in-degree count
        in_degree = {nid: 0 for nid in nodes}
        for parent, child in edges:
            in_degree[child] += 1
        
        # Start with nodes that have no incoming edges
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        result = []
        
        while queue:
            # Sort queue for deterministic order
            queue.sort()
            node = queue.pop(0)
            result.append(node)
            
            # Remove this node's edges
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
        
        # Color nodes by subgraph
        colors = plt.cm.tab10(np.linspace(0, 1, dag.n_subgraphs))
        node_colors = [colors[dag.nodes[nid].subgraph_id] for nid in G.nodes()]
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, 
                node_color=node_colors,
                with_labels=True,
                node_size=500,
                font_size=8,
                arrows=True,
                arrowsize=15)
        plt.title(f"DAG with {len(dag.nodes)} nodes, {len(dag.edges)} edges, {dag.n_subgraphs} subgraphs")
        
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

