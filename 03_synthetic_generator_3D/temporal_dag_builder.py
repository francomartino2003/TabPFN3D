"""
Temporal DAG Builder for 3D Synthetic Dataset Generation.

This module constructs the unrolled DAG for time series generation:
1. Create a base DAG (spatial structure within each timestep)
2. Unroll it T times
3. Add temporal connections between timesteps (always forward to maintain acyclicity)

The result is a large acyclic graph where:
- Nodes are indexed as (timestep, node_in_base_graph)
- Edges can be spatial (within timestep) or temporal (across timesteps)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
import numpy as np
import sys
import os

# Import base DAG builder from 2D generator
# Need to be careful with imports to avoid conflicts
_2d_path = os.path.join(os.path.dirname(__file__), '..', '02_synthetic_generator_2D')

# Temporarily modify path to import from 2D
_original_path = sys.path.copy()
sys.path.insert(0, _2d_path)

# Import 2D modules
import importlib.util
spec = importlib.util.spec_from_file_location("dag_builder_2d", os.path.join(_2d_path, "dag_builder.py"))
dag_builder_2d = importlib.util.module_from_spec(spec)

# We need the DAG and Node classes - let's define them locally to avoid import conflicts
# This is a cleaner approach

# Restore path
sys.path = _original_path

# Import local config
from config import DatasetConfig3D, TemporalConnectionConfig


# Redefine necessary classes from 2D (avoiding import conflicts)
@dataclass
class Node:
    """Represents a node in the causal DAG (from 2D)."""
    id: int
    parents: List[int]
    children: List[int]
    is_root: bool
    subgraph_id: int
    topological_order: int


@dataclass
class DAG:
    """DAG from 2D generator."""
    nodes: Dict[int, 'Node']
    edges: List[Tuple[int, int]]
    root_nodes: List[int]
    n_subgraphs: int
    topological_order: List[int]
    
    def get_ancestors(self, node_id: int) -> Set[int]:
        ancestors = set()
        to_visit = list(self.nodes[node_id].parents)
        while to_visit:
            parent = to_visit.pop()
            if parent not in ancestors:
                ancestors.add(parent)
                to_visit.extend(self.nodes[parent].parents)
        return ancestors
    
    def get_descendants(self, node_id: int) -> Set[int]:
        descendants = set()
        to_visit = list(self.nodes[node_id].children)
        while to_visit:
            child = to_visit.pop()
            if child not in descendants:
                descendants.add(child)
                to_visit.extend(self.nodes[child].children)
        return descendants
    
    def get_subgraph_nodes(self, subgraph_id: int) -> List[int]:
        return [nid for nid, node in self.nodes.items() if node.subgraph_id == subgraph_id]


@dataclass
class TemporalNode:
    """Represents a node in the unrolled temporal DAG."""
    global_id: int           # Unique ID in the full unrolled graph
    timestep: int            # Which timestep this node belongs to
    base_node_id: int        # ID in the base graph
    spatial_parents: List[int]   # Parent global IDs within same timestep
    temporal_parents: List[int]  # Parent global IDs from previous timesteps
    children: List[int]      # Children global IDs (spatial + temporal)
    is_root: bool            # True if no parents (spatial or temporal)
    subgraph_id: int         # Which subgraph in base graph
    

@dataclass
class TemporalEdgeInfo:
    """Information about a temporal edge."""
    from_timestep: int
    to_timestep: int
    from_base_id: int
    to_base_id: int
    pattern_id: str  # Which connection pattern created this edge
    weight: float    # Weight/strength of connection (for multi-skip decay)


@dataclass
class TemporalDAG:
    """
    Unrolled Directed Acyclic Graph with temporal structure.
    
    This represents the full computation graph across all timesteps.
    
    Attributes:
        nodes: Dict mapping global_id to TemporalNode
        spatial_edges: Edges within timesteps (t, parent_id, child_id)
        temporal_edges: Edges across timesteps with pattern info
        temporal_edge_info: Detailed info for each temporal edge
        base_dag: The underlying spatial DAG (repeated at each timestep)
        n_timesteps: Number of timesteps
        n_base_nodes: Number of nodes in base DAG
        topological_order: Global IDs in topological order
        connection_configs: Reference to the connection configurations
    """
    nodes: Dict[int, TemporalNode]
    spatial_edges: List[Tuple[int, int, int]]  # (timestep, parent_base_id, child_base_id)
    temporal_edges: List[Tuple[int, int, int, int]]  # (from_t, to_t, from_base_id, to_base_id) - legacy
    temporal_edge_info: List[TemporalEdgeInfo]  # Rich edge information
    base_dag: DAG
    n_timesteps: int
    n_base_nodes: int
    topological_order: List[int]
    connection_configs: List[Any] = field(default_factory=list)  # TemporalConnectionConfig list
    
    def get_global_id(self, timestep: int, base_node_id: int) -> int:
        """Convert (timestep, base_node_id) to global_id."""
        return timestep * self.n_base_nodes + base_node_id
    
    def get_timestep_and_base(self, global_id: int) -> Tuple[int, int]:
        """Convert global_id to (timestep, base_node_id)."""
        timestep = global_id // self.n_base_nodes
        base_node_id = global_id % self.n_base_nodes
        return timestep, base_node_id
    
    def get_nodes_at_timestep(self, timestep: int) -> List[int]:
        """Get all global IDs for nodes at a given timestep."""
        start = timestep * self.n_base_nodes
        return list(range(start, start + self.n_base_nodes))
    
    def get_ancestors(self, global_id: int) -> Set[int]:
        """Get all ancestors of a node (including temporal ancestors)."""
        ancestors = set()
        to_visit = list(self.nodes[global_id].spatial_parents + 
                       self.nodes[global_id].temporal_parents)
        while to_visit:
            parent = to_visit.pop()
            if parent not in ancestors:
                ancestors.add(parent)
                node = self.nodes[parent]
                to_visit.extend(node.spatial_parents + node.temporal_parents)
        return ancestors


class TemporalDAGBuilder:
    """
    Builds the unrolled temporal DAG.
    
    Process:
    1. Create base DAG using 2D DAGBuilder
    2. Create T copies of the base DAG
    3. Add temporal connections according to connection patterns
    4. Compute topological order for the full graph
    """
    
    def __init__(self, config: DatasetConfig3D, rng: Optional[np.random.Generator] = None):
        """
        Initialize the temporal DAG builder.
        
        Args:
            config: 3D dataset configuration
            rng: Random number generator
        """
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng(config.seed)
    
    def build(self) -> TemporalDAG:
        """
        Build the unrolled temporal DAG.
        
        Returns:
            TemporalDAG with full structure
        """
        n_timesteps = self.config.n_timesteps
        
        # Step 1: Build base DAG
        base_dag = self._build_base_dag()
        n_base_nodes = len(base_dag.nodes)
        
        # Step 2: Create temporal nodes (unroll the graph)
        nodes: Dict[int, TemporalNode] = {}
        spatial_edges: List[Tuple[int, int, int]] = []
        
        for t in range(n_timesteps):
            for base_node_id, base_node in base_dag.nodes.items():
                global_id = t * n_base_nodes + base_node_id
                
                # Spatial parents (within same timestep)
                spatial_parents = [t * n_base_nodes + p for p in base_node.parents]
                
                nodes[global_id] = TemporalNode(
                    global_id=global_id,
                    timestep=t,
                    base_node_id=base_node_id,
                    spatial_parents=spatial_parents,
                    temporal_parents=[],  # Will be filled in step 3
                    children=[],
                    is_root=base_node.is_root and t == 0,
                    subgraph_id=base_node.subgraph_id
                )
                
                # Record spatial edges
                for parent_base_id in base_node.parents:
                    spatial_edges.append((t, parent_base_id, base_node_id))
        
        # Step 3: Add temporal connections (returns tuples with pattern info)
        temporal_edge_tuples = self._add_temporal_connections(nodes, n_base_nodes, n_timesteps)
        
        # Convert to legacy format and rich format
        temporal_edges = [(t[0], t[1], t[2], t[3]) for t in temporal_edge_tuples]
        temporal_edge_info = [
            TemporalEdgeInfo(
                from_timestep=t[0],
                to_timestep=t[1],
                from_base_id=t[2],
                to_base_id=t[3],
                pattern_id=t[4] if len(t) > 4 else "",
                weight=t[5] if len(t) > 5 else 1.0
            )
            for t in temporal_edge_tuples
        ]
        
        # Step 4: Fill in children lists
        for global_id, node in nodes.items():
            for parent_id in node.spatial_parents + node.temporal_parents:
                if parent_id in nodes:
                    nodes[parent_id].children.append(global_id)
        
        # Step 5: Update is_root based on temporal parents
        for global_id, node in nodes.items():
            if node.temporal_parents:
                node.is_root = False
        
        # Step 6: Compute topological order
        topological_order = self._compute_topological_order(nodes)
        
        return TemporalDAG(
            nodes=nodes,
            spatial_edges=spatial_edges,
            temporal_edges=temporal_edges,
            temporal_edge_info=temporal_edge_info,
            base_dag=base_dag,
            n_timesteps=n_timesteps,
            n_base_nodes=n_base_nodes,
            topological_order=topological_order,
            connection_configs=self.config.temporal_connections
        )
    
    def _build_base_dag(self) -> DAG:
        """Build the base spatial DAG using preferential attachment."""
        n_nodes = self.config.n_nodes
        density = self.config.density
        n_disconnected = self.config.n_disconnected_subgraphs
        
        # Allocate nodes to subgraphs
        n_main_subgraph = n_nodes - n_disconnected * 2
        n_main_subgraph = max(n_main_subgraph, n_nodes // 2)
        
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
        
        # Build subgraphs
        all_nodes: Dict[int, Node] = {}
        all_edges: List[Tuple[int, int]] = []
        all_roots: List[int] = []
        node_id_offset = 0
        
        for subgraph_id, size in enumerate(subgraph_sizes):
            if size == 0:
                continue
                
            # Build this subgraph using preferential attachment
            for i in range(size):
                node_id = node_id_offset + i
                
                if i == 0:
                    # First node is a root
                    all_nodes[node_id] = Node(
                        id=node_id,
                        parents=[],
                        children=[],
                        is_root=True,
                        subgraph_id=subgraph_id,
                        topological_order=-1
                    )
                    all_roots.append(node_id)
                else:
                    # Connect to previous nodes
                    existing = list(range(node_id_offset, node_id))
                    n_parents = min(len(existing), max(1, int(self.rng.poisson(density))))
                    n_parents = min(n_parents, 4)  # Cap parents
                    
                    # Preferential attachment: prefer nodes with more connections
                    weights = np.array([1.0 + len(all_nodes[n].children) for n in existing])
                    weights = weights / weights.sum()
                    
                    parents = list(self.rng.choice(
                        existing, 
                        size=min(n_parents, len(existing)),
                        replace=False,
                        p=weights
                    ))
                    
                    all_nodes[node_id] = Node(
                        id=node_id,
                        parents=parents,
                        children=[],
                        is_root=False,
                        subgraph_id=subgraph_id,
                        topological_order=-1
                    )
                    
                    for p in parents:
                        all_nodes[p].children.append(node_id)
                        all_edges.append((p, node_id))
            
            node_id_offset += size
        
        # Compute topological order (nodes are already in order due to construction)
        topological_order = list(range(n_nodes))
        for order, nid in enumerate(topological_order):
            if nid in all_nodes:
                all_nodes[nid].topological_order = order
        
        return DAG(
            nodes=all_nodes,
            edges=all_edges,
            root_nodes=all_roots,
            n_subgraphs=len(subgraph_sizes),
            topological_order=topological_order
        )
    
    def _add_temporal_connections(
        self, 
        nodes: Dict[int, TemporalNode],
        n_base_nodes: int,
        n_timesteps: int
    ) -> List[Tuple[int, int, int, int, str, float]]:
        """
        Add temporal connections between timesteps.
        
        Handles all connection types:
        - self, cross, many_to_one, one_to_many: standard patterns
        - broadcast_multiskip: multiple skips with decay weights
        - conditional_lag: skip determined at runtime based on value
        - conditional_dest: target determined at runtime based on value
        
        Returns:
            List of (t_source, t_target, source_base, target_base, pattern_id, weight)
        """
        temporal_edges = []
        
        for conn in self.config.temporal_connections:
            # Get active timesteps for this connection
            active_timesteps = conn.get_active_timesteps(n_timesteps)
            
            # Get all skip values for this connection
            skip_values = conn.get_all_skips()
            
            # Handle different connection types
            if conn.connection_type in ['self', 'cross', 'many_to_one', 'one_to_many']:
                # Standard patterns - single skip
                for skip in skip_values:
                    weight = conn.get_skip_weight(skip)
                    for t in active_timesteps:
                        if t + skip < n_timesteps:
                            for src_base in conn.source_nodes:
                                for tgt_base in conn.target_nodes:
                                    if src_base < n_base_nodes and tgt_base < n_base_nodes:
                                        src_global = t * n_base_nodes + src_base
                                        tgt_global = (t + skip) * n_base_nodes + tgt_base
                                        
                                        if src_global in nodes and tgt_global in nodes:
                                            nodes[tgt_global].temporal_parents.append(src_global)
                                            temporal_edges.append((
                                                t, t + skip, src_base, tgt_base, 
                                                conn.pattern_id, weight
                                            ))
            
            elif conn.connection_type == 'broadcast_multiskip':
                # Multiple skips with decay weights (AR-like)
                for skip in skip_values:
                    weight = conn.get_skip_weight(skip)
                    for t in active_timesteps:
                        if t + skip < n_timesteps:
                            for src_base in conn.source_nodes:
                                for tgt_base in conn.target_nodes:
                                    if src_base < n_base_nodes and tgt_base < n_base_nodes:
                                        src_global = t * n_base_nodes + src_base
                                        tgt_global = (t + skip) * n_base_nodes + tgt_base
                                        
                                        if src_global in nodes and tgt_global in nodes:
                                            nodes[tgt_global].temporal_parents.append(src_global)
                                            temporal_edges.append((
                                                t, t + skip, src_base, tgt_base,
                                                conn.pattern_id, weight
                                            ))
            
            elif conn.connection_type == 'conditional_lag':
                # Skip depends on value - create edges for all possible skips
                # The actual skip will be determined at runtime by row_generator
                if conn.conditional_skips:
                    for skip in conn.conditional_skips:
                        for t in active_timesteps:
                            if t + skip < n_timesteps:
                                for src_base in conn.source_nodes:
                                    for tgt_base in conn.target_nodes:
                                        if src_base < n_base_nodes and tgt_base < n_base_nodes:
                                            src_global = t * n_base_nodes + src_base
                                            tgt_global = (t + skip) * n_base_nodes + tgt_base
                                            
                                            if src_global in nodes and tgt_global in nodes:
                                                # Mark as conditional parent
                                                if src_global not in nodes[tgt_global].temporal_parents:
                                                    nodes[tgt_global].temporal_parents.append(src_global)
                                                temporal_edges.append((
                                                    t, t + skip, src_base, tgt_base,
                                                    conn.pattern_id, conn.weight
                                                ))
            
            elif conn.connection_type == 'conditional_dest':
                # Target depends on value - create edges to all possible targets
                # Actual target determined at runtime
                skip = conn.skip
                if conn.conditional_targets:
                    all_targets = set()
                    for target_set in conn.conditional_targets:
                        all_targets.update(target_set)
                    
                    for t in active_timesteps:
                        if t + skip < n_timesteps:
                            for src_base in conn.source_nodes:
                                for tgt_base in all_targets:
                                    if src_base < n_base_nodes and tgt_base < n_base_nodes:
                                        src_global = t * n_base_nodes + src_base
                                        tgt_global = (t + skip) * n_base_nodes + tgt_base
                                        
                                        if src_global in nodes and tgt_global in nodes:
                                            if src_global not in nodes[tgt_global].temporal_parents:
                                                nodes[tgt_global].temporal_parents.append(src_global)
                                            temporal_edges.append((
                                                t, t + skip, src_base, tgt_base,
                                                conn.pattern_id, conn.weight
                                            ))
        
        return temporal_edges
    
    def _compute_topological_order(self, nodes: Dict[int, TemporalNode]) -> List[int]:
        """
        Compute topological ordering of the unrolled graph.
        
        Since we only add forward temporal edges, this is straightforward:
        process timesteps in order, and within each timestep use base graph order.
        """
        order = []
        
        # In-degree count
        in_degree = {gid: len(n.spatial_parents) + len(n.temporal_parents) 
                     for gid, n in nodes.items()}
        
        # Start with nodes that have no parents
        queue = [gid for gid, deg in in_degree.items() if deg == 0]
        
        while queue:
            # Sort for determinism (process lower timesteps first)
            queue.sort(key=lambda x: (nodes[x].timestep, nodes[x].base_node_id))
            node_id = queue.pop(0)
            order.append(node_id)
            
            # Decrease in-degree of children
            for child_id in nodes[node_id].children:
                in_degree[child_id] -= 1
                if in_degree[child_id] == 0:
                    queue.append(child_id)
        
        if len(order) != len(nodes):
            raise ValueError(f"Graph has cycles! Order has {len(order)} nodes, expected {len(nodes)}")
        
        return order


def visualize_temporal_dag(dag: TemporalDAG, max_timesteps: int = 5, filename: Optional[str] = None):
    """
    Visualize the temporal DAG structure.
    
    Args:
        dag: The temporal DAG
        max_timesteps: Maximum timesteps to show (for readability)
        filename: If provided, save to file
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib required for visualization")
        return
    
    n_show = min(max_timesteps, dag.n_timesteps)
    n_nodes = dag.n_base_nodes
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Node positions
    pos = {}
    for t in range(n_show):
        for i in range(n_nodes):
            global_id = t * n_nodes + i
            pos[global_id] = (t * 2, -i)
    
    # Draw spatial edges (within timestep)
    for t, parent_base, child_base in dag.spatial_edges:
        if t < n_show:
            parent_global = t * n_nodes + parent_base
            child_global = t * n_nodes + child_base
            if parent_global in pos and child_global in pos:
                ax.annotate('', xy=pos[child_global], xytext=pos[parent_global],
                           arrowprops=dict(arrowstyle='->', color='blue', alpha=0.5))
    
    # Draw temporal edges (across timesteps)
    for from_t, to_t, from_base, to_base in dag.temporal_edges:
        if from_t < n_show and to_t < n_show:
            from_global = from_t * n_nodes + from_base
            to_global = to_t * n_nodes + to_base
            if from_global in pos and to_global in pos:
                ax.annotate('', xy=pos[to_global], xytext=pos[from_global],
                           arrowprops=dict(arrowstyle='->', color='red', alpha=0.7,
                                         connectionstyle='arc3,rad=0.2'))
    
    # Draw nodes
    for global_id, (x, y) in pos.items():
        t, base_id = dag.get_timestep_and_base(global_id)
        color = plt.cm.tab10(dag.nodes[global_id].subgraph_id % 10)
        circle = plt.Circle((x, y), 0.15, color=color, ec='black', zorder=10)
        ax.add_patch(circle)
        ax.text(x, y, str(base_id), ha='center', va='center', fontsize=8, zorder=11)
    
    # Add timestep labels
    for t in range(n_show):
        ax.text(t * 2, 1, f't={t}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Legend
    spatial_patch = mpatches.Patch(color='blue', alpha=0.5, label='Spatial edges')
    temporal_patch = mpatches.Patch(color='red', alpha=0.7, label='Temporal edges')
    ax.legend(handles=[spatial_patch, temporal_patch], loc='upper right')
    
    ax.set_xlim(-0.5, n_show * 2 - 0.5)
    ax.set_ylim(-n_nodes - 0.5, 2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Temporal DAG ({dag.n_timesteps} timesteps, {n_nodes} nodes/timestep)\n'
                f'Showing first {n_show} timesteps')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

