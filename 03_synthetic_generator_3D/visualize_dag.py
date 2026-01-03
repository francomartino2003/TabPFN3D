"""
Visualize DAG Structure and Explain How It's Built.

This script generates several DAGs and shows:
1. The graph structure (nodes and edges)
2. Input types (noise, time, state)
3. Transformations on edges (NN, decision tree, discretization, identity)
4. How values propagate through the network
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple, Set
import sys
import os

# Add parent for imports
sys.path.insert(0, os.path.dirname(__file__))

from config import PriorConfig3D, DatasetConfig3D
from dag_utils import DAGBuilder, TransformationFactory, DAG
from temporal_inputs import TemporalInputManager


def visualize_dag(
    dag: DAG,
    input_manager: TemporalInputManager,
    transformations: Dict,
    title: str = "DAG Structure",
    ax=None
):
    """
    Visualize a DAG with colored nodes by type.
    
    Colors:
    - Red: Noise inputs
    - Blue: Time inputs  
    - Green: State inputs
    - Gray: Internal nodes
    - Yellow: Nodes with discretization (categorical)
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    else:
        fig = ax.figure
    
    noise_ids = set(input_manager.noise_node_ids or [])
    time_ids = set(input_manager.time_node_ids or [])
    state_ids = set(input_manager.state_node_ids or [])
    
    # Find categorical nodes (after discretization)
    categorical_ids = set()
    for node_id, transform in transformations.items():
        if hasattr(transform, '__class__') and 'Discretization' in transform.__class__.__name__:
            categorical_ids.add(node_id)
    
    # Compute node positions using topological layers
    layers = compute_layers(dag)
    positions = {}
    
    for layer_idx, layer_nodes in enumerate(layers):
        n_nodes = len(layer_nodes)
        for i, node_id in enumerate(layer_nodes):
            x = (i - (n_nodes - 1) / 2) * 1.5
            y = -layer_idx * 2
            positions[node_id] = (x, y)
    
    # Draw edges first
    for node_id, node in dag.nodes.items():
        for parent_id in node.parents:
            if parent_id in positions and node_id in positions:
                x1, y1 = positions[parent_id]
                x2, y2 = positions[node_id]
                
                # Get transformation type for this child node
                # (transformation is per child node, not per edge)
                edge_color = 'gray'
                edge_style = '-'
                
                if node_id in transformations:
                    t = transformations[node_id]
                    t_name = t.__class__.__name__
                    if 'NN' in t_name or 'Neural' in t_name:
                        edge_color = 'purple'
                    elif 'Tree' in t_name or 'Decision' in t_name:
                        edge_color = 'orange'
                    elif 'Discretization' in t_name:
                        edge_color = 'brown'
                        edge_style = '--'
                
                ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                           arrowprops=dict(arrowstyle="->", color=edge_color,
                                          linestyle=edge_style, lw=1.5))
    
    # Draw nodes
    for node_id, (x, y) in positions.items():
        if node_id in noise_ids:
            color = '#FF6B6B'  # Red
            label = f'N{node_id}'
        elif node_id in time_ids:
            color = '#4ECDC4'  # Teal/Blue
            label = f'T{node_id}'
        elif node_id in state_ids:
            color = '#95E86B'  # Green
            label = f'S{node_id}'
        elif node_id in categorical_ids:
            color = '#FFE66D'  # Yellow
            label = f'C{node_id}'
        else:
            color = '#CCCCCC'  # Gray
            label = f'{node_id}'
        
        circle = plt.Circle((x, y), 0.4, color=color, ec='black', lw=2, zorder=10)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold', zorder=11)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='#FF6B6B', label='Noise Input'),
        mpatches.Patch(color='#4ECDC4', label='Time Input'),
        mpatches.Patch(color='#95E86B', label='State Input'),
        mpatches.Patch(color='#CCCCCC', label='Internal Node'),
        mpatches.Patch(color='#FFE66D', label='Categorical (after discretization)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    ax.set_xlim(-10, 10)
    ax.set_ylim(-len(layers) * 2 - 1, 2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    return fig, ax


def compute_layers(dag: DAG) -> List[List[int]]:
    """Compute topological layers for positioning."""
    # Find roots
    roots = [nid for nid, node in dag.nodes.items() if not node.parents]
    
    layers = []
    current_layer = roots
    visited = set(roots)
    
    while current_layer:
        layers.append(current_layer)
        next_layer = []
        for node_id in current_layer:
            node = dag.nodes[node_id]
            for child_id in node.children:
                if child_id not in visited:
                    # Check if all parents are visited
                    child_node = dag.nodes[child_id]
                    if all(p in visited for p in child_node.parents):
                        next_layer.append(child_id)
                        visited.add(child_id)
        current_layer = next_layer
    
    # Add any remaining nodes
    remaining = [nid for nid in dag.nodes if nid not in visited]
    if remaining:
        layers.append(remaining)
    
    return layers


def explain_dag_construction():
    """Print explanation of how DAGs are constructed."""
    explanation = """
================================================================================
                    HOW THE DAG IS CONSTRUCTED
================================================================================

1. GRAPH GENERATION: Growing Network with Redirection
   -------------------------------------------------
   - Start with a small seed graph (few nodes)
   - Add new nodes one at a time
   - Each new node connects to existing nodes:
     * With probability p: connect to a random existing node
     * With probability (1-p): connect to a parent of that node ("redirection")
   - This creates scale-free networks (some nodes have many connections)
   
2. ROOT NODES (Inputs)
   -------------------
   Nodes with no parents are "root nodes". They receive external inputs:
   
   [NOISE INPUTS - Red]
   - Fresh random values each timestep
   - Different for each sample
   - Provide sample-to-sample variability
   - Types: Normal(0, sigma), Uniform(-a, a)
   
   [TIME INPUTS - Blue/Teal]
   - Deterministic function of t/T (normalized time)
   - SAME for all samples at each timestep
   - Functions: linear, quadratic, cubic, sin(2*pi*k*u), cos(2*pi*k*u), tanh, exp_decay
   - Provide temporal patterns
   
   [STATE INPUTS - Green]
   - Memory from previous timestep
   - At t=0: initialized with random noise
   - At t>0: tanh(alpha * state_{t-1}) where alpha controls smoothing
   - Provide temporal dependencies between timesteps

3. EDGE TRANSFORMATIONS
   --------------------
   Each edge (parent -> child) has a transformation:
   
   [NEURAL NETWORK - Purple edges]
   - MLP with 1-2 hidden layers
   - ReLU/tanh activations
   - Xavier initialization
   - Most common transformation
   
   [DECISION TREE - Orange edges]
   - Random decision tree structure
   - Splits based on thresholds
   - Creates piecewise constant regions
   
   [DISCRETIZATION - Brown dashed edges]
   - Converts continuous -> categorical
   - Random bin boundaries
   - Child becomes categorical node
   
   [IDENTITY - Light gray edges]
   - Direct copy (with optional noise)
   - Used for simple propagation

4. VALUE PROPAGATION
   -----------------
   For each timestep t:
   1. Generate root inputs (noise, time, state)
   2. Process in topological order:
      - For each non-root node:
        - Collect parent values
        - Apply edge transformation
        - Add small noise (innovation)
   3. Extract state for next timestep

5. FEATURE/TARGET SELECTION
   ------------------------
   - Root nodes: NEVER selected (they're inputs, not observations)
   - Time-only nodes: NEVER selected (constant across samples)
   - Target: preferably deep in the graph (complex dependencies)
   - Features: ancestors of target (relevant) + some irrelevant nodes
   - Equivalence: avoid selecting both pre/post discretization nodes

================================================================================
"""
    print(explanation)


def main():
    """Generate and visualize several DAGs."""
    
    explain_dag_construction()
    
    print("\n" + "="*60)
    print("GENERATING EXAMPLE DAGs")
    print("="*60 + "\n")
    
    # Generate 4 DAGs with different configurations
    configs = [
        {"n_nodes": 20, "seed": 1, "desc": "Small DAG (20 nodes)"},
        {"n_nodes": 40, "seed": 2, "desc": "Medium DAG (40 nodes)"},
        {"n_nodes": 80, "seed": 3, "desc": "Large DAG (80 nodes)"},
        {"n_nodes": 30, "seed": 4, "desc": "Another variation (30 nodes)"},
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    prior = PriorConfig3D()
    
    for idx, cfg_info in enumerate(configs):
        seed = cfg_info["seed"]
        rng = np.random.default_rng(seed)
        
        # Sample config with specific n_nodes
        # Create a modified prior for this example
        modified_prior = PriorConfig3D(
            n_nodes_range=(cfg_info["n_nodes"], cfg_info["n_nodes"]),
            prob_disconnected_subgraph=0.0,  # Single connected graph for clarity
        )
        cfg = DatasetConfig3D.sample_from_prior(modified_prior, rng)
        
        # Build DAG
        dag_builder = DAGBuilder(cfg)
        dag = dag_builder.build()
        
        # Build transformations - ONE per non-root node
        transform_factory = TransformationFactory(cfg, rng)
        transformations = {}
        for node_id, node in dag.nodes.items():
            if node.parents:  # Only non-root nodes need transformations
                transformations[node_id] = transform_factory.create(n_parents=len(node.parents))
        
        # Create input manager
        input_manager = TemporalInputManager.from_config(cfg, rng)
        root_nodes = [nid for nid, n in dag.nodes.items() if not n.parents]
        input_manager.assign_root_nodes(root_nodes)
        
        # Print stats
        print(f"\n{cfg_info['desc']}:")
        print(f"  Total nodes: {len(dag.nodes)}")
        print(f"  Root nodes: {len(root_nodes)}")
        print(f"  - Noise: {len(input_manager.noise_node_ids or [])}")
        print(f"  - Time: {len(input_manager.time_node_ids or [])}")
        print(f"  - State: {len(input_manager.state_node_ids or [])}")
        print(f"  Edges: {sum(len(n.parents) for n in dag.nodes.values())}")
        
        # Count transformation types
        t_counts = {}
        for t in transformations.values():
            t_name = t.__class__.__name__
            t_counts[t_name] = t_counts.get(t_name, 0) + 1
        print(f"  Transformations: {t_counts}")
        
        # Visualize
        title = f"{cfg_info['desc']}\nNoise:{len(input_manager.noise_node_ids or [])} Time:{len(input_manager.time_node_ids or [])} State:{len(input_manager.state_node_ids or [])}"
        visualize_dag(dag, input_manager, transformations, title=title, ax=axes[idx])
    
    plt.tight_layout()
    
    output_path = os.path.join(os.path.dirname(__file__), 'dag_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n\nVisualization saved to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    main()

