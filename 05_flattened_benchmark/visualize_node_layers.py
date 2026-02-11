#!/usr/bin/env python3
"""
Visualize time series for ALL nodes in DAGs.

Uses the normal generator (like changes_test) - no hardcoded structure.
Shows 5 samples per node to understand how values propagate.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '03_synthetic_generator_3D'))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from generator import SyntheticDatasetGenerator3D
from config import PriorConfig3D

def create_test_prior():
    """Create prior for testing."""
    return PriorConfig3D(
        # Small, simple structure for visualization
        max_samples=100,
        max_features=5,
        max_t_subseq=100,
        max_classes=5,
        
        n_samples_range=(50, 100),
        prob_univariate=0.5,
        n_features_range=(2, 4),
        
        T_total_range=(100, 200),
        t_subseq_range=(50, 100),
        
        n_nodes_range=(10, 15),
        density_range=(0.3, 0.5),
        
        # Sample mode
        prob_iid_mode=1.0,  # Use IID for cleaner visualization
        
        # Low noise
        noise_scale_range=(0.001, 0.01),
        prob_low_noise_dataset=0.8,
        
        force_classification=True,
        max_target_offset=5,
    )


def visualize_node_timeseries(
    propagated_values,
    node_id: int,
    node_type: str,
    n_samples: int = 5
):
    """Visualize time series for a specific node."""
    # Get timeseries for this node
    node_series = propagated_values.get_node_timeseries(node_id)  # Shape: (n_samples, T)
    T = node_series.shape[1]
    
    # Create figure
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 2 * n_samples))
    if n_samples == 1:
        axes = [axes]
    
    for i in range(min(n_samples, node_series.shape[0])):
        ax = axes[i]
        series = node_series[i, :]
        
        ax.plot(series, linewidth=1.5, alpha=0.8)
        ax.set_title(f'Node {node_id:03d} ({node_type:8s}) - Sample {i}', fontsize=10)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def visualize_all_nodes(graph_id, generator):
    """Visualize ALL nodes in a generated dataset."""
    print(f"\n{'='*60}")
    print(f"GRAPH {graph_id}")
    print(f"{'='*60}")
    
    # Generate dataset using normal generator
    dataset = generator.generate()
    config = dataset.config
    
    print(f"Config:")
    print(f"  Nodes: {config.n_nodes}")
    print(f"  T_total: {config.T_total}, t_subseq: {config.t_subseq}")
    print(f"  Shape: {dataset.shape}")
    
    # We need to propagate again to get all node values
    # Use the generator's internal components
    dataset_rng = np.random.default_rng(config.seed)
    
    # Build DAG
    from dag_utils import DAGBuilder
    dag_builder = DAGBuilder(config, dataset_rng)
    dag = dag_builder.build()
    
    # Select target
    from feature_selector import FeatureSelector3D
    selector = FeatureSelector3D(dag, {}, config, None, dataset_rng)
    target_node = selector.select_target_only()
    
    # Create input manager and propagator
    from temporal_inputs import TemporalInputManager
    from temporal_propagator import BatchTemporalPropagator
    input_manager = TemporalInputManager.from_config(config, dataset_rng)
    
    # Build transformations
    from dag_utils import TransformationFactory
    transform_factory = TransformationFactory(config, dataset_rng)
    transformations = {}
    transformation_types = {}  # Store transformation type info
    
    # Check if we have 2-layer structure
    n_roots = config.n_time_inputs + config.n_state_inputs
    n_combination = getattr(config, 'n_combination_nodes', 0)
    if n_combination > 0:
        combination_node_ids = set(range(n_roots, n_roots + n_combination))
    else:
        combination_node_ids = set()
    
    for node_id, node in dag.nodes.items():
        if node.parents:
            n_parents = len(node.parents)
            if node_id in combination_node_ids:
                transform = transform_factory._create_nn(n_parents, config.noise_scale, force_smooth=True)
                transformation_types[node_id] = "NN (smooth)"
            else:
                transform = transform_factory.create(n_parents)
                # Get transformation type
                if hasattr(transform, 'activation'):
                    transformation_types[node_id] = f"NN ({transform.activation})"
                elif hasattr(transform, 'depth'):
                    transformation_types[node_id] = f"Tree (depth={transform.depth})"
                elif hasattr(transform, 'n_clusters'):
                    transformation_types[node_id] = f"Discretization (k={transform.n_clusters})"
                else:
                    transformation_types[node_id] = "Unknown"
            transformations[node_id] = transform
    
    # Create propagator
    propagator = BatchTemporalPropagator(
        config, dag, transformations, input_manager, dataset_rng, target_node=target_node
    )
    
    # Assign root nodes
    root_ids = [nid for nid, node in dag.nodes.items() if not node.parents]
    non_root_ids = [nid for nid, node in dag.nodes.items() if node.parents]
    input_manager.assign_root_nodes(root_ids, non_root_ids, dag=dag, target_node=target_node)
    
    # Generate propagated values
    n_samples_viz = 5
    propagated = propagator.generate_iid_sequences(n_samples_viz, config.T_total)
    
    # Identify node types
    root_ids_set = set(root_ids)
    time_ids = set(input_manager.time_node_ids or [])
    state_ids = set(input_manager.state_node_ids or [])
    
    # Visualize ALL nodes
    output_dir = Path(__file__).parent / 'results' / 'node_layers'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_node_ids = sorted(dag.nodes.keys())
    print(f"\nVisualizing {len(all_node_ids)} nodes...")
    
    for node_id in all_node_ids:
        node = dag.nodes[node_id]
        is_root = node_id in root_ids_set
        is_time = node_id in time_ids
        is_state = node_id in state_ids
        
        node_type = "TIME" if is_time else ("STATE" if is_state else ("ROOT" if is_root else "REGULAR"))
        transform_type = transformation_types.get(node_id, "N/A")
        
        print(f"  Node {node_id:03d} ({node_type:8s}, {transform_type:20s}): {len(node.parents)} parents, {len(node.children)} children")
        
        # Visualize this node
        fig = visualize_node_timeseries(propagated, node_id, node_type, n_samples_viz)
        
        filename = f'graph{graph_id:02d}_node{node_id:03d}.png'
        filepath = output_dir / filename
        fig.savefig(filepath, dpi=120, bbox_inches='tight')
        plt.close(fig)
    
    print(f"\nVisualizations saved to: {output_dir}")
    return config, dag


def main():
    print("="*60)
    print("VISUALIZING ALL NODES IN 5 DAGs")
    print("="*60)
    
    prior = create_test_prior()
    
    # Generate 5 graphs with different seeds
    seeds = [42, 123, 456, 789, 999]
    for graph_id in range(5):
        try:
            # Use different seed for each graph
            generator = SyntheticDatasetGenerator3D(prior=prior, seed=seeds[graph_id])
            config, dag = visualize_all_nodes(graph_id, generator)
        except Exception as e:
            print(f"Error generating graph {graph_id}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)


if __name__ == '__main__':
    main()
