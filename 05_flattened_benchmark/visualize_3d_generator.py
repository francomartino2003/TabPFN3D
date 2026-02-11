"""
Visualize outputs from the 3D Synthetic Generator.

Shows time series for each node across samples, similar to random_nn_generator visualizations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '03_synthetic_generator_3D'))

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from dataclasses import dataclass

# Import 3D generator components
from config import PriorConfig3D, DatasetConfig3D
from generator import SyntheticDatasetGenerator3D


def visualize_propagated_values(
    generator: SyntheticDatasetGenerator3D,
    config: DatasetConfig3D,
    n_samples: int = 5,
    output_dir: str = "./3d_generator_vis",
    dataset_id: int = 0,
    max_nodes_per_page: int = 6
):
    """
    Visualize propagated node values over time.
    
    Similar to random_nn_generator visualization but for 3D generator.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate dataset to get propagated values
    # We need to access the propagator directly
    dataset_rng = np.random.default_rng(config.seed)
    
    # Import components
    from dag_utils import DAGBuilder, TransformationFactory
    from temporal_inputs import TemporalInputManager
    from temporal_propagator import BatchTemporalPropagator
    
    # Build DAG
    dag_builder = DAGBuilder(config, dataset_rng)
    dag = dag_builder.build()
    
    # Create transformations
    transform_factory = TransformationFactory(config, dataset_rng)
    transformations = {}
    for node_id, node in dag.nodes.items():
        if len(node.parents) > 0:
            transformations[node_id] = transform_factory.create(len(node.parents))
    
    # Create input manager
    input_manager = TemporalInputManager.from_config(config, dataset_rng)
    
    # Create propagator
    propagator = BatchTemporalPropagator(
        config, dag, transformations, input_manager, dataset_rng
    )
    
    # Propagate
    T = min(config.T_total, 100)  # Limit for visualization
    propagated = propagator.propagate(n_samples, T)
    
    # Get values for each node
    # propagated.node_values is Dict[node_id, np.ndarray of shape (n_samples, T)]
    node_ids = sorted(propagated.node_values.keys())
    n_nodes = len(node_ids)
    
    # Save config summary
    with open(f"{output_dir}/dataset{dataset_id:02d}_config.txt", 'w') as f:
        f.write(f"Dataset {dataset_id}\n")
        f.write(f"=" * 40 + "\n")
        f.write(f"n_samples: {config.n_samples}\n")
        f.write(f"n_features: {config.n_features}\n")
        f.write(f"T_total: {config.T_total}\n")
        f.write(f"t_subseq: {config.t_subseq}\n")
        f.write(f"n_nodes: {config.n_nodes}\n")
        f.write(f"memory_dim: {config.memory_dim}\n")
        f.write(f"n_extra_time_inputs: {config.n_extra_time_inputs}\n")
        f.write(f"time_input_activations: {config.time_input_activations}\n")
        f.write(f"sample_mode: {config.sample_mode}\n")
        f.write(f"n_classes: {config.n_classes}\n")
        f.write(f"target_offset: {config.target_offset}\n")
        f.write(f"noise_scale: {config.noise_scale}\n")
    
    # Create visualizations - one page per group of nodes
    n_pages = (n_nodes + max_nodes_per_page - 1) // max_nodes_per_page
    
    for page in range(n_pages):
        start_node_idx = page * max_nodes_per_page
        end_node_idx = min(start_node_idx + max_nodes_per_page, n_nodes)
        nodes_this_page = end_node_idx - start_node_idx
        
        fig, axes = plt.subplots(n_samples, nodes_this_page,
                                figsize=(3 * nodes_this_page, 2.5 * n_samples))
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        if nodes_this_page == 1:
            axes = axes.reshape(-1, 1)
        
        title = f"Dataset {dataset_id} | Nodes {start_node_idx}-{end_node_idx-1}"
        if n_pages > 1:
            title += f" [page {page+1}/{n_pages}]"
        fig.suptitle(title, fontsize=12)
        
        for sample_idx in range(n_samples):
            for node_offset, node_idx in enumerate(range(start_node_idx, end_node_idx)):
                ax = axes[sample_idx, node_offset]
                node_id = node_ids[node_idx]
                
                # Get time series for this node and sample
                ts = propagated.node_values[node_id][sample_idx, :T]
                
                color = plt.cm.tab10(node_offset % 10)
                ax.plot(ts, color=color, linewidth=1)
                
                if sample_idx == 0:
                    # Check if this is a root node
                    node = dag.nodes[node_id]
                    if len(node.parents) == 0:
                        ax.set_title(f"Node {node_id} (root)", fontsize=9)
                    else:
                        ax.set_title(f"Node {node_id}", fontsize=9)
                if node_offset == 0:
                    ax.set_ylabel(f"S{sample_idx}", fontsize=9)
                if sample_idx == n_samples - 1:
                    ax.set_xlabel("Time", fontsize=8)
        
        plt.tight_layout()
        
        if n_pages > 1:
            filename = f"{output_dir}/dataset{dataset_id:02d}_page{page}.png"
        else:
            filename = f"{output_dir}/dataset{dataset_id:02d}.png"
        
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"  Dataset {dataset_id}: {n_nodes} nodes, {n_pages} pages saved")


def main():
    print("=" * 60)
    print("3D Generator Visualization")
    print("=" * 60)
    
    output_dir = "/Users/franco/Documents/TabPFN3D/05_flattened_benchmark/results/3d_generator_vis"
    
    # Create generator with default prior
    prior = PriorConfig3D()
    generator = SyntheticDatasetGenerator3D(prior, seed=42)
    
    n_datasets = 5
    print(f"\nGenerating {n_datasets} datasets...")
    
    for dataset_id in range(n_datasets):
        # Sample config
        rng = np.random.default_rng(42 + dataset_id)
        config = DatasetConfig3D.sample_from_prior(prior, rng)
        
        print(f"\n  Dataset {dataset_id}:")
        print(f"    n_samples={config.n_samples}, n_features={config.n_features}")
        print(f"    T_total={config.T_total}, t_subseq={config.t_subseq}")
        print(f"    n_nodes={config.n_nodes}, memory_dim={config.memory_dim}")
        print(f"    sample_mode={config.sample_mode}, n_classes={config.n_classes}")
        
        try:
            visualize_propagated_values(
                generator, config, 
                n_samples=5, 
                output_dir=output_dir,
                dataset_id=dataset_id
            )
        except Exception as e:
            print(f"    Error: {e}")
    
    print(f"\nDone! Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
