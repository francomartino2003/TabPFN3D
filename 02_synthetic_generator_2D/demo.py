"""
Demo script for the synthetic dataset generator.

This script demonstrates the main functionality of the generator
and provides examples of how to use it.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Import from parent directory
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from __init__ import (
    SyntheticDatasetGenerator,
    PriorConfig,
    DatasetConfig,
)
from dag_builder import DAGBuilder


def demo_basic_generation():
    """Demonstrate basic dataset generation."""
    print("=" * 60)
    print("DEMO 1: Basic Dataset Generation")
    print("=" * 60)
    
    # Create generator with default prior
    generator = SyntheticDatasetGenerator(seed=42, verbose=True)
    
    # Generate a single dataset
    dataset = generator.generate()
    
    print(f"\nGenerated dataset:")
    print(f"  X shape: {dataset.X.shape}")
    print(f"  y shape: {dataset.y.shape}")
    print(f"  Task: {'Classification' if dataset.is_classification else 'Regression'}")
    if dataset.is_classification:
        print(f"  Number of classes: {dataset.n_classes}")
    print(f"  Feature names: {dataset.feature_names[:5]}...")
    
    # Check for missing values
    n_missing = np.isnan(dataset.X).sum()
    if n_missing > 0:
        print(f"  Missing values: {n_missing} ({100 * n_missing / dataset.X.size:.1f}%)")
    
    return dataset


def demo_custom_prior():
    """Demonstrate generation with custom prior configuration."""
    print("\n" + "=" * 60)
    print("DEMO 2: Custom Prior Configuration")
    print("=" * 60)
    
    # Create a custom prior for small datasets
    small_prior = PriorConfig(
        n_rows_range=(50, 200),
        n_features_range=(5, 15),
        n_nodes_range=(10, 30),
        prob_classification=0.7,  # More classification tasks
        prob_missing_values=0.0,  # No missing values
        prob_warping=0.0,  # No warping
    )
    
    generator = SyntheticDatasetGenerator(prior=small_prior, seed=123)
    
    print("Generating 5 small datasets...")
    for i, dataset in enumerate(generator.generate_many(5)):
        task = 'C' if dataset.is_classification else 'R'
        print(f"  Dataset {i+1}: {dataset.X.shape[0]:4d} rows, {dataset.X.shape[1]:2d} features, task={task}")
    
    return generator


def demo_dag_visualization():
    """Demonstrate DAG construction and visualization."""
    print("\n" + "=" * 60)
    print("DEMO 3: DAG Structure")
    print("=" * 60)
    
    # Create a simple config
    prior = PriorConfig()
    rng = np.random.default_rng(42)
    config = prior.sample_hyperparams(rng)
    
    # Build DAG
    dag_builder = DAGBuilder(config, rng)
    dag = dag_builder.build()
    
    print(f"DAG Statistics:")
    print(f"  Number of nodes: {len(dag.nodes)}")
    print(f"  Number of edges: {len(dag.edges)}")
    print(f"  Number of subgraphs: {dag.n_subgraphs}")
    print(f"  Root nodes: {dag.root_nodes[:5]}..." if len(dag.root_nodes) > 5 else f"  Root nodes: {dag.root_nodes}")
    
    # Analyze node degrees
    in_degrees = [len(node.parents) for node in dag.nodes.values()]
    out_degrees = [len(node.children) for node in dag.nodes.values()]
    
    print(f"\n  In-degree stats: min={min(in_degrees)}, max={max(in_degrees)}, mean={np.mean(in_degrees):.1f}")
    print(f"  Out-degree stats: min={min(out_degrees)}, max={max(out_degrees)}, mean={np.mean(out_degrees):.1f}")
    
    # Show first few nodes
    print("\n  First 5 nodes:")
    for node_id in list(dag.nodes.keys())[:5]:
        node = dag.nodes[node_id]
        print(f"    Node {node_id}: parents={node.parents}, children={node.children[:3]}...")
    
    return dag


def demo_dataset_statistics():
    """Demonstrate statistics across multiple generated datasets."""
    print("\n" + "=" * 60)
    print("DEMO 4: Statistics Across Datasets")
    print("=" * 60)
    
    generator = SyntheticDatasetGenerator(seed=42)
    
    n_datasets = 50
    stats = {
        'n_rows': [],
        'n_features': [],
        'n_relevant': [],
        'n_irrelevant': [],
        'is_classification': [],
        'missing_rate': [],
    }
    
    print(f"Generating {n_datasets} datasets...")
    
    for dataset in generator.generate_many(n_datasets):
        stats['n_rows'].append(dataset.X.shape[0])
        stats['n_features'].append(dataset.X.shape[1])
        stats['n_relevant'].append(dataset.metadata['n_relevant_features'])
        stats['n_irrelevant'].append(dataset.metadata['n_irrelevant_features'])
        stats['is_classification'].append(int(dataset.is_classification))
        stats['missing_rate'].append(np.isnan(dataset.X).mean())
    
    print(f"\nStatistics over {n_datasets} datasets:")
    print(f"  Rows: min={min(stats['n_rows'])}, max={max(stats['n_rows'])}, mean={np.mean(stats['n_rows']):.0f}")
    print(f"  Features: min={min(stats['n_features'])}, max={max(stats['n_features'])}, mean={np.mean(stats['n_features']):.0f}")
    print(f"  Relevant features: mean={np.mean(stats['n_relevant']):.1f}")
    print(f"  Irrelevant features: mean={np.mean(stats['n_irrelevant']):.1f}")
    print(f"  Classification tasks: {sum(stats['is_classification'])}/{n_datasets} ({100*np.mean(stats['is_classification']):.0f}%)")
    print(f"  Missing rate: mean={100*np.mean(stats['missing_rate']):.1f}%")
    
    return stats


def demo_visualization():
    """Visualize generated datasets."""
    print("\n" + "=" * 60)
    print("DEMO 5: Dataset Visualization")
    print("=" * 60)
    
    generator = SyntheticDatasetGenerator(seed=42)
    
    # Generate datasets with specific characteristics
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i in range(6):
        ax = axes[i // 3, i % 3]
        dataset = generator.generate()
        
        # Handle NaN for plotting
        X = dataset.X.copy()
        X = np.nan_to_num(X, nan=0)
        
        # Use first two features for 2D scatter
        if X.shape[1] >= 2:
            x1, x2 = X[:, 0], X[:, 1]
        else:
            x1 = X[:, 0]
            x2 = np.zeros_like(x1)
        
        if dataset.is_classification:
            scatter = ax.scatter(x1, x2, c=dataset.y, cmap='viridis', alpha=0.6, s=10)
            ax.set_title(f"Classification ({dataset.n_classes} classes)\n{dataset.X.shape[0]} rows, {dataset.X.shape[1]} features")
        else:
            scatter = ax.scatter(x1, x2, c=dataset.y, cmap='RdYlBu', alpha=0.6, s=10)
            ax.set_title(f"Regression\n{dataset.X.shape[0]} rows, {dataset.X.shape[1]} features")
        
        ax.set_xlabel(dataset.feature_names[0] if dataset.feature_names else "Feature 0")
        ax.set_ylabel(dataset.feature_names[1] if len(dataset.feature_names) > 1 else "Feature 1")
        plt.colorbar(scatter, ax=ax, label='Target')
    
    plt.tight_layout()
    plt.savefig("synthetic_datasets_demo.png", dpi=150, bbox_inches='tight')
    print("Saved visualization to 'synthetic_datasets_demo.png'")
    plt.close()
    
    return fig


def demo_transformation_variety():
    """Show the variety of transformations applied."""
    print("\n" + "=" * 60)
    print("DEMO 6: Transformation Variety")
    print("=" * 60)
    
    generator = SyntheticDatasetGenerator(seed=42)
    
    transform_counts = {
        'warping': 0,
        'quantization': 0,
        'missing_values': 0,
    }
    
    n_datasets = 100
    for dataset in generator.generate_many(n_datasets):
        post = dataset.metadata['post_processing']
        for key in transform_counts:
            if post.get(key, False):
                transform_counts[key] += 1
    
    print(f"Transformation application rates ({n_datasets} datasets):")
    for key, count in transform_counts.items():
        print(f"  {key}: {count}/{n_datasets} ({100*count/n_datasets:.0f}%)")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("SYNTHETIC DATASET GENERATOR - DEMO")
    print("TabPFN-style Causal DAG Generator")
    print("=" * 60 + "\n")
    
    # Run demos
    dataset = demo_basic_generation()
    demo_custom_prior()
    demo_dag_visualization()
    demo_dataset_statistics()
    demo_transformation_variety()
    
    try:
        demo_visualization()
    except Exception as e:
        print(f"Visualization demo skipped (matplotlib not available or error): {e}")
    
    print("\n" + "=" * 60)
    print("ALL DEMOS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()

