#!/usr/bin/env python3
"""
Analyze propagation step-by-step to understand why synthetic data
doesn't look like real data (smooth, seasonal, etc.).

Tracks:
- Value statistics per node per timestep
- Clipping events
- Activation compression
- Clustering issues (same cluster always)
- Evolution of patterns
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '03_synthetic_generator_3D'))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from generator import SyntheticDatasetGenerator3D
from config import PriorConfig3D, DatasetConfig3D
from dag_utils import DAGBuilder, TransformationFactory
from temporal_inputs import TemporalInputManager
from temporal_propagator import BatchTemporalPropagator
from feature_selector import FeatureSelector3D


@dataclass
class TimestepStats:
    """Statistics for a single timestep."""
    t: int
    node_stats: Dict[int, Dict[str, float]] = field(default_factory=dict)  # node_id -> {min, max, mean, std}
    clipping_events: Dict[int, Tuple[float, float]] = field(default_factory=dict)  # node_id -> (before_clip, after_clip)
    activation_outputs: Dict[int, Dict[str, float]] = field(default_factory=dict)  # node_id -> {activation_type, compression_ratio}
    clustering_issues: Dict[int, int] = field(default_factory=dict)  # node_id -> n_unique_clusters


@dataclass
class PropagationAnalysis:
    """Full analysis of propagation."""
    config: any
    dag: any
    timestep_stats: List[TimestepStats] = field(default_factory=list)
    node_types: Dict[int, str] = field(default_factory=dict)
    transformations: Dict[int, str] = field(default_factory=dict)


def create_test_prior():
    """Create prior for testing."""
    return PriorConfig3D(
        max_samples=10,  # Small for analysis
        max_features=5,
        max_t_subseq=100,
        max_classes=5,
        
        n_samples_range=(5, 10),
        prob_univariate=0.5,
        n_features_range=(2, 4),
        
        T_total_range=(80, 120),
        t_subseq_range=(50, 80),
        
        n_nodes_range=(10, 15),
        density_range=(0.3, 0.5),
        
        prob_iid_mode=1.0,
        noise_scale_range=(0.001, 0.01),
        prob_low_noise_dataset=0.8,
        
        force_classification=True,
        max_target_offset=5,
    )


def analyze_propagation_step_by_step(
    generator: SyntheticDatasetGenerator3D,
    graph_id: int
) -> PropagationAnalysis:
    """Analyze propagation step by step for a single DAG."""
    print(f"\n{'='*60}")
    print(f"ANALYZING GRAPH {graph_id}")
    print(f"{'='*60}")
    
    # Generate config
    prior = generator.prior
    config = DatasetConfig3D.sample_from_prior(prior, generator.rng)
    dataset_rng = np.random.default_rng(config.seed)
    
    print(f"Config: {config.n_nodes} nodes, T_total={config.T_total}, t_subseq={config.t_subseq}")
    
    # Build DAG
    dag_builder = DAGBuilder(config, dataset_rng)
    dag = dag_builder.build()
    
    # Select target
    selector = FeatureSelector3D(dag, {}, config, None, dataset_rng)
    target_node = selector.select_target_only()
    
    # Create input manager
    input_manager = TemporalInputManager.from_config(config, dataset_rng)
    
    # Build transformations
    transform_factory = TransformationFactory(config, dataset_rng)
    transformations = {}
    transformation_types = {}
    
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
    
    # Identify node types
    node_types = {}
    root_ids_set = set(root_ids)
    time_ids = set(input_manager.time_node_ids or [])
    state_ids = set(input_manager.state_node_ids or [])
    
    for node_id in dag.nodes.keys():
        if node_id in time_ids:
            node_types[node_id] = "TIME"
        elif node_id in state_ids:
            node_types[node_id] = "STATE"
        elif node_id in root_ids_set:
            node_types[node_id] = "ROOT"
        else:
            node_types[node_id] = "REGULAR"
    
    # Now do step-by-step propagation with tracking
    n_samples = 5
    T = config.T_total
    
    # Pre-compute topological order
    topo_order = propagator.topo_order
    node_to_idx = propagator.node_to_idx
    parent_indices = propagator.parent_indices
    root_node_ids = propagator.root_node_ids
    
    # Storage for analysis
    values = np.zeros((len(topo_order), n_samples, T), dtype=np.float32)
    history: Dict[Tuple[int, int], np.ndarray] = {}
    timestep_stats_list = []
    
    print(f"\nPropagating {T} timesteps with {n_samples} samples...")
    
    for t in range(T):
        stats = TimestepStats(t=t)
        
        # Generate root inputs
        root_inputs = input_manager.generate_inputs_for_timestep(
            t=t, T=T, n_samples=n_samples, history=history
        )
        
        # Process each node in topological order
        for node_id in topo_order:
            idx = node_to_idx[node_id]
            
            if node_id in root_node_ids:
                if node_id in root_inputs:
                    values[idx, :, t] = root_inputs[node_id]
                else:
                    values[idx, :, t] = dataset_rng.normal(0, 1, size=n_samples)
            else:
                # Non-root: compute from parents
                parent_idxs = parent_indices[node_id]
                parent_vals = values[parent_idxs, :, t].T  # (n_samples, n_parents)
                
                # Track parent stats before transformation
                parent_mean = np.mean(np.abs(parent_vals))
                parent_std = np.std(parent_vals)
                
                if node_id in transformations:
                    transform = transformations[node_id]
                    result = transform.forward(parent_vals)
                    if result.ndim > 1:
                        result = result[:, 0] if result.shape[1] > 0 else result.flatten()
                    
                    # Track transformation output
                    result_mean = np.mean(np.abs(result))
                    result_std = np.std(result)
                    
                    # Check for clustering issues (if discretization)
                    if hasattr(transform, 'n_clusters'):
                        unique_vals = len(np.unique(result))
                        if unique_vals < transform.n_clusters:
                            stats.clustering_issues[node_id] = unique_vals
                    
                    # Track compression
                    if parent_mean > 1e-8:
                        compression = result_mean / parent_mean
                        stats.activation_outputs[node_id] = {
                            'type': transformation_types[node_id],
                            'compression': compression,
                            'parent_mean': parent_mean,
                            'result_mean': result_mean,
                        }
                    
                    values[idx, :, t] = result
                else:
                    # Fallback
                    weights = dataset_rng.normal(0, 1, size=len(parent_idxs))
                    weights = weights / (np.linalg.norm(weights) + 1e-8)
                    values[idx, :, t] = parent_vals @ weights
            
            # Track node statistics
            node_values = values[idx, :, t]
            node_min = np.min(node_values)
            node_max = np.max(node_values)
            node_mean = np.mean(node_values)
            node_std = np.std(node_values)
            
            stats.node_stats[node_id] = {
                'min': node_min,
                'max': node_max,
                'mean': node_mean,
                'std': node_std,
                'range': node_max - node_min,
            }
            
            # Check for clipping
            before_clip = node_values.copy()
            node_values = np.clip(node_values, -1000, 1000)
            node_values = np.nan_to_num(node_values, nan=0.0, posinf=1000, neginf=-1000)
            
            if np.any(before_clip != node_values):
                clipped_min = np.min(before_clip[before_clip < -1000]) if np.any(before_clip < -1000) else None
                clipped_max = np.max(before_clip[before_clip > 1000]) if np.any(before_clip > 1000) else None
                stats.clipping_events[node_id] = (
                    clipped_min if clipped_min is not None else -1000,
                    clipped_max if clipped_max is not None else 1000
                )
            
            values[idx, :, t] = node_values
            
            # Store in history
            history[(t, node_id)] = node_values.copy()
        
        timestep_stats_list.append(stats)
        
        # Print progress every 10 timesteps
        if (t + 1) % 10 == 0 or t == 0:
            print(f"  Timestep {t+1}/{T}: ", end="")
            n_clipped = len(stats.clipping_events)
            n_clustered = len(stats.clustering_issues)
            if n_clipped > 0:
                print(f"{n_clipped} clipping events, ", end="")
            if n_clustered > 0:
                print(f"{n_clustered} clustering issues, ", end="")
            print()
    
    analysis = PropagationAnalysis(
        config=config,
        dag=dag,
        timestep_stats=timestep_stats_list,
        node_types=node_types,
        transformations=transformation_types
    )
    
    return analysis


def visualize_analysis(analysis: PropagationAnalysis, graph_id: int, output_dir: Path):
    """Create visualizations of the analysis."""
    print(f"\nCreating visualizations for graph {graph_id}...")
    
    # 1. Value evolution over time for each node
    n_nodes = len(analysis.dag.nodes)
    T = len(analysis.timestep_stats)
    
    # Get node values over time
    node_means = np.zeros((n_nodes, T))
    node_stds = np.zeros((n_nodes, T))
    node_ranges = np.zeros((n_nodes, T))
    clipping_mask = np.zeros((n_nodes, T), dtype=bool)
    
    node_ids = sorted(analysis.dag.nodes.keys())
    node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    
    for t, stats in enumerate(analysis.timestep_stats):
        for node_id, node_stat in stats.node_stats.items():
            idx = node_id_to_idx[node_id]
            node_means[idx, t] = node_stat['mean']
            node_stds[idx, t] = node_stat['std']
            node_ranges[idx, t] = node_stat['range']
        
        for node_id in stats.clipping_events:
            idx = node_id_to_idx[node_id]
            clipping_mask[idx, t] = True
    
    # Plot 1: Mean values over time (all nodes)
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # Mean values
    ax = axes[0]
    for i, node_id in enumerate(node_ids):
        node_type = analysis.node_types.get(node_id, "UNKNOWN")
        color = 'red' if node_type == 'TIME' else ('blue' if node_type == 'STATE' else 'green')
        ax.plot(node_means[i, :], label=f'Node {node_id:03d} ({node_type})', 
                color=color, alpha=0.6, linewidth=1)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Mean Value')
    ax.set_title('Mean Value Evolution Over Time (All Nodes)')
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Standard deviation
    ax = axes[1]
    for i, node_id in enumerate(node_ids):
        node_type = analysis.node_types.get(node_id, "UNKNOWN")
        color = 'red' if node_type == 'TIME' else ('blue' if node_type == 'STATE' else 'green')
        ax.plot(node_stds[i, :], label=f'Node {node_id:03d} ({node_type})', 
                color=color, alpha=0.6, linewidth=1)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Std Dev')
    ax.set_title('Standard Deviation Evolution Over Time')
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Range
    ax = axes[2]
    for i, node_id in enumerate(node_ids):
        node_type = analysis.node_types.get(node_id, "UNKNOWN")
        color = 'red' if node_type == 'TIME' else ('blue' if node_type == 'STATE' else 'green')
        ax.plot(node_ranges[i, :], label=f'Node {node_id:03d} ({node_type})', 
                color=color, alpha=0.6, linewidth=1)
        # Mark clipping events
        if np.any(clipping_mask[i, :]):
            clipped_times = np.where(clipping_mask[i, :])[0]
            ax.scatter(clipped_times, node_ranges[i, clipped_times], 
                      color='red', marker='x', s=50, zorder=10)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Value Range (max - min)')
    ax.set_title('Value Range Over Time (X = clipping event)')
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / f'graph{graph_id:02d}_evolution.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Plot 2: Clipping events summary
    total_clips = defaultdict(int)
    for stats in analysis.timestep_stats:
        for node_id in stats.clipping_events:
            total_clips[node_id] += 1
    
    if total_clips:
        fig, ax = plt.subplots(figsize=(12, 6))
        node_ids_clipped = sorted(total_clips.keys())
        clip_counts = [total_clips[nid] for nid in node_ids_clipped]
        ax.bar(range(len(node_ids_clipped)), clip_counts)
        ax.set_xticks(range(len(node_ids_clipped)))
        ax.set_xticklabels([f'Node {nid:03d}' for nid in node_ids_clipped], rotation=45)
        ax.set_ylabel('Number of Clipping Events')
        ax.set_title('Clipping Events per Node')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        fig.savefig(output_dir / f'graph{graph_id:02d}_clipping.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # Plot 3: Activation compression analysis
    compression_data = defaultdict(list)
    for stats in analysis.timestep_stats:
        for node_id, info in stats.activation_outputs.items():
            compression_data[node_id].append(info['compression'])
    
    if compression_data:
        fig, ax = plt.subplots(figsize=(12, 6))
        node_ids_compressed = sorted(compression_data.keys())
        compressions = [np.mean(compression_data[nid]) for nid in node_ids_compressed]
        ax.bar(range(len(node_ids_compressed)), compressions)
        ax.set_xticks(range(len(node_ids_compressed)))
        ax.set_xticklabels([f'Node {nid:03d}\n({analysis.transformations.get(nid, "?")})' 
                           for nid in node_ids_compressed], rotation=45, ha='right')
        ax.set_ylabel('Mean Compression Ratio (output/input)')
        ax.set_title('Activation Compression per Node (<1 = compresses, >1 = expands)')
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='No compression')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        fig.savefig(output_dir / f'graph{graph_id:02d}_compression.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # Plot 4: Clustering issues
    clustering_issues = defaultdict(int)
    for stats in analysis.timestep_stats:
        for node_id, n_unique in stats.clustering_issues.items():
            clustering_issues[node_id] = max(clustering_issues[node_id], n_unique)
    
    if clustering_issues:
        fig, ax = plt.subplots(figsize=(12, 6))
        node_ids_clustered = sorted(clustering_issues.keys())
        n_unique_vals = [clustering_issues[nid] for nid in node_ids_clustered]
        ax.bar(range(len(node_ids_clustered)), n_unique_vals)
        ax.set_xticks(range(len(node_ids_clustered)))
        ax.set_xticklabels([f'Node {nid:03d}' for nid in node_ids_clustered], rotation=45)
        ax.set_ylabel('Number of Unique Clusters')
        ax.set_title('Clustering Issues (should match n_clusters)')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        fig.savefig(output_dir / f'graph{graph_id:02d}_clustering.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"  Saved visualizations to {output_dir}")


def print_summary(analysis: PropagationAnalysis, graph_id: int):
    """Print summary statistics."""
    print(f"\n{'='*60}")
    print(f"SUMMARY FOR GRAPH {graph_id}")
    print(f"{'='*60}")
    
    # Count clipping events
    total_clips = defaultdict(int)
    for stats in analysis.timestep_stats:
        for node_id in stats.clipping_events:
            total_clips[node_id] += 1
    
    if total_clips:
        print(f"\nClipping Events:")
        for node_id in sorted(total_clips.keys()):
            node_type = analysis.node_types.get(node_id, "UNKNOWN")
            transform = analysis.transformations.get(node_id, "?")
            print(f"  Node {node_id:03d} ({node_type:8s}, {transform:20s}): {total_clips[node_id]} events")
    
    # Compression statistics
    compression_by_node = defaultdict(list)
    for stats in analysis.timestep_stats:
        for node_id, info in stats.activation_outputs.items():
            compression_by_node[node_id].append(info['compression'])
    
    if compression_by_node:
        print(f"\nActivation Compression (mean):")
        for node_id in sorted(compression_by_node.keys()):
            compressions = compression_by_node[node_id]
            mean_comp = np.mean(compressions)
            node_type = analysis.node_types.get(node_id, "UNKNOWN")
            transform = analysis.transformations.get(node_id, "?")
            print(f"  Node {node_id:03d} ({node_type:8s}, {transform:20s}): {mean_comp:.4f}")
    
    # Clustering issues
    clustering_issues = defaultdict(int)
    for stats in analysis.timestep_stats:
        for node_id, n_unique in stats.clustering_issues.items():
            clustering_issues[node_id] = max(clustering_issues[node_id], n_unique)
    
    if clustering_issues:
        print(f"\nClustering Issues:")
        for node_id in sorted(clustering_issues.keys()):
            n_unique = clustering_issues[node_id]
            transform = analysis.transformations.get(node_id, "?")
            print(f"  Node {node_id:03d} ({transform:20s}): only {n_unique} unique values")
    
    # Value convergence
    print(f"\nValue Convergence Analysis:")
    node_ids = sorted(analysis.dag.nodes.keys())
    for node_id in node_ids[:5]:  # First 5 nodes
        node_type = analysis.node_types.get(node_id, "UNKNOWN")
        # Get mean values at start and end
        start_mean = analysis.timestep_stats[0].node_stats[node_id]['mean']
        end_mean = analysis.timestep_stats[-1].node_stats[node_id]['mean']
        start_std = analysis.timestep_stats[0].node_stats[node_id]['std']
        end_std = analysis.timestep_stats[-1].node_stats[node_id]['std']
        
        print(f"  Node {node_id:03d} ({node_type:8s}): "
              f"mean {start_mean:.3f}→{end_mean:.3f}, "
              f"std {start_std:.3f}→{end_std:.3f}")


def main():
    print("="*60)
    print("PROPAGATION ANALYSIS")
    print("="*60)
    
    prior = create_test_prior()
    output_dir = Path(__file__).parent / 'results' / 'propagation_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze 2 graphs
    for graph_id in range(2):
        try:
            generator = SyntheticDatasetGenerator3D(prior=prior, seed=42 + graph_id * 100)
            analysis = analyze_propagation_step_by_step(generator, graph_id)
            visualize_analysis(analysis, graph_id, output_dir)
            print_summary(analysis, graph_id)
        except Exception as e:
            print(f"Error analyzing graph {graph_id}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)


if __name__ == '__main__':
    main()
