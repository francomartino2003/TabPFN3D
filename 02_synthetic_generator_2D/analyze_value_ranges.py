"""
Analyze value ranges at different stages of the synthetic data generation pipeline.

This script shows:
1. Input values at root nodes
2. Values after each transformation layer
3. Final feature values
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

from config import PriorConfig, DatasetConfig
from dag_builder import DAGBuilder
from transformations import TransformationFactory
from row_generator import RowGenerator
from feature_selector import FeatureSelector, TableBuilder


def analyze_single_dataset(seed: int = 42, verbose: bool = True) -> Dict:
    """Generate one dataset and analyze value ranges at each stage."""
    
    rng = np.random.default_rng(seed)
    prior = PriorConfig()
    config = DatasetConfig.sample_from_prior(prior, rng)
    
    if verbose:
        print("="*70)
        print("DATASET CONFIGURATION")
        print("="*70)
        print(f"  n_rows: {config.n_rows}")
        print(f"  n_features: {config.n_features}")
        print(f"  n_nodes: {config.n_nodes}")
        print(f"  noise_type: {config.noise_type}")
        print(f"  init_sigma (for Normal): {config.init_sigma:.3f}")
        print(f"  init_a (for Uniform): {config.init_a:.3f}")
    
    # Build DAG
    dag_builder = DAGBuilder(config, rng)
    dag = dag_builder.build()
    
    if verbose:
        print(f"\n  DAG built with {len(dag.nodes)} nodes")
        root_nodes = [n for n in dag.nodes.values() if not n.parents]
        print(f"  Root nodes: {len(root_nodes)}")
    
    # Create transformations
    transform_factory = TransformationFactory(config, rng)
    transformations = {}
    transform_types = defaultdict(int)
    
    for node_id, node in dag.nodes.items():
        for parent_id in node.parents:
            transform = transform_factory.create(n_parents=len(node.parents))
            transformations[(parent_id, node_id)] = transform
            transform_types[type(transform).__name__] += 1
    
    if verbose:
        print(f"\n  Transformations created:")
        for t_type, count in sorted(transform_types.items()):
            print(f"    {t_type}: {count}")
    
    # Generate values
    row_generator = RowGenerator(config, dag, transformations, rng)
    propagated = row_generator.generate(config.n_rows)
    
    # Analyze values at each node
    if verbose:
        print("\n" + "="*70)
        print("VALUE RANGES BY NODE DEPTH")
        print("="*70)
    
    # Calculate node depths
    def get_depth(node_id: int, memo: Dict = {}) -> int:
        if node_id in memo:
            return memo[node_id]
        node = dag.nodes[node_id]
        if not node.parents:
            memo[node_id] = 0
        else:
            memo[node_id] = 1 + max(get_depth(p, memo) for p in node.parents)
        return memo[node_id]
    
    depths = {nid: get_depth(nid) for nid in dag.nodes}
    max_depth = max(depths.values())
    
    # Group nodes by depth
    nodes_by_depth = defaultdict(list)
    for nid, depth in depths.items():
        nodes_by_depth[depth].append(nid)
    
    # Analyze values at each depth
    depth_stats = {}
    
    for depth in range(max_depth + 1):
        nodes_at_depth = nodes_by_depth[depth]
        all_values = []
        
        for nid in nodes_at_depth:
            if nid in propagated.values:
                all_values.extend(propagated.values[nid].flatten())
        
        if all_values:
            arr = np.array(all_values)
            stats = {
                'n_nodes': len(nodes_at_depth),
                'n_values': len(all_values),
                'min': float(np.min(arr)),
                'p1': float(np.percentile(arr, 1)),
                'p25': float(np.percentile(arr, 25)),
                'median': float(np.median(arr)),
                'p75': float(np.percentile(arr, 75)),
                'p99': float(np.percentile(arr, 99)),
                'max': float(np.max(arr)),
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'has_nan': bool(np.any(np.isnan(arr))),
                'has_inf': bool(np.any(np.isinf(arr)))
            }
            depth_stats[depth] = stats
            
            if verbose:
                label = "ROOT" if depth == 0 else f"Depth {depth}"
                print(f"\n{label} ({stats['n_nodes']} nodes, {stats['n_values']} values):")
                print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
                print(f"  Percentiles: p1={stats['p1']:.3f}, p25={stats['p25']:.3f}, "
                      f"median={stats['median']:.3f}, p75={stats['p75']:.3f}, p99={stats['p99']:.3f}")
                print(f"  Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
                if stats['has_nan'] or stats['has_inf']:
                    print(f"  WARNING: has_nan={stats['has_nan']}, has_inf={stats['has_inf']}")
    
    # Analyze by transformation type
    if verbose:
        print("\n" + "="*70)
        print("VALUE RANGES BY TRANSFORMATION TYPE")
        print("="*70)
    
    transform_output_values = defaultdict(list)
    
    for (parent_id, child_id), transform in transformations.items():
        if child_id in propagated.values:
            t_name = type(transform).__name__
            transform_output_values[t_name].extend(propagated.values[child_id].flatten())
    
    transform_stats = {}
    for t_name, values in transform_output_values.items():
        if values:
            arr = np.array(values)
            # Remove inf/nan for stats
            arr_clean = arr[np.isfinite(arr)]
            if len(arr_clean) > 0:
                stats = {
                    'n_values': len(values),
                    'n_finite': len(arr_clean),
                    'min': float(np.min(arr_clean)),
                    'max': float(np.max(arr_clean)),
                    'mean': float(np.mean(arr_clean)),
                    'std': float(np.std(arr_clean)),
                    'p1': float(np.percentile(arr_clean, 1)),
                    'p99': float(np.percentile(arr_clean, 99))
                }
                transform_stats[t_name] = stats
                
                if verbose:
                    print(f"\n{t_name} ({stats['n_values']} values):")
                    print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
                    print(f"  p1-p99: [{stats['p1']:.3f}, {stats['p99']:.3f}]")
                    print(f"  Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
    
    # Final feature values - use full generator
    if verbose:
        print("\n" + "="*70)
        print("FINAL FEATURE VALUES (X matrix)")
        print("="*70)
    
    from generator import SyntheticDatasetGenerator
    gen = SyntheticDatasetGenerator(seed=seed)
    dataset = gen.generate()
    X, y = dataset.X, dataset.y
    
    feature_stats = []
    for i in range(X.shape[1]):
        col = X[:, i]
        col_clean = col[np.isfinite(col)]
        if len(col_clean) > 0:
            stats = {
                'feature_idx': i,
                'min': float(np.min(col_clean)),
                'max': float(np.max(col_clean)),
                'mean': float(np.mean(col_clean)),
                'std': float(np.std(col_clean))
            }
            feature_stats.append(stats)
    
    if verbose and feature_stats:
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"\nPer-feature statistics (first 10):")
        print(f"  {'Idx':>4} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
        for s in feature_stats[:10]:
            print(f"  {s['feature_idx']:4d} {s['min']:10.3f} {s['max']:10.3f} "
                  f"{s['mean']:10.3f} {s['std']:10.3f}")
        
        # Overall X stats
        X_clean = X[np.isfinite(X)]
        print(f"\nOverall X matrix:")
        print(f"  Range: [{np.min(X_clean):.3f}, {np.max(X_clean):.3f}]")
        print(f"  Mean: {np.mean(X_clean):.3f}, Std: {np.std(X_clean):.3f}")
        print(f"  p1-p99: [{np.percentile(X_clean, 1):.3f}, {np.percentile(X_clean, 99):.3f}]")
    
    # Target values
    if verbose:
        print("\n" + "="*70)
        print("TARGET VALUES (y)")
        print("="*70)
        print(f"  Unique values: {np.unique(y)}")
        print(f"  Value counts:")
        unique, counts = np.unique(y, return_counts=True)
        for val, count in zip(unique, counts):
            print(f"    Class {int(val)}: {count} ({100*count/len(y):.1f}%)")
    
    return {
        'config': config,
        'depth_stats': depth_stats,
        'transform_stats': transform_stats,
        'feature_stats': feature_stats,
        'X_shape': X.shape,
        'y_unique': np.unique(y).tolist()
    }


def analyze_multiple_datasets(n_datasets: int = 10, seed: int = 42):
    """Analyze value ranges across multiple datasets."""
    
    print("="*70)
    print(f"ANALYZING {n_datasets} DATASETS")
    print("="*70)
    
    all_root_ranges = []
    all_final_ranges = []
    all_max_depths = []
    
    for i in range(n_datasets):
        result = analyze_single_dataset(seed=seed + i, verbose=False)
        
        if 0 in result['depth_stats']:
            root_stats = result['depth_stats'][0]
            all_root_ranges.append((root_stats['min'], root_stats['max']))
        
        max_depth = max(result['depth_stats'].keys())
        all_max_depths.append(max_depth)
        
        if max_depth in result['depth_stats']:
            final_stats = result['depth_stats'][max_depth]
            all_final_ranges.append((final_stats['min'], final_stats['max']))
    
    print("\nROOT NODE VALUE RANGES (across all datasets):")
    if all_root_ranges:
        mins = [r[0] for r in all_root_ranges]
        maxs = [r[1] for r in all_root_ranges]
        print(f"  Min values: [{min(mins):.3f}, {max(mins):.3f}]")
        print(f"  Max values: [{min(maxs):.3f}, {max(maxs):.3f}]")
    
    print(f"\nMAX DEPTH: {min(all_max_depths)} - {max(all_max_depths)}")
    
    print("\nDEEPEST NODE VALUE RANGES (across all datasets):")
    if all_final_ranges:
        mins = [r[0] for r in all_final_ranges]
        maxs = [r[1] for r in all_final_ranges]
        print(f"  Min values: [{min(mins):.3f}, {max(mins):.3f}]")
        print(f"  Max values: [{min(maxs):.3f}, {max(maxs):.3f}]")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze value ranges in synthetic data generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--multi", type=int, default=0, help="Analyze multiple datasets")
    args = parser.parse_args()
    
    if args.multi > 0:
        analyze_multiple_datasets(n_datasets=args.multi, seed=args.seed)
    else:
        analyze_single_dataset(seed=args.seed, verbose=True)

