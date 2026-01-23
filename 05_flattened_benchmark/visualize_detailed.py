"""
Generate detailed visualizations of synthetic datasets with full DAG/config info.

Shows:
- Time series plots
- DAG structure information
- Node types (time, state, internal)
- State configurations (source nodes, lags, alpha)
- Target offset and feature distances
- Noise parameters
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
GENERATOR_3D_DIR = PROJECT_ROOT / "03_synthetic_generator_3D"
RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_DIR = RESULTS_DIR / "detailed_visualizations"

sys.path.insert(0, str(GENERATOR_3D_DIR))

# Visualization settings
N_SAMPLES_TO_PLOT = 5
FIG_WIDTH = 18
FIG_HEIGHT = 14
DPI = 150


def generate_dataset_with_full_info(generator, seed: int) -> Dict:
    """Generate a dataset and extract all configuration info."""
    from config import DatasetConfig3D
    
    # Generate dataset
    dataset = generator.generate()
    
    # Get the config that was used
    config = generator.last_config if hasattr(generator, 'last_config') else None
    
    # Extract metadata
    metadata = dataset.metadata
    
    # Build comprehensive info dict
    info = {
        'X': dataset.X,
        'y': dataset.y,
        'seed': seed,
        
        # Basic stats
        'n_samples': dataset.X.shape[0],
        'n_features': dataset.X.shape[1],
        'length': dataset.X.shape[2],
        'n_classes': len(np.unique(dataset.y)),
        'sample_mode': metadata.get('sample_mode', 'unknown'),
        
        # DAG structure
        'n_nodes': metadata.get('n_nodes', 0),
        'n_roots': metadata.get('n_roots', 0),
        'density': metadata.get('density', 0),
        
        # Root node types
        'n_time_inputs': metadata.get('n_time_inputs', 0),
        'n_state_inputs': metadata.get('n_state_inputs', 0),
        
        # Time input activations
        'time_activations': metadata.get('time_activations', []),
        
        # State configurations
        'state_configs': metadata.get('state_configs', []),
        'state_alpha': metadata.get('state_alpha', 0),
        
        # Target info
        'target_node': metadata.get('target_node', -1),
        'target_offset': metadata.get('target_offset', 0),
        'feature_nodes': metadata.get('feature_nodes', []),
        
        # Noise
        'noise_scale': metadata.get('noise_scale', 0),
        
        # Distance info (if available)
        'spatial_distance_alpha': metadata.get('spatial_distance_alpha', 0),
        'state_source_distance_alpha': metadata.get('state_source_distance_alpha', 0),
        
        # Full metadata for reference
        'metadata': metadata,
    }
    
    return info


def compute_dag_distances(metadata: Dict) -> Dict[int, int]:
    """Compute shortest path distances from target to all nodes."""
    # This would require the actual DAG structure
    # For now, return empty if not available in metadata
    return metadata.get('node_distances', {})


def create_detailed_visualization(info: Dict, dataset_id: int, output_path: Path):
    """Create a detailed visualization with all config info."""
    
    X = info['X']  # (n_samples, n_features, length)
    y = info['y']
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[2, 1, 1], width_ratios=[2, 1])
    
    # ===== LEFT COLUMN: Time series plots =====
    ax_ts = fig.add_subplot(gs[0, 0])
    
    # Select samples to plot (one per class if possible)
    unique_classes = np.unique(y)
    samples_to_plot = []
    for cls in unique_classes[:N_SAMPLES_TO_PLOT]:
        idx = np.where(y == cls)[0]
        if len(idx) > 0:
            samples_to_plot.append(idx[0])
    
    # Fill remaining with random samples
    while len(samples_to_plot) < N_SAMPLES_TO_PLOT and len(samples_to_plot) < len(y):
        idx = np.random.randint(len(y))
        if idx not in samples_to_plot:
            samples_to_plot.append(idx)
    
    # Plot time series
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, sample_idx in enumerate(samples_to_plot):
        X_sample = X[sample_idx]  # (n_features, length)
        y_sample = y[sample_idx]
        
        offset = i * 3  # Vertical offset for visibility
        for feat_idx in range(X_sample.shape[0]):
            label = f"Sample {sample_idx} (class {y_sample})" if feat_idx == 0 else None
            ax_ts.plot(X_sample[feat_idx] + offset, 
                      color=colors[int(y_sample) % 10],
                      alpha=0.8, linewidth=1.2, label=label)
    
    ax_ts.set_xlabel('Time', fontsize=11)
    ax_ts.set_ylabel('Value (offset for visibility)', fontsize=11)
    ax_ts.set_title(f'Time Series Samples', fontsize=12, fontweight='bold')
    ax_ts.legend(loc='upper right', fontsize=8)
    ax_ts.grid(True, alpha=0.3)
    
    # ===== RIGHT COLUMN: Configuration info =====
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis('off')
    
    # Build info text
    info_lines = []
    info_lines.append("═" * 40)
    info_lines.append(f"DATASET: synthetic_{dataset_id:04d}")
    info_lines.append("═" * 40)
    
    # Basic stats
    info_lines.append("")
    info_lines.append("【BASIC STATS】")
    info_lines.append(f"  Samples:     {info['n_samples']}")
    info_lines.append(f"  Features:    {info['n_features']}")
    info_lines.append(f"  Length:      {info['length']}")
    info_lines.append(f"  Classes:     {info['n_classes']}")
    info_lines.append(f"  Sample mode: {info['sample_mode']}")
    
    # DAG structure
    info_lines.append("")
    info_lines.append("【DAG STRUCTURE】")
    info_lines.append(f"  Total nodes:    {info['n_nodes']}")
    info_lines.append(f"  Root nodes:     {info['n_roots']}")
    info_lines.append(f"    - Time inputs:  {info['n_time_inputs']}")
    info_lines.append(f"    - State inputs: {info['n_state_inputs']}")
    info_lines.append(f"  Density:        {info['density']:.3f}")
    
    # Time activations
    info_lines.append("")
    info_lines.append("【TIME ACTIVATIONS】")
    time_acts = info['time_activations']
    if time_acts:
        for i, act in enumerate(time_acts[:6]):  # Show first 6
            info_lines.append(f"  [{i}] {act}")
        if len(time_acts) > 6:
            info_lines.append(f"  ... (+{len(time_acts)-6} more)")
    else:
        info_lines.append("  (none)")
    
    # State configurations
    info_lines.append("")
    info_lines.append("【STATE INPUTS】")
    info_lines.append(f"  State alpha (tanh): {info['state_alpha']:.3f}")
    state_configs = info['state_configs']
    if state_configs:
        for i, sc in enumerate(state_configs[:5]):  # Show first 5
            if isinstance(sc, dict):
                src = sc.get('source_node_id', '?')
                lag = sc.get('lag', '?')
                root = sc.get('root_node_id', '?')
                info_lines.append(f"  [{root}] ← node {src} at t-{lag}")
            else:
                info_lines.append(f"  {sc}")
        if len(state_configs) > 5:
            info_lines.append(f"  ... (+{len(state_configs)-5} more)")
    else:
        info_lines.append("  (none)")
    
    # Target info
    info_lines.append("")
    info_lines.append("【TARGET & FEATURES】")
    info_lines.append(f"  Target node:   {info['target_node']}")
    info_lines.append(f"  Target offset: {info['target_offset']} timesteps")
    offset = info['target_offset']
    if offset == 0:
        offset_type = "within window"
    elif offset > 0:
        offset_type = f"future (+{offset})"
    else:
        offset_type = f"past ({offset})"
    info_lines.append(f"                 ({offset_type})")
    
    feat_nodes = info['feature_nodes']
    info_lines.append(f"  Feature nodes: {feat_nodes[:5]}")
    if len(feat_nodes) > 5:
        info_lines.append(f"                 ... (+{len(feat_nodes)-5} more)")
    
    # Noise
    info_lines.append("")
    info_lines.append("【NOISE & DISTANCES】")
    info_lines.append(f"  Noise scale:       {info['noise_scale']:.4f}")
    info_lines.append(f"  Spatial dist α:    {info['spatial_distance_alpha']:.2f}")
    info_lines.append(f"  State source α:    {info['state_source_distance_alpha']:.2f}")
    
    # Display info text
    info_text = '\n'.join(info_lines)
    ax_info.text(0.02, 0.98, info_text, transform=ax_info.transAxes,
                fontsize=9, fontfamily='monospace',
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ===== BOTTOM LEFT: Class distribution =====
    ax_dist = fig.add_subplot(gs[1, 0])
    unique, counts = np.unique(y, return_counts=True)
    bars = ax_dist.bar(unique, counts, color=[colors[int(c) % 10] for c in unique])
    ax_dist.set_xlabel('Class', fontsize=11)
    ax_dist.set_ylabel('Count', fontsize=11)
    ax_dist.set_title('Class Distribution', fontsize=12, fontweight='bold')
    for bar, count in zip(bars, counts):
        ax_dist.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(count), ha='center', va='bottom', fontsize=9)
    
    # ===== BOTTOM CENTER: Feature statistics =====
    ax_stats = fig.add_subplot(gs[1, 1])
    ax_stats.axis('off')
    
    # Compute feature statistics
    stats_lines = []
    stats_lines.append("【FEATURE STATISTICS】")
    stats_lines.append("")
    
    X_flat = X.reshape(X.shape[0], -1)
    stats_lines.append(f"  Mean:   {np.mean(X_flat):.4f}")
    stats_lines.append(f"  Std:    {np.std(X_flat):.4f}")
    stats_lines.append(f"  Min:    {np.min(X_flat):.4f}")
    stats_lines.append(f"  Max:    {np.max(X_flat):.4f}")
    stats_lines.append(f"  NaN:    {np.sum(np.isnan(X_flat))}")
    stats_lines.append(f"  Inf:    {np.sum(np.isinf(X_flat))}")
    
    # Per-feature variance
    if X.shape[1] > 1:
        stats_lines.append("")
        stats_lines.append("  Per-feature variance:")
        for f in range(min(X.shape[1], 5)):
            var = np.var(X[:, f, :])
            stats_lines.append(f"    Feature {f}: {var:.4f}")
    
    stats_text = '\n'.join(stats_lines)
    ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes,
                 fontsize=10, fontfamily='monospace',
                 verticalalignment='top')
    
    # ===== BOTTOM ROW: Additional metadata =====
    ax_meta = fig.add_subplot(gs[2, :])
    ax_meta.axis('off')
    
    # Show additional metadata if available
    meta = info['metadata']
    meta_items = []
    
    important_keys = ['is_classification', 'T_total', 't_subseq', 'window_stride',
                      'n_sequences', 'transform_probs', 'init_sigma', 'init_a']
    
    for key in important_keys:
        if key in meta:
            val = meta[key]
            if isinstance(val, float):
                meta_items.append(f"{key}={val:.3f}")
            elif isinstance(val, dict):
                meta_items.append(f"{key}={{...}}")
            else:
                meta_items.append(f"{key}={val}")
    
    meta_text = "【ADDITIONAL METADATA】  " + "  |  ".join(meta_items)
    ax_meta.text(0.5, 0.5, meta_text, transform=ax_meta.transAxes,
                fontsize=9, fontfamily='monospace',
                verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    import argparse
    from tqdm import tqdm
    
    parser = argparse.ArgumentParser(description='Generate detailed synthetic dataset visualizations')
    parser.add_argument('--n-datasets', type=int, default=30, help='Number of datasets to generate')
    parser.add_argument('--seed', type=int, default=456, help='Random seed')
    args = parser.parse_args()
    
    print("=" * 70)
    print("GENERATING DETAILED SYNTHETIC DATASET VISUALIZATIONS")
    print("=" * 70)
    
    # Setup output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Clear old visualizations
    for f in OUTPUT_DIR.glob("*.png"):
        f.unlink()
    
    # Import generator
    from config import PriorConfig3D
    from generator import SyntheticDatasetGenerator3D
    
    # Create prior matching finetuning constraints
    prior = PriorConfig3D(
        max_samples=1000,
        max_features=12,
        max_t_subseq=500,
        
        # Matching real data distributions
        prob_univariate=0.65,
        n_features_beta_a=1.5,
        n_features_beta_b=4.0,
        n_features_range=(2, 10),
        
        n_samples_range=(100, 1000),
        t_subseq_range=(50, 300),
        T_total_range=(100, 500),
        
        # DAG structure
        n_nodes_range=(8, 25),
        n_roots_range=(4, 15),
        min_roots_fraction=0.30,
        max_roots_fraction=0.70,
        
        # Time inputs
        time_fraction_range=(0.2, 0.45),
        min_time_inputs=1,
        
        # State inputs
        min_state_inputs=1,
        state_lag_range=(1, 5),
        state_lag_geometric_p=0.6,
        state_alpha_range=(0.3, 0.8),
        state_source_distance_alpha=2.0,
        
        # Transformations
        prob_nn_transform=0.40,
        prob_tree_transform=0.45,
        prob_discretization=0.15,
        
        # Noise
        noise_scale_range=(0.001, 0.03),
        prob_low_noise_dataset=0.5,
        
        # Distance preferences
        distance_alpha=1.5,
        
        # Classes
        max_classes=10,
        min_samples_per_class=25,
        
        # Sample modes
        prob_iid_mode=0.35,
        prob_sliding_window_mode=0.45,
        prob_mixed_mode=0.20,
    )
    
    print(f"\n[1] Generating {args.n_datasets} datasets with detailed info...")
    
    generator = SyntheticDatasetGenerator3D(prior, seed=args.seed)
    
    datasets_info = []
    for i in tqdm(range(args.n_datasets), desc="Generating"):
        # Create new generator instance for each dataset
        gen = SyntheticDatasetGenerator3D(prior, seed=args.seed + i)
        info = generate_dataset_with_full_info(gen, args.seed + i)
        info['id'] = i
        datasets_info.append(info)
    
    print(f"\n[2] Creating visualizations...")
    
    for info in tqdm(datasets_info, desc="Visualizing"):
        output_path = OUTPUT_DIR / f"detailed_{info['id']:04d}.png"
        create_detailed_visualization(info, info['id'], output_path)
    
    # Summary statistics
    print(f"\n[3] Summary of generated datasets:")
    print("-" * 50)
    
    n_samples_list = [d['n_samples'] for d in datasets_info]
    n_features_list = [d['n_features'] for d in datasets_info]
    n_classes_list = [d['n_classes'] for d in datasets_info]
    n_nodes_list = [d['n_nodes'] for d in datasets_info]
    n_time_list = [d['n_time_inputs'] for d in datasets_info]
    n_state_list = [d['n_state_inputs'] for d in datasets_info]
    
    print(f"  Samples:      median={np.median(n_samples_list):.0f}, range=[{min(n_samples_list)}, {max(n_samples_list)}]")
    print(f"  Features:     median={np.median(n_features_list):.0f}, univariate={100*sum(1 for f in n_features_list if f==1)/len(n_features_list):.0f}%")
    print(f"  Classes:      median={np.median(n_classes_list):.0f}, range=[{min(n_classes_list)}, {max(n_classes_list)}]")
    print(f"  DAG nodes:    median={np.median(n_nodes_list):.0f}, range=[{min(n_nodes_list)}, {max(n_nodes_list)}]")
    print(f"  Time inputs:  median={np.median(n_time_list):.0f}, range=[{min(n_time_list)}, {max(n_time_list)}]")
    print(f"  State inputs: median={np.median(n_state_list):.0f}, range=[{min(n_state_list)}, {max(n_state_list)}]")
    
    # Sample mode distribution
    mode_counts = {}
    for d in datasets_info:
        mode = d['sample_mode']
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    print(f"  Sample modes: {mode_counts}")
    
    print(f"\n✅ Visualizations saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
