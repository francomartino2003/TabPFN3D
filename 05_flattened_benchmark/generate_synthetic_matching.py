"""
Generate synthetic 3D datasets with distributions matching real datasets.

This script:
1. Analyzes the distribution of real usable datasets (n_samples, n_features, length, n_classes)
2. Configures the 3D generator (v2) to produce datasets with similar distributions
3. Generates N synthetic datasets
4. Saves and visualizes them for qualitative comparison with real data

v2 Generator Changes:
- Only time and state inputs (no noise roots)
- State inputs reference specific nodes at t-k
- No passthrough transformation
- Smaller DAGs for simpler/interpretable structure
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
GENERATOR_3D_DIR = PROJECT_ROOT / "03_synthetic_generator_3D"
RESULTS_DIR = Path(__file__).parent / "results"
USABLE_DATASETS_FILE = RESULTS_DIR / "usable_datasets.json"
SYNTHETIC_DIR = RESULTS_DIR / "synthetic_datasets"
SYNTHETIC_VIZ_DIR = RESULTS_DIR / "synthetic_visualizations"

# Add generator to path
sys.path.insert(0, str(GENERATOR_3D_DIR))

# Visualization settings
N_SAMPLES_TO_VISUALIZE = 5
FIG_SIZE = (15, 10)
DPI = 150


def load_real_dataset_stats() -> Dict:
    """Load statistics from real usable datasets."""
    if not USABLE_DATASETS_FILE.exists():
        raise FileNotFoundError(f"Usable datasets file not found: {USABLE_DATASETS_FILE}")
    
    with open(USABLE_DATASETS_FILE, 'r') as f:
        data = json.load(f)
    
    return data


def analyze_real_distributions(data: Dict) -> Dict:
    """
    Analyze distributions of real datasets to configure the generator.
    
    Returns dict with distribution parameters for:
    - n_samples
    - n_features (channels)
    - length (timesteps)
    - n_classes
    """
    datasets = data['datasets']
    
    # Extract values
    n_samples_list = [d['n_samples'] for d in datasets]
    n_features_list = [d['n_dimensions'] for d in datasets]
    lengths_list = [d['length'] for d in datasets]
    n_classes_list = [d['n_classes'] for d in datasets if d.get('n_classes')]
    
    # Calculate statistics
    stats = {
        'n_samples': {
            'min': int(np.min(n_samples_list)),
            'max': int(min(np.max(n_samples_list), 10000)),
            'median': int(np.median(n_samples_list)),
            'mean': float(np.mean(n_samples_list)),
            'p25': int(np.percentile(n_samples_list, 25)),
            'p75': int(np.percentile(n_samples_list, 75)),
        },
        'n_features': {
            'min': int(np.min(n_features_list)),
            'max': int(np.max(n_features_list)),
            'median': int(np.median(n_features_list)),
            'mean': float(np.mean(n_features_list)),
            'prob_univariate': sum(1 for f in n_features_list if f == 1) / len(n_features_list),
        },
        'length': {
            'min': int(np.min(lengths_list)),
            'max': int(np.max(lengths_list)),
            'median': int(np.median(lengths_list)),
            'mean': float(np.mean(lengths_list)),
            'p25': max(10, int(np.percentile(lengths_list, 25))),
            'p75': int(np.percentile(lengths_list, 75)),
        },
        'n_classes': {
            'min': int(np.min(n_classes_list)),
            'max': int(np.max(n_classes_list)),
            'median': int(np.median(n_classes_list)),
            'mean': float(np.mean(n_classes_list)),
        },
        'n_datasets': len(datasets),
    }
    
    return stats


def create_matching_prior(real_stats: Dict):
    """
    Create a PriorConfig3D (v2) that matches real dataset distributions.
    
    v2 changes:
    - No passthrough, no noise roots
    - Only time and state inputs
    - Smaller DAGs
    """
    from config import PriorConfig3D
    
    n_samples = real_stats['n_samples']
    n_features = real_stats['n_features']
    length = real_stats['length']
    n_classes = real_stats['n_classes']
    
    prior = PriorConfig3D(
        # === Size constraints ===
        max_samples=10000,
        max_features=12,  # Max features (will be constrained by 500 rule)
        max_t_subseq=500,
        max_T_total=600,
        max_classes=6,
        max_complexity=50_000_000,
        
        # === Sample size (wider range, more samples) ===
        # Real: median=600, p25=211, p75=1931
        n_samples_range=(200, 3000),  # Increased upper bound
        
        # === Feature count (more multivariate than real data) ===
        # Real is 94% univariate, but we want more variety
        prob_univariate=0.65,  # 65% univariate (was 94%)
        n_features_beta_a=1.5,  # Shape for multivariate count
        n_features_beta_b=4.0,  # Favors smaller values (2-5 features)
        n_features_range=(2, 8),  # 2-8 features when multivariate
        
        # === Temporal parameters ===
        # Constraint: n_features * t_subseq < 500
        # If n_features=5, max t_subseq=100
        # If n_features=2, max t_subseq=250
        T_total_range=(length['p25'], length['p75'] + 50),
        T_total_log_uniform=True,
        t_subseq_range=(50, min(300, length['p75'])),  # Adjusted for multivariate
        t_subseq_log_uniform=True,
        
        # === Graph structure (v3: distance preference controls complexity) ===
        n_nodes_range=(8, 30),
        n_nodes_log_uniform=True,
        density_range=(0.1, 0.6),
        n_roots_range=(4, 18),
        min_roots_fraction=0.25,
        max_roots_fraction=0.60,
        
        # === Root input distribution (v3: favor state inputs) ===
        min_time_inputs=2,
        min_state_inputs=2,
        time_fraction_range=(0.20, 0.40),  # Less time = more state inputs
        
        # === State input parameters ===
        state_lag_range=(1, 8),
        state_lag_distribution='geometric',
        state_lag_geometric_p=0.4,
        state_alpha_range=(0.3, 0.8),  # Reduced to preserve information
        
        # === Transformations ===
        prob_nn_transform=0.40,
        prob_tree_transform=0.45,
        prob_discretization=0.20,
        prob_identity_activation=0.5,
        
        # === Noise (v3: variable per dataset) ===
        prob_edge_noise=0.05,
        noise_scale_range=(0.001, 0.08),
        prob_low_noise_dataset=0.3,  # 30% clean datasets
        low_noise_scale_max=0.005,
        
        # === Distance preference (v3: unified for spatial and temporal) ===
        distance_alpha=1.5,  # Controls preference for closer distances
        
        # === Sample generation mode ===
        prob_iid_mode=0.5,
        prob_sliding_window_mode=0.3,
        prob_mixed_mode=0.2,
        
        # === Target (classification only) ===
        prob_classification=1.0,
        force_classification=True,
        min_samples_per_class=25,
        max_target_offset=20,
        prob_future=0.75,  # 75% future, 25% past for non-zero offsets
        
        # === Post-processing (minimal) ===
        prob_warping=0.15,
        warping_intensity_range=(0.1, 0.25),
        prob_quantization=0.1,
        prob_missing_values=0.0,
        
        # === Train/test split ===
        train_ratio_range=(0.4, 0.7),
    )
    
    return prior


def generate_synthetic_datasets(n_datasets: int, prior, seed: int = 42) -> List:
    """Generate N synthetic datasets using the configured prior."""
    from generator import SyntheticDatasetGenerator3D
    
    generator = SyntheticDatasetGenerator3D(prior=prior, seed=seed)
    
    datasets = []
    for i, dataset in enumerate(tqdm(generator.generate_many(n_datasets), 
                                      total=n_datasets, desc="Generating")):
        datasets.append({
            'id': i,
            'X': dataset.X,
            'y': dataset.y,
            'config': dataset.config,
            'shape': dataset.shape,
            'n_classes': dataset.n_classes,
        })
    
    return datasets


def visualize_synthetic_dataset(dataset: Dict, output_path: Path, 
                                n_samples: int = N_SAMPLES_TO_VISUALIZE):
    """Create visualization for a synthetic dataset (same format as real)."""
    X = dataset['X']
    y = dataset['y']
    config = dataset['config']
    
    n_total, n_features, length = X.shape
    
    # Select samples to visualize
    n_samples = min(n_samples, n_total)
    if n_total > n_samples:
        indices = np.random.choice(n_total, size=n_samples, replace=False)
        indices = np.sort(indices)
    else:
        indices = np.arange(n_total)
    
    fig, axes = plt.subplots(n_samples, 1, figsize=FIG_SIZE, sharex=True)
    if n_samples == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, min(n_features, 10)))
    if n_features > 10:
        colors = plt.cm.tab20(np.linspace(0, 1, n_features))
    
    for idx, (ax, sample_idx) in enumerate(zip(axes, indices)):
        sample = X[sample_idx]
        label = y[sample_idx] if y is not None else None
        time = np.arange(length)
        
        for ch in range(n_features):
            channel_data = sample[ch, :]
            valid_mask = ~np.isnan(channel_data)
            
            if valid_mask.sum() == 0:
                continue
            
            if n_features == 1:
                label_str = f"Sample {sample_idx}"
                if label is not None:
                    label_str += f" (Class: {int(label)})"
                ax.plot(time[valid_mask], channel_data[valid_mask],
                       color=colors[ch], linewidth=1.5, label=label_str)
            else:
                ax.plot(time[valid_mask], channel_data[valid_mask],
                       color=colors[ch], linewidth=1.5, label=f"Channel {ch}", alpha=0.8)
        
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        
        info_text = f"Sample {sample_idx}"
        if label is not None:
            info_text += f" | Class: {int(label)}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    axes[-1].set_xlabel('Time Step', fontsize=12)
    
    title = f"Synthetic Dataset {dataset['id']}\n"
    title += f"Shape: {n_total} samples × {n_features} channels × {length} timesteps"
    if dataset['n_classes'] > 0:
        title += f" | {dataset['n_classes']} classes"
    title += f" | Mode: {config.sample_mode}"
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def save_synthetic_datasets(datasets: List, output_dir: Path):
    """Save synthetic datasets to numpy files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for dataset in datasets:
        np.savez(
            output_dir / f"synthetic_{dataset['id']:04d}.npz",
            X=dataset['X'],
            y=dataset['y'],
            shape=dataset['shape'],
            n_classes=dataset['n_classes'],
        )


def print_comparison(real_stats: Dict, synthetic_datasets: List):
    """Print comparison between real and synthetic distributions."""
    syn_samples = [d['shape'][0] for d in synthetic_datasets]
    syn_features = [d['shape'][1] for d in synthetic_datasets]
    syn_lengths = [d['shape'][2] for d in synthetic_datasets]
    syn_classes = [d['n_classes'] for d in synthetic_datasets]
    
    print("\n" + "=" * 70)
    print("COMPARISON: REAL vs SYNTHETIC DISTRIBUTIONS")
    print("=" * 70)
    
    print(f"\n{'Metric':<20} {'Real':<25} {'Synthetic':<25}")
    print("-" * 70)
    
    real_s = real_stats['n_samples']
    print(f"{'n_samples':<20} "
          f"median={real_s['median']}, range=[{real_s['min']},{real_s['max']}] "
          f"median={int(np.median(syn_samples))}, range=[{min(syn_samples)},{max(syn_samples)}]")
    
    real_f = real_stats['n_features']
    syn_univariate = sum(1 for f in syn_features if f == 1) / len(syn_features)
    print(f"{'n_features':<20} "
          f"median={real_f['median']}, univar={real_f['prob_univariate']:.0%} "
          f"median={int(np.median(syn_features))}, univar={syn_univariate:.0%}")
    
    real_l = real_stats['length']
    print(f"{'length':<20} "
          f"median={real_l['median']}, range=[{real_l['min']},{real_l['max']}] "
          f"median={int(np.median(syn_lengths))}, range=[{min(syn_lengths)},{max(syn_lengths)}]")
    
    real_c = real_stats['n_classes']
    print(f"{'n_classes':<20} "
          f"median={real_c['median']:.1f}, range=[{real_c['min']},{real_c['max']}] "
          f"median={np.median(syn_classes):.1f}, range=[{min(syn_classes)},{max(syn_classes)}]")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate synthetic 3D datasets matching real dataset distributions (v2 generator)"
    )
    parser.add_argument('--n-datasets', type=int, default=20,
                       help='Number of synthetic datasets to generate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--visualize-all', action='store_true',
                       help='Visualize all generated datasets')
    parser.add_argument('--n-visualize', type=int, default=10,
                       help='Number of datasets to visualize (if not --visualize-all)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("GENERATING SYNTHETIC DATASETS (v2 Generator)")
    print("=" * 70)
    
    print("\n[1] Loading real dataset statistics...")
    real_data = load_real_dataset_stats()
    real_stats = analyze_real_distributions(real_data)
    
    print(f"    Real datasets: {real_stats['n_datasets']}")
    print(f"    n_samples: median={real_stats['n_samples']['median']}, "
          f"range=[{real_stats['n_samples']['min']}, {real_stats['n_samples']['max']}]")
    print(f"    n_features: median={real_stats['n_features']['median']}, "
          f"univariate={real_stats['n_features']['prob_univariate']:.0%}")
    print(f"    length: median={real_stats['length']['median']}, "
          f"range=[{real_stats['length']['min']}, {real_stats['length']['max']}]")
    print(f"    n_classes: median={real_stats['n_classes']['median']:.1f}, "
          f"range=[{real_stats['n_classes']['min']}, {real_stats['n_classes']['max']}]")
    
    print("\n[2] Creating generator prior (v2: time+state only, no passthrough)...")
    prior = create_matching_prior(real_stats)
    
    print(f"\n[3] Generating {args.n_datasets} synthetic datasets...")
    np.random.seed(args.seed)
    datasets = generate_synthetic_datasets(args.n_datasets, prior, args.seed)
    
    print(f"\n[4] Saving datasets to {SYNTHETIC_DIR}...")
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    save_synthetic_datasets(datasets, SYNTHETIC_DIR)
    
    summary = {
        'n_datasets': len(datasets),
        'seed': args.seed,
        'generator_version': 'v3',
        'real_stats': real_stats,
        'datasets': [
            {
                'id': int(d['id']),
                'shape': [int(x) for x in d['shape']],
                'n_classes': int(d['n_classes']),
                'sample_mode': d['config'].sample_mode,
                'n_time_inputs': d['config'].n_time_inputs,
                'n_state_inputs': d['config'].n_state_inputs,
                'target_offset': d['config'].target_offset,
                'spatial_distance_alpha': d['config'].spatial_distance_alpha,
                'noise_scale': float(d['config'].noise_scale),
            }
            for d in datasets
        ]
    }
    with open(SYNTHETIC_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    n_viz = len(datasets) if args.visualize_all else min(args.n_visualize, len(datasets))
    print(f"\n[5] Visualizing {n_viz} datasets to {SYNTHETIC_VIZ_DIR}...")
    SYNTHETIC_VIZ_DIR.mkdir(parents=True, exist_ok=True)
    
    for dataset in tqdm(datasets[:n_viz], desc="Visualizing"):
        output_path = SYNTHETIC_VIZ_DIR / f"synthetic_{dataset['id']:04d}.png"
        visualize_synthetic_dataset(dataset, output_path)
    
    print_comparison(real_stats, datasets)
    
    print("\n" + "=" * 70)
    print("✅ GENERATION COMPLETE!")
    print("=" * 70)
    print(f"    Datasets saved to: {SYNTHETIC_DIR}")
    print(f"    Visualizations saved to: {SYNTHETIC_VIZ_DIR}")
    
    return datasets


if __name__ == "__main__":
    main()
