#!/usr/bin/env python3
"""
Generate visualizations after changes:
1. Clipping unified to [-1e10, 1e10]
2. Time-only nodes NOT excluded in sliding_window/mixed
3. Unified activations (NN + time)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '03_synthetic_generator_3D'))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from generator import SyntheticDatasetGenerator3D
from config import PriorConfig3D

def create_viz_prior():
    """Create prior for visualization with NEW 2-layer structure."""
    return PriorConfig3D(
        # Size constraints for TabPFN
        max_samples=1000,
        max_features=15,
        max_t_subseq=100,
        max_classes=10,
        
        # Sample sizes
        n_samples_range=(200, 800),
        
        # Feature count - mix of univariate and multivariate
        prob_univariate=0.4,
        n_features_range=(2, 6),
        
        # Temporal - longer for smoother patterns
        T_total_range=(80, 400),
        t_subseq_range=(40, 100),
        
        # DAG structure - smaller DAG after combination layer
        n_nodes_range=(12, 25),
        density_range=(0.2, 0.4),
        
        # NEW 2-layer structure
        n_time_inputs=1,  # Fixed: 1 raw time input
        n_state_inputs_range=(3, 6),  # Multiple state inputs
        n_combination_nodes_range=(4, 7),  # Combination layer
        
        # Short lags for smoothness
        state_lag_range=(1, 3),
        state_lag_geometric_p=0.7,
        
        # Sample modes - test all
        prob_iid_mode=0.33,
        prob_sliding_window_mode=0.34,
        prob_mixed_mode=0.33,
        
        # Very low noise for cleaner visualization
        noise_scale_range=(0.001, 0.01),
        prob_low_noise_dataset=0.7,
        
        # Classification only for cleaner viz
        force_classification=True,
        
        # Target offset - prefer close offsets
        max_target_offset=10,
        prob_future=0.6,
        
        # Favor smooth transformations
        prob_nn_transform=0.70,
        prob_tree_transform=0.20,
        prob_discretization=0.10,
    )


def visualize_dataset(dataset, idx, output_dir):
    """Create visualization for a single dataset."""
    X = dataset.X  # (n_samples, n_features, t_subseq)
    y = dataset.y
    config = dataset.config
    metadata = dataset.metadata
    
    n_samples, n_features, t_subseq = X.shape
    n_classes = len(np.unique(y))
    
    # Select 5 random samples to visualize
    n_viz = min(5, n_samples)
    viz_indices = np.random.choice(n_samples, n_viz, replace=False)
    
    # Create figure
    fig, axes = plt.subplots(n_viz, 1, figsize=(14, 3 * n_viz))
    if n_viz == 1:
        axes = [axes]
    
    # Color map for features
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_features, 10)))
    
    for i, sample_idx in enumerate(viz_indices):
        ax = axes[i]
        class_label = int(y[sample_idx])
        
        for f in range(n_features):
            series = X[sample_idx, f, :]
            ax.plot(series, color=colors[f], alpha=0.8, linewidth=1.2,
                   label=f'F{f}' if i == 0 else None)
        
        ax.set_title(f'Sample {sample_idx} | Class {class_label}', fontsize=10)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    # Add legend to first plot
    if n_features <= 10:
        axes[0].legend(loc='upper right', ncol=min(n_features, 5), fontsize=8)
    
    # Title with metadata
    offset_type = metadata.get('target_offset_type', 'unknown')
    n_combo = getattr(config, 'n_combination_nodes', 0)
    
    title = (
        f"Dataset {idx:04d} | Mode: {config.sample_mode} | "
        f"Shape: {n_samples}×{n_features}×{t_subseq}\n"
        f"Classes: {n_classes} | Offset: {config.target_offset} ({offset_type})\n"
        f"Structure: 1 TIME + {config.n_state_inputs} STATE → {n_combo} COMBO → "
        f"{config.n_nodes - 1 - config.n_state_inputs - n_combo} DAG"
    )
    
    fig.suptitle(title, fontsize=11, y=1.02)
    plt.tight_layout()
    
    # Save
    output_path = output_dir / f'changes_test_{idx:04d}.png'
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    
    return {
        'idx': idx,
        'mode': config.sample_mode,
        'n_features': n_features,
        't_subseq': t_subseq,
        'n_classes': n_classes,
        'offset': config.target_offset,
        'time_activations': config.time_activations,
    }


def main():
    print("Testing changes with visualizations...")
    print("=" * 60)
    
    output_dir = Path(__file__).parent / 'results' / 'changes_test'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean old files
    for f in output_dir.glob('changes_test_*.png'):
        f.unlink()
    
    prior = create_viz_prior()
    generator = SyntheticDatasetGenerator3D(prior=prior, seed=42)
    
    n_datasets = 15
    results = []
    
    print(f"\nGenerating {n_datasets} datasets...")
    
    for i in range(n_datasets):
        try:
            dataset = generator.generate()
            info = visualize_dataset(dataset, i, output_dir)
            results.append(info)
            
            mode = info['mode']
            acts = info['time_activations'][:2]
            print(f"  [{i:02d}] {mode:15s} | {info['n_features']}×{info['t_subseq']} | "
                  f"offset={info['offset']:+3d} | acts: {acts}")
            
        except Exception as e:
            print(f"  [{i:02d}] ERROR: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    
    modes = {}
    for r in results:
        mode = r['mode']
        modes[mode] = modes.get(mode, 0) + 1
    
    for mode, count in modes.items():
        print(f"  {mode}: {count}")
    
    # Check if new activations appear
    all_acts = set()
    for r in results:
        all_acts.update(r['time_activations'])
    
    new_acts = {'sigmoid', 'relu', 'abs', 'step', 'softplus'} & all_acts
    print(f"\nNew activations used: {new_acts}")
    
    print(f"\nVisualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
