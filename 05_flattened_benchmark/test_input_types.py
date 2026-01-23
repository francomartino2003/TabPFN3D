"""
Test different input type configurations:
1. Few inputs with random mix of time/state

This helps understand what patterns different combinations produce.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
GENERATOR_3D_DIR = PROJECT_ROOT / "03_synthetic_generator_3D"
OUTPUT_DIR = Path(__file__).parent / "results" / "input_type_comparison"

sys.path.insert(0, str(GENERATOR_3D_DIR))

from config import PriorConfig3D
from generator import SyntheticDatasetGenerator3D


def create_few_random_inputs_prior():
    """Prior with FEW inputs, random mix of time/state."""
    return PriorConfig3D(
        max_samples=500,
        max_features=5,
        max_t_subseq=300,
        
        prob_univariate=0.7,
        n_samples_range=(100, 500),
        t_subseq_range=(50, 200),
        T_total_range=(100, 300),
        
        # Smaller DAGs
        n_nodes_range=(5, 12),
        n_roots_range=(2, 5),  # FEW root inputs (2-5)
        min_roots_fraction=0.3,
        max_roots_fraction=0.6,
        
        # NO MINIMUMS - let it be random
        min_time_inputs=0,
        min_state_inputs=0,
        time_fraction_range=(0.0, 1.0),  # FULLY RANDOM: 0% to 100% time
        
        state_lag_range=(1, 5),
        state_lag_geometric_p=0.6,
        state_alpha_range=(0.3, 0.8),
        
        prob_nn_transform=0.40,
        prob_tree_transform=0.45,
        prob_discretization=0.15,
        
        noise_scale_range=(0.001, 0.02),
        max_classes=6,
        min_samples_per_class=20,
        
        prob_iid_mode=0.5,
        prob_sliding_window_mode=0.3,
        prob_mixed_mode=0.2,
    )


def visualize_dataset(X, y, title, subtitle, output_path):
    """Create visualization for a dataset."""
    n_samples = min(5, X.shape[0])
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(14, 3*n_samples))
    if n_samples == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Get one sample per class if possible
    unique_classes = np.unique(y)
    sample_indices = []
    for cls in unique_classes[:n_samples]:
        idx = np.where(y == cls)[0]
        if len(idx) > 0:
            sample_indices.append(idx[0])
    
    while len(sample_indices) < n_samples:
        idx = np.random.randint(len(y))
        if idx not in sample_indices:
            sample_indices.append(idx)
    
    for i, sample_idx in enumerate(sample_indices):
        ax = axes[i]
        X_sample = X[sample_idx]  # (n_features, length)
        y_sample = y[sample_idx]
        
        for feat_idx in range(X_sample.shape[0]):
            ax.plot(X_sample[feat_idx], color=colors[int(y_sample) % 10], 
                   alpha=0.8, linewidth=1.2)
        
        ax.set_ylabel(f'Class {y_sample}')
        ax.grid(True, alpha=0.3)
        
        # Stats
        mean_val = np.mean(X_sample)
        std_val = np.std(X_sample)
        ax.set_title(f'Sample {sample_idx} | mean={mean_val:.2f}, std={std_val:.2f}')
    
    axes[-1].set_xlabel('Time')
    fig.suptitle(f"{title}\n{subtitle}", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Clear old files
    for f in OUTPUT_DIR.glob("*.png"):
        f.unlink()
    
    n_datasets = 30
    seed = 42
    
    print("=" * 60)
    print("FEW INPUTS WITH RANDOM TIME/STATE MIX")
    print("=" * 60)
    
    prior = create_few_random_inputs_prior()
    
    print(f"\nGenerating {n_datasets} datasets with 2-5 root inputs (random time/state mix)...")
    
    for i in tqdm(range(n_datasets), desc="Generating"):
        gen = SyntheticDatasetGenerator3D(prior, seed=seed+i)
        try:
            dataset = gen.generate()
            meta = dataset.metadata
            
            n_time = meta.get('n_time_inputs', 0)
            n_state = meta.get('n_state_inputs', 0)
            n_nodes = meta.get('n_nodes', 0)
            sample_mode = meta.get('sample_mode', '?')
            
            title = f"#{i:02d} | {dataset.X.shape[0]} samples, {dataset.X.shape[1]} feat, {dataset.X.shape[2]} len, {len(np.unique(dataset.y))} classes"
            subtitle = f"Roots: {n_time} time + {n_state} state | Nodes: {n_nodes} | Mode: {sample_mode}"
            
            visualize_dataset(dataset.X, dataset.y, title, subtitle, OUTPUT_DIR / f"few_inputs_{i:02d}.png")
        except Exception as e:
            print(f"  Error generating {i}: {e}")
    
    print(f"\n✅ Visualizations saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
