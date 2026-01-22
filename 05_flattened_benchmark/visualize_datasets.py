"""
Visualize real 3D time series datasets (before flattening)

For each valid dataset, creates a visualization showing 5 sample observations.
For multivariate datasets, all channels are shown in the same plot.
"""
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
REAL_DATA_DIR = PROJECT_ROOT / "01_real_data"
DATASETS_PKL = REAL_DATA_DIR / "AEON" / "data" / "classification_datasets.pkl"
RESULTS_DIR = Path(__file__).parent / "results"
USABLE_DATASETS_FILE = RESULTS_DIR / "usable_datasets.json"
VISUALIZATIONS_DIR = RESULTS_DIR / "dataset_visualizations"

# Add paths for imports
sys.path.insert(0, str(REAL_DATA_DIR))

# Visualization settings
N_SAMPLES_TO_VISUALIZE = 5
FIG_SIZE = (15, 10)
DPI = 150


def load_usable_dataset_names() -> List[str]:
    """Load list of usable dataset names from analysis results"""
    if not USABLE_DATASETS_FILE.exists():
        raise FileNotFoundError(
            f"Usable datasets file not found: {USABLE_DATASETS_FILE}\n"
            f"Run 'python analyze_flattenable_datasets.py' first."
        )
    
    with open(USABLE_DATASETS_FILE, 'r') as f:
        data = json.load(f)
    
    return [ds['name'] for ds in data['datasets']]


def load_all_datasets_from_pkl() -> Dict:
    """Load all datasets from pickle file into a dictionary"""
    if not DATASETS_PKL.exists():
        raise FileNotFoundError(
            f"Datasets pickle not found: {DATASETS_PKL}\n"
            f"Run 'python 01_real_data/src/analyze_all_datasets.py' first."
        )
    
    print(f"Loading datasets from {DATASETS_PKL}...")
    with open(DATASETS_PKL, 'rb') as f:
        datasets_list = pickle.load(f)
    
    # Convert to dict for O(1) lookup
    datasets_dict = {ds.name: ds for ds in datasets_list}
    print(f"Loaded {len(datasets_dict)} datasets")
    
    return datasets_dict


def visualize_single_dataset(dataset_name: str, dataset, output_path: Path, 
                             n_samples: int = N_SAMPLES_TO_VISUALIZE):
    """
    Create visualization for a single dataset.
    
    Args:
        dataset_name: Name of the dataset
        dataset: TimeSeriesDataset object
        output_path: Path to save the visualization
        n_samples: Number of sample observations to visualize
    """
    # Get train data (use train for visualization)
    X_train = dataset.X_train  # Shape: (n_train, n_channels, length)
    y_train = dataset.y_train
    
    n_train, n_channels, length = X_train.shape
    
    # Select samples to visualize
    n_samples = min(n_samples, n_train)
    if n_train > n_samples:
        # Randomly select samples
        indices = np.random.choice(n_train, size=n_samples, replace=False)
        indices = np.sort(indices)  # Sort for better visualization
    else:
        indices = np.arange(n_train)
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_samples, 1, figsize=FIG_SIZE, sharex=True)
    
    # Handle case where n_samples == 1
    if n_samples == 1:
        axes = [axes]
    
    # Color palette for channels
    colors = plt.cm.tab10(np.linspace(0, 1, min(n_channels, 10)))
    if n_channels > 10:
        # Extend colors if needed
        colors = plt.cm.tab20(np.linspace(0, 1, n_channels))
    
    # Plot each sample
    for idx, (ax, sample_idx) in enumerate(zip(axes, indices)):
        # Get the sample
        sample = X_train[sample_idx]  # Shape: (n_channels, length)
        label = y_train[sample_idx] if y_train is not None else None
        
        # Time axis
        time = np.arange(length)
        
        # Plot each channel
        for ch in range(n_channels):
            channel_data = sample[ch, :]
            
            # Check for NaN values
            valid_mask = ~np.isnan(channel_data)
            if valid_mask.sum() == 0:
                continue  # Skip if all NaN
            
            # Plot the channel
            if n_channels == 1:
                label_str = f"Sample {sample_idx}"
                if label is not None:
                    label_str += f" (Class: {label})"
                ax.plot(time[valid_mask], channel_data[valid_mask], 
                       color=colors[ch], linewidth=1.5, label=label_str)
            else:
                label_str = f"Channel {ch}"
                ax.plot(time[valid_mask], channel_data[valid_mask], 
                       color=colors[ch], linewidth=1.5, label=label_str, alpha=0.8)
        
        # Customize subplot
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        
        # Add sample info
        info_text = f"Sample {sample_idx}"
        if label is not None:
            info_text += f" | Class: {label}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Set common x-axis label
    axes[-1].set_xlabel('Time Step', fontsize=12)
    
    # Set main title
    title = f"{dataset_name}\n"
    title += f"Shape: {n_train} samples × {n_channels} channels × {length} timesteps"
    if y_train is not None:
        n_classes = len(np.unique(y_train))
        title += f" | {n_classes} classes"
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def visualize_all_datasets(dataset_names: Optional[List[str]] = None,
                          n_samples: int = N_SAMPLES_TO_VISUALIZE,
                          random_state: int = 42):
    """
    Visualize all usable datasets.
    
    Args:
        dataset_names: List of dataset names to visualize (None = all usable)
        n_samples: Number of sample observations per dataset
        random_state: Random seed for sample selection
    """
    np.random.seed(random_state)
    
    # Create output directory
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    if dataset_names is None:
        dataset_names = load_usable_dataset_names()
    
    datasets = load_all_datasets_from_pkl()
    
    print("=" * 70)
    print("VISUALIZING REAL 3D TIME SERIES DATASETS")
    print("=" * 70)
    print(f"Total datasets to visualize: {len(dataset_names)}")
    print(f"Samples per dataset: {n_samples}")
    print(f"Output directory: {VISUALIZATIONS_DIR}")
    print("=" * 70)
    
    successful = []
    failed = []
    
    for name in tqdm(dataset_names, desc="Visualizing"):
        try:
            if name not in datasets:
                print(f"\n⚠️  Dataset '{name}' not found in pickle, skipping...")
                failed.append(name)
                continue
            
            dataset = datasets[name]
            
            # Check if dataset has train data
            if dataset.X_train is None:
                print(f"\n⚠️  Dataset '{name}' has no train data, skipping...")
                failed.append(name)
                continue
            
            # Create visualization
            output_path = VISUALIZATIONS_DIR / f"{name}.png"
            visualize_single_dataset(name, dataset, output_path, n_samples)
            successful.append(name)
            
        except Exception as e:
            print(f"\n✗ Error visualizing {name}: {e}")
            failed.append(name)
    
    # Summary
    print("\n" + "=" * 70)
    print("VISUALIZATION SUMMARY")
    print("=" * 70)
    print(f"✓ Successful: {len(successful)}")
    print(f"✗ Failed: {len(failed)}")
    print(f"\nVisualizations saved to: {VISUALIZATIONS_DIR}")
    
    if failed:
        print(f"\nFailed datasets ({len(failed)}):")
        for name in failed[:10]:  # Show first 10
            print(f"  - {name}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
    
    return successful, failed


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize real 3D time series datasets (before flattening)"
    )
    parser.add_argument('--datasets', nargs='+', 
                       help='Specific dataset names to visualize')
    parser.add_argument('--n-datasets', type=int, 
                       help='Number of datasets to visualize (for testing)')
    parser.add_argument('--n-samples', type=int, default=N_SAMPLES_TO_VISUALIZE,
                       help='Number of sample observations per dataset')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for sample selection')
    args = parser.parse_args()
    
    # Determine which datasets to visualize
    if args.datasets:
        dataset_names = args.datasets
    else:
        dataset_names = load_usable_dataset_names()
        if args.n_datasets:
            dataset_names = dataset_names[:args.n_datasets]
    
    # Visualize
    successful, failed = visualize_all_datasets(
        dataset_names=dataset_names,
        n_samples=args.n_samples,
        random_state=args.random_state
    )
    
    print("\n✅ Visualization complete!")
    
    return successful, failed


if __name__ == "__main__":
    main()
