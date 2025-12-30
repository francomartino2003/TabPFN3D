"""
Demo script for the 3D Synthetic Dataset Generator.

Demonstrates generation of time series classification datasets.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List
import sys
import os

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PriorConfig3D
from generator import SyntheticDatasetGenerator3D


def demo_basic_generation():
    """Demonstrate basic 3D dataset generation."""
    print("=" * 70)
    print("DEMO 1: Basic 3D Dataset Generation")
    print("=" * 70)
    
    generator = SyntheticDatasetGenerator3D(seed=42, verbose=True)
    dataset = generator.generate()
    
    print(f"\nGenerated dataset:")
    print(f"  X shape: {dataset.X.shape} (samples, features, timesteps)")
    print(f"  y shape: {dataset.y.shape}")
    print(f"  Classes: {dataset.n_classes} (unique: {len(np.unique(dataset.y))})")
    print(f"  Feature names: {dataset.feature_names}")
    
    # Check for NaN
    n_nan = np.isnan(dataset.X).sum()
    if n_nan > 0:
        print(f"  Missing values: {n_nan} ({100 * n_nan / dataset.X.size:.2f}%)")
    
    return dataset


def demo_custom_prior():
    """Demonstrate generation with custom prior."""
    print("\n" + "=" * 70)
    print("DEMO 2: Custom Prior Configuration")
    print("=" * 70)
    
    # Create prior for smaller, univariate time series (common in real data)
    small_prior = PriorConfig3D(
        n_samples_range=(100, 500),
        n_features_range=(1, 3),
        prob_univariate=0.7,  # 70% chance of univariate
        n_timesteps_range=(50, 200),
        n_nodes_range=(5, 15),
        max_classes=5,
    )
    
    generator = SyntheticDatasetGenerator3D(prior=small_prior, seed=123)
    
    print("Generating 5 small time series datasets...")
    for i, dataset in enumerate(generator.generate_many(5)):
        n_samples, n_features, n_timesteps = dataset.shape
        print(f"  Dataset {i+1}: samples={n_samples:4d}, features={n_features}, "
              f"timesteps={n_timesteps:3d}, classes={dataset.n_classes}")


def demo_temporal_structure():
    """Examine the temporal structure of generated data."""
    print("\n" + "=" * 70)
    print("DEMO 3: Temporal Structure Analysis")
    print("=" * 70)
    
    prior = PriorConfig3D(
        n_samples_range=(200, 200),
        n_features_range=(3, 3),
        n_timesteps_range=(100, 100),
    )
    
    generator = SyntheticDatasetGenerator3D(prior=prior, seed=42, verbose=True)
    dataset = generator.generate()
    
    print(f"\nTemporal structure info:")
    print(f"  Feature window: {dataset.metadata['feature_selection']['feature_window']}")
    print(f"  Target timestep: {dataset.metadata['feature_selection']['target_timestep']}")
    print(f"  Target position: {dataset.metadata['feature_selection']['target_position']}")
    print(f"  Temporal patterns: {dataset.metadata['n_temporal_patterns']}")
    print(f"  Correlated noise: {dataset.metadata['has_correlated_noise']}")
    
    return dataset


def demo_statistics():
    """Generate statistics across multiple datasets."""
    print("\n" + "=" * 70)
    print("DEMO 4: Statistics Across Datasets")
    print("=" * 70)
    
    generator = SyntheticDatasetGenerator3D(seed=42)
    
    n_datasets = 30
    stats = {
        'n_samples': [],
        'n_features': [],
        'n_timesteps': [],
        'n_classes': [],
        'target_position': {'before': 0, 'within': 0, 'after': 0},
        'univariate': 0,
    }
    
    print(f"Generating {n_datasets} datasets...")
    
    for dataset in generator.generate_many(n_datasets):
        stats['n_samples'].append(dataset.n_samples)
        stats['n_features'].append(dataset.n_features)
        stats['n_timesteps'].append(dataset.n_timesteps)
        stats['n_classes'].append(dataset.n_classes)
        pos = dataset.metadata['feature_selection']['target_position']
        stats['target_position'][pos] += 1
        if dataset.n_features == 1:
            stats['univariate'] += 1
    
    print(f"\nStatistics over {n_datasets} datasets:")
    print(f"  Samples: min={min(stats['n_samples'])}, max={max(stats['n_samples'])}, "
          f"median={np.median(stats['n_samples']):.0f}")
    print(f"  Features: min={min(stats['n_features'])}, max={max(stats['n_features'])}, "
          f"median={np.median(stats['n_features']):.0f}")
    print(f"  Timesteps: min={min(stats['n_timesteps'])}, max={max(stats['n_timesteps'])}, "
          f"median={np.median(stats['n_timesteps']):.0f}")
    print(f"  Classes: min={min(stats['n_classes'])}, max={max(stats['n_classes'])}")
    print(f"\n  Univariate datasets: {stats['univariate']}/{n_datasets} "
          f"({100*stats['univariate']/n_datasets:.0f}%)")
    print(f"\n  Target position distribution:")
    for pos, count in stats['target_position'].items():
        print(f"    {pos}: {count}/{n_datasets} ({100*count/n_datasets:.0f}%)")
    
    return stats


def demo_visualization():
    """Visualize generated time series."""
    print("\n" + "=" * 70)
    print("DEMO 5: Visualization")
    print("=" * 70)
    
    # Generate a small multivariate dataset for visualization
    prior = PriorConfig3D(
        n_samples_range=(100, 100),
        n_features_range=(3, 3),
        n_timesteps_range=(100, 100),
        prob_missing_values=0.0,  # No missing for cleaner viz
    )
    
    generator = SyntheticDatasetGenerator3D(prior=prior, seed=42)
    dataset = generator.generate()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Top row: Sample time series by class
    classes = np.unique(dataset.y)
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    
    for feat_idx in range(min(3, dataset.n_features)):
        ax = axes[0, feat_idx]
        for class_idx, c in enumerate(classes[:5]):  # Show up to 5 classes
            mask = dataset.y == c
            sample_idx = np.where(mask)[0][0]  # First sample of this class
            ax.plot(dataset.X[sample_idx, feat_idx, :], 
                   color=colors[class_idx], alpha=0.8, label=f'Class {c}')
        ax.set_title(f'Feature {feat_idx}: {dataset.feature_names[feat_idx]}')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Value')
        if feat_idx == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    # Bottom row: Class distribution and feature correlations
    ax = axes[1, 0]
    unique, counts = np.unique(dataset.y, return_counts=True)
    ax.bar(unique, counts, color='steelblue')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution')
    
    # Feature mean over time by class
    ax = axes[1, 1]
    for class_idx, c in enumerate(classes[:5]):
        mask = dataset.y == c
        mean_series = dataset.X[mask, 0, :].mean(axis=0)
        ax.plot(mean_series, color=colors[class_idx], label=f'Class {c}')
    ax.set_title('Mean of Feature 0 by Class')
    ax.set_xlabel('Timestep')
    ax.legend(fontsize=8)
    
    # Target position indicator
    ax = axes[1, 2]
    target_t = dataset.metadata['feature_selection']['target_timestep']
    window = dataset.metadata['feature_selection']['feature_window']
    
    ax.axvspan(window[0], window[1], alpha=0.3, color='green', label='Feature window')
    ax.axvline(target_t, color='red', linewidth=2, label=f'Target (t={target_t})')
    ax.set_xlim(0, dataset.config.n_timesteps)
    ax.set_xlabel('Timestep')
    ax.set_title(f'Target Position: {dataset.metadata["feature_selection"]["target_position"]}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('synthetic_3d_demo.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to 'synthetic_3d_demo.png'")
    plt.close()


def demo_2d_compatibility():
    """Show conversion to 2D format for standard ML."""
    print("\n" + "=" * 70)
    print("DEMO 6: 2D Compatibility")
    print("=" * 70)
    
    generator = SyntheticDatasetGenerator3D(seed=42)
    dataset = generator.generate()
    
    print(f"Original 3D shape: {dataset.X.shape}")
    
    X_2d, y = dataset.to_2d()
    print(f"Flattened 2D shape: {X_2d.shape}")
    print(f"  = {dataset.n_features} features × {dataset.n_timesteps} timesteps")
    
    # Quick ML test
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import accuracy_score
        
        # Prepare data
        X = X_2d.copy()
        if np.any(np.isnan(X)):
            X = SimpleImputer(strategy='mean').fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        rf.fit(X_train, y_train)
        acc = accuracy_score(y_test, rf.predict(X_test))
        
        # Majority class baseline
        from collections import Counter
        most_common_class = Counter(y_train).most_common(1)[0][0]
        baseline_acc = accuracy_score(y_test, [most_common_class] * len(y_test))
        
        print(f"\nQuick ML test (RandomForest on flattened data):")
        print(f"  Test accuracy: {acc:.3f}")
        print(f"  Baseline (majority class): {baseline_acc:.3f}")
        
    except ImportError:
        print("sklearn not available for ML test")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("3D SYNTHETIC DATASET GENERATOR - DEMO")
    print("Time Series Classification Datasets")
    print("=" * 70 + "\n")
    
    dataset = demo_basic_generation()
    demo_custom_prior()
    demo_temporal_structure()
    demo_statistics()
    demo_2d_compatibility()
    
    try:
        demo_visualization()
    except Exception as e:
        print(f"Visualization demo skipped: {e}")
    
    print("\n" + "=" * 70)
    print("ALL DEMOS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()

