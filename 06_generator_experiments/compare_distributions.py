"""
Compare distributions of real AEON datasets vs synthetic datasets from dataset_generator_v2.
Includes TabPFN benchmark comparison.
"""

import json
import numpy as np
import os
from dataclasses import dataclass
from typing import List, Dict, Optional
from sklearn.metrics import accuracy_score, roc_auc_score

from dataset_generator_v2 import generate_dataset, GeneratedDataset, get_default_nn_config, get_default_dataset_params
from random_nn_generator_v2 import RandomNNConfig


@dataclass
class DatasetStats:
    """Stats for a single dataset."""
    name: str
    n_samples: int
    n_features: int
    t_length: int
    m_times_t: int
    n_classes: int
    class_balance: float  # min_class / max_class
    missing_pct: float = 0.0  # percentage of missing values
    min_samples_per_class: int = 0  # minimum samples in any class
    accuracy: Optional[float] = None
    auc: Optional[float] = None


def load_aeon_stats(json_path: str) -> List[Dict]:
    """Load AEON dataset stats from JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)


# Global cache for loaded datasets
_AEON_DATASETS_CACHE = None

def load_aeon_datasets_pkl(pkl_path: str = None):
    """Load all AEON datasets from pkl file."""
    global _AEON_DATASETS_CACHE
    if _AEON_DATASETS_CACHE is not None:
        return _AEON_DATASETS_CACHE
    
    if pkl_path is None:
        pkl_path = "/Users/franco/Documents/TabPFN3D/01_real_data/AEON/data/classification_datasets.pkl"
    
    # Add src path for TimeSeriesDataset class
    import sys
    sys.path.insert(0, "/Users/franco/Documents/TabPFN3D/01_real_data")
    
    import pickle
    with open(pkl_path, 'rb') as f:
        datasets = pickle.load(f)
    
    # Build dict by name for fast lookup
    _AEON_DATASETS_CACHE = {ds.name: ds for ds in datasets}
    return _AEON_DATASETS_CACHE

def load_aeon_dataset(name: str):
    """Load actual AEON dataset for benchmarking from pkl cache."""
    try:
        datasets = load_aeon_datasets_pkl()
        if name not in datasets:
            return None, None, None, None
        
        ds = datasets[name]
        # TimeSeriesDataset has X_train, y_train, X_test, y_test
        return ds.X_train, ds.y_train, ds.X_test, ds.y_test
    except Exception as e:
        return None, None, None, None


def run_tabpfn_on_dataset(X_train, y_train, X_test, y_test, n_classes, tabpfn):
    """Run TabPFN on a dataset and return accuracy and AUC."""
    try:
        # Flatten 3D to 2D if needed
        if len(X_train.shape) == 3:
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
        else:
            X_train_flat = X_train
            X_test_flat = X_test
        
        # Skip if too many features
        if X_train_flat.shape[1] > 500:
            return None, None
        
        # Encode labels if string
        if isinstance(y_train[0], str):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)
        
        tabpfn.fit(X_train_flat, y_train)
        y_pred = tabpfn.predict(X_test_flat)
        y_proba = tabpfn.predict_proba(X_test_flat)
        
        acc = accuracy_score(y_test, y_pred)
        
        if n_classes == 2:
            auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            try:
                auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
            except:
                auc = None
        
        return acc, auc
    except Exception as e:
        return None, None


def filter_aeon_datasets(
    datasets: List[Dict],
    max_samples: int = 1000,
    max_features: int = 12,
    max_m_times_t: int = 500,
    max_classes: int = 10,
    run_benchmark: bool = False,
    tabpfn = None
) -> List[DatasetStats]:
    """Filter AEON datasets by constraints and return stats."""
    # First pass: filter datasets
    candidates = []
    for ds in datasets:
        n_samples = ds['n_samples']
        n_features = ds['n_dimensions']
        t_length = ds['length']
        n_classes = ds['n_classes']
        m_times_t = n_features * t_length
        
        # Apply filters
        if n_samples > max_samples:
            continue
        if n_features > max_features:
            continue
        if m_times_t > max_m_times_t:
            continue
        if n_classes > max_classes:
            continue
        candidates.append(ds)
    
    print(f"  {len(candidates)} datasets pass filters")
    
    filtered = []
    for i, ds in enumerate(candidates):
        n_samples = ds['n_samples']
        n_features = ds['n_dimensions']
        t_length = ds['length']
        n_classes = ds['n_classes']
        m_times_t = n_features * t_length
        class_balance = ds.get('class_balance', 1.0)
        missing_pct = ds.get('missing_pct', 0.0)
        
        # Calculate min_samples_per_class from class_balance
        avg_per_class = n_samples / n_classes
        min_samples_per_class = int(class_balance * avg_per_class)
        
        acc, auc = None, None
        if run_benchmark and tabpfn is not None:
            X_train, y_train, X_test, y_test = load_aeon_dataset(ds['name'])
            if X_train is not None:
                acc, auc = run_tabpfn_on_dataset(
                    X_train, y_train, X_test, y_test, n_classes, tabpfn
                )
                status = f"acc={acc:.3f}" if acc else "skipped"
                print(f"  [{i+1}/{len(candidates)}] {ds['name']}: {status}")
        
        filtered.append(DatasetStats(
            name=ds['name'],
            n_samples=n_samples,
            n_features=n_features,
            t_length=t_length,
            m_times_t=m_times_t,
            n_classes=n_classes,
            class_balance=class_balance,
            missing_pct=missing_pct,
            min_samples_per_class=min_samples_per_class,
            accuracy=acc,
            auc=auc
        ))
    
    return filtered


def generate_synthetic_stats(n_datasets: int = 100, run_benchmark: bool = False, tabpfn = None) -> List[DatasetStats]:
    """Generate synthetic datasets and collect stats."""
    
    # Import config from dataset_generator_v2 (single source of truth)
    nn_config = get_default_nn_config(seq_length=200)
    dataset_params = get_default_dataset_params()
    
    stats = []
    
    for i in range(n_datasets):
        for attempt in range(10):
            try:
                seed = 1000 + i * 100 + attempt
                dataset = generate_dataset(
                    nn_config=nn_config,
                    max_samples=dataset_params['max_samples'],
                    max_features=dataset_params['max_features'],
                    feature_geometric_p=dataset_params['feature_geometric_p'],
                    t_subseq_range=dataset_params['t_subseq_range'],
                    t_excess_range=dataset_params['t_excess_range'],
                    max_m_times_t=dataset_params['max_m_times_t'],
                    train_ratio=dataset_params['train_ratio'],
                    seed=seed
                )
                
                n_total = len(dataset.X_train) + len(dataset.X_test)
                y_all = np.concatenate([dataset.y_train, dataset.y_test])
                
                # Calculate class balance and min samples per class
                unique, counts = np.unique(y_all, return_counts=True)
                if len(counts) > 1:
                    class_balance = counts.min() / counts.max()
                    min_samples_per_class = int(counts.min())
                else:
                    class_balance = 1.0
                    min_samples_per_class = n_total
                
                # Synthetic datasets have no missing values
                missing_pct = 0.0
                
                # Run benchmark if requested
                acc, auc = None, None
                if run_benchmark and tabpfn is not None:
                    acc, auc = run_tabpfn_on_dataset(
                        dataset.X_train, dataset.y_train,
                        dataset.X_test, dataset.y_test,
                        dataset.n_classes, tabpfn
                    )
                    status = f"acc={acc:.3f}" if acc else "skipped"
                    print(f"  [{i+1}/{n_datasets}] synthetic_{i}: {status}")
                
                stats.append(DatasetStats(
                    name=f"synthetic_{i}",
                    n_samples=n_total,
                    n_features=dataset.n_features,
                    t_length=dataset.t_length,
                    m_times_t=dataset.n_features * dataset.t_length,
                    n_classes=dataset.n_classes,
                    class_balance=class_balance,
                    missing_pct=missing_pct,
                    min_samples_per_class=min_samples_per_class,
                    accuracy=acc,
                    auc=auc
                ))
                break
                
            except ValueError:
                continue
    
    return stats


def print_distribution_stats(stats: List[DatasetStats], name: str):
    """Print distribution statistics."""
    if not stats:
        print(f"\n{name}: No datasets")
        return
    
    n_samples = [s.n_samples for s in stats]
    n_features = [s.n_features for s in stats]
    t_length = [s.t_length for s in stats]
    m_times_t = [s.m_times_t for s in stats]
    n_classes = [s.n_classes for s in stats]
    class_balance = [s.class_balance for s in stats]
    missing_pct = [s.missing_pct for s in stats]
    min_samples_class = [s.min_samples_per_class for s in stats]
    accuracies = [s.accuracy for s in stats if s.accuracy is not None]
    aucs = [s.auc for s in stats if s.auc is not None]
    
    print(f"\n{'='*60}")
    print(f"{name} ({len(stats)} datasets)")
    print(f"{'='*60}")
    
    print(f"\n{'Metric':<20} {'Min':>10} {'Max':>10} {'Mean':>10} {'Median':>10}")
    print("-" * 60)
    print(f"{'n_samples':<20} {min(n_samples):>10} {max(n_samples):>10} {np.mean(n_samples):>10.1f} {np.median(n_samples):>10.1f}")
    print(f"{'n_features':<20} {min(n_features):>10} {max(n_features):>10} {np.mean(n_features):>10.1f} {np.median(n_features):>10.1f}")
    print(f"{'t_length':<20} {min(t_length):>10} {max(t_length):>10} {np.mean(t_length):>10.1f} {np.median(t_length):>10.1f}")
    print(f"{'m_times_t':<20} {min(m_times_t):>10} {max(m_times_t):>10} {np.mean(m_times_t):>10.1f} {np.median(m_times_t):>10.1f}")
    print(f"{'n_classes':<20} {min(n_classes):>10} {max(n_classes):>10} {np.mean(n_classes):>10.1f} {np.median(n_classes):>10.1f}")
    print(f"{'class_balance':<20} {min(class_balance):>10.3f} {max(class_balance):>10.3f} {np.mean(class_balance):>10.3f} {np.median(class_balance):>10.3f}")
    print(f"{'missing_pct':<20} {min(missing_pct):>10.3f} {max(missing_pct):>10.3f} {np.mean(missing_pct):>10.3f} {np.median(missing_pct):>10.3f}")
    print(f"{'min_samples/class':<20} {min(min_samples_class):>10} {max(min_samples_class):>10} {np.mean(min_samples_class):>10.1f} {np.median(min_samples_class):>10.1f}")
    
    # TabPFN benchmark results
    if accuracies:
        print(f"{'accuracy':<20} {min(accuracies):>10.3f} {max(accuracies):>10.3f} {np.mean(accuracies):>10.3f} {np.median(accuracies):>10.3f}")
    if aucs:
        print(f"{'auc_roc':<20} {min(aucs):>10.3f} {max(aucs):>10.3f} {np.mean(aucs):>10.3f} {np.median(aucs):>10.3f}")
    
    # Distribution of discrete values
    print(f"\n--- Distribution ---")
    
    print(f"\nn_features distribution:")
    for v in sorted(set(n_features)):
        count = n_features.count(v)
        pct = 100 * count / len(n_features)
        bar = '█' * int(pct / 2)
        print(f"  {v:>3}: {count:>4} ({pct:>5.1f}%) {bar}")
    
    print(f"\nn_classes distribution:")
    for v in sorted(set(n_classes)):
        count = n_classes.count(v)
        pct = 100 * count / len(n_classes)
        bar = '█' * int(pct / 2)
        print(f"  {v:>3}: {count:>4} ({pct:>5.1f}%) {bar}")
    
    # Accuracy distribution (binned)
    if accuracies:
        print(f"\naccuracy distribution:")
        bins = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.01)]
        for lo, hi in bins:
            count = sum(1 for a in accuracies if lo <= a < hi)
            pct = 100 * count / len(accuracies)
            bar = '█' * int(pct / 2)
            print(f"  {lo:.2f}-{hi:.2f}: {count:>4} ({pct:>5.1f}%) {bar}")


def print_detailed_benchmark_results(stats: List[DatasetStats], title: str):
    """Print detailed benchmark results per dataset."""
    # Filter only datasets with benchmark results
    with_results = [s for s in stats if s.accuracy is not None]
    if not with_results:
        return
    
    print(f"\n{'='*80}")
    print(f"DETAILED BENCHMARK RESULTS: {title}")
    print(f"{'='*80}")
    print(f"\n{'Dataset':<35} {'Samples':>8} {'Feat':>5} {'T':>5} {'Cls':>4} {'Acc':>7} {'AUC':>7}")
    print("-" * 80)
    
    # Sort by accuracy descending
    sorted_stats = sorted(with_results, key=lambda x: x.accuracy or 0, reverse=True)
    
    for s in sorted_stats:
        auc_str = f"{s.auc:.3f}" if s.auc is not None else "  N/A"
        print(f"{s.name:<35} {s.n_samples:>8} {s.n_features:>5} {s.t_length:>5} {s.n_classes:>4} {s.accuracy:>7.3f} {auc_str:>7}")
    
    print("-" * 80)
    accs = [s.accuracy for s in sorted_stats]
    aucs = [s.auc for s in sorted_stats if s.auc is not None]
    print(f"{'MEAN':<35} {'':<8} {'':<5} {'':<5} {'':<4} {np.mean(accs):>7.3f} {np.mean(aucs) if aucs else 0:>7.3f}")
    print(f"{'MEDIAN':<35} {'':<8} {'':<5} {'':<5} {'':<4} {np.median(accs):>7.3f} {np.median(aucs) if aucs else 0:>7.3f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', action='store_true', help='Run TabPFN benchmark')
    parser.add_argument('--n-synthetic', type=int, default=50, help='Number of synthetic datasets')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Distribution Comparison: AEON vs Synthetic")
    print("=" * 60)
    
    # Constraints
    max_samples = 1000
    max_features = 12
    max_m_times_t = 500
    max_classes = 10
    
    print(f"\nConstraints:")
    print(f"  max_samples: {max_samples}")
    print(f"  max_features: {max_features}")
    print(f"  max_m_times_t: {max_m_times_t}")
    print(f"  max_classes: {max_classes}")
    print(f"  run_benchmark: {args.benchmark}")
    
    # Initialize TabPFN if benchmarking
    tabpfn = None
    if args.benchmark:
        print("\nInitializing TabPFN...")
        from tabpfn import TabPFNClassifier
        tabpfn = TabPFNClassifier(n_estimators=4, ignore_pretraining_limits=True)
    
    # Load and filter AEON
    json_path = "/Users/franco/Documents/TabPFN3D/01_real_data/AEON/data/classification_stats.json"
    aeon_all = load_aeon_stats(json_path)
    print(f"\nLoaded {len(aeon_all)} AEON datasets")
    
    print("Filtering and benchmarking AEON datasets...")
    aeon_filtered = filter_aeon_datasets(
        aeon_all,
        max_samples=max_samples,
        max_features=max_features,
        max_m_times_t=max_m_times_t,
        max_classes=max_classes,
        run_benchmark=args.benchmark,
        tabpfn=tabpfn
    )
    print(f"After filtering: {len(aeon_filtered)} datasets")
    
    # Generate synthetic
    print(f"\nGenerating {args.n_synthetic} synthetic datasets...")
    synthetic_stats = generate_synthetic_stats(
        n_datasets=args.n_synthetic,
        run_benchmark=args.benchmark,
        tabpfn=tabpfn
    )
    print(f"Generated: {len(synthetic_stats)} datasets")
    
    # Print comparisons
    print_distribution_stats(aeon_filtered, "AEON (filtered)")
    print_distribution_stats(synthetic_stats, "Synthetic (dataset_generator_v2)")
    
    # Print detailed benchmark results if available
    if args.benchmark:
        print_detailed_benchmark_results(aeon_filtered, "AEON (Real Datasets)")
        print_detailed_benchmark_results(synthetic_stats, "Synthetic Datasets")


if __name__ == "__main__":
    main()
