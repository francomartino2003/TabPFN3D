"""
Analyze which real 3D datasets can be flattened for TabPFN

This script loads the pre-computed statistics from 01_real_data and identifies
datasets that satisfy TabPFN's constraints:
  - n_dimensions × length ≤ 500 (feature limit)
  - n_classes ≤ 10 (class limit)
  - n_samples ≤ 10,000 (sample limit - can be subsampled if exceeded)
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
REAL_DATA_DIR = PROJECT_ROOT / "01_real_data"
STATS_FILE = REAL_DATA_DIR / "AEON" / "data" / "classification_stats.json"
OUTPUT_DIR = Path(__file__).parent / "results"

# TabPFN constraints
TABPFN_MAX_FEATURES = 500
TABPFN_MAX_CLASSES = 10
TABPFN_MAX_SAMPLES = 10000  # Can be subsampled if exceeded


def load_dataset_stats() -> List[Dict]:
    """Load pre-computed statistics from 01_real_data"""
    if not STATS_FILE.exists():
        raise FileNotFoundError(
            f"Stats file not found: {STATS_FILE}\n"
            f"Run 'python 01_real_data/src/analyze_all_datasets.py' first."
        )
    
    with open(STATS_FILE, 'r') as f:
        stats = json.load(f)
    
    return stats


def compute_flattened_features(dataset: Dict) -> int:
    """Compute number of features when flattening (channels × timesteps)"""
    n_dimensions = dataset.get('n_dimensions', 1)
    length = dataset.get('length', 0)
    return n_dimensions * length


def analyze_flattenability(stats: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Categorize datasets by TabPFN compatibility.
    
    Returns:
        Dict with categories:
        - 'fully_compatible': Meets all constraints (features ≤500, classes ≤10, samples ≤10000)
        - 'needs_subsampling': Meets features & classes, but samples >10000 (can subsample)
        - 'too_many_classes': Features OK but classes >10
        - 'too_many_features': Features >500
    """
    categories = {
        'fully_compatible': [],
        'needs_subsampling': [],
        'too_many_classes': [],
        'too_many_features': [],
    }
    
    for ds in stats:
        # Compute flattened feature count
        flattened_features = compute_flattened_features(ds)
        n_classes = ds.get('n_classes', 0) or 0
        n_samples = ds.get('n_samples', 0) or 0
        
        # Add computed fields
        ds_enriched = ds.copy()
        ds_enriched['flattened_features'] = flattened_features
        ds_enriched['meets_feature_limit'] = flattened_features <= TABPFN_MAX_FEATURES
        ds_enriched['meets_class_limit'] = n_classes <= TABPFN_MAX_CLASSES
        ds_enriched['meets_sample_limit'] = n_samples <= TABPFN_MAX_SAMPLES
        ds_enriched['needs_subsampling'] = n_samples > TABPFN_MAX_SAMPLES
        
        # Categorize
        if flattened_features > TABPFN_MAX_FEATURES:
            categories['too_many_features'].append(ds_enriched)
        elif n_classes > TABPFN_MAX_CLASSES:
            categories['too_many_classes'].append(ds_enriched)
        elif n_samples > TABPFN_MAX_SAMPLES:
            categories['needs_subsampling'].append(ds_enriched)
        else:
            categories['fully_compatible'].append(ds_enriched)
    
    return categories


def compute_statistics(datasets: List[Dict]) -> Dict:
    """Compute summary statistics for a group of datasets"""
    if not datasets:
        return {}
    
    n_datasets = len(datasets)
    
    # Extract values (handle None)
    features = [d['flattened_features'] for d in datasets]
    samples = [d.get('n_samples', 0) for d in datasets]
    dimensions = [d.get('n_dimensions', 1) for d in datasets]
    lengths = [d.get('length', 0) for d in datasets]
    n_classes = [d.get('n_classes') for d in datasets if d.get('n_classes') is not None]
    train_sizes = [d.get('train_size') for d in datasets if d.get('train_size') is not None]
    test_sizes = [d.get('test_size') for d in datasets if d.get('test_size') is not None]
    
    # Count multivariate
    multivariate_count = sum(1 for d in datasets if d.get('n_dimensions', 1) > 1)
    univariate_count = n_datasets - multivariate_count
    
    # Count variable length
    variable_length_count = sum(1 for d in datasets if d.get('has_variable_length', False))
    
    stats = {
        'n_datasets': n_datasets,
        'n_univariate': univariate_count,
        'n_multivariate': multivariate_count,
        'n_variable_length': variable_length_count,
        'flattened_features': {
            'min': int(min(features)),
            'max': int(max(features)),
            'mean': float(np.mean(features)),
            'median': float(np.median(features)),
            'std': float(np.std(features)),
        },
        'n_samples': {
            'min': int(min(samples)),
            'max': int(max(samples)),
            'mean': float(np.mean(samples)),
            'median': float(np.median(samples)),
        },
        'n_dimensions': {
            'min': int(min(dimensions)),
            'max': int(max(dimensions)),
            'mean': float(np.mean(dimensions)),
        },
        'length': {
            'min': int(min(lengths)),
            'max': int(max(lengths)),
            'mean': float(np.mean(lengths)),
            'median': float(np.median(lengths)),
        },
    }
    
    if n_classes:
        stats['n_classes'] = {
            'min': int(min(n_classes)),
            'max': int(max(n_classes)),
            'mean': float(np.mean(n_classes)),
        }
    
    if train_sizes:
        stats['train_size'] = {
            'min': int(min(train_sizes)),
            'max': int(max(train_sizes)),
            'mean': float(np.mean(train_sizes)),
        }
    
    if test_sizes:
        stats['test_size'] = {
            'min': int(min(test_sizes)),
            'max': int(max(test_sizes)),
            'mean': float(np.mean(test_sizes)),
        }
    
    return stats


def print_summary(categories: Dict[str, List[Dict]], all_stats: List[Dict]):
    """Print a summary of the analysis"""
    total = len(all_stats)
    fully_compatible = categories['fully_compatible']
    needs_subsampling = categories['needs_subsampling']
    too_many_classes = categories['too_many_classes']
    too_many_features = categories['too_many_features']
    
    # Usable = fully compatible + needs subsampling (can be fixed)
    usable = fully_compatible + needs_subsampling
    
    print("=" * 80)
    print("TABPFN COMPATIBILITY ANALYSIS")
    print("=" * 80)
    print(f"\nTabPFN Constraints:")
    print(f"  • Features (channels × timesteps): ≤{TABPFN_MAX_FEATURES}")
    print(f"  • Classes: ≤{TABPFN_MAX_CLASSES}")
    print(f"  • Samples: ≤{TABPFN_MAX_SAMPLES} (can subsample if exceeded)")
    
    print(f"\n{'='*80}")
    print("📊 OVERVIEW")
    print(f"{'='*80}")
    print(f"   Total datasets analyzed: {total}")
    print(f"\n   ✅ FULLY COMPATIBLE: {len(fully_compatible)} ({100*len(fully_compatible)/total:.1f}%)")
    print(f"      (features ≤{TABPFN_MAX_FEATURES}, classes ≤{TABPFN_MAX_CLASSES}, samples ≤{TABPFN_MAX_SAMPLES})")
    print(f"\n   ⚠️  NEEDS SUBSAMPLING: {len(needs_subsampling)} ({100*len(needs_subsampling)/total:.1f}%)")
    print(f"      (features ≤{TABPFN_MAX_FEATURES}, classes ≤{TABPFN_MAX_CLASSES}, samples >{TABPFN_MAX_SAMPLES})")
    print(f"\n   ❌ TOO MANY CLASSES: {len(too_many_classes)} ({100*len(too_many_classes)/total:.1f}%)")
    print(f"      (features ≤{TABPFN_MAX_FEATURES}, classes >{TABPFN_MAX_CLASSES})")
    print(f"\n   ❌ TOO MANY FEATURES: {len(too_many_features)} ({100*len(too_many_features)/total:.1f}%)")
    print(f"      (features >{TABPFN_MAX_FEATURES})")
    
    print(f"\n   {'─'*60}")
    print(f"   🎯 TOTAL USABLE (fully + subsampling): {len(usable)} ({100*len(usable)/total:.1f}%)")
    print(f"   {'─'*60}")
    
    # Statistics for usable datasets
    if usable:
        usable_stats = compute_statistics(usable)
        print(f"\n{'='*80}")
        print("📋 USABLE DATASETS STATISTICS")
        print(f"{'='*80}")
        print(f"   Univariate: {usable_stats['n_univariate']}")
        print(f"   Multivariate: {usable_stats['n_multivariate']}")
        print(f"   Variable length: {usable_stats['n_variable_length']}")
        print(f"\n   Flattened features:")
        print(f"     Range: {usable_stats['flattened_features']['min']} - {usable_stats['flattened_features']['max']}")
        print(f"     Mean: {usable_stats['flattened_features']['mean']:.1f}")
        print(f"     Median: {usable_stats['flattened_features']['median']:.1f}")
        print(f"\n   Samples (train + test):")
        print(f"     Range: {usable_stats['n_samples']['min']} - {usable_stats['n_samples']['max']}")
        print(f"     Mean: {usable_stats['n_samples']['mean']:.1f}")
        print(f"\n   Temporal length:")
        print(f"     Range: {usable_stats['length']['min']} - {usable_stats['length']['max']}")
        print(f"     Mean: {usable_stats['length']['mean']:.1f}")
        print(f"\n   Dimensions/Channels:")
        print(f"     Range: {usable_stats['n_dimensions']['min']} - {usable_stats['n_dimensions']['max']}")
        print(f"     Mean: {usable_stats['n_dimensions']['mean']:.1f}")
        if 'n_classes' in usable_stats:
            print(f"\n   Classes:")
            print(f"     Range: {usable_stats['n_classes']['min']} - {usable_stats['n_classes']['max']}")
            print(f"     Mean: {usable_stats['n_classes']['mean']:.1f}")
    
    # List fully compatible datasets
    print(f"\n{'='*80}")
    print(f"📝 FULLY COMPATIBLE DATASETS ({len(fully_compatible)})")
    print(f"{'='*80}")
    print(f"{'Dataset':<35} {'Dims':<5} {'Len':<6} {'Feat':<6} {'Samp':<7} {'Cls':<5} {'Train':<6} {'Test':<6}")
    print("-" * 80)
    
    sorted_compat = sorted(fully_compatible, key=lambda x: x['flattened_features'])
    for ds in sorted_compat:
        name = ds['name'][:34]
        dims = ds.get('n_dimensions', 1)
        length = ds.get('length', 0)
        features = ds['flattened_features']
        samples = ds.get('n_samples', 0)
        classes = ds.get('n_classes', '-')
        train = ds.get('train_size', '-')
        test = ds.get('test_size', '-')
        print(f"{name:<35} {dims:<5} {length:<6} {features:<6} {samples:<7} {classes:<5} {train:<6} {test:<6}")
    
    print("-" * 80)
    
    # List datasets needing subsampling
    if needs_subsampling:
        print(f"\n{'='*80}")
        print(f"⚠️  DATASETS NEEDING SUBSAMPLING ({len(needs_subsampling)})")
        print(f"{'='*80}")
        print(f"{'Dataset':<35} {'Dims':<5} {'Len':<6} {'Feat':<6} {'Samp':<7} {'Cls':<5} {'Train':<6} {'Test':<6}")
        print("-" * 80)
        
        sorted_sub = sorted(needs_subsampling, key=lambda x: x['n_samples'])
        for ds in sorted_sub:
            name = ds['name'][:34]
            dims = ds.get('n_dimensions', 1)
            length = ds.get('length', 0)
            features = ds['flattened_features']
            samples = ds.get('n_samples', 0)
            classes = ds.get('n_classes', '-')
            train = ds.get('train_size', '-')
            test = ds.get('test_size', '-')
            print(f"{name:<35} {dims:<5} {length:<6} {features:<6} {samples:<7} {classes:<5} {train:<6} {test:<6}")
        
        print("-" * 80)
    
    # Breakdown by feature ranges for usable
    print(f"\n📈 USABLE DATASETS BY FEATURE RANGE:")
    ranges = [(0, 50), (51, 100), (101, 200), (201, 300), (301, 400), (401, 500)]
    for low, high in ranges:
        count = sum(1 for d in usable if low <= d['flattened_features'] <= high)
        bar = "█" * (count // 2) + ("▌" if count % 2 else "")
        print(f"   {low:3d}-{high:3d} features: {count:3d} datasets  {bar}")
    
    # Too many classes summary
    if too_many_classes:
        print(f"\n❌ EXCLUDED: TOO MANY CLASSES (>{TABPFN_MAX_CLASSES})")
        print(f"   Total: {len(too_many_classes)}")
        sorted_classes = sorted(too_many_classes, key=lambda x: x.get('n_classes', 0))
        print(f"   Examples (closest to limit):")
        for ds in sorted_classes[:5]:
            print(f"     {ds['name']}: {ds.get('n_classes', '?')} classes, {ds['flattened_features']} features")
    
    # Too many features summary
    if too_many_features:
        print(f"\n❌ EXCLUDED: TOO MANY FEATURES (>{TABPFN_MAX_FEATURES})")
        print(f"   Total: {len(too_many_features)}")
        sorted_feat = sorted(too_many_features, key=lambda x: x['flattened_features'])
        print(f"   Closest to limit:")
        for ds in sorted_feat[:5]:
            print(f"     {ds['name']}: {ds['flattened_features']} features ({ds.get('n_dimensions', 1)} dims × {ds.get('length', 0)} length)")


def save_results(categories: Dict[str, List[Dict]]):
    """Save results to JSON files"""
    import csv
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    fully_compatible = categories['fully_compatible']
    needs_subsampling = categories['needs_subsampling']
    usable = fully_compatible + needs_subsampling
    
    # Save fully compatible datasets
    compat_output = OUTPUT_DIR / "fully_compatible_datasets.json"
    with open(compat_output, 'w') as f:
        json.dump({
            'description': f'Datasets fully compatible with TabPFN (features ≤{TABPFN_MAX_FEATURES}, classes ≤{TABPFN_MAX_CLASSES}, samples ≤{TABPFN_MAX_SAMPLES})',
            'constraints': {
                'max_features': TABPFN_MAX_FEATURES,
                'max_classes': TABPFN_MAX_CLASSES,
                'max_samples': TABPFN_MAX_SAMPLES,
            },
            'n_datasets': len(fully_compatible),
            'statistics': compute_statistics(fully_compatible) if fully_compatible else {},
            'datasets': sorted(fully_compatible, key=lambda x: x['flattened_features'])
        }, f, indent=2)
    print(f"\n💾 Saved fully compatible datasets to: {compat_output}")
    
    # Save datasets needing subsampling
    subsample_output = OUTPUT_DIR / "needs_subsampling_datasets.json"
    with open(subsample_output, 'w') as f:
        json.dump({
            'description': f'Datasets that need subsampling (features ≤{TABPFN_MAX_FEATURES}, classes ≤{TABPFN_MAX_CLASSES}, samples >{TABPFN_MAX_SAMPLES})',
            'constraints': {
                'max_features': TABPFN_MAX_FEATURES,
                'max_classes': TABPFN_MAX_CLASSES,
                'max_samples': TABPFN_MAX_SAMPLES,
            },
            'n_datasets': len(needs_subsampling),
            'statistics': compute_statistics(needs_subsampling) if needs_subsampling else {},
            'datasets': sorted(needs_subsampling, key=lambda x: x['n_samples'])
        }, f, indent=2)
    print(f"💾 Saved datasets needing subsampling to: {subsample_output}")
    
    # Save all usable datasets (combined)
    usable_output = OUTPUT_DIR / "usable_datasets.json"
    with open(usable_output, 'w') as f:
        json.dump({
            'description': f'All usable datasets for TabPFN (fully compatible + needs subsampling)',
            'constraints': {
                'max_features': TABPFN_MAX_FEATURES,
                'max_classes': TABPFN_MAX_CLASSES,
                'max_samples': f'{TABPFN_MAX_SAMPLES} (can be subsampled)',
            },
            'n_datasets': len(usable),
            'n_fully_compatible': len(fully_compatible),
            'n_needs_subsampling': len(needs_subsampling),
            'statistics': compute_statistics(usable) if usable else {},
            'datasets': sorted(usable, key=lambda x: x['flattened_features'])
        }, f, indent=2)
    print(f"💾 Saved all usable datasets to: {usable_output}")
    
    # Save excluded datasets
    excluded = categories['too_many_classes'] + categories['too_many_features']
    excluded_output = OUTPUT_DIR / "excluded_datasets.json"
    with open(excluded_output, 'w') as f:
        json.dump({
            'description': 'Datasets excluded due to TabPFN constraints',
            'constraints': {
                'max_features': TABPFN_MAX_FEATURES,
                'max_classes': TABPFN_MAX_CLASSES,
            },
            'n_datasets': len(excluded),
            'n_too_many_classes': len(categories['too_many_classes']),
            'n_too_many_features': len(categories['too_many_features']),
            'too_many_classes': sorted(categories['too_many_classes'], key=lambda x: x.get('n_classes', 0)),
            'too_many_features': sorted(categories['too_many_features'], key=lambda x: x['flattened_features']),
        }, f, indent=2)
    print(f"💾 Saved excluded datasets to: {excluded_output}")
    
    # Save summary CSV for usable datasets
    csv_output = OUTPUT_DIR / "usable_datasets_summary.csv"
    with open(csv_output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'name', 'n_dimensions', 'length', 'flattened_features', 
            'n_samples', 'train_size', 'test_size', 'n_classes',
            'needs_subsampling', 'has_variable_length', 'missing_pct'
        ])
        writer.writeheader()
        for ds in sorted(usable, key=lambda x: x['flattened_features']):
            writer.writerow({
                'name': ds['name'],
                'n_dimensions': ds.get('n_dimensions', 1),
                'length': ds.get('length', 0),
                'flattened_features': ds['flattened_features'],
                'n_samples': ds.get('n_samples', 0),
                'train_size': ds.get('train_size', ''),
                'test_size': ds.get('test_size', ''),
                'n_classes': ds.get('n_classes', ''),
                'needs_subsampling': ds.get('needs_subsampling', False),
                'has_variable_length': ds.get('has_variable_length', False),
                'missing_pct': ds.get('missing_pct', 0),
            })
    print(f"💾 Saved usable datasets CSV to: {csv_output}")


def main():
    """Main analysis pipeline"""
    print("Loading dataset statistics...")
    stats = load_dataset_stats()
    print(f"Loaded {len(stats)} datasets from {STATS_FILE}")
    
    print("\nAnalyzing TabPFN compatibility...")
    categories = analyze_flattenability(stats)
    
    print_summary(categories, stats)
    
    save_results(categories)
    
    # Summary counts
    n_fully = len(categories['fully_compatible'])
    n_sub = len(categories['needs_subsampling'])
    n_usable = n_fully + n_sub
    
    print(f"\n{'='*80}")
    print("✅ ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"   🎯 Total usable datasets: {n_usable}")
    print(f"      • Fully compatible: {n_fully}")
    print(f"      • Needs subsampling: {n_sub}")
    
    return categories


if __name__ == "__main__":
    main()
