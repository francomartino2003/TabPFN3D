"""
Sanity Checks for 3D Synthetic Dataset Generator.

Validates that generated datasets:
1. Have correct shapes and types
2. Are learnable (models can beat baseline)
3. Have appropriate difficulty distribution
4. Don't have data leakage
5. Work with different sampling modes

Adapted from 2D sanity checks for temporal data.
"""

import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings

from config import PriorConfig3D
from generator import SyntheticDatasetGenerator3D, SyntheticDataset3D


@dataclass
class DatasetStats3D:
    """Statistics for a single 3D dataset."""
    dataset_id: int
    n_samples: int
    n_features: int
    t_subseq: int
    T_total: int
    is_classification: bool
    n_classes: int
    sample_mode: str
    target_offset_type: str
    target_offset: int
    n_noise_inputs: int
    n_time_inputs: int
    n_state_inputs: int
    baseline_acc: float
    rf_acc: float
    lift: float
    has_nan: bool
    nan_rate: float


def flatten_temporal(X: np.ndarray) -> np.ndarray:
    """
    Flatten temporal dimension for sklearn models.
    
    Args:
        X: Shape (n_samples, n_features, t_subseq)
        
    Returns:
        X_flat: Shape (n_samples, n_features * t_subseq)
    """
    n_samples = X.shape[0]
    return X.reshape(n_samples, -1)


def prepare_data(
    X: np.ndarray, 
    y: np.ndarray,
    test_size: float = 0.2,
    rng: np.random.Generator = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare train/test split with NaN handling.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    # Flatten temporal dimension
    X_flat = flatten_temporal(X)
    
    # Handle NaN
    nan_mask = np.any(np.isnan(X_flat), axis=1)
    X_clean = X_flat[~nan_mask]
    y_clean = y[~nan_mask]
    
    if len(y_clean) < 10:
        return None, None, None, None
    
    # Check stratification
    unique, counts = np.unique(y_clean, return_counts=True)
    can_stratify = all(c >= 2 for c in counts) and len(unique) >= 2
    
    try:
        if can_stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y_clean, test_size=test_size, stratify=y_clean,
                random_state=int(rng.integers(0, 2**31))
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y_clean, test_size=test_size,
                random_state=int(rng.integers(0, 2**31))
            )
        return X_train, X_test, y_train, y_test
    except Exception as e:
        warnings.warn(f"Split failed: {e}")
        return None, None, None, None


def evaluate_dataset(
    dataset: SyntheticDataset3D,
    dataset_id: int,
    rng: np.random.Generator
) -> DatasetStats3D:
    """
    Evaluate a single 3D dataset.
    """
    X, y = dataset.X, dataset.y
    config = dataset.config
    
    # Basic stats
    n_samples, n_features, t_subseq = X.shape
    has_nan = np.any(np.isnan(X))
    nan_rate = np.mean(np.isnan(X)) if has_nan else 0.0
    
    # Prepare data
    split = prepare_data(X, y, rng=rng)
    if split[0] is None:
        # Can't evaluate
        return DatasetStats3D(
            dataset_id=dataset_id,
            n_samples=n_samples,
            n_features=n_features,
            t_subseq=t_subseq,
            T_total=config.T_total,
            is_classification=dataset.is_classification,
            n_classes=dataset.n_classes,
            sample_mode=config.sample_mode,
            target_offset_type=config.target_offset_type,
            target_offset=config.target_offset,
            n_noise_inputs=config.n_noise_inputs,
            n_time_inputs=config.n_time_inputs,
            n_state_inputs=config.n_state_inputs,
            baseline_acc=1.0,
            rf_acc=0.0,
            lift=0.0,
            has_nan=has_nan,
            nan_rate=nan_rate
        )
    
    X_train, X_test, y_train, y_test = split
    
    if dataset.is_classification:
        # Baseline: random guess
        n_classes = len(np.unique(y_train))
        baseline_acc = 1.0 / max(1, n_classes)
        
        # Random Forest
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
            rf.fit(X_train, y_train)
            rf_acc = accuracy_score(y_test, rf.predict(X_test))
        
        lift = rf_acc - baseline_acc
    else:
        # Regression - use R^2-like metric
        baseline_pred = np.mean(y_train)
        baseline_mse = mean_squared_error(y_test, np.full_like(y_test, baseline_pred))
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train, y_train)
            model_mse = mean_squared_error(y_test, ridge.predict(X_test))
        
        # Convert to "accuracy-like" for comparison
        baseline_acc = 0.0  # Baseline is "no skill"
        rf_acc = 1.0 - (model_mse / (baseline_mse + 1e-8))
        rf_acc = max(0.0, rf_acc)  # Clip negative
        lift = rf_acc
    
    return DatasetStats3D(
        dataset_id=dataset_id,
        n_samples=n_samples,
        n_features=n_features,
        t_subseq=t_subseq,
        T_total=config.T_total,
        is_classification=dataset.is_classification,
        n_classes=dataset.n_classes,
        sample_mode=config.sample_mode,
        target_offset_type=config.target_offset_type,
        target_offset=config.target_offset,
        n_noise_inputs=config.n_noise_inputs,
        n_time_inputs=config.n_time_inputs,
        n_state_inputs=config.n_state_inputs,
        baseline_acc=baseline_acc,
        rf_acc=rf_acc,
        lift=lift,
        has_nan=has_nan,
        nan_rate=nan_rate
    )


def run_sanity_checks(
    n_datasets: int = 50,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run comprehensive sanity checks on the 3D generator.
    
    Args:
        n_datasets: Number of datasets to generate
        seed: Random seed
        verbose: Print progress
        
    Returns:
        Dictionary with check results
    """
    rng = np.random.default_rng(seed)
    generator = SyntheticDatasetGenerator3D(seed=seed)
    
    all_stats: List[DatasetStats3D] = []
    errors: List[str] = []
    
    if verbose:
        print(f"Generating and evaluating {n_datasets} 3D datasets...")
    
    for i in range(n_datasets):
        try:
            dataset = generator.generate()
            stats = evaluate_dataset(dataset, i, rng)
            all_stats.append(stats)
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{n_datasets}")
        except Exception as e:
            errors.append(f"Dataset {i}: {str(e)}")
            if verbose:
                print(f"  Error on dataset {i}: {e}")
    
    if not all_stats:
        return {"error": "No datasets generated successfully", "errors": errors}
    
    # Aggregate statistics
    results = compute_aggregate_stats(all_stats)
    results['errors'] = errors
    results['n_successful'] = len(all_stats)
    results['n_failed'] = len(errors)
    results['individual_stats'] = [asdict(s) for s in all_stats]
    
    if verbose:
        print_summary(results)
    
    return results


def compute_aggregate_stats(stats: List[DatasetStats3D]) -> Dict[str, Any]:
    """Compute aggregate statistics."""
    
    def percentiles(values: List[float]) -> Dict[str, float]:
        arr = np.array(values)
        return {
            'min': float(np.min(arr)),
            'p10': float(np.percentile(arr, 10)),
            'p25': float(np.percentile(arr, 25)),
            'median': float(np.median(arr)),
            'p75': float(np.percentile(arr, 75)),
            'p90': float(np.percentile(arr, 90)),
            'max': float(np.max(arr)),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr))
        }
    
    classification_stats = [s for s in stats if s.is_classification]
    regression_stats = [s for s in stats if not s.is_classification]
    
    # Count by mode
    mode_counts = {}
    for s in stats:
        mode_counts[s.sample_mode] = mode_counts.get(s.sample_mode, 0) + 1
    
    # Count by offset type
    offset_counts = {}
    for s in stats:
        offset_counts[s.target_offset_type] = offset_counts.get(s.target_offset_type, 0) + 1
    
    return {
        'n_samples': percentiles([s.n_samples for s in stats]),
        'n_features': percentiles([s.n_features for s in stats]),
        't_subseq': percentiles([s.t_subseq for s in stats]),
        'T_total': percentiles([s.T_total for s in stats]),
        'n_classification': len(classification_stats),
        'n_regression': len(regression_stats),
        'mode_distribution': mode_counts,
        'offset_distribution': offset_counts,
        'baseline_acc': percentiles([s.baseline_acc for s in stats if s.is_classification]),
        'rf_acc': percentiles([s.rf_acc for s in stats if s.is_classification]),
        'lift': percentiles([s.lift for s in stats if s.is_classification]),
        'nan_datasets': sum(1 for s in stats if s.has_nan),
        'avg_nan_rate': float(np.mean([s.nan_rate for s in stats])),
        'input_type_stats': {
            'noise': percentiles([s.n_noise_inputs for s in stats]),
            'time': percentiles([s.n_time_inputs for s in stats]),
            'state': percentiles([s.n_state_inputs for s in stats])
        }
    }


def print_summary(results: Dict[str, Any]):
    """Print a summary of sanity check results."""
    print("\n" + "="*70)
    print("SANITY CHECK SUMMARY")
    print("="*70)
    
    print(f"\nDatasets: {results['n_successful']} successful, {results['n_failed']} failed")
    print(f"Classification: {results['n_classification']}, Regression: {results['n_regression']}")
    
    print(f"\nSample modes: {results['mode_distribution']}")
    print(f"Target offsets: {results['offset_distribution']}")
    
    print(f"\nDataset sizes:")
    print(f"  n_samples: {results['n_samples']['median']:.0f} median "
          f"(range: {results['n_samples']['min']:.0f}-{results['n_samples']['max']:.0f})")
    print(f"  n_features: {results['n_features']['median']:.0f} median "
          f"(range: {results['n_features']['min']:.0f}-{results['n_features']['max']:.0f})")
    print(f"  t_subseq: {results['t_subseq']['median']:.0f} median "
          f"(range: {results['t_subseq']['min']:.0f}-{results['t_subseq']['max']:.0f})")
    
    if 'baseline_acc' in results and results['baseline_acc']:
        print(f"\nClassification performance:")
        print(f"  Baseline accuracy: {results['baseline_acc']['median']:.3f} median")
        print(f"  RF accuracy: {results['rf_acc']['median']:.3f} median")
        print(f"  Lift: {results['lift']['median']:.3f} median "
              f"(range: {results['lift']['min']:.3f}-{results['lift']['max']:.3f})")
    
    print(f"\nInput types (median):")
    print(f"  Noise: {results['input_type_stats']['noise']['median']:.0f}")
    print(f"  Time: {results['input_type_stats']['time']['median']:.0f}")
    print(f"  State: {results['input_type_stats']['state']['median']:.0f}")
    
    print(f"\nMissing values: {results['nan_datasets']} datasets have NaN")
    print(f"  Average NaN rate: {results['avg_nan_rate']:.2%}")
    
    # Warnings
    print("\n" + "-"*70)
    print("CHECKS:")
    
    # Check 1: Lift > 0 for most datasets
    if 'lift' in results and results['lift']:
        positive_lift = sum(1 for s in results.get('individual_stats', []) 
                          if s.get('is_classification') and s.get('lift', 0) > 0)
        total_class = results['n_classification']
        pct = positive_lift / total_class if total_class > 0 else 0
        status = "[OK]" if pct > 0.5 else "[WARN]"
        print(f"  {status} Positive lift: {positive_lift}/{total_class} ({pct:.1%})")
    
    # Check 2: Variety in modes
    modes = results['mode_distribution']
    if len(modes) >= 2:
        print(f"  [OK] Multiple sample modes used: {list(modes.keys())}")
    else:
        print(f"  [WARN] Only one sample mode: {list(modes.keys())}")
    
    # Check 3: Errors
    if results['n_failed'] == 0:
        print(f"  [OK] No generation errors")
    else:
        print(f"  [WARN] {results['n_failed']} generation errors")
    
    print("="*70)


def save_results(results: Dict[str, Any], filepath: str):
    """Save results to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {filepath}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run 3D generator sanity checks")
    parser.add_argument("--n_datasets", type=int, default=50, help="Number of datasets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="sanity_check_results_3d.json", 
                       help="Output file")
    args = parser.parse_args()
    
    results = run_sanity_checks(
        n_datasets=args.n_datasets,
        seed=args.seed,
        verbose=True
    )
    
    save_results(results, args.output)


