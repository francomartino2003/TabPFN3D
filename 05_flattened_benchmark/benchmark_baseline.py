"""
Baseline benchmark: TabPFN on flattened 3D time series datasets

This script:
1. Loads usable datasets from pre-downloaded pickle (train/test already separated)
2. Subsamples if needed (>10000 samples) - SEPARATELY for train and test
3. Flattens 3D data: (n, channels, length) -> (n, channels * length)
4. Runs TabPFN classifier (fit on TRAIN only, predict on TEST only)
5. Reports AUC-ROC and Accuracy

IMPORTANT: No data leakage - train and test are kept strictly separate throughout:
- Datasets come pre-split from UCR/UEA archive
- LabelEncoder is fit ONLY on train labels
- Missing value imputation uses ONLY train statistics
- TabPFN.fit() uses ONLY train data
- Metrics computed ONLY on test predictions
"""
import json
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
REAL_DATA_DIR = PROJECT_ROOT / "01_real_data"
DATASETS_PKL = REAL_DATA_DIR / "AEON" / "data" / "classification_datasets.pkl"
RESULTS_DIR = Path(__file__).parent / "results"
USABLE_DATASETS_FILE = RESULTS_DIR / "usable_datasets.json"

# Add paths for imports
sys.path.insert(0, str(PROJECT_ROOT / "00_TabPFN" / "src"))
sys.path.insert(0, str(REAL_DATA_DIR))

# TabPFN constraints
TABPFN_MAX_SAMPLES = 10000
RANDOM_STATE = 42

# Global cache for loaded datasets (to avoid re-reading pickle for each dataset)
_DATASETS_CACHE = None


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


def _load_all_datasets_from_pkl() -> Dict:
    """
    Load all datasets from pickle file into a dictionary.
    Uses caching to avoid re-reading for each dataset.
    
    Returns:
        Dict mapping dataset name -> TimeSeriesDataset object
    """
    global _DATASETS_CACHE
    
    if _DATASETS_CACHE is not None:
        return _DATASETS_CACHE
    
    if not DATASETS_PKL.exists():
        raise FileNotFoundError(
            f"Datasets pickle not found: {DATASETS_PKL}\n"
            f"Run 'python 01_real_data/src/analyze_all_datasets.py' first."
        )
    
    print(f"Loading datasets from {DATASETS_PKL}...")
    with open(DATASETS_PKL, 'rb') as f:
        datasets_list = pickle.load(f)
    
    # Convert to dict for O(1) lookup
    _DATASETS_CACHE = {ds.name: ds for ds in datasets_list}
    print(f"Loaded {len(_DATASETS_CACHE)} datasets into cache")
    
    return _DATASETS_CACHE


def load_dataset(name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Load a dataset from the pre-downloaded pickle file.
    
    The pickle contains TimeSeriesDataset objects with train/test ALREADY SEPARATED
    as they come from the UCR/UEA archive. This ensures we use the official splits
    and avoid any data leakage.
    
    Returns:
        X_train, y_train, X_test, y_test, metadata
        
    Data flow:
        pickle (pre-split) -> X_train, y_train (TRAIN ONLY)
                           -> X_test, y_test   (TEST ONLY)
    """
    datasets = _load_all_datasets_from_pkl()
    
    if name not in datasets:
        raise ValueError(f"Dataset '{name}' not found in pickle file")
    
    dataset = datasets[name]
    
    # =========================================================================
    # TRAIN DATA - from official UCR/UEA train split
    # =========================================================================
    X_train = dataset.X_train.copy()  # Shape: (n_train, channels, length)
    y_train = dataset.y_train.copy()  # Shape: (n_train,)
    
    # =========================================================================
    # TEST DATA - from official UCR/UEA test split (NEVER seen during training)
    # =========================================================================
    X_test = dataset.X_test.copy()    # Shape: (n_test, channels, length)
    y_test = dataset.y_test.copy()    # Shape: (n_test,)
    
    # Metadata (computed from train only to avoid leakage)
    metadata = {
        'name': name,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_channels': X_train.shape[1],
        'length': X_train.shape[2],
        'n_classes': len(np.unique(y_train)),  # Count classes from TRAIN only
    }
    
    return X_train, y_train, X_test, y_test, metadata


def flatten_3d(X: np.ndarray) -> np.ndarray:
    """
    Flatten 3D array to 2D.
    
    Input: (n_samples, n_channels, length)
    Output: (n_samples, n_channels * length)
    
    Flattening strategy: concatenate channels, preserving temporal order within each channel.
    Result: [ch0_t0, ch0_t1, ..., ch0_tT, ch1_t0, ch1_t1, ..., ch1_tT, ...]
    """
    n_samples, n_channels, length = X.shape
    return X.reshape(n_samples, n_channels * length)


def subsample_stratified(X: np.ndarray, y: np.ndarray, max_samples: int, 
                         random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stratified subsampling to reduce dataset size while maintaining class proportions.
    
    Args:
        X: Features array
        y: Labels array
        max_samples: Maximum number of samples to keep
        random_state: Random seed for reproducibility
        
    Returns:
        Subsampled X, y (maintaining class proportions)
    """
    if len(X) <= max_samples:
        return X, y
    
    from sklearn.model_selection import train_test_split
    
    ratio = max_samples / len(X)
    
    # Stratified sampling - keep ratio portion, discard the rest
    X_keep, _, y_keep, _ = train_test_split(
        X, y, 
        train_size=ratio,
        stratify=y,
        random_state=random_state
    )
    
    return X_keep, y_keep


def preprocess_labels(y_train: np.ndarray, y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    Encode labels as integers (required for TabPFN).
    
    IMPORTANT: LabelEncoder is fit ONLY on train labels to avoid data leakage.
    Test labels are then transformed using the encoder fit on train.
    
    Args:
        y_train: Training labels (used for fitting encoder)
        y_test: Test labels (transformed only, not used for fitting)
        
    Returns:
        y_train_encoded, y_test_encoded, label_encoder
    """
    le = LabelEncoder()
    
    # =========================================================================
    # FIT ONLY ON TRAIN - no test data used here
    # =========================================================================
    le.fit(y_train)
    
    # Transform both using encoder fit on train
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    
    return y_train_enc, y_test_enc, le


def handle_missing_values(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Handle missing values (NaN) in the data.
    
    IMPORTANT: Statistics (column means) are computed ONLY from train data
    to avoid data leakage. The same train-derived means are used to impute
    both train and test.
    
    Args:
        X_train: Training features (used to compute imputation statistics)
        X_test: Test features (imputed using train statistics)
        
    Returns:
        X_train_imputed, X_test_imputed
    """
    has_nan_train = np.isnan(X_train).any()
    has_nan_test = np.isnan(X_test).any()
    
    if not has_nan_train and not has_nan_test:
        return X_train, X_test
    
    X_train_clean = X_train.copy()
    X_test_clean = X_test.copy()
    
    # =========================================================================
    # COMPUTE IMPUTATION VALUES FROM TRAIN ONLY
    # =========================================================================
    for col in range(X_train.shape[1]):
        # Compute mean from TRAIN data only
        train_col = X_train_clean[:, col]
        col_mean = np.nanmean(train_col)
        
        if np.isnan(col_mean):  # All train values are NaN
            col_mean = 0.0
        
        # Impute train NaNs
        train_mask = np.isnan(train_col)
        if train_mask.any():
            X_train_clean[train_mask, col] = col_mean
        
        # Impute test NaNs using TRAIN-derived mean
        test_col = X_test_clean[:, col]
        test_mask = np.isnan(test_col)
        if test_mask.any():
            X_test_clean[test_mask, col] = col_mean
    
    return X_train_clean, X_test_clean


def compute_roc_auc(y_true: np.ndarray, y_proba: np.ndarray, n_classes: int) -> float:
    """
    Compute ROC AUC score, handling binary and multiclass cases.
    
    Args:
        y_true: True labels (encoded as integers)
        y_proba: Predicted probabilities (n_samples, n_classes)
        n_classes: Number of classes
        
    Returns:
        ROC AUC score
    """
    try:
        if n_classes == 2:
            # Binary: use probability of positive class
            return roc_auc_score(y_true, y_proba[:, 1])
        else:
            # Multiclass: one-vs-rest strategy
            return roc_auc_score(y_true, y_proba, multi_class='ovr')
    except ValueError as e:
        print(f"    Warning: Could not compute ROC AUC: {e}")
        return np.nan


def run_tabpfn_benchmark(X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray,
                         n_classes: int) -> Dict:
    """
    Run TabPFN classifier on preprocessed data.
    
    IMPORTANT: 
    - fit() uses ONLY X_train, y_train
    - predict() uses ONLY X_test
    - Metrics computed comparing predictions to y_test
    
    Returns:
        Dict with metrics and timing
    """
    from tabpfn import TabPFNClassifier
    
    clf = TabPFNClassifier()
    
    # =========================================================================
    # FIT ON TRAIN ONLY
    # =========================================================================
    start_fit = time.time()
    clf.fit(X_train, y_train)
    fit_time = time.time() - start_fit
    
    # =========================================================================
    # PREDICT ON TEST ONLY
    # =========================================================================
    start_pred = time.time()
    y_proba = clf.predict_proba(X_test)
    pred_time = time.time() - start_pred
    
    y_pred = clf.predict(X_test)
    
    # =========================================================================
    # EVALUATE ON TEST ONLY
    # =========================================================================
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = compute_roc_auc(y_test, y_proba, n_classes)
    
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'fit_time': fit_time,
        'predict_time': pred_time,
    }


def benchmark_single_dataset(name: str, verbose: bool = True) -> Optional[Dict]:
    """
    Run complete benchmark pipeline for a single dataset.
    
    Pipeline (train/test kept separate throughout):
    1. Load from pickle (pre-split train/test)
    2. Encode labels (fit on train only)
    3. Subsample if needed (separately for train and test)
    4. Flatten 3D -> 2D
    5. Impute missing (statistics from train only)
    6. TabPFN fit (train only) -> predict (test only)
    7. Evaluate (test only)
    
    Returns:
        Dict with all results or None if failed
    """
    try:
        if verbose:
            print(f"\n{'─'*60}")
            print(f"Dataset: {name}")
            print(f"{'─'*60}")
        
        # =====================================================================
        # STEP 1: Load pre-split data from pickle
        # =====================================================================
        X_train, y_train, X_test, y_test, metadata = load_dataset(name)
        
        if verbose:
            print(f"  [LOAD] Train: {X_train.shape}, Test: {X_test.shape}")
            print(f"  [LOAD] Channels: {metadata['n_channels']}, Length: {metadata['length']}")
            print(f"  [LOAD] Classes (from train): {metadata['n_classes']}")
        
        # =====================================================================
        # STEP 2: Encode labels (fit on TRAIN only)
        # =====================================================================
        y_train_enc, y_test_enc, _ = preprocess_labels(y_train, y_test)
        
        # =====================================================================
        # STEP 3: Subsample if needed (SEPARATELY for train and test)
        # =====================================================================
        n_train_original = len(X_train)
        n_test_original = len(X_test)
        
        if len(X_train) > TABPFN_MAX_SAMPLES:
            X_train, y_train_enc = subsample_stratified(
                X_train, y_train_enc, TABPFN_MAX_SAMPLES, RANDOM_STATE
            )
            if verbose:
                print(f"  [SUBSAMPLE] Train: {n_train_original} -> {len(X_train)}")
        
        if len(X_test) > TABPFN_MAX_SAMPLES:
            X_test, y_test_enc = subsample_stratified(
                X_test, y_test_enc, TABPFN_MAX_SAMPLES, RANDOM_STATE
            )
            if verbose:
                print(f"  [SUBSAMPLE] Test: {n_test_original} -> {len(X_test)}")
        
        # =====================================================================
        # STEP 4: Flatten 3D -> 2D (independently for train and test)
        # =====================================================================
        X_train_flat = flatten_3d(X_train)
        X_test_flat = flatten_3d(X_test)
        
        if verbose:
            print(f"  [FLATTEN] Train: {X_train_flat.shape}, Test: {X_test_flat.shape}")
        
        # =====================================================================
        # STEP 5: Handle missing values (statistics from TRAIN only)
        # =====================================================================
        X_train_flat, X_test_flat = handle_missing_values(X_train_flat, X_test_flat)
        
        # =====================================================================
        # STEP 6 & 7: TabPFN (fit on TRAIN, predict on TEST, evaluate on TEST)
        # =====================================================================
        if verbose:
            print(f"  [TABPFN] Fitting on train ({len(X_train_flat)} samples)...")
            print(f"  [TABPFN] Predicting on test ({len(X_test_flat)} samples)...")
        
        results = run_tabpfn_benchmark(
            X_train_flat, y_train_enc,
            X_test_flat, y_test_enc,
            metadata['n_classes']
        )
        
        if verbose:
            print(f"  [RESULT] Accuracy: {results['accuracy']:.4f}")
            if not np.isnan(results['roc_auc']):
                print(f"  [RESULT] ROC AUC:  {results['roc_auc']:.4f}")
            else:
                print(f"  [RESULT] ROC AUC:  N/A")
            print(f"  [TIMING] Fit: {results['fit_time']:.2f}s, Predict: {results['predict_time']:.2f}s")
        
        return {
            'name': name,
            'n_channels': metadata['n_channels'],
            'length': metadata['length'],
            'flattened_features': metadata['n_channels'] * metadata['length'],
            'n_classes': metadata['n_classes'],
            'n_train_original': n_train_original,
            'n_test_original': n_test_original,
            'n_train_used': len(X_train),
            'n_test_used': len(X_test),
            'was_subsampled': n_train_original > TABPFN_MAX_SAMPLES or n_test_original > TABPFN_MAX_SAMPLES,
            **results,
            'status': 'success',
        }
        
    except Exception as e:
        if verbose:
            print(f"  [ERROR] {e}")
        return {
            'name': name,
            'status': 'failed',
            'error': str(e),
        }


def run_full_benchmark(dataset_names: Optional[List[str]] = None, 
                       verbose: bool = True) -> pd.DataFrame:
    """
    Run benchmark on all (or specified) datasets.
    
    Args:
        dataset_names: List of dataset names to benchmark (None = all usable)
        verbose: Print progress
        
    Returns:
        DataFrame with all results
    """
    if dataset_names is None:
        dataset_names = load_usable_dataset_names()
    
    print("=" * 70)
    print("TABPFN BASELINE BENCHMARK ON FLATTENED 3D DATASETS")
    print("=" * 70)
    print(f"Total datasets to benchmark: {len(dataset_names)}")
    print(f"TabPFN max samples: {TABPFN_MAX_SAMPLES}")
    print(f"Random state: {RANDOM_STATE}")
    print("=" * 70)
    print("\nData leakage prevention:")
    print("  • Train/test pre-split from UCR/UEA archive")
    print("  • LabelEncoder fit on TRAIN only")
    print("  • Missing value imputation from TRAIN statistics only")
    print("  • TabPFN fit on TRAIN only, predict on TEST only")
    print("=" * 70)
    
    # Pre-load all datasets into cache
    _load_all_datasets_from_pkl()
    
    results = []
    
    for name in tqdm(dataset_names, desc="Benchmarking", disable=verbose):
        result = benchmark_single_dataset(name, verbose=verbose)
        if result:
            results.append(result)
    
    return pd.DataFrame(results)


def print_summary(df: pd.DataFrame):
    """Print summary statistics of benchmark results"""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    success_df = df[df['status'] == 'success']
    failed_df = df[df['status'] == 'failed']
    
    print(f"\nDatasets processed: {len(df)}")
    print(f"  ✓ Successful: {len(success_df)}")
    print(f"  ✗ Failed: {len(failed_df)}")
    
    if len(failed_df) > 0:
        print(f"\nFailed datasets:")
        for _, row in failed_df.iterrows():
            print(f"  - {row['name']}: {row.get('error', 'Unknown error')}")
    
    if len(success_df) == 0:
        print("\nNo successful runs to summarize.")
        return
    
    print(f"\n{'─'*70}")
    print("OVERALL METRICS (evaluated on TEST set only)")
    print(f"{'─'*70}")
    
    acc = success_df['accuracy']
    print(f"\nAccuracy:")
    print(f"  Mean:   {acc.mean():.4f}")
    print(f"  Std:    {acc.std():.4f}")
    print(f"  Median: {acc.median():.4f}")
    print(f"  Min:    {acc.min():.4f} ({success_df.loc[acc.idxmin(), 'name']})")
    print(f"  Max:    {acc.max():.4f} ({success_df.loc[acc.idxmax(), 'name']})")
    
    roc = success_df['roc_auc'].dropna()
    if len(roc) > 0:
        print(f"\nROC AUC (n={len(roc)}):")
        print(f"  Mean:   {roc.mean():.4f}")
        print(f"  Std:    {roc.std():.4f}")
        print(f"  Median: {roc.median():.4f}")
        print(f"  Min:    {roc.min():.4f}")
        print(f"  Max:    {roc.max():.4f}")
    
    print(f"\nTiming:")
    print(f"  Mean fit time:     {success_df['fit_time'].mean():.2f}s")
    print(f"  Mean predict time: {success_df['predict_time'].mean():.2f}s")
    print(f"  Total time:        {success_df['fit_time'].sum() + success_df['predict_time'].sum():.1f}s")
    
    print(f"\n{'─'*70}")
    print("BREAKDOWN BY DATASET TYPE")
    print(f"{'─'*70}")
    
    uni = success_df[success_df['n_channels'] == 1]
    multi = success_df[success_df['n_channels'] > 1]
    
    print(f"\nUnivariate (n={len(uni)}):")
    print(f"  Mean Accuracy: {uni['accuracy'].mean():.4f}")
    print(f"  Mean ROC AUC:  {uni['roc_auc'].dropna().mean():.4f}")
    
    if len(multi) > 0:
        print(f"\nMultivariate (n={len(multi)}):")
        print(f"  Mean Accuracy: {multi['accuracy'].mean():.4f}")
        print(f"  Mean ROC AUC:  {multi['roc_auc'].dropna().mean():.4f}")
        print(f"\n  Multivariate datasets:")
        for _, row in multi.iterrows():
            print(f"    {row['name']}: Acc={row['accuracy']:.4f}, AUC={row['roc_auc']:.4f}")
    
    binary = success_df[success_df['n_classes'] == 2]
    multiclass = success_df[success_df['n_classes'] > 2]
    
    print(f"\nBinary classification (n={len(binary)}):")
    print(f"  Mean Accuracy: {binary['accuracy'].mean():.4f}")
    print(f"  Mean ROC AUC:  {binary['roc_auc'].dropna().mean():.4f}")
    
    print(f"\nMulticlass (n={len(multiclass)}):")
    print(f"  Mean Accuracy: {multiclass['accuracy'].mean():.4f}")
    print(f"  Mean ROC AUC:  {multiclass['roc_auc'].dropna().mean():.4f}")
    
    print(f"\n{'─'*70}")
    print("TOP 10 DATASETS BY ACCURACY")
    print(f"{'─'*70}")
    top10 = success_df.nlargest(10, 'accuracy')[['name', 'accuracy', 'roc_auc', 'n_classes', 'flattened_features']]
    print(top10.to_string(index=False))
    
    print(f"\n{'─'*70}")
    print("BOTTOM 10 DATASETS BY ACCURACY")
    print(f"{'─'*70}")
    bottom10 = success_df.nsmallest(10, 'accuracy')[['name', 'accuracy', 'roc_auc', 'n_classes', 'flattened_features']]
    print(bottom10.to_string(index=False))


def save_results(df: pd.DataFrame, suffix: str = ""):
    """
    Save results to files.
    Always overwrites the same files (no timestamps) to keep only the latest results.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Always use the same filenames (overwrite previous runs)
    csv_path = RESULTS_DIR / "baseline_benchmark_latest.csv"
    json_path = RESULTS_DIR / "baseline_benchmark_latest.json"
    
    # Save CSV
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Saved CSV to: {csv_path}")
    
    # Save JSON
    results_dict = {
        'timestamp': timestamp,
        'n_datasets': len(df),
        'n_successful': len(df[df['status'] == 'success']),
        'n_failed': len(df[df['status'] == 'failed']),
        'config': {
            'tabpfn_max_samples': TABPFN_MAX_SAMPLES,
            'random_state': RANDOM_STATE,
        },
        'data_leakage_prevention': {
            'train_test_split': 'Pre-split from UCR/UEA archive',
            'label_encoding': 'Fit on train only',
            'missing_imputation': 'Statistics from train only',
            'model_fitting': 'Train data only',
            'evaluation': 'Test data only',
        },
        'summary': {},
        'results': df.to_dict(orient='records'),
    }
    
    success_df = df[df['status'] == 'success']
    if len(success_df) > 0:
        results_dict['summary'] = {
            'accuracy': {
                'mean': float(success_df['accuracy'].mean()),
                'std': float(success_df['accuracy'].std()),
                'median': float(success_df['accuracy'].median()),
                'min': float(success_df['accuracy'].min()),
                'max': float(success_df['accuracy'].max()),
            },
            'roc_auc': {
                'mean': float(success_df['roc_auc'].dropna().mean()),
                'std': float(success_df['roc_auc'].dropna().std()),
                'median': float(success_df['roc_auc'].dropna().median()),
            }
        }
    
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    print(f"💾 Saved JSON to: {json_path}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run TabPFN baseline benchmark on flattened 3D datasets")
    parser.add_argument('--datasets', nargs='+', help='Specific dataset names to benchmark')
    parser.add_argument('--n-datasets', type=int, help='Number of datasets to benchmark (for testing)')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    args = parser.parse_args()
    
    if args.datasets:
        dataset_names = args.datasets
    else:
        dataset_names = load_usable_dataset_names()
        if args.n_datasets:
            dataset_names = dataset_names[:args.n_datasets]
    
    df = run_full_benchmark(dataset_names, verbose=not args.quiet)
    
    print_summary(df)
    
    save_results(df)
    
    print("\n✅ Benchmark complete!")
    
    return df


if __name__ == "__main__":
    main()
