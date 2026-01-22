"""
Benchmark TabPFN on synthetic 3D datasets (flattened).

Similar to benchmark_baseline.py but uses synthetic datasets generated
by our 3D generator instead of real UCR/UEA datasets.

Process:
1. Load synthetic datasets from npz files
2. Flatten from (n, m, t) to (n, m*t)
3. Apply TabPFN
4. Report AUC ROC and Accuracy
"""

import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# Paths
RESULTS_DIR = Path(__file__).parent / "results"
SYNTHETIC_DIR = RESULTS_DIR / "synthetic_datasets"
OUTPUT_DIR = RESULTS_DIR


def load_synthetic_datasets() -> List[Dict]:
    """Load all synthetic datasets from npz files."""
    datasets = []
    
    # Load summary for metadata
    summary_file = SYNTHETIC_DIR / "summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        dataset_info = {d['id']: d for d in summary.get('datasets', [])}
    else:
        dataset_info = {}
    
    # Load each npz file
    npz_files = sorted(SYNTHETIC_DIR.glob("synthetic_*.npz"))
    
    for npz_file in npz_files:
        data = np.load(npz_file)
        dataset_id = int(npz_file.stem.split('_')[1])
        
        info = dataset_info.get(dataset_id, {})
        
        datasets.append({
            'id': dataset_id,
            'name': f"synthetic_{dataset_id:04d}",
            'X': data['X'],
            'y': data['y'],
            'n_classes': int(data['n_classes']),
            'sample_mode': info.get('sample_mode', 'unknown'),
        })
    
    return datasets


def flatten_3d_to_2d(X: np.ndarray) -> np.ndarray:
    """
    Flatten 3D array (n_samples, n_features, n_timesteps) to 2D (n_samples, n_features * n_timesteps).
    """
    n_samples, n_features, n_timesteps = X.shape
    return X.reshape(n_samples, n_features * n_timesteps)


def handle_missing_values(X: np.ndarray, train_means: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Handle missing values by imputing with column means.
    
    Args:
        X: Data array
        train_means: Pre-computed means from training data (for test set)
        
    Returns:
        (X_imputed, column_means)
    """
    X_imputed = X.copy()
    
    if train_means is None:
        # Compute means from this data (training)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            column_means = np.nanmean(X_imputed, axis=0)
        # Replace NaN means with 0
        column_means = np.nan_to_num(column_means, nan=0.0)
    else:
        column_means = train_means
    
    # Impute missing values
    for col in range(X_imputed.shape[1]):
        mask = np.isnan(X_imputed[:, col])
        if mask.any():
            X_imputed[mask, col] = column_means[col]
    
    # Final check: replace any remaining NaN with 0
    X_imputed = np.nan_to_num(X_imputed, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X_imputed, column_means


def benchmark_single_dataset(dataset: Dict, test_ratio: float = 0.3) -> Dict:
    """
    Benchmark TabPFN on a single synthetic dataset.
    
    Args:
        dataset: Dict with 'X', 'y', 'n_classes', etc.
        test_ratio: Fraction of data to use for testing
        
    Returns:
        Dict with results
    """
    from tabpfn import TabPFNClassifier
    
    X = dataset['X']
    y = dataset['y']
    n_classes = dataset['n_classes']
    
    n_samples, n_features, n_timesteps = X.shape
    n_flat_features = n_features * n_timesteps
    
    # Check if flattenable (TabPFN limit: 500 features)
    if n_flat_features > 500:
        return {
            'status': 'skipped',
            'reason': f'Too many features after flattening: {n_flat_features} > 500',
        }
    
    # Check classes
    if n_classes > 10:
        return {
            'status': 'skipped',
            'reason': f'Too many classes: {n_classes} > 10',
        }
    
    # Train/test split (stratified)
    n_test = max(int(n_samples * test_ratio), n_classes)  # At least n_classes in test
    n_train = n_samples - n_test
    
    # Stratified split
    unique_classes = np.unique(y)
    train_indices = []
    test_indices = []
    
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        n_cls = len(cls_indices)
        n_cls_test = max(1, int(n_cls * test_ratio))
        n_cls_train = n_cls - n_cls_test
        
        np.random.shuffle(cls_indices)
        train_indices.extend(cls_indices[:n_cls_train])
        test_indices.extend(cls_indices[n_cls_train:])
    
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    # Flatten
    X_train_flat = flatten_3d_to_2d(X_train)
    X_test_flat = flatten_3d_to_2d(X_test)
    
    # Handle missing values (impute with train means)
    X_train_flat, train_means = handle_missing_values(X_train_flat)
    X_test_flat, _ = handle_missing_values(X_test_flat, train_means=train_means)
    
    # Encode labels
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    
    # Subsample if needed (TabPFN limit: 10000 samples)
    if len(y_train_enc) > 10000:
        indices = np.random.choice(len(y_train_enc), size=10000, replace=False)
        X_train_flat = X_train_flat[indices]
        y_train_enc = y_train_enc[indices]
    
    # Train TabPFN
    try:
        start_time = time.time()
        
        clf = TabPFNClassifier(
            device='cpu',  # Force CPU to avoid MPS memory issues
            ignore_pretraining_limits=True  # Allow >1000 samples on CPU
        )
        clf.fit(X_train_flat, y_train_enc)
        
        # Predict
        y_pred = clf.predict(X_test_flat)
        y_proba = clf.predict_proba(X_test_flat)
        
        train_time = time.time() - start_time
        
        # Metrics
        accuracy = accuracy_score(y_test_enc, y_pred)
        
        # ROC AUC
        if n_classes == 2:
            roc_auc = roc_auc_score(y_test_enc, y_proba[:, 1])
        else:
            try:
                roc_auc = roc_auc_score(y_test_enc, y_proba, multi_class='ovr', average='weighted')
            except ValueError:
                roc_auc = None
        
        return {
            'status': 'success',
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'train_time': train_time,
            'n_train': len(y_train_enc),
            'n_test': len(y_test_enc),
            'n_features_flat': n_flat_features,
            'n_classes': n_classes,
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'reason': str(e),
        }


def run_benchmark(datasets: List[Dict], seed: int = 42) -> pd.DataFrame:
    """Run benchmark on all synthetic datasets."""
    np.random.seed(seed)
    
    results = []
    
    for dataset in tqdm(datasets, desc="Benchmarking"):
        result = benchmark_single_dataset(dataset)
        result['dataset_id'] = dataset['id']
        result['dataset_name'] = dataset['name']
        result['sample_mode'] = dataset.get('sample_mode', 'unknown')
        result['shape'] = f"{dataset['X'].shape}"
        results.append(result)
    
    return pd.DataFrame(results)


def save_results(df: pd.DataFrame, output_dir: Path):
    """Save benchmark results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    csv_path = output_dir / "synthetic_benchmark_latest.csv"
    df.to_csv(csv_path, index=False)
    
    # Save JSON summary
    successful = df[df['status'] == 'success']
    
    summary = {
        'total_datasets': len(df),
        'successful': len(successful),
        'skipped': len(df[df['status'] == 'skipped']),
        'errors': len(df[df['status'] == 'error']),
        'mean_accuracy': float(successful['accuracy'].mean()) if len(successful) > 0 else None,
        'std_accuracy': float(successful['accuracy'].std()) if len(successful) > 0 else None,
        'mean_roc_auc': float(successful['roc_auc'].dropna().mean()) if len(successful) > 0 else None,
        'std_roc_auc': float(successful['roc_auc'].dropna().std()) if len(successful) > 0 else None,
        'mean_train_time': float(successful['train_time'].mean()) if len(successful) > 0 else None,
    }
    
    json_path = output_dir / "synthetic_benchmark_latest.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return csv_path, json_path


def main():
    """Main entry point."""
    print("=" * 70)
    print("BENCHMARK: TabPFN on SYNTHETIC 3D Datasets (Flattened)")
    print("=" * 70)
    
    # Load datasets
    print("\n[1] Loading synthetic datasets...")
    datasets = load_synthetic_datasets()
    print(f"    Loaded {len(datasets)} datasets")
    
    if len(datasets) == 0:
        print("    ERROR: No synthetic datasets found!")
        print(f"    Please run generate_synthetic_matching.py first.")
        return
    
    # Show sample info
    print("\n    Sample dataset info:")
    for ds in datasets[:3]:
        print(f"      {ds['name']}: shape={ds['X'].shape}, classes={ds['n_classes']}")
    
    # Run benchmark
    print("\n[2] Running TabPFN benchmark...")
    results_df = run_benchmark(datasets)
    
    # Save results
    print("\n[3] Saving results...")
    csv_path, json_path = save_results(results_df, OUTPUT_DIR)
    print(f"    CSV: {csv_path}")
    print(f"    JSON: {json_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    successful = results_df[results_df['status'] == 'success']
    
    print(f"\nTotal datasets: {len(results_df)}")
    print(f"Successful: {len(successful)}")
    print(f"Skipped: {len(results_df[results_df['status'] == 'skipped'])}")
    print(f"Errors: {len(results_df[results_df['status'] == 'error'])}")
    
    if len(successful) > 0:
        print(f"\nAccuracy: {successful['accuracy'].mean():.4f} ± {successful['accuracy'].std():.4f}")
        
        roc_values = successful['roc_auc'].dropna()
        if len(roc_values) > 0:
            print(f"ROC AUC:  {roc_values.mean():.4f} ± {roc_values.std():.4f}")
        
        print(f"Avg time: {successful['train_time'].mean():.2f}s")
        
        print("\n" + "-" * 70)
        print("Per-dataset results:")
        print("-" * 70)
        for _, row in successful.iterrows():
            roc_str = f"{row['roc_auc']:.4f}" if pd.notna(row['roc_auc']) else "N/A"
            print(f"  {row['dataset_name']}: Acc={row['accuracy']:.4f}, AUC={roc_str}, "
                  f"shape={row['shape']}, mode={row['sample_mode']}")
    
    print("\n" + "=" * 70)
    print("✅ BENCHMARK COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
