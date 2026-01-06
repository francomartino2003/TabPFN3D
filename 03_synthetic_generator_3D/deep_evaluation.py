"""
Deep Evaluation of Synthetic 3D Datasets

Generates datasets and performs extensive hyperparameter tuning
to determine maximum achievable lift with proper preprocessing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
import time
import sys
import os
import pickle
from pathlib import Path

warnings.filterwarnings('ignore')

# Check CUDA availability
def check_cuda():
    """Check if CUDA is available for GPU acceleration."""
    cuda_available = False
    gpu_support = False
    
    # Check PyTorch CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"CUDA available via PyTorch (GPU: {gpu_name})")
            gpu_support = True
    except ImportError:
        pass
    
    # Note: XGBoost and LightGBM GPU will be tested at runtime
    # If they fail, they'll fall back to CPU automatically
    
    if not cuda_available:
        print("CUDA not available - using CPU only")
    
    return gpu_support

CUDA_AVAILABLE = check_cuda()

# Sklearn
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# XGBoost / LightGBM
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not available")

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("LightGBM not available")

# Time series classifiers
try:
    from sktime.classification.kernel_based import RocketClassifier
    from sktime.classification.interval_based import TimeSeriesForestClassifier
    from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
    from sktime.classification.hybrid import HIVECOTEV2
    HAS_SKTIME = True
except ImportError:
    HAS_SKTIME = False
    print("sktime not available - time series classifiers disabled")

try:
    from aeon.classification.convolution_based import RocketClassifier as AeonRocket
    from aeon.classification.hybrid import HIVECOTEV2 as AeonHC2
    from aeon.classification.interval_based import TimeSeriesForestClassifier as AeonTSF
    from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier as AeonKNN
    HAS_AEON = True
except ImportError:
    HAS_AEON = False
    print("aeon not available")

from config import PriorConfig3D
from generator import SyntheticDatasetGenerator3D, SyntheticDataset3D

# Custom unpickler for loading real datasets
class CustomUnpickler(pickle.Unpickler):
    """Custom unpickler that handles missing modules."""
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            # Return a dummy class for missing modules
            class DummyClass:
                pass
            return DummyClass


@dataclass
class EvaluationResult:
    """Result of evaluating a single dataset."""
    dataset_id: int
    n_samples: int
    n_features: int
    t_length: int
    n_classes: int
    baseline: float
    best_model: str
    best_score: float
    lift: float
    all_scores: Dict[str, float]
    preprocessing: str
    time_elapsed: float


def preprocess_data(X: np.ndarray, y: np.ndarray, 
                    strategy: str = 'robust') -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess 3D data for classification.
    
    Args:
        X: Shape (n_samples, n_features, t_length)
        y: Shape (n_samples,)
        strategy: 'standard', 'robust', 'minmax'
    
    Returns:
        Preprocessed X, y
    """
    n_samples, n_features, t_length = X.shape
    
    # Handle NaN values
    nan_ratio = np.isnan(X).mean()
    if nan_ratio > 0:
        print(f"    NaN ratio: {nan_ratio:.1%}")
        # Simple imputation: replace with feature mean
        for f in range(n_features):
            for t in range(t_length):
                col = X[:, f, t]
                mask = np.isnan(col)
                if mask.any():
                    col[mask] = np.nanmean(col)
                    X[:, f, t] = col
        
        # If still NaN, replace with 0
        X = np.nan_to_num(X, nan=0.0)
    
    # Normalize per feature
    if strategy == 'robust':
        for f in range(n_features):
            data = X[:, f, :].flatten()
            median = np.median(data)
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1 + 1e-10
            X[:, f, :] = (X[:, f, :] - median) / iqr
    elif strategy == 'standard':
        for f in range(n_features):
            data = X[:, f, :].flatten()
            mean, std = np.mean(data), np.std(data) + 1e-10
            X[:, f, :] = (X[:, f, :] - mean) / std
    elif strategy == 'minmax':
        for f in range(n_features):
            data = X[:, f, :].flatten()
            xmin, xmax = np.min(data), np.max(data)
            X[:, f, :] = (X[:, f, :] - xmin) / (xmax - xmin + 1e-10)
    
    # Clip extreme values
    X = np.clip(X, -10, 10)
    
    return X, y


def flatten_for_sklearn(X: np.ndarray) -> np.ndarray:
    """Flatten 3D data to 2D for sklearn classifiers."""
    n_samples, n_features, t_length = X.shape
    return X.reshape(n_samples, n_features * t_length)


def convert_for_sktime(X: np.ndarray) -> np.ndarray:
    """Convert to sktime format (n_samples, n_features, t_length)."""
    # sktime expects (n_samples, n_channels, n_timepoints)
    return X


def get_tabular_models(n_samples: int, use_gpu: bool = False) -> Dict[str, Tuple[Any, Dict]]:
    """
    Get tabular models with parameter grids for tuning.
    
    Args:
        n_samples: Number of samples (to adjust complexity)
        use_gpu: Whether to use GPU acceleration (if available)
    """
    models = {}
    
    # Random Forest - minimal baseline only (no tuning for speed)
    # Focus is on time series models, tabular is just for comparison
    models['RF_baseline'] = (
        RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        {}  # No tuning, just baseline
    )
    
    return models


def get_ts_models(include_hc2: bool = False) -> Dict[str, Any]:
    """
    Get time series specific models.
    
    Args:
        include_hc2: Whether to include HC2 (HIVECOTEV2) - very slow but state-of-the-art
    """
    models = {}
    
    if HAS_AEON:
        # ROCKET - fast and effective
        models['ROCKET'] = AeonRocket(random_state=42, n_jobs=-1)
        
        # Time Series Forest
        models['TSF'] = AeonTSF(random_state=42, n_jobs=-1, n_estimators=100)
        
        # KNN-DTW (slow but good)
        models['KNN-DTW'] = AeonKNN(n_neighbors=3, distance='dtw')
        
        # HC2 - state of the art (very slow but best performance)
        if include_hc2:
            models['HC2'] = AeonHC2(random_state=42, n_jobs=-1)
        
    elif HAS_SKTIME:
        models['ROCKET'] = RocketClassifier(random_state=42)
        models['TSF'] = TimeSeriesForestClassifier(random_state=42, n_estimators=100)
        models['KNN-DTW'] = KNeighborsTimeSeriesClassifier(n_neighbors=3, distance='dtw')
        if include_hc2:
            models['HC2'] = HIVECOTEV2(random_state=42, n_jobs=-1)
    
    return models


def evaluate_tabular_baseline(X_train: np.ndarray, X_test: np.ndarray,
                              y_train: np.ndarray, y_test: np.ndarray,
                              baseline: float = 0.0) -> Dict[str, float]:
    """
    Quick baseline evaluation with RF (no tuning for speed).
    Focus is on time series models.
    """
    results = {}
    
    try:
        print(f"      RF baseline...", end=" ", flush=True)
        start_time = time.time()
        
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        test_score = accuracy_score(y_test, model.predict(X_test))
        results['RF_baseline'] = test_score
        
        elapsed = time.time() - start_time
        lift = test_score - baseline
        print(f"Test: {test_score:.3f} (lift: {lift:+.3f}) [{elapsed:.1f}s]", flush=True)
        
    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        results['RF_baseline'] = baseline
    
    return results


def evaluate_ts_models(X_train: np.ndarray, X_test: np.ndarray,
                       y_train: np.ndarray, y_test: np.ndarray,
                       timeout: int = 300, include_hc2: bool = False,
                       baseline: float = 0.0) -> Dict[str, float]:
    """
    Evaluate time series models (primary focus).
    
    Args:
        X_train, X_test: Shape (n_samples, n_features, t_length)
        timeout: Max seconds per model (longer for TS models)
        include_hc2: Whether to include HC2 (very slow but best)
        baseline: Baseline score for lift calculation
    """
    results = {}
    best_overall_score = baseline
    best_overall_model = None
    models = get_ts_models(include_hc2=include_hc2)
    
    # Different timeouts for different models
    model_timeouts = {
        'ROCKET': 120,
        'TSF': 180,
        'KNN-DTW': 240,
        'HC2': 600  # HC2 can take 5-10 minutes
    }
    
    print(f"      Evaluating {len(models)} time series models...")
    sys.stdout.flush()
    
    for idx, (name, model) in enumerate(models.items(), 1):
        try:
            print(f"        [{idx}/{len(models)}] {name}...", end=" ", flush=True)
            start = time.time()
            
            model_timeout = model_timeouts.get(name, timeout)
            model.fit(X_train, y_train)
            
            elapsed = time.time() - start
            if elapsed > model_timeout:
                print(f"timeout ({elapsed:.1f}s > {model_timeout}s)", flush=True)
                continue
            
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            results[name] = score
            lift = score - baseline
            
            print(f"Test: {score:.3f} (lift: {lift:+.3f}) [{elapsed:.1f}s]", flush=True)
            
            # Track best overall
            if score > best_overall_score:
                best_overall_score = score
                best_overall_model = name
                print(f"        >>> NEW BEST: {name} = {score:.3f} (lift: {lift:+.3f})", flush=True)
            
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
            results[name] = 0.0
    
    if best_overall_model:
        print(f"      Best TS model: {best_overall_model} = {best_overall_score:.3f}")
    
    return results


def evaluate_dataset_raw(X: np.ndarray, y: np.ndarray, dataset_id: int,
                         dataset_name: str = "unknown",
                     min_samples: int = 50,
                     max_baseline: float = 0.7,
                     tuning_iter: int = 20) -> Optional[EvaluationResult]:
    """
    Evaluate a raw dataset (X, y) - works for both synthetic and real datasets.
    
    Args:
        X: Feature tensor of shape (n_samples, n_features, t_length)
        y: Target vector of shape (n_samples,)
        dataset_id: ID for tracking
        dataset_name: Name of the dataset (for real datasets)
        min_samples: Minimum samples required
        max_baseline: Maximum acceptable baseline (skip if higher)
        tuning_iter: Number of iterations for RandomizedSearchCV (not used)
    
    Returns:
        EvaluationResult or None if dataset is skipped
    """
    print(f"\n{'='*60}")
    print(f"Dataset {dataset_id}: {dataset_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Ensure 3D shape
    if X.ndim == 2:
        X = X[:, np.newaxis, :]  # (n_samples, t_length) -> (n_samples, 1, t_length)
    elif X.ndim == 1:
        X = X[:, np.newaxis, np.newaxis]  # (n_samples,) -> (n_samples, 1, 1)
    
    n_samples, n_features, t_length = X.shape
    
    print(f"  Shape: {X.shape}")
    print(f"  Classes: {len(np.unique(y))}")
    
    # Check minimum samples
    if n_samples < min_samples:
        print(f"  SKIPPED: Too few samples ({n_samples} < {min_samples})")
        return None
    
    # Check baseline
    baseline = 1.0 / len(np.unique(y))
    if baseline > max_baseline:
        print(f"  SKIPPED: Baseline too high ({baseline:.2%} > {max_baseline:.0%})")
        return None
    
    # Use robust preprocessing by default
    best_preprocess = 'robust'
    print(f"  Preprocessing: {best_preprocess}")
    
    # Apply preprocessing
    X_proc, y_proc = preprocess_data(X.copy(), y.copy(), best_preprocess)
    
    # Split data
    X_flat = flatten_for_sklearn(X_proc)
    X_train_flat, X_test_flat, y_train, y_test = train_test_split(
        X_flat, y_proc, test_size=0.3, stratify=y_proc, random_state=42
    )
    
    # For TS models
    X_train_3d, X_test_3d, _, _ = train_test_split(
        X_proc, y_proc, test_size=0.3, stratify=y_proc, random_state=42
    )
    
    print(f"\n  Train: {len(X_train_flat)}, Test: {len(X_test_flat)}")
    
    # Calculate baseline
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train_flat, y_train)
    baseline_score = dummy.score(X_test_flat, y_test)
    print(f"  Baseline (majority): {baseline_score:.3f}")
    
    # Evaluate TS models FIRST (primary focus)
    ts_scores = {}
    if HAS_AEON or HAS_SKTIME:
        print("\n  Evaluating TIME SERIES models (primary focus)...")
        # Only include HC2 for smaller datasets (it's very slow)
        include_hc2 = (n_samples <= 1000 and t_length <= 300)
        if include_hc2:
            print("    (HC2 enabled for this dataset size)")
        ts_scores = evaluate_ts_models(
            X_train_3d, X_test_3d, y_train, y_test,
            timeout=300, include_hc2=include_hc2, baseline=baseline_score
        )
    else:
        print("\n  WARNING: No time series libraries available (aeon/sktime)")
        print("    Install with: pip install aeon")
    
    # Quick tabular baseline for comparison
    print("\n  Evaluating tabular baseline (for comparison)...")
    tabular_scores = evaluate_tabular_baseline(
        X_train_flat, X_test_flat, y_train, y_test, baseline=baseline_score
    )
    
    # Combine all scores
    all_scores = {**tabular_scores, **ts_scores}
    
    # Find best
    best_model = max(all_scores, key=all_scores.get)
    best_score = all_scores[best_model]
    lift = best_score - baseline_score
    
    elapsed = time.time() - start_time
    
    print(f"\n  BEST: {best_model} = {best_score:.3f} (lift: {lift:+.3f})")
    print(f"  Time: {elapsed:.1f}s")
    
    return EvaluationResult(
        dataset_id=dataset_id,
        n_samples=n_samples,
        n_features=n_features,
        t_length=t_length,
        n_classes=len(np.unique(y)),
        baseline=baseline_score,
        best_model=best_model,
        best_score=best_score,
        lift=lift,
        all_scores=all_scores,
        preprocessing=best_preprocess,
        time_elapsed=elapsed
    )


def evaluate_dataset(ds: SyntheticDataset3D, dataset_id: int,
                     min_samples: int = 50,
                     max_baseline: float = 0.7,
                     tuning_iter: int = 20) -> Optional[EvaluationResult]:
    """
    Evaluate a single dataset with extensive tuning.
    
    Args:
        ds: Dataset to evaluate
        dataset_id: ID for tracking
        min_samples: Minimum samples required
        max_baseline: Maximum acceptable baseline (skip if higher)
        tuning_iter: Number of iterations for RandomizedSearchCV
    
    Returns:
        EvaluationResult or None if dataset is skipped
    """
    print(f"\n{'='*60}")
    print(f"Dataset {dataset_id}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    X, y = ds.X, ds.y
    n_samples, n_features, t_length = X.shape
    
    print(f"  Shape: {X.shape}")
    print(f"  Classes: {len(np.unique(y))}")
    print(f"  Mode: {ds.config.sample_mode}")
    
    # Check minimum samples
    if n_samples < min_samples:
        print(f"  SKIPPED: Too few samples ({n_samples} < {min_samples})")
        return None
    
    # Check baseline
    baseline = 1.0 / len(np.unique(y))
    if baseline > max_baseline:
        print(f"  SKIPPED: Baseline too high ({baseline:.2%} > {max_baseline:.0%})")
        return None
    
    # Use robust preprocessing by default (fastest option)
    # Skip preprocessing comparison to save time
    best_preprocess = 'robust'
    print(f"  Preprocessing: {best_preprocess} (fast mode)")
    
    # Apply best preprocessing
    X_proc, y_proc = preprocess_data(X.copy(), y.copy(), best_preprocess)
    
    # Split data
    X_flat = flatten_for_sklearn(X_proc)
    X_train_flat, X_test_flat, y_train, y_test = train_test_split(
        X_flat, y_proc, test_size=0.3, stratify=y_proc, random_state=42
    )
    
    # For TS models
    X_train_3d, X_test_3d, _, _ = train_test_split(
        X_proc, y_proc, test_size=0.3, stratify=y_proc, random_state=42
    )
    
    print(f"\n  Train: {len(X_train_flat)}, Test: {len(X_test_flat)}")
    
    # Calculate baseline
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train_flat, y_train)
    baseline_score = dummy.score(X_test_flat, y_test)
    print(f"  Baseline (majority): {baseline_score:.3f}")
    
    # Evaluate TS models FIRST (primary focus)
    ts_scores = {}
    if HAS_AEON or HAS_SKTIME:
        print("\n  Evaluating TIME SERIES models (primary focus)...")
        # Only include HC2 for smaller datasets (it's very slow)
        include_hc2 = (n_samples <= 1000 and t_length <= 300)
        if include_hc2:
            print("    (HC2 enabled for this dataset size)")
        ts_scores = evaluate_ts_models(
            X_train_3d, X_test_3d, y_train, y_test,
            timeout=300, include_hc2=include_hc2, baseline=baseline_score
        )
    else:
        print("\n  WARNING: No time series libraries available (aeon/sktime)")
        print("    Install with: pip install aeon")
    
    # Quick tabular baseline for comparison
    print("\n  Evaluating tabular baseline (for comparison)...")
    tabular_scores = evaluate_tabular_baseline(
        X_train_flat, X_test_flat, y_train, y_test, baseline=baseline_score
    )
    
    # Combine all scores
    all_scores = {**tabular_scores, **ts_scores}
    
    # Find best
    best_model = max(all_scores, key=all_scores.get)
    best_score = all_scores[best_model]
    lift = best_score - baseline_score
    
    elapsed = time.time() - start_time
    
    print(f"\n  BEST: {best_model} = {best_score:.3f} (lift: {lift:+.3f})")
    print(f"  Time: {elapsed:.1f}s")
    
    return EvaluationResult(
        dataset_id=dataset_id,
        n_samples=n_samples,
        n_features=n_features,
        t_length=t_length,
        n_classes=len(np.unique(y)),
        baseline=baseline_score,
        best_model=best_model,
        best_score=best_score,
        lift=lift,
        all_scores=all_scores,
        preprocessing=best_preprocess,
        time_elapsed=elapsed
    )


def load_real_datasets(pkl_path: str) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """
    Load real datasets from pickle file.
    
    Returns:
        List of (name, X, y) tuples where X has shape (n_samples, n_features, t_length)
    """
    if not os.path.exists(pkl_path):
        print(f"  Real datasets file not found: {pkl_path}")
        return []
    
    try:
        with open(pkl_path, 'rb') as f:
            data = CustomUnpickler(f).load()
    except Exception as e:
        print(f"  Error loading pickle: {e}")
        return []
    
    datasets = []
    
    if isinstance(data, dict):
        items = list(data.items())
    elif isinstance(data, list):
        items = list(enumerate(data))
    else:
        print(f"  Unexpected data type: {type(data)}")
        return []
    
    for name, content in items:
        try:
            X, y = None, None
            
            # Try different formats
            if isinstance(content, dict):
                X = content.get('X', content.get('data', content.get('X_train', None)))
                y = content.get('y', content.get('target', content.get('y_train', content.get('labels', None))))
                # If split into train/test, concatenate
                if X is None and 'X_train' in content and 'X_test' in content:
                    X = np.concatenate([content['X_train'], content['X_test']], axis=0)
                    y = np.concatenate([content['y_train'], content['y_test']], axis=0)
            elif isinstance(content, (tuple, list)) and len(content) >= 2:
                X, y = content[0], content[1]
            elif hasattr(content, 'X') and hasattr(content, 'y'):
                X, y = content.X, content.y
            elif hasattr(content, 'data') and hasattr(content, 'target'):
                X, y = content.data, content.target
            
            if X is None or y is None:
                continue
            
            X = np.array(X)
            y = np.array(y)
            
            # Skip if shapes don't make sense
            if X.size == 0 or y.size == 0:
                continue
            if len(X) != len(y):
                continue
            
            # Ensure 3D shape (n_samples, n_features, t_length)
            if X.ndim == 1:
                X = X[:, np.newaxis, np.newaxis]
            elif X.ndim == 2:
                # Assume (n_samples, time_steps) -> (n_samples, 1, time_steps)
                X = X[:, np.newaxis, :]
            elif X.ndim == 3 and X.shape[1] > X.shape[2]:
                # Might be (n_samples, t_len, n_features), transpose
                X = np.transpose(X, (0, 2, 1))
            
            # Convert y to numeric if needed
            if not np.issubdtype(y.dtype, np.number):
                unique_labels = np.unique(y)
                label_map = {l: i for i, l in enumerate(unique_labels)}
                y = np.array([label_map[l] for l in y])
            
            datasets.append((str(name), X, y))
        except Exception as e:
            continue
    
    return datasets


def run_deep_evaluation(
    n_datasets: int = 4,
    n_real_datasets: int = 2,
    seed: int = 42,
    min_samples: int = 100,
    max_baseline: float = 0.6,
    tuning_iter: int = 30,
    real_pkl_path: str = "../01_real_data/AEON/data/classification_datasets.pkl"
) -> List[EvaluationResult]:
    """
    Run deep evaluation on multiple datasets (synthetic and real).
    
    Args:
        n_datasets: Number of synthetic datasets to evaluate
        n_real_datasets: Number of real datasets to evaluate
        seed: Random seed
        min_samples: Minimum samples required
        max_baseline: Maximum acceptable baseline (skip if higher)
        tuning_iter: Number of iterations for RandomizedSearchCV (not used)
        real_pkl_path: Path to real datasets pickle file
    """
    print("="*70)
    print("DEEP EVALUATION OF 3D DATASETS (SYNTHETIC + REAL)")
    print("="*70)
    print(f"Generating {n_datasets} synthetic datasets...")
    print(f"Filters: min_samples={min_samples}, max_baseline={max_baseline:.0%}")
    
    results = []
    dataset_counter = 0
    
    # Evaluate synthetic datasets
    prior = PriorConfig3D()
    prior.prob_classification = 1.0
    generator = SyntheticDatasetGenerator3D(seed=seed, prior=prior)
    
    attempts = 0
    max_attempts = n_datasets * 5
    
    while len([r for r in results if r.dataset_id < 1000]) < n_datasets and attempts < max_attempts:
        attempts += 1
        dataset_counter += 1
        
        try:
            ds = generator.generate()
            result = evaluate_dataset(
                ds, 
                dataset_id=dataset_counter,
                min_samples=min_samples,
                max_baseline=max_baseline,
                tuning_iter=tuning_iter
            )
            
            if result is not None:
                results.append(result)
                
        except Exception as e:
            print(f"\nError generating/evaluating synthetic dataset: {e}")
    
    # Evaluate real datasets
    if n_real_datasets > 0:
        print(f"\n{'='*70}")
        print(f"Loading and evaluating {n_real_datasets} REAL datasets...")
        print("="*70)
        
        script_dir = Path(__file__).parent
        pkl_path = script_dir / real_pkl_path
        
        real_datasets = load_real_datasets(str(pkl_path))
        
        if real_datasets:
            print(f"  Loaded {len(real_datasets)} real datasets from pickle")
            
            # Use IDs starting from 10000 to distinguish from synthetic
            real_id_start = 10000
            for i, (name, X, y) in enumerate(real_datasets[:n_real_datasets]):
                real_id = real_id_start + i
                try:
                    result = evaluate_dataset_raw(
                        X, y,
                        dataset_id=real_id,
                        dataset_name=f"REAL_{name}",
                        min_samples=min_samples,
                        max_baseline=max_baseline,
                        tuning_iter=tuning_iter
                    )
                    
                    if result is not None:
                        results.append(result)
                        
                except Exception as e:
                    print(f"\nError evaluating real dataset {name}: {e}")
        else:
            print(f"  WARNING: Could not load real datasets from {pkl_path}")
            print("    Skipping real dataset evaluation")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if results:
        lifts = [r.lift for r in results]
        print(f"\nDatasets evaluated: {len(results)}")
        print(f"Mean lift: {np.mean(lifts):+.3f}")
        print(f"Median lift: {np.median(lifts):+.3f}")
        print(f"Max lift: {np.max(lifts):+.3f}")
        print(f"Min lift: {np.min(lifts):+.3f}")
        
        print("\nDetail per dataset:")
        for r in results:
            source = "REAL" if r.dataset_id >= 10000 else "SYNTH"
            print(f"  Dataset {r.dataset_id} ({source}): {r.best_model} = {r.best_score:.3f} "
                  f"(lift: {r.lift:+.3f}, baseline: {r.baseline:.3f})")
        
        print("\nModel wins:")
        model_wins = {}
        for r in results:
            model_wins[r.best_model] = model_wins.get(r.best_model, 0) + 1
        for model, wins in sorted(model_wins.items(), key=lambda x: -x[1]):
            print(f"  {model}: {wins}")
        
        # Separate stats for synthetic vs real
        synth_results = [r for r in results if r.dataset_id < 10000]
        real_results = [r for r in results if r.dataset_id >= 10000]
        
        if synth_results:
            synth_lifts = [r.lift for r in synth_results]
            print(f"\nSynthetic datasets ({len(synth_results)}):")
            print(f"  Mean lift: {np.mean(synth_lifts):+.3f}")
            print(f"  Median lift: {np.median(synth_lifts):+.3f}")
        
        if real_results:
            real_lifts = [r.lift for r in real_results]
            print(f"\nReal datasets ({len(real_results)}):")
            print(f"  Mean lift: {np.mean(real_lifts):+.3f}")
            print(f"  Median lift: {np.median(real_lifts):+.3f}")
    else:
        print("No datasets passed the filters.")
    
    return results


if __name__ == "__main__":
    results = run_deep_evaluation(
        n_datasets=4,
        n_real_datasets=2,  # Also evaluate 2 real datasets
        seed=42,
        min_samples=100,
        max_baseline=0.6,
        tuning_iter=15  # Not used for tabular (TS models are primary focus)
    )

