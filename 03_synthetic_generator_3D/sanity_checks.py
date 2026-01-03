"""
Sanity Checks for 3D Synthetic Dataset Generator.

Validates that generated datasets:
1. Have correct shapes and types
2. Are learnable (models can beat baseline)
3. Have appropriate difficulty distribution
4. Don't have data leakage
5. Work with different sampling modes
6. Have realistic temporal characteristics
7. Match distributions of real time series datasets

Adapted from 2D sanity checks for temporal data.
"""

import numpy as np
import pandas as pd
import json
import pickle
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.signal import correlate
import warnings

warnings.filterwarnings('ignore')

from config import PriorConfig3D
from generator import SyntheticDatasetGenerator3D, SyntheticDataset3D

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Using RandomForest as replacement.")

# Time series classifiers
try:
    from aeon.classification.convolution_based import RocketClassifier
    from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
    HAS_AEON = True
except ImportError:
    HAS_AEON = False
    print("Warning: aeon not installed. Time series classifiers unavailable.")


# =============================================================================
# Data Classes
# =============================================================================

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
    
    # Model performance
    baseline_acc: float
    logistic_acc: float
    rf_acc: float
    xgb_acc: float
    lift: float
    
    # Data quality
    has_nan: bool
    nan_rate: float
    
    # Temporal statistics
    mean_autocorr_lag1: float = 0.0
    mean_autocorr_lag5: float = 0.0
    trend_strength: float = 0.0
    seasonality_score: float = 0.0
    
    # Feature stats
    mean_feature_std: float = 0.0
    mean_feature_range: float = 0.0


@dataclass
class TemporalStats:
    """Temporal statistics for a time series."""
    autocorr_lag1: float
    autocorr_lag5: float
    autocorr_lag10: float
    trend_strength: float
    stationarity_pvalue: float
    mean: float
    std: float
    range: float
    zero_crossing_rate: float


# =============================================================================
# Utility Functions
# =============================================================================

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


def compute_autocorr(x: np.ndarray, lag: int) -> float:
    """Compute autocorrelation at given lag."""
    if len(x) <= lag:
        return 0.0
    x = x - np.nanmean(x)
    x = np.nan_to_num(x, nan=0.0)
    if np.std(x) < 1e-10:
        return 0.0
    autocorr = np.correlate(x, x, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    if autocorr[0] > 0:
        return autocorr[lag] / autocorr[0]
    return 0.0


def compute_temporal_stats(series: np.ndarray) -> TemporalStats:
    """Compute temporal statistics for a single time series."""
    series = np.nan_to_num(series, nan=0.0)
    
    # Autocorrelation
    ac1 = compute_autocorr(series, 1)
    ac5 = compute_autocorr(series, min(5, len(series)-1))
    ac10 = compute_autocorr(series, min(10, len(series)-1))
    
    # Trend strength (linear regression R²)
    t = np.arange(len(series))
    if np.std(series) > 1e-10:
        slope, intercept, r_value, _, _ = stats.linregress(t, series)
        trend = r_value ** 2
    else:
        trend = 0.0
    
    # Stationarity (simplified - just check variance stability)
    mid = len(series) // 2
    if mid > 5:
        var1 = np.var(series[:mid])
        var2 = np.var(series[mid:])
        stat_pvalue = 1.0 if abs(var1 - var2) < 0.5 * (var1 + var2 + 1e-10) else 0.0
    else:
        stat_pvalue = 0.5
    
    # Zero crossing rate
    zero_crossings = np.sum(np.abs(np.diff(np.sign(series - np.mean(series)))) > 0)
    zcr = zero_crossings / (len(series) - 1) if len(series) > 1 else 0
    
    return TemporalStats(
        autocorr_lag1=ac1,
        autocorr_lag5=ac5,
        autocorr_lag10=ac10,
        trend_strength=trend,
        stationarity_pvalue=stat_pvalue,
        mean=float(np.mean(series)),
        std=float(np.std(series)),
        range=float(np.max(series) - np.min(series)),
        zero_crossing_rate=zcr
    )


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
    
    # Handle NaN - replace with 0
    X_flat = np.nan_to_num(X_flat, nan=0.0)
    
    if len(y) < 10:
        return None, None, None, None
    
    # Check stratification
    unique, counts = np.unique(y, return_counts=True)
    can_stratify = all(c >= 2 for c in counts) and len(unique) >= 2
    
    try:
        if can_stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X_flat, y, test_size=test_size, stratify=y,
                random_state=int(rng.integers(0, 2**31))
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_flat, y, test_size=test_size,
                random_state=int(rng.integers(0, 2**31))
            )
        
        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    except:
        return None, None, None, None


def train_models_flat(
    X_train: np.ndarray, 
    X_test: np.ndarray,
    y_train: np.ndarray, 
    y_test: np.ndarray
) -> Dict[str, float]:
    """Train tabular models on flattened data."""
    results = {}
    
    # Baseline
    baseline = DummyClassifier(strategy='most_frequent')
    baseline.fit(X_train, y_train)
    results['baseline'] = accuracy_score(y_test, baseline.predict(X_test))
    
    # Random Forest (on flattened)
    try:
        rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        results['rf_flat'] = accuracy_score(y_test, rf.predict(X_test))
    except:
        results['rf_flat'] = results['baseline']
    
    return results


def train_ts_models(
    X_train_3d: np.ndarray,
    X_test_3d: np.ndarray, 
    y_train: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Train time series classifiers.
    
    Args:
        X_train_3d: Shape (n_samples, n_features, n_timesteps)
        X_test_3d: Shape (n_samples, n_features, n_timesteps)
    """
    results = {}
    
    # Baseline
    baseline = DummyClassifier(strategy='most_frequent')
    baseline.fit(X_train_3d.reshape(len(X_train_3d), -1), y_train)
    results['baseline'] = accuracy_score(y_test, baseline.predict(X_test_3d.reshape(len(X_test_3d), -1)))
    
    if not HAS_AEON:
        return results
    
    # ROCKET - very fast and effective
    try:
        # aeon expects (n_samples, n_channels, n_timesteps)
        rocket = RocketClassifier(num_kernels=500, random_state=42, n_jobs=-1)
        rocket.fit(X_train_3d, y_train)
        results['rocket'] = accuracy_score(y_test, rocket.predict(X_test_3d))
    except Exception as e:
        results['rocket'] = results['baseline']
    
    # 1-NN DTW (slower but interpretable) - only on small datasets
    if len(X_train_3d) <= 200 and X_train_3d.shape[2] <= 200:
        try:
            knn = KNeighborsTimeSeriesClassifier(n_neighbors=1, distance='dtw')
            knn.fit(X_train_3d, y_train)
            results['1nn_dtw'] = accuracy_score(y_test, knn.predict(X_test_3d))
        except Exception as e:
            results['1nn_dtw'] = results['baseline']
    
    return results


def train_models(
    X_train: np.ndarray, 
    X_test: np.ndarray,
    y_train: np.ndarray, 
    y_test: np.ndarray,
    is_classification: bool = True
) -> Dict[str, float]:
    """Train models and return accuracies (legacy, uses flattened)."""
    results = {}
    
    if is_classification:
        # Baseline
        baseline = DummyClassifier(strategy='most_frequent')
        baseline.fit(X_train, y_train)
        results['baseline'] = accuracy_score(y_test, baseline.predict(X_test))
        
        # Logistic Regression
        try:
            lr = LogisticRegression(max_iter=500, random_state=42)
            lr.fit(X_train, y_train)
            results['logistic'] = accuracy_score(y_test, lr.predict(X_test))
        except:
            results['logistic'] = results['baseline']
        
        # Random Forest
        try:
            rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            results['rf'] = accuracy_score(y_test, rf.predict(X_test))
        except:
            results['rf'] = results['baseline']
        
        # XGBoost
        if HAS_XGBOOST:
            try:
                xgb = XGBClassifier(n_estimators=50, max_depth=5, random_state=42, 
                                   verbosity=0, use_label_encoder=False)
                xgb.fit(X_train, y_train)
                results['xgb'] = accuracy_score(y_test, xgb.predict(X_test))
            except:
                results['xgb'] = results['baseline']
        else:
            results['xgb'] = results['rf']
    else:
        # Regression - not used for TS
        mean_pred = np.mean(y_train)
        results['baseline'] = -mean_squared_error(y_test, np.full_like(y_test, mean_pred))
        results['rf'] = results['baseline']
        results['xgb'] = results['baseline']
    
    return results


class CustomUnpickler(pickle.Unpickler):
    """Custom unpickler that handles missing modules."""
    def find_class(self, module, name):
        # Handle missing modules by returning a dummy class
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            # Return a simple class that stores attributes
            class DummyClass:
                def __init__(self, *args, **kwargs):
                    pass
                def __reduce__(self):
                    return (self.__class__, ())
            return DummyClass


def load_real_datasets(pkl_path: str) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """
    Load real datasets from pickle file.
    
    Returns:
        List of (name, X, y) tuples
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
            
            # Ensure at least 2D
            if X.ndim == 1:
                X = X[:, np.newaxis]
            if X.ndim == 2:
                # Assume (n_samples, time_steps) -> (n_samples, 1, time_steps)
                X = X[:, np.newaxis, :]
            
            datasets.append((str(name), X, y))
        except Exception as e:
            continue
    
    return datasets


# =============================================================================
# Sanity Check Functions
# =============================================================================

def sanity_check_1_basic_stats(datasets: List[SyntheticDataset3D]) -> Dict[str, Any]:
    """
    SANITY CHECK 1: Basic statistics and shape validation.
    """
    print("\n" + "="*60)
    print("SANITY CHECK 1: Basic Statistics")
    print("="*60)
    
    results = {
        'n_datasets': len(datasets),
        'shapes': [],
        'modes': defaultdict(int),
        'classification_ratio': 0,
        'nan_stats': [],
        'all_valid': True
    }
    
    for i, ds in enumerate(datasets):
        # Shape validation
        X, y = ds.X, ds.y
        shape = X.shape
        results['shapes'].append(shape)
        
        # Mode distribution
        results['modes'][ds.config.sample_mode] += 1
        
        # Classification ratio
        if ds.is_classification:
            results['classification_ratio'] += 1
        
        # NaN stats
        nan_rate = np.sum(np.isnan(X)) / X.size if X.size > 0 else 0
        results['nan_stats'].append(nan_rate)
        
        # Validation
        if len(shape) != 3:
            results['all_valid'] = False
            print(f"  [FAIL] Dataset {i}: Invalid shape {shape}")
        elif shape[0] < 10:
            print(f"  [WARN] Dataset {i}: Few samples ({shape[0]})")
    
    results['classification_ratio'] /= len(datasets) if datasets else 1
    results['modes'] = dict(results['modes'])
    
    print(f"\n  Total datasets: {results['n_datasets']}")
    print(f"  Mode distribution: {results['modes']}")
    print(f"  Classification ratio: {results['classification_ratio']:.1%}")
    print(f"  Mean NaN rate: {np.mean(results['nan_stats']):.2%}")
    print(f"  Shape range: {min(results['shapes'])} to {max(results['shapes'])}")
    
    return results


def sanity_check_2_learnability(datasets: List[SyntheticDataset3D], n_test: int = 20) -> Dict[str, Any]:
    """
    SANITY CHECK 2: Models can learn from the data.
    """
    print("\n" + "="*60)
    print("SANITY CHECK 2: Learnability")
    print("="*60)
    
    stats_list = []
    
    for i, ds in enumerate(datasets[:n_test]):
        if not ds.is_classification:
            continue
        
        data = prepare_data(ds.X, ds.y.astype(int))
        if data[0] is None:
            continue
        
        X_train, X_test, y_train, y_test = data
        perf = train_models(X_train, X_test, y_train, y_test, is_classification=True)
        
        lift = max(perf['rf'], perf['xgb']) - perf['baseline']
        
        stats_list.append({
            'dataset_id': i,
            'baseline': perf['baseline'],
            'logistic': perf['logistic'],
            'rf': perf['rf'],
            'xgb': perf['xgb'],
            'lift': lift,
            'mode': ds.config.sample_mode
        })
        
        if i % 5 == 0:
            print(f"  Dataset {i}: baseline={perf['baseline']:.3f}, "
                  f"rf={perf['rf']:.3f}, lift={lift:+.3f}")
    
    if not stats_list:
        return {'error': 'No valid datasets'}
    
    lifts = [s['lift'] for s in stats_list]
    results = {
        'n_tested': len(stats_list),
        'mean_lift': float(np.mean(lifts)),
        'std_lift': float(np.std(lifts)),
        'pct_positive_lift': float(np.mean([l > 0 for l in lifts])),
        'mean_baseline': float(np.mean([s['baseline'] for s in stats_list])),
        'mean_rf': float(np.mean([s['rf'] for s in stats_list])),
        'mean_xgb': float(np.mean([s['xgb'] for s in stats_list])),
        'stats': stats_list
    }
    
    print(f"\n  Mean lift over baseline: {results['mean_lift']:+.3f}")
    print(f"  Datasets with positive lift: {results['pct_positive_lift']:.1%}")
    print(f"  Mean accuracy: baseline={results['mean_baseline']:.3f}, "
          f"RF={results['mean_rf']:.3f}, XGB={results['mean_xgb']:.3f}")
    
    return results


def sanity_check_3_temporal_characteristics(datasets: List[SyntheticDataset3D], 
                                            n_test: int = 10) -> Dict[str, Any]:
    """
    SANITY CHECK 3: Temporal characteristics of generated data.
    """
    print("\n" + "="*60)
    print("SANITY CHECK 3: Temporal Characteristics")
    print("="*60)
    
    all_stats = []
    
    for i, ds in enumerate(datasets[:n_test]):
        X = ds.X
        n_samples, n_features, t_len = X.shape
        
        # Compute stats for sample of series
        sample_size = min(100, n_samples)
        sample_idx = np.random.choice(n_samples, sample_size, replace=False)
        
        ac1_list, ac5_list, trend_list, zcr_list = [], [], [], []
        
        for idx in sample_idx:
            for f in range(n_features):
                series = X[idx, f, :]
                tstats = compute_temporal_stats(series)
                ac1_list.append(tstats.autocorr_lag1)
                ac5_list.append(tstats.autocorr_lag5)
                trend_list.append(tstats.trend_strength)
                zcr_list.append(tstats.zero_crossing_rate)
        
        all_stats.append({
            'dataset_id': i,
            'mode': ds.config.sample_mode,
            'mean_autocorr_lag1': float(np.mean(ac1_list)),
            'mean_autocorr_lag5': float(np.mean(ac5_list)),
            'mean_trend': float(np.mean(trend_list)),
            'mean_zcr': float(np.mean(zcr_list)),
            'std_autocorr_lag1': float(np.std(ac1_list))
        })
        
        print(f"  Dataset {i} ({ds.config.sample_mode}): "
              f"AC(1)={np.mean(ac1_list):.3f}, "
              f"AC(5)={np.mean(ac5_list):.3f}, "
              f"Trend={np.mean(trend_list):.3f}")
    
    results = {
        'n_tested': len(all_stats),
        'mean_autocorr_lag1': float(np.mean([s['mean_autocorr_lag1'] for s in all_stats])),
        'mean_autocorr_lag5': float(np.mean([s['mean_autocorr_lag5'] for s in all_stats])),
        'mean_trend': float(np.mean([s['mean_trend'] for s in all_stats])),
        'stats_by_mode': defaultdict(list),
        'all_stats': all_stats
    }
    
    # Group by mode
    for s in all_stats:
        results['stats_by_mode'][s['mode']].append(s)
    results['stats_by_mode'] = dict(results['stats_by_mode'])
    
    print(f"\n  Overall mean AC(1): {results['mean_autocorr_lag1']:.3f}")
    print(f"  Overall mean AC(5): {results['mean_autocorr_lag5']:.3f}")
    print(f"  Overall mean Trend: {results['mean_trend']:.3f}")
    
    return results


def sanity_check_4_mode_comparison(datasets: List[SyntheticDataset3D]) -> Dict[str, Any]:
    """
    SANITY CHECK 4: Compare characteristics across sampling modes.
    """
    print("\n" + "="*60)
    print("SANITY CHECK 4: Sampling Mode Comparison")
    print("="*60)
    
    by_mode = defaultdict(list)
    
    for ds in datasets:
        mode = ds.config.sample_mode
        by_mode[mode].append({
            'n_samples': ds.X.shape[0],
            'n_features': ds.X.shape[1],
            't_subseq': ds.X.shape[2],
            'n_classes': ds.n_classes,
            'T_total': ds.config.T_total
        })
    
    results = {}
    for mode, stats in by_mode.items():
        results[mode] = {
            'count': len(stats),
            'mean_samples': float(np.mean([s['n_samples'] for s in stats])),
            'mean_features': float(np.mean([s['n_features'] for s in stats])),
            'mean_t_subseq': float(np.mean([s['t_subseq'] for s in stats])),
            'mean_T_total': float(np.mean([s['T_total'] for s in stats]))
        }
        
        print(f"\n  {mode}:")
        print(f"    Count: {results[mode]['count']}")
        print(f"    Mean samples: {results[mode]['mean_samples']:.0f}")
        print(f"    Mean features: {results[mode]['mean_features']:.1f}")
        print(f"    Mean t_subseq: {results[mode]['mean_t_subseq']:.0f}")
    
    return results


def sanity_check_5_label_permutation(datasets: List[SyntheticDataset3D], 
                                     n_test: int = 10) -> Dict[str, Any]:
    """
    SANITY CHECK 5: Label permutation test - shuffled labels should hurt performance.
    """
    print("\n" + "="*60)
    print("SANITY CHECK 5: Label Permutation Test")
    print("="*60)
    
    results_list = []
    
    for i, ds in enumerate(datasets[:n_test]):
        if not ds.is_classification:
            continue
        
        data = prepare_data(ds.X, ds.y.astype(int))
        if data[0] is None:
            continue
        
        X_train, X_test, y_train, y_test = data
        
        # Original performance
        perf_orig = train_models(X_train, X_test, y_train, y_test)
        
        # Permuted performance
        rng = np.random.default_rng(42)
        y_train_perm = rng.permutation(y_train)
        perf_perm = train_models(X_train, X_test, y_train_perm, y_test)
        
        drop = perf_orig['rf'] - perf_perm['rf']
        
        results_list.append({
            'dataset_id': i,
            'rf_original': perf_orig['rf'],
            'rf_permuted': perf_perm['rf'],
            'drop': drop
        })
        
        print(f"  Dataset {i}: RF original={perf_orig['rf']:.3f}, "
              f"permuted={perf_perm['rf']:.3f}, drop={drop:+.3f}")
    
    if not results_list:
        return {'error': 'No valid datasets'}
    
    drops = [r['drop'] for r in results_list]
    results = {
        'n_tested': len(results_list),
        'mean_drop': float(np.mean(drops)),
        'pct_positive_drop': float(np.mean([d > 0 for d in drops])),
        'details': results_list
    }
    
    print(f"\n  Mean performance drop: {results['mean_drop']:+.3f}")
    print(f"  Datasets with expected behavior: {results['pct_positive_drop']:.1%}")
    
    return results


def sanity_check_6_compare_with_real(
    synthetic_datasets: List[SyntheticDataset3D],
    real_pkl_path: str = "../01_real_data/AEON/data/classification_datasets.pkl",
    n_synthetic: int = 20,
    n_real: int = 20
) -> Dict[str, Any]:
    """
    SANITY CHECK 6: Compare synthetic vs real dataset distributions.
    """
    print("\n" + "="*60)
    print("SANITY CHECK 6: Comparison with Real Datasets")
    print("="*60)
    
    # Load real datasets
    script_dir = Path(__file__).parent
    pkl_path = script_dir / real_pkl_path
    
    real_datasets = load_real_datasets(str(pkl_path))
    
    if not real_datasets:
        print("  [WARN] Could not load real datasets")
        return {'error': 'No real datasets loaded'}
    
    print(f"  Loaded {len(real_datasets)} real datasets")
    
    # Extract features from synthetic
    synthetic_stats = []
    for i, ds in enumerate(synthetic_datasets[:n_synthetic]):
        X = ds.X
        n_samples, n_features, t_len = X.shape
        
        # Sample temporal stats
        sample_idx = np.random.choice(n_samples, min(50, n_samples), replace=False)
        ac1_list = []
        for idx in sample_idx:
            for f in range(n_features):
                series = X[idx, f, :]
                ac1_list.append(compute_autocorr(series, 1))
        
        synthetic_stats.append({
            'source': 'synthetic',
            'n_samples': n_samples,
            'n_features': n_features,
            't_length': t_len,
            'mean_ac1': float(np.mean(ac1_list)),
            'std_ac1': float(np.std(ac1_list)),
            'mean_value': float(np.nanmean(X)),
            'std_value': float(np.nanstd(X))
        })
    
    # Extract features from real
    real_stats = []
    for name, X, y in real_datasets[:n_real]:
        n_samples = X.shape[0]
        n_features = X.shape[1] if X.ndim > 1 else 1
        t_len = X.shape[2] if X.ndim > 2 else (X.shape[1] if X.ndim == 2 else 1)
        
        # Reshape if needed
        if X.ndim == 2:
            X = X[:, :, np.newaxis]
        
        # Sample temporal stats
        sample_idx = np.random.choice(n_samples, min(50, n_samples), replace=False)
        ac1_list = []
        for idx in sample_idx:
            for f in range(min(n_features, 5)):  # Limit features
                series = X[idx, f, :] if X.ndim == 3 else X[idx, :]
                if len(series) > 1:
                    ac1_list.append(compute_autocorr(series, 1))
        
        real_stats.append({
            'source': 'real',
            'name': name,
            'n_samples': n_samples,
            'n_features': n_features,
            't_length': t_len,
            'mean_ac1': float(np.mean(ac1_list)) if ac1_list else 0,
            'std_ac1': float(np.std(ac1_list)) if ac1_list else 0,
            'mean_value': float(np.nanmean(X)),
            'std_value': float(np.nanstd(X))
        })
    
    # Compare distributions
    def compare_feature(synth, real, feature_name):
        synth_vals = [s[feature_name] for s in synth]
        real_vals = [s[feature_name] for s in real]
        
        # Filter NaN
        synth_vals = [v for v in synth_vals if np.isfinite(v)]
        real_vals = [v for v in real_vals if np.isfinite(v)]
        
        if not synth_vals or not real_vals:
            return {'statistic': 0, 'pvalue': 1.0}
        
        stat, pvalue = stats.ks_2samp(synth_vals, real_vals)
        return {
            'synth_mean': float(np.mean(synth_vals)),
            'synth_std': float(np.std(synth_vals)),
            'real_mean': float(np.mean(real_vals)),
            'real_std': float(np.std(real_vals)),
            'ks_statistic': float(stat),
            'ks_pvalue': float(pvalue)
        }
    
    comparisons = {
        'n_samples': compare_feature(synthetic_stats, real_stats, 'n_samples'),
        't_length': compare_feature(synthetic_stats, real_stats, 't_length'),
        'mean_ac1': compare_feature(synthetic_stats, real_stats, 'mean_ac1'),
        'std_value': compare_feature(synthetic_stats, real_stats, 'std_value')
    }
    
    print(f"\n  Distribution Comparisons (KS test):")
    for feature, comp in comparisons.items():
        print(f"    {feature}:")
        print(f"      Synthetic: {comp.get('synth_mean', 0):.3f} ± {comp.get('synth_std', 0):.3f}")
        print(f"      Real:      {comp.get('real_mean', 0):.3f} ± {comp.get('real_std', 0):.3f}")
        print(f"      KS p-value: {comp.get('ks_pvalue', 0):.4f}")
    
    return {
        'n_synthetic': len(synthetic_stats),
        'n_real': len(real_stats),
        'comparisons': comparisons,
        'synthetic_stats': synthetic_stats,
        'real_stats': real_stats
    }


def sanity_check_7_difficulty_spectrum(datasets: List[SyntheticDataset3D], 
                                       n_test: int = 30) -> Dict[str, Any]:
    """
    SANITY CHECK 7: Check variety of dataset difficulties.
    """
    print("\n" + "="*60)
    print("SANITY CHECK 7: Difficulty Spectrum")
    print("="*60)
    
    difficulties = []
    
    for i, ds in enumerate(datasets[:n_test]):
        if not ds.is_classification:
            continue
        
        data = prepare_data(ds.X, ds.y.astype(int))
        if data[0] is None:
            continue
        
        X_train, X_test, y_train, y_test = data
        perf = train_models(X_train, X_test, y_train, y_test)
        
        difficulties.append({
            'dataset_id': i,
            'baseline': perf['baseline'],
            'best_model': max(perf['rf'], perf['xgb']),
            'lift': max(perf['rf'], perf['xgb']) - perf['baseline']
        })
    
    if not difficulties:
        return {'error': 'No valid datasets'}
    
    lifts = [d['lift'] for d in difficulties]
    baselines = [d['baseline'] for d in difficulties]
    
    # Categorize difficulty
    easy = sum(1 for l in lifts if l > 0.3)
    medium = sum(1 for l in lifts if 0.1 <= l <= 0.3)
    hard = sum(1 for l in lifts if 0 < l < 0.1)
    impossible = sum(1 for l in lifts if l <= 0)
    
    results = {
        'n_tested': len(difficulties),
        'easy': easy,
        'medium': medium,
        'hard': hard,
        'impossible': impossible,
        'lift_min': float(min(lifts)),
        'lift_max': float(max(lifts)),
        'lift_mean': float(np.mean(lifts)),
        'baseline_mean': float(np.mean(baselines)),
        'details': difficulties
    }
    
    print(f"\n  Difficulty distribution:")
    print(f"    Easy (lift > 0.3):    {easy} ({easy/len(difficulties)*100:.0f}%)")
    print(f"    Medium (0.1-0.3):     {medium} ({medium/len(difficulties)*100:.0f}%)")
    print(f"    Hard (0-0.1):         {hard} ({hard/len(difficulties)*100:.0f}%)")
    print(f"    Impossible (<=0):     {impossible} ({impossible/len(difficulties)*100:.0f}%)")
    print(f"  Lift range: [{results['lift_min']:.3f}, {results['lift_max']:.3f}]")
    
    return results


def sanity_check_8_input_type_impact(datasets: List[SyntheticDataset3D]) -> Dict[str, Any]:
    """
    SANITY CHECK 8: Impact of different input types (noise, time, state).
    """
    print("\n" + "="*60)
    print("SANITY CHECK 8: Input Type Distribution")
    print("="*60)
    
    stats = []
    
    for ds in datasets:
        cfg = ds.config
        total_inputs = cfg.n_noise_inputs + cfg.n_time_inputs + cfg.n_state_inputs
        
        stats.append({
            'n_noise': cfg.n_noise_inputs,
            'n_time': cfg.n_time_inputs,
            'n_state': cfg.n_state_inputs,
            'total': total_inputs,
            'noise_ratio': cfg.n_noise_inputs / total_inputs if total_inputs > 0 else 0,
            'time_ratio': cfg.n_time_inputs / total_inputs if total_inputs > 0 else 0,
            'state_ratio': cfg.n_state_inputs / total_inputs if total_inputs > 0 else 0
        })
    
    results = {
        'n_datasets': len(stats),
        'mean_noise': float(np.mean([s['n_noise'] for s in stats])),
        'mean_time': float(np.mean([s['n_time'] for s in stats])),
        'mean_state': float(np.mean([s['n_state'] for s in stats])),
        'mean_noise_ratio': float(np.mean([s['noise_ratio'] for s in stats])),
        'mean_time_ratio': float(np.mean([s['time_ratio'] for s in stats])),
        'mean_state_ratio': float(np.mean([s['state_ratio'] for s in stats]))
    }
    
    print(f"\n  Mean input counts:")
    print(f"    Noise inputs: {results['mean_noise']:.1f} ({results['mean_noise_ratio']*100:.0f}%)")
    print(f"    Time inputs:  {results['mean_time']:.1f} ({results['mean_time_ratio']*100:.0f}%)")
    print(f"    State inputs: {results['mean_state']:.1f} ({results['mean_state_ratio']*100:.0f}%)")
    
    return results


def sanity_check_9_ts_classifiers(
    synthetic_datasets: List[SyntheticDataset3D],
    real_pkl_path: str = "../01_real_data/AEON/data/classification_datasets.pkl",
    n_synthetic: int = 15,
    n_real: int = 15
) -> Dict[str, Any]:
    """
    SANITY CHECK 9: Compare classification performance using proper TS classifiers.
    
    Uses ROCKET and 1NN-DTW instead of flattened RF/XGB.
    Compares lift distributions between synthetic and real datasets.
    """
    print("\n" + "="*60)
    print("SANITY CHECK 9: Time Series Classifiers")
    print("="*60)
    
    if not HAS_AEON:
        print("  [SKIP] aeon not installed - cannot run TS classifiers")
        return {'error': 'aeon not installed'}
    
    # Evaluate synthetic datasets
    print("\n  Evaluating synthetic datasets with ROCKET...")
    synthetic_results = []
    
    for i, ds in enumerate(synthetic_datasets[:n_synthetic]):
        if not ds.is_classification:
            continue
        
        X, y = ds.X, ds.y.astype(int)
        
        # Handle NaN
        X = np.nan_to_num(X, nan=0.0)
        
        # Need at least 2 classes with 2 samples each
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2 or np.min(counts) < 2:
            continue
        
        # Split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, stratify=y, random_state=42
            )
        except:
            continue
        
        # Train TS models
        perf = train_ts_models(X_train, X_test, y_train, y_test)
        
        best_acc = max(perf.get('rocket', perf['baseline']), 
                       perf.get('1nn_dtw', perf['baseline']))
        lift = best_acc - perf['baseline']
        
        synthetic_results.append({
            'dataset_id': i,
            'source': 'synthetic',
            'baseline': perf['baseline'],
            'rocket': perf.get('rocket', None),
            '1nn_dtw': perf.get('1nn_dtw', None),
            'best': best_acc,
            'lift': lift
        })
        
        if i < 5:
            print(f"    Synthetic {i}: baseline={perf['baseline']:.3f}, "
                  f"ROCKET={perf.get('rocket', 'N/A')}, lift={lift:+.3f}")
    
    # Evaluate real datasets
    print("\n  Evaluating real datasets with ROCKET...")
    script_dir = Path(__file__).parent
    pkl_path = script_dir / real_pkl_path
    real_datasets = load_real_datasets(str(pkl_path))
    
    real_results = []
    
    for name, X, y in real_datasets[:n_real]:
        # Handle NaN
        X = np.nan_to_num(X, nan=0.0)
        y = np.array(y)
        
        # Encode labels if needed
        if y.dtype == object or not np.issubdtype(y.dtype, np.integer):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2 or np.min(counts) < 2:
            continue
        
        # Limit size for speed
        if len(X) > 500:
            idx = np.random.choice(len(X), 500, replace=False)
            X, y = X[idx], y[idx]
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, stratify=y, random_state=42
            )
        except:
            continue
        
        perf = train_ts_models(X_train, X_test, y_train, y_test)
        
        best_acc = max(perf.get('rocket', perf['baseline']),
                       perf.get('1nn_dtw', perf['baseline']))
        lift = best_acc - perf['baseline']
        
        real_results.append({
            'name': name,
            'source': 'real',
            'baseline': perf['baseline'],
            'rocket': perf.get('rocket', None),
            '1nn_dtw': perf.get('1nn_dtw', None),
            'best': best_acc,
            'lift': lift
        })
    
    if len(real_results) > 0:
        print(f"    Evaluated {len(real_results)} real datasets")
    
    # Compare distributions
    synth_lifts = [r['lift'] for r in synthetic_results]
    real_lifts = [r['lift'] for r in real_results]
    
    synth_baselines = [r['baseline'] for r in synthetic_results]
    real_baselines = [r['baseline'] for r in real_results]
    
    results = {
        'n_synthetic': len(synthetic_results),
        'n_real': len(real_results),
        'synthetic_mean_lift': float(np.mean(synth_lifts)) if synth_lifts else 0,
        'synthetic_std_lift': float(np.std(synth_lifts)) if synth_lifts else 0,
        'real_mean_lift': float(np.mean(real_lifts)) if real_lifts else 0,
        'real_std_lift': float(np.std(real_lifts)) if real_lifts else 0,
        'synthetic_mean_baseline': float(np.mean(synth_baselines)) if synth_baselines else 0,
        'real_mean_baseline': float(np.mean(real_baselines)) if real_baselines else 0,
        'synthetic_results': synthetic_results,
        'real_results': real_results
    }
    
    # KS test on lifts
    if synth_lifts and real_lifts:
        ks_stat, ks_pval = stats.ks_2samp(synth_lifts, real_lifts)
        results['lift_ks_statistic'] = float(ks_stat)
        results['lift_ks_pvalue'] = float(ks_pval)
    
    print(f"\n  Lift Comparison (ROCKET):")
    print(f"    Synthetic: {results['synthetic_mean_lift']:+.3f} ± {results['synthetic_std_lift']:.3f}")
    print(f"    Real:      {results['real_mean_lift']:+.3f} ± {results['real_std_lift']:.3f}")
    print(f"    Baseline - Synthetic: {results['synthetic_mean_baseline']:.3f}, Real: {results['real_mean_baseline']:.3f}")
    if 'lift_ks_pvalue' in results:
        print(f"    KS test p-value: {results['lift_ks_pvalue']:.4f}")
    
    return results


# =============================================================================
# Main Orchestration
# =============================================================================

def run_all_checks(
    n_datasets: int = 50,
    seed: int = 42,
    output_dir: str = "sanity_check_results",
    real_data_path: str = "../01_real_data/AEON/data/classification_datasets.pkl"
) -> Dict[str, Any]:
    """
    Run all sanity checks and save results.
    """
    print("="*60)
    print("3D SYNTHETIC GENERATOR SANITY CHECKS")
    print("="*60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate datasets
    print(f"\nGenerating {n_datasets} synthetic datasets...")
    generator = SyntheticDatasetGenerator3D(seed=seed)
    datasets = []
    
    for i in range(n_datasets):
        try:
            ds = generator.generate()
            datasets.append(ds)
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{n_datasets}")
        except Exception as e:
            print(f"  Error generating dataset {i}: {e}")
    
    print(f"Successfully generated {len(datasets)} datasets")
    
    # Run checks
    all_results = {}
    
    all_results['check_1_basic'] = sanity_check_1_basic_stats(datasets)
    all_results['check_2_learnability'] = sanity_check_2_learnability(datasets)
    all_results['check_3_temporal'] = sanity_check_3_temporal_characteristics(datasets)
    all_results['check_4_modes'] = sanity_check_4_mode_comparison(datasets)
    all_results['check_5_permutation'] = sanity_check_5_label_permutation(datasets)
    all_results['check_6_vs_real'] = sanity_check_6_compare_with_real(
        datasets, real_data_path
    )
    all_results['check_7_difficulty'] = sanity_check_7_difficulty_spectrum(datasets)
    all_results['check_8_inputs'] = sanity_check_8_input_type_impact(datasets)
    all_results['check_9_ts_classifiers'] = sanity_check_9_ts_classifiers(
        datasets, real_data_path
    )
    
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    # Save results
    results_path = os.path.join(output_dir, "sanity_check_results.json")
    with open(results_path, 'w') as f:
        json.dump(convert(all_results), f, indent=2)
    
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Results saved to: {results_path}")
    print(f"Total datasets generated: {len(datasets)}")
    
    # Print summary statistics
    if 'check_2_learnability' in all_results and 'mean_lift' in all_results['check_2_learnability']:
        print(f"Mean lift over baseline: {all_results['check_2_learnability']['mean_lift']:+.3f}")
    
    if 'check_7_difficulty' in all_results and 'lift_mean' in all_results['check_7_difficulty']:
        print(f"Difficulty spectrum covered: lift range "
              f"[{all_results['check_7_difficulty']['lift_min']:.3f}, "
              f"{all_results['check_7_difficulty']['lift_max']:.3f}]")
    
    return all_results


if __name__ == "__main__":
    results = run_all_checks(n_datasets=50, seed=42)
