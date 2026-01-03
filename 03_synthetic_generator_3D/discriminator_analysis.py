"""
Discriminator Analysis: Synthetic vs Real Time Series Datasets

This script:
1. Generates 20 synthetic classification datasets with visualizations
2. Loads real classification datasets from 01_real_data
3. Extracts dataset-level features for discrimination
4. Trains classifiers to distinguish synthetic from real
5. Evaluates and shows feature importance
"""

import numpy as np
import math
import pickle
import os
import sys
import time
import signal
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Timeout for dataset generation (seconds)
GENERATION_TIMEOUT = 90  # 1.5 minutes

# Visualization - use non-interactive backend to avoid tkinter errors
import matplotlib
matplotlib.use('Agg')  # Must be before pyplot import
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ML
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import pandas as pd

# Signal processing
from scipy import stats
from scipy.signal import welch
from scipy.fft import fft

# Local imports
from config import PriorConfig3D
from generator import SyntheticDatasetGenerator3D


# =============================================================================
# PART 1: Generate Synthetic Datasets + Visualizations
# =============================================================================

class GenerationResult:
    """Container for generation result with timeout support."""
    def __init__(self):
        self.dataset = None
        self.error = None
        self.config = None


def generate_with_timeout(generator, result: GenerationResult, config=None):
    """Generate dataset in a thread."""
    try:
        result.dataset = generator.generate(config=config)
        result.config = result.dataset.config
    except Exception as e:
        result.error = str(e)


def generate_synthetic_datasets(n_datasets: int = 20, seed: int = 42, output_dir: str = "visualizations"):
    """Generate synthetic classification datasets and save visualizations."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Force classification
    prior = PriorConfig3D(force_classification=True)
    generator = SyntheticDatasetGenerator3D(prior=prior, seed=seed)
    
    datasets = []
    
    print(f"Generating {n_datasets} synthetic classification datasets...")
    print(f"Visualizations will be saved to: {os.path.abspath(output_dir)}")
    print(f"Timeout per dataset: {GENERATION_TIMEOUT} seconds")
    print("-" * 60)
    sys.stdout.flush()
    
    for i in range(n_datasets):
        start_time = time.time()
        
        # Generate with timeout using threading
        result = GenerationResult()
        thread = threading.Thread(target=generate_with_timeout, args=(generator, result))
        thread.start()
        thread.join(timeout=GENERATION_TIMEOUT)
        
        elapsed = time.time() - start_time
        
        if thread.is_alive():
            # Timeout occurred
            print(f"[{i+1:2d}/{n_datasets}] TIMEOUT ({GENERATION_TIMEOUT}s exceeded)")
            # Try to get config from generator's last sampled config
            try:
                from config import DatasetConfig3D
                temp_config = DatasetConfig3D.sample_from_prior(prior, np.random.default_rng(seed + i))
                total_cells = temp_config.n_samples * temp_config.n_features * temp_config.t_subseq
                n_propagations = temp_config.T_total * temp_config.n_samples if temp_config.sample_mode == 'iid' else temp_config.T_total
                print(f"    n_samples={temp_config.n_samples}, T_total={temp_config.T_total}, "
                      f"t_subseq={temp_config.t_subseq}, n_features={temp_config.n_features}")
                print(f"    n_nodes={temp_config.n_nodes}, mode={temp_config.sample_mode}")
                print(f"    n_noise={temp_config.n_noise_inputs}, n_time={temp_config.n_time_inputs}, "
                      f"n_state={temp_config.n_state_inputs}")
                print(f"    total_cells={total_cells:,}, n_propagations~={n_propagations:,}")
            except Exception as e:
                print(f"    (Could not get config: {e})")
            sys.stdout.flush()
            # Thread will eventually finish, but we move on
            continue
        
        if result.error:
            print(f"[{i+1:2d}/{n_datasets}] ERROR: {result.error}")
            # Try to get config info
            try:
                from config import DatasetConfig3D
                temp_config = DatasetConfig3D.sample_from_prior(prior, np.random.default_rng(seed + i))
                total_cells = temp_config.n_samples * temp_config.n_features * temp_config.t_subseq
                n_propagations = temp_config.T_total * temp_config.n_samples if temp_config.sample_mode == 'iid' else temp_config.T_total
                print(f"    Config: n_samples={temp_config.n_samples}, T_total={temp_config.T_total}, "
                      f"t_subseq={temp_config.t_subseq}, n_features={temp_config.n_features}")
                print(f"    n_nodes={temp_config.n_nodes}, mode={temp_config.sample_mode}")
                print(f"    n_noise={temp_config.n_noise_inputs}, n_time={temp_config.n_time_inputs}, "
                      f"n_state={temp_config.n_state_inputs}")
                print(f"    total_cells={total_cells:,}, n_propagations~={n_propagations:,}")
            except Exception as e:
                print(f"    (Could not get config: {e})")
            sys.stdout.flush()
            continue
        
        if result.dataset is None:
            print(f"[{i+1:2d}/{n_datasets}] ERROR: No dataset generated")
            sys.stdout.flush()
            continue
        
        dataset = result.dataset
        cfg = dataset.config
        datasets.append(dataset)
        
        # Create and save visualization IMMEDIATELY
        try:
            png_path = os.path.join(output_dir, f"synthetic_{i+1:02d}.png")
            fig = visualize_dataset(dataset, title=f"Synthetic Dataset {i+1}")
            fig.savefig(png_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Detailed metrics - same format as timeout/error for comparison
            total_cells = cfg.n_samples * cfg.n_features * cfg.t_subseq
            n_propagations = cfg.T_total * cfg.n_samples if cfg.sample_mode == 'iid' else cfg.T_total
            
            print(f"[{i+1:2d}/{n_datasets}] OK {elapsed:.1f}s | Shape: {dataset.shape} | Classes: {dataset.n_classes}")
            print(f"    n_samples={cfg.n_samples}, T_total={cfg.T_total}, "
                  f"t_subseq={cfg.t_subseq}, n_features={cfg.n_features}")
            print(f"    n_nodes={cfg.n_nodes}, mode={cfg.sample_mode}")
            print(f"    n_noise={cfg.n_noise_inputs}, n_time={cfg.n_time_inputs}, "
                  f"n_state={cfg.n_state_inputs}")
            print(f"    total_cells={total_cells:,}, n_propagations~={n_propagations:,}")
        except Exception as e:
            print(f"[{i+1:2d}/{n_datasets}] Generated but viz failed: {e}")
            # Still print config
            total_cells = cfg.n_samples * cfg.n_features * cfg.t_subseq
            n_propagations = cfg.T_total * cfg.n_samples if cfg.sample_mode == 'iid' else cfg.T_total
            print(f"    n_samples={cfg.n_samples}, T_total={cfg.T_total}, "
                  f"t_subseq={cfg.t_subseq}, n_features={cfg.n_features}")
            print(f"    n_nodes={cfg.n_nodes}, mode={cfg.sample_mode}")
            print(f"    n_noise={cfg.n_noise_inputs}, n_time={cfg.n_time_inputs}, "
                  f"n_state={cfg.n_state_inputs}")
            print(f"    total_cells={total_cells:,}, n_propagations~={n_propagations:,}")
        
        sys.stdout.flush()
    
    print("-" * 60)
    print(f"Generated {len(datasets)} datasets. Visualizations in {output_dir}/")
    return datasets


def visualize_dataset(dataset, title: str = "Dataset", n_series: int = 3):
    """
    Visualize a 3D dataset showing n_series examples per feature.
    
    Args:
        dataset: SyntheticDataset3D object
        title: Plot title
        n_series: Number of series to show per feature
    """
    X, y = dataset.X, dataset.y
    n_samples, n_features, t_len = X.shape
    
    # Select random samples to visualize
    n_show = min(n_series, n_samples)
    indices = np.random.choice(n_samples, n_show, replace=False)
    
    # Create figure with subplots: n_features rows x n_series columns
    fig, axes = plt.subplots(n_features, n_show, figsize=(4*n_show, 2.5*n_features))
    
    if n_features == 1:
        axes = axes.reshape(1, -1)
    if n_show == 1:
        axes = axes.reshape(-1, 1)
    
    # Color by class
    unique_classes = np.unique(y)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
    class_to_color = {c: colors[i] for i, c in enumerate(unique_classes)}
    
    for row, feat_idx in enumerate(range(n_features)):
        for col, sample_idx in enumerate(indices):
            ax = axes[row, col]
            series = X[sample_idx, feat_idx, :]
            class_label = y[sample_idx]
            color = class_to_color[class_label]
            
            # Handle NaN
            valid_mask = ~np.isnan(series)
            t = np.arange(t_len)
            
            ax.plot(t[valid_mask], series[valid_mask], color=color, linewidth=0.8)
            ax.set_title(f"F{feat_idx+1}, Class {int(class_label)}", fontsize=9)
            
            if row == n_features - 1:
                ax.set_xlabel("Time", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"Feature {feat_idx+1}", fontsize=8)
            
            ax.tick_params(labelsize=7)
    
    fig.suptitle(f"{title}\nShape: {X.shape}, Classes: {len(unique_classes)}", fontsize=11)
    plt.tight_layout()
    return fig


# =============================================================================
# PART 2: Load Real Datasets
# =============================================================================

def load_real_datasets(pkl_path: str = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Load real classification datasets from pickle file.
    
    Returns list of (X, y) tuples where X has shape (n_samples, n_features, t_len)
    """
    if pkl_path is None:
        pkl_path = os.path.join(os.path.dirname(__file__), 
                                '..', '01_real_data', 'AEON', 'data', 
                                'classification_datasets.pkl')
    
    print(f"Loading real datasets from {pkl_path}...")
    
    # Add src to path so pickle can find the module references
    real_data_path = os.path.join(os.path.dirname(__file__), '..', '01_real_data')
    if real_data_path not in sys.path:
        sys.path.insert(0, real_data_path)
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    datasets = []
    
    # Handle both dict and list formats
    if isinstance(data, dict):
        items = data.items()
    elif isinstance(data, list):
        # List of dicts with 'name' key, or list of tuples
        items = []
        for i, item in enumerate(data):
            if isinstance(item, dict):
                name = item.get('name', item.get('dataset_name', f'dataset_{i}'))
                items.append((name, item))
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                items.append((item[0], item[1]))
            else:
                items.append((f'dataset_{i}', item))
    else:
        print(f"Unknown data format: {type(data)}")
        return []
    
    for name, content in items:
        try:
            # Handle different content formats
            if isinstance(content, dict):
                X_train = content.get('X_train')
                X_test = content.get('X_test')
                y_train = content.get('y_train')
                y_test = content.get('y_test')
            else:
                # Try to extract from object attributes
                X_train = getattr(content, 'X_train', None)
                X_test = getattr(content, 'X_test', None)
                y_train = getattr(content, 'y_train', None)
                y_test = getattr(content, 'y_test', None)
            
            if X_train is None or X_test is None:
                continue
            
            # Concatenate train and test
            X = np.concatenate([X_train, X_test], axis=0)
            y = np.concatenate([y_train, y_test], axis=0)
            
            # Ensure 3D shape (n_samples, n_features, t_len)
            if X.ndim == 2:
                X = X[:, np.newaxis, :]  # Add feature dimension
            elif X.ndim == 3 and X.shape[1] > X.shape[2]:
                # Might be (n_samples, t_len, n_features), transpose
                X = np.transpose(X, (0, 2, 1))
            
            # Convert y to numeric if needed
            if not np.issubdtype(y.dtype, np.number):
                unique_labels = np.unique(y)
                label_map = {l: i for i, l in enumerate(unique_labels)}
                y = np.array([label_map[l] for l in y])
            
            datasets.append((X, y, name))
            
        except Exception as e:
            print(f"  Warning: Could not load {name}: {e}")
            continue
    
    print(f"Loaded {len(datasets)} real datasets")
    return datasets


# =============================================================================
# PART 3: Feature Extraction
# =============================================================================

@dataclass
class DatasetFeatures:
    """Features extracted from a single dataset."""
    name: str
    is_synthetic: bool
    features: Dict[str, float]


def extract_features(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Extract dataset-level features for discrimination.
    
    Args:
        X: Shape (n_samples, n_features, t_len)
        y: Shape (n_samples,)
    
    Returns:
        Dictionary of feature name -> value
    """
    n_samples, n_channels, t_len = X.shape
    features = {}
    
    # Flatten for some aggregate stats
    X_flat = X.reshape(-1)
    X_flat_clean = X_flat[~np.isnan(X_flat)]
    
    # ==========================================================================
    # 1. Basic Statistics
    # ==========================================================================
    features['mean'] = np.nanmean(X)
    features['std'] = np.nanstd(X)
    features['var'] = np.nanvar(X)
    features['skewness'] = stats.skew(X_flat_clean) if len(X_flat_clean) > 0 else 0
    features['kurtosis'] = stats.kurtosis(X_flat_clean) if len(X_flat_clean) > 0 else 0
    features['range'] = np.nanmax(X) - np.nanmin(X)
    features['coef_variation'] = features['std'] / (abs(features['mean']) + 1e-8)
    
    # Proportion near saturation (|x| > 0.9)
    features['prop_saturated'] = np.mean(np.abs(X_flat_clean) > 0.9) if len(X_flat_clean) > 0 else 0
    
    # Proportion of repeated values (discretization artifact)
    if len(X_flat_clean) > 0:
        unique_ratio = len(np.unique(X_flat_clean)) / len(X_flat_clean)
        features['unique_ratio'] = unique_ratio
        features['prop_repeated'] = 1 - unique_ratio
    else:
        features['unique_ratio'] = 1
        features['prop_repeated'] = 0
    
    # ==========================================================================
    # 2. Distribution Shape
    # ==========================================================================
    if len(X_flat_clean) > 10:
        # KS test against normal
        ks_stat, _ = stats.kstest(X_flat_clean[:10000], 'norm', 
                                   args=(np.mean(X_flat_clean), np.std(X_flat_clean) + 1e-8))
        features['ks_normal'] = ks_stat
        
        # Excess mass near 0
        features['mass_near_zero'] = np.mean(np.abs(X_flat_clean) < 0.1)
        
        # Symmetry
        median = np.median(X_flat_clean)
        features['asymmetry'] = (np.mean(X_flat_clean) - median) / (np.std(X_flat_clean) + 1e-8)
    else:
        features['ks_normal'] = 0
        features['mass_near_zero'] = 0
        features['asymmetry'] = 0
    
    # ==========================================================================
    # 3. Temporal Stability
    # ==========================================================================
    # Variance by window (rolling)
    window_size = max(10, t_len // 10)
    
    vars_start = []
    vars_end = []
    
    for sample_idx in range(min(50, n_samples)):
        for ch in range(n_channels):
            series = X[sample_idx, ch, :]
            if not np.any(np.isnan(series)):
                var_start = np.var(series[:window_size])
                var_end = np.var(series[-window_size:])
                vars_start.append(var_start)
                vars_end.append(var_end)
    
    if vars_start and vars_end:
        features['var_start'] = np.mean(vars_start)
        features['var_end'] = np.mean(vars_end)
        features['var_ratio_end_start'] = np.mean(vars_end) / (np.mean(vars_start) + 1e-8)
    else:
        features['var_start'] = 0
        features['var_end'] = 0
        features['var_ratio_end_start'] = 1
    
    # ==========================================================================
    # 4. Autocorrelation Features
    # ==========================================================================
    acf_features = compute_acf_features(X)
    features.update(acf_features)
    
    # ==========================================================================
    # 5. Spectral Features
    # ==========================================================================
    spectral_features = compute_spectral_features(X)
    features.update(spectral_features)
    
    # ==========================================================================
    # 6. Complexity Features
    # ==========================================================================
    complexity_features = compute_complexity_features(X)
    features.update(complexity_features)
    
    # ==========================================================================
    # 7. Multivariate Features
    # ==========================================================================
    if n_channels > 1:
        mv_features = compute_multivariate_features(X)
        features.update(mv_features)
    
    # ==========================================================================
    # 8. Target-related Features
    # ==========================================================================
    features['n_classes'] = len(np.unique(y))
    features['class_balance'] = np.min(np.bincount(y.astype(int))) / (np.max(np.bincount(y.astype(int))) + 1)
    
    # Dataset size features
    features['n_samples'] = n_samples
    features['n_channels'] = n_channels
    features['t_len'] = t_len
    features['total_points'] = n_samples * n_channels * t_len
    
    return features


def compute_acf_features(X: np.ndarray) -> Dict[str, float]:
    """Compute autocorrelation features."""
    features = {}
    
    n_samples, n_channels, t_len = X.shape
    lags = [1, 2, 3, 5, 10, 20]
    lags = [l for l in lags if l < t_len // 2]
    
    acf_values = {l: [] for l in lags}
    
    # Sample a subset for efficiency
    for sample_idx in range(min(30, n_samples)):
        for ch in range(n_channels):
            series = X[sample_idx, ch, :]
            if np.any(np.isnan(series)):
                continue
            
            # Normalize
            series = (series - np.mean(series)) / (np.std(series) + 1e-8)
            
            for lag in lags:
                if lag < len(series):
                    acf = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                    if not np.isnan(acf):
                        acf_values[lag].append(acf)
    
    # Aggregate
    for lag in lags:
        if acf_values[lag]:
            features[f'acf_lag{lag}'] = np.mean(acf_values[lag])
        else:
            features[f'acf_lag{lag}'] = 0
    
    # ACF decay rate (slope of ACF vs lag)
    if len(lags) >= 2:
        acf_means = [features[f'acf_lag{l}'] for l in lags]
        if len(acf_means) > 1:
            slope, _ = np.polyfit(lags, acf_means, 1)
            features['acf_decay_slope'] = slope
        else:
            features['acf_decay_slope'] = 0
    else:
        features['acf_decay_slope'] = 0
    
    # Area under ACF curve
    features['acf_auc'] = sum(abs(features.get(f'acf_lag{l}', 0)) for l in lags)
    
    return features


def compute_spectral_features(X: np.ndarray) -> Dict[str, float]:
    """Compute spectral/frequency features."""
    features = {}
    
    n_samples, n_channels, t_len = X.shape
    
    psd_slopes = []
    spectral_entropies = []
    peak_ratios = []
    
    for sample_idx in range(min(30, n_samples)):
        for ch in range(n_channels):
            series = X[sample_idx, ch, :]
            if np.any(np.isnan(series)) or len(series) < 16:
                continue
            
            try:
                # Compute PSD using Welch's method
                freqs, psd = welch(series, nperseg=min(256, len(series)//2))
                
                if len(psd) > 2 and np.all(psd > 0):
                    # Log-log slope of PSD
                    log_freqs = np.log(freqs[1:] + 1e-8)
                    log_psd = np.log(psd[1:] + 1e-8)
                    slope, _ = np.polyfit(log_freqs, log_psd, 1)
                    psd_slopes.append(slope)
                    
                    # Spectral entropy
                    psd_norm = psd / (np.sum(psd) + 1e-8)
                    entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-8))
                    spectral_entropies.append(entropy)
                    
                    # Peak ratio (main peak / total energy)
                    peak_ratio = np.max(psd) / (np.sum(psd) + 1e-8)
                    peak_ratios.append(peak_ratio)
                    
            except Exception:
                continue
    
    features['psd_slope'] = np.mean(psd_slopes) if psd_slopes else 0
    features['psd_slope_std'] = np.std(psd_slopes) if psd_slopes else 0
    features['spectral_entropy'] = np.mean(spectral_entropies) if spectral_entropies else 0
    features['peak_ratio'] = np.mean(peak_ratios) if peak_ratios else 0
    
    return features


def compute_complexity_features(X: np.ndarray) -> Dict[str, float]:
    """Compute complexity and predictability features."""
    features = {}
    
    n_samples, n_channels, t_len = X.shape
    
    # Sample entropy approximation (using permutation patterns)
    perm_entropies = []
    ar1_errors = []
    
    for sample_idx in range(min(20, n_samples)):
        for ch in range(n_channels):
            series = X[sample_idx, ch, :]
            if np.any(np.isnan(series)):
                continue
            
            # Permutation entropy (simplified)
            if len(series) >= 10:
                m = 3  # embedding dimension
                patterns = []
                for i in range(len(series) - m):
                    pattern = tuple(np.argsort(series[i:i+m]))
                    patterns.append(pattern)
                
                if patterns:
                    unique_patterns = len(set(patterns))
                    max_patterns = math.factorial(m)
                    perm_entropy = np.log(unique_patterns) / np.log(max_patterns)
                    perm_entropies.append(perm_entropy)
            
            # AR(1) prediction error
            if len(series) >= 5:
                X_ar = series[:-1].reshape(-1, 1)
                y_ar = series[1:]
                try:
                    coef = np.linalg.lstsq(X_ar, y_ar, rcond=None)[0][0]
                    pred = series[:-1] * coef
                    error = np.mean((y_ar - pred) ** 2) / (np.var(series) + 1e-8)
                    ar1_errors.append(error)
                except:
                    pass
    
    features['permutation_entropy'] = np.mean(perm_entropies) if perm_entropies else 0
    features['ar1_error'] = np.mean(ar1_errors) if ar1_errors else 0
    
    return features


def compute_multivariate_features(X: np.ndarray) -> Dict[str, float]:
    """Compute multivariate (cross-channel) features."""
    features = {}
    
    n_samples, n_channels, t_len = X.shape
    
    if n_channels < 2:
        return features
    
    # Cross-correlation statistics
    cross_corrs = []
    
    for sample_idx in range(min(20, n_samples)):
        corr_matrix = np.zeros((n_channels, n_channels))
        
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                s1 = X[sample_idx, i, :]
                s2 = X[sample_idx, j, :]
                
                if not np.any(np.isnan(s1)) and not np.any(np.isnan(s2)):
                    corr = np.corrcoef(s1, s2)[0, 1]
                    if not np.isnan(corr):
                        cross_corrs.append(abs(corr))
    
    if cross_corrs:
        features['mean_cross_corr'] = np.mean(cross_corrs)
        features['std_cross_corr'] = np.std(cross_corrs)
        features['max_cross_corr'] = np.max(cross_corrs)
    else:
        features['mean_cross_corr'] = 0
        features['std_cross_corr'] = 0
        features['max_cross_corr'] = 0
    
    return features


# =============================================================================
# PART 4: Train Discriminator
# =============================================================================

def prepare_discrimination_data(
    synthetic_datasets: List,
    real_datasets: List[Tuple[np.ndarray, np.ndarray, str]]
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Extract features from all datasets and prepare for training.
    
    Returns:
        X_df: DataFrame of features
        y: Binary labels (0=real, 1=synthetic)
    """
    all_features = []
    all_labels = []
    all_names = []
    
    print("\nExtracting features from synthetic datasets...")
    for i, ds in enumerate(synthetic_datasets):
        try:
            feats = extract_features(ds.X, ds.y)
            all_features.append(feats)
            all_labels.append(1)  # Synthetic
            all_names.append(f"synthetic_{i}")
        except Exception as e:
            print(f"  Error on synthetic {i}: {e}")
    
    print("Extracting features from real datasets...")
    for X, y, name in real_datasets:
        try:
            feats = extract_features(X, y)
            all_features.append(feats)
            all_labels.append(0)  # Real
            all_names.append(name)
        except Exception as e:
            print(f"  Error on {name}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(all_features)
    df['name'] = all_names
    df['is_synthetic'] = all_labels
    
    # Handle NaN/Inf
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    return df


def train_discriminator(df: pd.DataFrame, test_size: float = 0.5):
    """
    Train multiple classifiers to discriminate synthetic from real.
    """
    # Separate features and labels
    feature_cols = [c for c in df.columns if c not in ['name', 'is_synthetic']]
    X = df[feature_cols].values
    y = df['is_synthetic'].values
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(y_train)} ({sum(y_train)} synthetic, {len(y_train)-sum(y_train)} real)")
    print(f"Test set: {len(y_test)} ({sum(y_test)} synthetic, {len(y_test)-sum(y_test)} real)")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Models to try
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    results = {}
    
    print("\n" + "="*60)
    print("DISCRIMINATOR RESULTS")
    print("="*60)
    
    best_model = None
    best_auc = 0
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        results[name] = {'accuracy': acc, 'auc': auc, 'model': model}
        
        print(f"\n{name}:")
        print(f"  Accuracy: {acc:.3f}")
        print(f"  AUC-ROC:  {auc:.3f}")
        
        if auc > best_auc:
            best_auc = auc
            best_model = (name, model)
    
    # Feature importance from best model
    print("\n" + "="*60)
    print(f"FEATURE IMPORTANCE (from {best_model[0]})")
    print("="*60)
    
    if hasattr(best_model[1], 'feature_importances_'):
        importances = best_model[1].feature_importances_
    elif hasattr(best_model[1], 'coef_'):
        importances = np.abs(best_model[1].coef_[0])
    else:
        importances = np.zeros(len(feature_cols))
    
    # Sort by importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 most important features:")
    for i, row in importance_df.head(20).iterrows():
        print(f"  {row['feature']:30s}: {row['importance']:.4f}")
    
    return results, importance_df, scaler, best_model


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution."""
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "discriminator_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Generate synthetic datasets
    print("\n" + "="*60)
    print("STEP 1: Generating Synthetic Datasets")
    print("="*60)
    
    vis_dir = os.path.join(output_dir, "visualizations")
    synthetic_datasets = generate_synthetic_datasets(
        n_datasets=20, 
        seed=42, 
        output_dir=vis_dir
    )
    
    # 2. Load real datasets
    print("\n" + "="*60)
    print("STEP 2: Loading Real Datasets")
    print("="*60)
    
    real_datasets = load_real_datasets()
    
    if not real_datasets:
        print("ERROR: No real datasets loaded. Check the pkl path.")
        return
    
    # Use same number of real as synthetic for balance
    n_real = min(len(real_datasets), len(synthetic_datasets))
    real_datasets = real_datasets[:n_real]
    print(f"Using {n_real} real datasets to match {len(synthetic_datasets)} synthetic")
    
    # 3. Extract features
    print("\n" + "="*60)
    print("STEP 3: Extracting Dataset Features")
    print("="*60)
    
    df = prepare_discrimination_data(synthetic_datasets, real_datasets)
    
    # Save features
    df.to_csv(os.path.join(output_dir, "dataset_features.csv"), index=False)
    print(f"\nFeatures saved to {output_dir}/dataset_features.csv")
    print(f"Total datasets: {len(df)} ({sum(df['is_synthetic'])} synthetic, {len(df)-sum(df['is_synthetic'])} real)")
    print(f"Total features: {len([c for c in df.columns if c not in ['name', 'is_synthetic']])}")
    
    # 4. Train discriminator
    print("\n" + "="*60)
    print("STEP 4: Training Discriminator")
    print("="*60)
    
    results, importance_df, scaler, best_model = train_discriminator(df)
    
    # Save importance
    importance_df.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)
    
    # 5. Visualize feature importance
    print("\n" + "="*60)
    print("STEP 5: Saving Feature Importance Plot")
    print("="*60)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    top_n = 20
    top_features = importance_df.head(top_n)
    
    ax.barh(range(len(top_features)), top_features['importance'].values)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Features for Synthetic vs Real Discrimination\n(Model: {best_model[0]})')
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nAll outputs saved to {output_dir}/")
    print("\nDone!")


if __name__ == "__main__":
    main()

