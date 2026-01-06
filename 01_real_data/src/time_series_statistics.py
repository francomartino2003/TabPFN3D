"""
Statistical analysis for time series datasets
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import acf, pacf
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesStatistics:
    """Calculates statistics for time series datasets"""
    
    @staticmethod
    def compute_dataset_stats_simple(dataset) -> Dict:
        """
        Calculates basic and lightweight statistics for a dataset
        Extracts information directly from data, not from metadata
        
        Args:
            dataset: TimeSeriesDataset object
            
        Returns:
            Dict with basic statistics
        """
        # Use train if available, otherwise use combined X
        if dataset.X_train is not None:
            X = dataset.X_train  # Use train for statistics
            n_train = dataset.n_train
            n_test = dataset.n_test
            n_total = n_train + n_test
        else:
            X = dataset.X
            n_train = None
            n_test = None
            n_total = dataset.n
        
        # aeon format: (n, channels, length)
        # n = number of samples
        # channels = number of dimensions/variables/channels
        # length = temporal length (timesteps)
        if X is not None and X.size > 0:
            n_samples_in_X, channels, length = X.shape
        else:
            channels, length = 0, 0
            n_samples_in_X = 0
        
        # Detect missing values/padding
        # Missing values can be NaN or '?' (which aeon may convert to NaN or keep as string)
        # For variable length series in ARFF format, they are padded with '?' which aeon converts to NaN
        # Weka ARFF format: all series of unequal length are padded with missing values ('?')
        # aeon ts format: allows series of unequal length without padding
        
        # Detect missing values: NaN or special values
        # aeon generally converts '?' to NaN, but we check both cases
        missing_mask = np.isnan(X)
        
        # If there are strings, search for '?' or other special values
        if X.dtype == object or (hasattr(X.dtype, 'kind') and X.dtype.kind == 'U'):
            try:
                # Convert to string and search for '?'
                X_str = X.astype(str)
                missing_mask = missing_mask | (X_str == '?') | (X_str == 'nan') | (X_str == '') | (X_str == 'None')
            except:
                pass
        
        missing_pct = float(missing_mask.sum() / (n_samples_in_X * channels * length) * 100) if (n_samples_in_X * channels * length) > 0 else 0.0
        
        # Calculate real lengths of each series (excluding missing values/padding)
        # aeon format: (n, channels, length)
        # For each sample, count how many timesteps are valid (not NaN, not '?')
        real_lengths = []
        for i in range(n_samples_in_X):
            # For each channel, count valid timesteps
            # Generally all channels have the same real length
            valid_timesteps = []
            for ch in range(channels):
                valid = ~missing_mask[i, ch, :]  # Format (n, channels, length)
                valid_timesteps.append(valid.sum())
            # Real length is the maximum among channels (if there are differences, use the maximum)
            real_length = max(valid_timesteps) if valid_timesteps else 0
            real_lengths.append(real_length)
        
        min_length = int(min(real_lengths)) if real_lengths and min(real_lengths) > 0 else length
        max_length = int(max(real_lengths)) if real_lengths and max(real_lengths) > 0 else length
        
        # Detect variable length:
        # 1. If min_length != max_length, has variable length series
        # 2. If metadata says equallength=False, it's variable length
        # 3. If there are many missing values at the end of series, probably variable length padding
        has_variable_length = False
        
        if dataset.metadata and 'equallength' in dataset.metadata:
            # If metadata says equallength=False, it's variable length
            has_variable_length = not dataset.metadata['equallength']
        elif min_length != max_length:
            # If real lengths differ, it's variable length
            has_variable_length = True
        elif missing_pct > 0:
            # If there are missing values, check if they're at the end (padding) or distributed (real missing values)
            # For variable length, padding is usually at the end of series
            # aeon format: (n, channels, length)
            # Count how many series have NaN at the end (last 20% of timesteps)
            samples_with_end_padding = 0
            for i in range(n):
                for ch in range(channels):
                    series = missing_mask[i, ch, :]  # Format (n, channels, length)
                    if series.sum() > 0:
                        # Check if NaN are mainly at the end
                        last_portion = int(length * 0.2)  # Last 20%
                        if last_portion > 0:
                            end_missing = series[-last_portion:].sum()
                            total_missing = series.sum()
                            # If more than 50% of NaN are at the end, probably padding
                            if total_missing > 0 and (end_missing / total_missing) > 0.5:
                                samples_with_end_padding += 1
                                break
            
            # If more than 30% of series have padding at the end, probably variable length
            if n_samples_in_X > 0 and (samples_with_end_padding / n_samples_in_X) > 0.3:
                has_variable_length = True
        
        stats_dict = {
            'name': dataset.name,
            'shape': f"({n_total}, {channels}, {length})",  # (samples, channels, length) - formato aeon
            'n_samples': n_total,
            'length': length,  # Maximum temporal length (may be padded for variable length)
            'min_length': min_length,  # Minimum real length (without padding)
            'max_length': max_length,  # Maximum real length (without padding)
            'n_dimensions': channels,  # Number of dimensions/variables/channels
            'missing_pct': missing_pct,
            'has_variable_length': has_variable_length,
        }
        
        # Count samples with missing values (may be padding or real missing values)
        # Format (n, channels, length)
        nan_per_sample = missing_mask.sum(axis=(1, 2))
        samples_with_missing = (nan_per_sample > 0).sum()
        stats_dict['samples_with_missing'] = int(samples_with_missing)
        
        # Basic statistics per dimension (mean, std, min, max)
        # aeon format: (n, channels, length)
        dimension_stats = []
        for ch in range(channels):
            ch_data = X[:, ch, :]  # Channel ch: (n, length)
            ch_missing = missing_mask[:, ch, :]
            valid_data = ch_data[~ch_missing]
            
            if len(valid_data) > 0:
                # Convert to float if necessary
                try:
                    valid_data_float = valid_data.astype(float)
                    dimension_stats.append({
                        'dimension': int(ch),
                        'mean': float(np.mean(valid_data_float)),
                        'std': float(np.std(valid_data_float)),
                        'min': float(np.min(valid_data_float)),
                        'max': float(np.max(valid_data_float)),
                    })
                except (ValueError, TypeError):
                    # If cannot convert to float, use values as they are
                    dimension_stats.append({
                        'dimension': int(ch),
                        'mean': None,
                        'std': None,
                        'min': None,
                        'max': None,
                    })
            else:
                dimension_stats.append({
                    'dimension': int(ch),
                    'mean': None,
                    'std': None,
                    'min': None,
                    'max': None,
                })
        stats_dict['dimension_stats'] = dimension_stats
        
        # Info about targets and classes (use train+test if separated)
        if dataset.X_train is not None:
            # Combine y_train and y_test to calculate classes
            if dataset.y_train is not None and dataset.y_test is not None:
                y = np.concatenate([dataset.y_train, dataset.y_test])
            elif dataset.y_train is not None:
                y = dataset.y_train
            elif dataset.y_test is not None:
                y = dataset.y_test
            else:
                y = None
        else:
            y = dataset.y
        
        if y is not None:
            unique_classes = np.unique(y)
            n_classes = len(unique_classes)
            
            stats_dict['n_classes'] = int(n_classes)
            
            # Calcular balance de clases
            if len(y) > 0:
                unique, counts = np.unique(y, return_counts=True)
                class_balance = float(np.min(counts) / np.max(counts)) if len(counts) > 0 else 0.0
                stats_dict['class_balance'] = class_balance
        else:
            stats_dict['n_classes'] = None
        
        # Train/Test split - use directly from dataset
        if dataset.X_train is not None:
            stats_dict['train_size'] = int(n_train)
            stats_dict['test_size'] = int(n_test) if n_test is not None else None
        elif dataset.metadata:
            train_size = dataset.metadata.get('train_size')
            test_size = dataset.metadata.get('test_size')
            stats_dict['train_size'] = int(train_size) if train_size is not None else None
            stats_dict['test_size'] = int(test_size) if test_size is not None else None
        else:
            stats_dict['train_size'] = None
            stats_dict['test_size'] = None
        
        return stats_dict
    
    @staticmethod
    def compute_dataset_stats(dataset, sample_size: int = 10) -> Dict:
        """
        Calculates statistics for a complete dataset
        
        Args:
            dataset: TimeSeriesDataset object
            sample_size: Number of samples for stationarity analysis
        """
        X = dataset.X
        n, s, m = X.shape
        
        stats_dict = {
            'name': dataset.name,
            'n_samples': n,
            'n_timesteps': s,
            'n_channels': m,
            'total_values': n * s * m,
            'missing_values': int(np.isnan(X).sum()),
            'missing_pct': float(np.isnan(X).sum() / (n * s * m) * 100),
            'dtype': str(X.dtype),
        }
        
        # Statistics per channel
        channel_stats = []
        for ch in range(m):
            channel_data = X[:, :, ch]
            valid_data = channel_data[~np.isnan(channel_data)]
            
            if len(valid_data) > 0:
                channel_stats.append({
                    'channel': int(ch),
                    'mean': float(np.nanmean(channel_data)),
                    'std': float(np.nanstd(channel_data)),
                    'min': float(np.nanmin(channel_data)),
                    'max': float(np.nanmax(channel_data)),
                    'median': float(np.nanmedian(channel_data)),
                    'q25': float(np.nanpercentile(channel_data, 25)),
                    'q75': float(np.nanpercentile(channel_data, 75)),
                    'skewness': float(stats.skew(valid_data)),
                    'kurtosis': float(stats.kurtosis(valid_data)),
                    'missing_pct': float(np.isnan(channel_data).sum() / (n * s) * 100),
                })
            else:
                channel_stats.append({
                    'channel': int(ch),
                    'mean': np.nan,
                    'std': np.nan,
                    'min': np.nan,
                    'max': np.nan,
                    'median': np.nan,
                    'q25': np.nan,
                    'q75': np.nan,
                    'skewness': np.nan,
                    'kurtosis': np.nan,
                    'missing_pct': 100.0,
                })
        
        stats_dict['channel_stats'] = channel_stats
        
        # Temporal statistics
        # Check if there are series of different lengths
        if X.ndim == 3:
            # For datasets with series of equal length
            temporal_stats = {
                'mean_length': int(s),
                'min_length': int(s),
                'max_length': int(s),
                'length_variance': 0.0,
                'equal_length': True,
            }
        else:
            temporal_stats = {
                'mean_length': float(np.mean([len(series) for series in X])),
                'min_length': int(np.min([len(series) for series in X])),
                'max_length': int(np.max([len(series) for series in X])),
                'length_variance': float(np.var([len(series) for series in X])),
                'equal_length': False,
            }
        
        stats_dict['temporal_stats'] = temporal_stats
        
        # Stationarity statistics (representative sample)
        stationarity_stats = []
        sample_indices = np.linspace(0, n-1, min(sample_size, n), dtype=int)
        
        for idx in sample_indices:
            for ch in range(min(3, m)):  # Limit to 3 channels to avoid overloading
                series = X[idx, :, ch]
                valid_series = series[~np.isnan(series)]
                
                if len(valid_series) > 10:  # Minimum for stationarity test
                    try:
                        adf_result = adfuller(valid_series, autolag='AIC')
                        stationarity_stats.append({
                            'sample': int(idx),
                            'channel': int(ch),
                            'adf_statistic': float(adf_result[0]),
                            'adf_pvalue': float(adf_result[1]),
                            'is_stationary': adf_result[1] < 0.05,
                            'n_observations': len(valid_series),
                        })
                    except Exception as e:
                        pass
        
        stats_dict['stationarity'] = stationarity_stats
        stats_dict['n_stationarity_tests'] = len(stationarity_stats)
        
        # Correlation between channels (temporal average)
        if m > 1:
            try:
                mean_series = np.nanmean(X, axis=1)  # (n, m)
                # Remove rows with NaN
                valid_rows = ~np.isnan(mean_series).any(axis=1)
                if valid_rows.sum() > 1:
                    channel_corr = np.corrcoef(mean_series[valid_rows].T)
                    stats_dict['channel_correlation'] = channel_corr.tolist()
                    stats_dict['channel_correlation_mean'] = float(np.nanmean(channel_corr[np.triu_indices(m, k=1)]))
                else:
                    stats_dict['channel_correlation'] = None
                    stats_dict['channel_correlation_mean'] = np.nan
            except:
                stats_dict['channel_correlation'] = None
                stats_dict['channel_correlation_mean'] = np.nan
        else:
            stats_dict['channel_correlation'] = None
            stats_dict['channel_correlation_mean'] = np.nan
        
        # Target statistics if they exist
        if dataset.y is not None:
            y = dataset.y
            target_stats = {
                'has_targets': True,
                'target_dtype': str(y.dtype),
            }
            
            # Check if it's numeric or string
            is_numeric = np.issubdtype(y.dtype, np.number)
            is_integer = np.issubdtype(y.dtype, np.integer) if is_numeric else False
            
            # If it's string or object, treat as classification
            if not is_numeric or is_integer:
                # Classification (can be integer or string)
                try:
                    unique, counts = np.unique(y, return_counts=True)
                    target_stats['target_type'] = 'classification'
                    target_stats['n_classes'] = int(len(unique))
                    # Convert to dict, handling strings and integers
                    class_dist = {}
                    for k, v in zip(unique, counts):
                        if isinstance(k, (str, np.str_)):
                            class_dist[str(k)] = int(v)
                        else:
                            class_dist[int(k)] = int(v)
                    target_stats['class_distribution'] = class_dist
                    target_stats['class_balance'] = float(np.min(counts) / np.max(counts)) if len(counts) > 0 else 0.0
                except Exception as e:
                    # If it fails, only save basic info
                    target_stats['target_type'] = 'classification'
                    target_stats['n_classes'] = len(np.unique(y)) if hasattr(y, '__len__') else None
            else:
                # Regression (numeric but not integer)
                try:
                    y_numeric = y.astype(float)
                    target_stats['target_type'] = 'regression'
                    target_stats['mean'] = float(np.nanmean(y_numeric))
                    target_stats['std'] = float(np.nanstd(y_numeric))
                    target_stats['min'] = float(np.nanmin(y_numeric))
                    target_stats['max'] = float(np.nanmax(y_numeric))
                except Exception as e:
                    # If conversion fails, treat as classification
                    unique, counts = np.unique(y, return_counts=True)
                    target_stats['target_type'] = 'classification'
                    target_stats['n_classes'] = int(len(unique))
            
            stats_dict['target_stats'] = target_stats
        else:
            stats_dict['target_stats'] = {'has_targets': False}
        
        return stats_dict
    
    @staticmethod
    def compute_global_stats(datasets: List) -> pd.DataFrame:
        """
        Aggregates statistics from multiple datasets
        
        Args:
            datasets: List of TimeSeriesDataset objects
            
        Returns:
            DataFrame with aggregated statistics
        """
        all_stats = []
        
        for dataset in datasets:
            stats_dict = TimeSeriesStatistics.compute_dataset_stats(dataset)
            
            # Extract channel statistics
            channel_means = [ch['mean'] for ch in stats_dict['channel_stats'] if not np.isnan(ch['mean'])]
            channel_stds = [ch['std'] for ch in stats_dict['channel_stats'] if not np.isnan(ch['std'])]
            
            row = {
                'name': stats_dict['name'],
                'n_samples': stats_dict['n_samples'],
                'n_timesteps': stats_dict['n_timesteps'],
                'n_channels': stats_dict['n_channels'],
                'missing_pct': stats_dict['missing_pct'],
                'mean_value': np.mean(channel_means) if channel_means else np.nan,
                'std_value': np.mean(channel_stds) if channel_stds else np.nan,
                'channel_corr_mean': stats_dict.get('channel_correlation_mean', np.nan),
                'n_stationarity_tests': stats_dict.get('n_stationarity_tests', 0),
            }
            
            # Add target info
            if stats_dict['target_stats']['has_targets']:
                row['has_targets'] = True
                row['target_type'] = stats_dict['target_stats'].get('target_type', 'unknown')
                if row['target_type'] == 'classification':
                    row['n_classes'] = stats_dict['target_stats'].get('n_classes', np.nan)
                else:
                    row['n_classes'] = np.nan
            else:
                row['has_targets'] = False
                row['target_type'] = None
                row['n_classes'] = np.nan
            
            all_stats.append(row)
        
        return pd.DataFrame(all_stats)

