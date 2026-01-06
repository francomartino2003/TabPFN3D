"""
Preprocessing for 3D Time Series Data.

This module adapts TabPFN's 2D preprocessing to 3D time series:
- Normalization across temporal dimension
- Missing value handling with temporal-aware imputation
- Missing value flags as additional features
- Outlier clipping

The preprocessing is designed to be compatible with the TemporalEncoder.
"""
import numpy as np
import torch
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass

try:
    from .training_config import Preprocessing3DConfig
except ImportError:
    from training_config import Preprocessing3DConfig


@dataclass
class PreprocessingResult:
    """Result of preprocessing a 3D dataset."""
    X: np.ndarray  # Preprocessed features (n, m', t) where m' may include missing flags
    original_shape: Tuple[int, int, int]
    n_features_original: int
    n_features_output: int
    feature_names: List[str]
    stats: dict  # Normalization statistics for inverse transform


class Preprocessor3D:
    """
    Preprocessor for 3D time series data.
    
    Handles:
    - Normalization: z-norm, minmax, quantile per feature
    - Missing values: forward-fill, interpolation, mean imputation
    - Missing flags: optional binary features indicating missing positions
    - Outlier clipping
    
    Usage:
        preprocessor = Preprocessor3D(config)
        preprocessor.fit(X_train)
        X_train_processed = preprocessor.transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
    """
    
    def __init__(self, config: Optional[Preprocessing3DConfig] = None):
        if config is None:
            config = Preprocessing3DConfig()
        self.config = config
        
        # Statistics computed during fit
        self.is_fitted = False
        self.means_: Optional[np.ndarray] = None  # (n_features,)
        self.stds_: Optional[np.ndarray] = None   # (n_features,)
        self.mins_: Optional[np.ndarray] = None   # (n_features,)
        self.maxs_: Optional[np.ndarray] = None   # (n_features,)
        self.constant_features_: Optional[np.ndarray] = None  # Boolean mask
        self.n_features_original_: int = 0
        self.n_features_removed_: int = 0  # Count of removed constant features
    
    def fit(self, X: np.ndarray) -> 'Preprocessor3D':
        """
        Fit preprocessor on training data.
        
        Args:
            X: (n_samples, n_features, n_timesteps)
        Returns:
            self
        """
        assert X.ndim == 3, f"Expected 3D array, got shape {X.shape}"
        n_samples, n_features, n_timesteps = X.shape
        self.n_features_original_ = n_features
        
        # Handle missing values temporarily for computing statistics
        X_filled = self._impute_for_stats(X)
        
        # Compute statistics per feature (across all samples and timesteps)
        # Reshape to (n_samples * n_timesteps, n_features) for statistics
        X_flat = X_filled.transpose(0, 2, 1).reshape(-1, n_features)
        
        self.means_ = np.nanmean(X_flat, axis=0)
        self.stds_ = np.nanstd(X_flat, axis=0)
        self.mins_ = np.nanmin(X_flat, axis=0)
        self.maxs_ = np.nanmax(X_flat, axis=0)
        
        # Identify constant or near-constant features using multiple criteria
        if self.config.remove_constant_features:
            self.constant_features_ = self._detect_constant_features(X_flat, n_features)
        else:
            self.constant_features_ = np.zeros(n_features, dtype=bool)
        
        # Handle zero std (after detecting constants, for normalization)
        self.stds_ = np.where(self.stds_ < 1e-8, 1.0, self.stds_)
        
        self.is_fitted = True
        self.n_features_removed_ = int(self.constant_features_.sum())
        
        return self
    
    def _detect_constant_features(self, X_flat: np.ndarray, n_features: int) -> np.ndarray:
        """
        Detect constant or near-constant features using multiple criteria.
        
        A feature is considered constant/near-constant if ANY of:
        1. std < constant_threshold_std
        2. (max - min) < constant_threshold_range
        3. n_unique / n_total < constant_threshold_unique_ratio
        
        Args:
            X_flat: (n_samples * n_timesteps, n_features) flattened data
            n_features: number of features
        
        Returns:
            Boolean mask where True = constant feature to remove
        """
        n_total = X_flat.shape[0]
        constant_mask = np.zeros(n_features, dtype=bool)
        
        # Get thresholds from config (with defaults for backward compatibility)
        std_thresh = getattr(self.config, 'constant_threshold_std', 1e-6)
        range_thresh = getattr(self.config, 'constant_threshold_range', 1e-6)
        unique_ratio_thresh = getattr(self.config, 'constant_threshold_unique_ratio', 0.01)
        
        for f in range(n_features):
            feature_data = X_flat[:, f]
            # Remove NaN for analysis
            valid_data = feature_data[~np.isnan(feature_data)]
            
            if len(valid_data) == 0:
                # All NaN - mark as constant
                constant_mask[f] = True
                continue
            
            # Criterion 1: Standard deviation
            if self.stds_[f] < std_thresh:
                constant_mask[f] = True
                continue
            
            # Criterion 2: Range (max - min)
            feature_range = self.maxs_[f] - self.mins_[f]
            if feature_range < range_thresh:
                constant_mask[f] = True
                continue
            
            # Criterion 3: Unique values ratio
            # Only check if we have enough data points
            if len(valid_data) > 10:
                n_unique = len(np.unique(valid_data))
                unique_ratio = n_unique / len(valid_data)
                if unique_ratio < unique_ratio_thresh:
                    constant_mask[f] = True
                    continue
        
        return constant_mask
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted statistics.
        
        Args:
            X: (n_samples, n_features, n_timesteps)
        Returns:
            X_transformed: (n_samples, n_features', n_timesteps)
                          where n_features' may be larger if missing flags are added
        """
        assert self.is_fitted, "Preprocessor must be fitted before transform"
        assert X.ndim == 3, f"Expected 3D array, got shape {X.shape}"
        
        X = X.copy().astype(np.float32)
        n_samples, n_features, n_timesteps = X.shape
        
        # Track missing values before imputation
        if self.config.add_missing_flags:
            missing_mask = np.isnan(X)  # (n_samples, n_features, n_timesteps)
        
        # 1. Handle missing values
        if self.config.handle_missing:
            X = self._impute_missing(X)
        
        # 2. Clip outliers
        if self.config.clip_outliers:
            X = self._clip_outliers(X)
        
        # 3. Normalize
        X = self._normalize(X)
        
        # 4. Remove constant features
        if self.config.remove_constant_features and self.constant_features_.any():
            X = X[:, ~self.constant_features_, :]
            if self.config.add_missing_flags:
                missing_mask = missing_mask[:, ~self.constant_features_, :]
        
        # 5. Add missing flags as additional features
        if self.config.add_missing_flags:
            missing_flags = missing_mask.astype(np.float32)
            X = np.concatenate([X, missing_flags], axis=1)
        
        return X
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
    
    def _impute_for_stats(self, X: np.ndarray) -> np.ndarray:
        """Impute missing values for computing statistics."""
        X = X.copy()
        
        # Simple mean imputation for stats computation
        for f in range(X.shape[1]):
            feature_data = X[:, f, :]
            mean_val = np.nanmean(feature_data)
            if np.isnan(mean_val):
                mean_val = 0.0
            feature_data[np.isnan(feature_data)] = mean_val
            X[:, f, :] = feature_data
        
        return X
    
    def _impute_missing(self, X: np.ndarray) -> np.ndarray:
        """
        Impute missing values using configured strategy.
        
        Args:
            X: (n_samples, n_features, n_timesteps)
        Returns:
            X with missing values filled
        """
        method = self.config.missing_imputation
        
        if method == "forward_fill":
            return self._forward_fill(X)
        elif method == "backward_fill":
            return self._backward_fill(X)
        elif method == "interpolate":
            return self._interpolate(X)
        elif method == "mean":
            return self._mean_impute(X)
        elif method == "zero":
            X[np.isnan(X)] = 0.0
            return X
        else:
            raise ValueError(f"Unknown imputation method: {method}")
    
    def _forward_fill(self, X: np.ndarray) -> np.ndarray:
        """Forward-fill missing values along time axis."""
        n_samples, n_features, n_timesteps = X.shape
        
        for s in range(n_samples):
            for f in range(n_features):
                series = X[s, f, :]
                mask = np.isnan(series)
                if mask.any():
                    # Forward fill
                    idx = np.where(~mask, np.arange(n_timesteps), 0)
                    np.maximum.accumulate(idx, out=idx)
                    series_filled = series[idx]
                    
                    # Handle leading NaNs with backward fill or mean
                    if np.isnan(series_filled[0]):
                        first_valid = np.where(~np.isnan(series))[0]
                        if len(first_valid) > 0:
                            series_filled[:first_valid[0]] = series[first_valid[0]]
                        else:
                            series_filled[:] = self.means_[f] if self.means_ is not None else 0.0
                    
                    X[s, f, :] = series_filled
        
        return X
    
    def _backward_fill(self, X: np.ndarray) -> np.ndarray:
        """Backward-fill missing values along time axis."""
        # Reverse, forward-fill, reverse back
        X_rev = X[:, :, ::-1].copy()
        X_filled = self._forward_fill(X_rev)
        return X_filled[:, :, ::-1].copy()
    
    def _interpolate(self, X: np.ndarray) -> np.ndarray:
        """Linear interpolation for missing values."""
        n_samples, n_features, n_timesteps = X.shape
        
        for s in range(n_samples):
            for f in range(n_features):
                series = X[s, f, :]
                mask = np.isnan(series)
                if mask.any():
                    valid_idx = np.where(~mask)[0]
                    if len(valid_idx) > 0:
                        # Interpolate
                        X[s, f, :] = np.interp(
                            np.arange(n_timesteps),
                            valid_idx,
                            series[valid_idx]
                        )
                    else:
                        X[s, f, :] = self.means_[f] if self.means_ is not None else 0.0
        
        return X
    
    def _mean_impute(self, X: np.ndarray) -> np.ndarray:
        """Replace missing values with feature mean."""
        for f in range(X.shape[1]):
            mean_val = self.means_[f] if self.means_ is not None else 0.0
            X[:, f, :][np.isnan(X[:, f, :])] = mean_val
        return X
    
    def _clip_outliers(self, X: np.ndarray) -> np.ndarray:
        """Clip values beyond clip_std standard deviations."""
        for f in range(X.shape[1]):
            mean = self.means_[f]
            std = self.stds_[f]
            lower = mean - self.config.clip_std * std
            upper = mean + self.config.clip_std * std
            X[:, f, :] = np.clip(X[:, f, :], lower, upper)
        return X
    
    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Normalize data using configured method."""
        method = self.config.normalize_type
        
        if method == "none":
            return X
        elif method == "z_norm":
            return self._z_normalize(X)
        elif method == "minmax":
            return self._minmax_normalize(X)
        else:
            raise ValueError(f"Unknown normalization: {method}")
    
    def _z_normalize(self, X: np.ndarray) -> np.ndarray:
        """Z-score normalization per feature."""
        for f in range(X.shape[1]):
            X[:, f, :] = (X[:, f, :] - self.means_[f]) / self.stds_[f]
        return X
    
    def _minmax_normalize(self, X: np.ndarray) -> np.ndarray:
        """Min-max normalization to [0, 1] per feature."""
        for f in range(X.shape[1]):
            range_val = self.maxs_[f] - self.mins_[f]
            if range_val < 1e-8:
                range_val = 1.0
            X[:, f, :] = (X[:, f, :] - self.mins_[f]) / range_val
        return X
    
    def get_output_n_features(self, n_input_features: int) -> int:
        """Calculate number of output features."""
        n_out = n_input_features
        
        if self.config.remove_constant_features and self.constant_features_ is not None:
            n_out -= self.constant_features_.sum()
        
        if self.config.add_missing_flags:
            n_out *= 2
        
        return int(n_out)
    
    def get_feature_removal_info(self) -> dict:
        """
        Get information about removed features.
        
        Returns:
            dict with:
                - n_original: original number of features
                - n_removed: number of features removed
                - n_remaining: number of features after removal
                - removed_indices: indices of removed features
        """
        if not self.is_fitted:
            return {"error": "Preprocessor not fitted"}
        
        removed_indices = np.where(self.constant_features_)[0].tolist()
        
        return {
            "n_original": self.n_features_original_,
            "n_removed": getattr(self, 'n_features_removed_', 0),
            "n_remaining": self.n_features_original_ - getattr(self, 'n_features_removed_', 0),
            "removed_indices": removed_indices,
            "removal_fraction": getattr(self, 'n_features_removed_', 0) / max(1, self.n_features_original_)
        }


def preprocess_batch(
    X_batch: List[np.ndarray],
    config: Optional[Preprocessing3DConfig] = None
) -> Tuple[List[np.ndarray], List[Preprocessor3D]]:
    """
    Preprocess a batch of datasets, fitting each independently.
    
    Args:
        X_batch: List of arrays, each (n_samples, n_features, n_timesteps)
        config: Preprocessing configuration
    
    Returns:
        List of preprocessed arrays and list of fitted preprocessors
    """
    if config is None:
        config = Preprocessing3DConfig()
    
    preprocessed = []
    preprocessors = []
    
    for X in X_batch:
        preprocessor = Preprocessor3D(config)
        X_proc = preprocessor.fit_transform(X)
        preprocessed.append(X_proc)
        preprocessors.append(preprocessor)
    
    return preprocessed, preprocessors


def numpy_to_torch(
    X: np.ndarray, 
    device: str = "cpu"
) -> torch.Tensor:
    """Convert numpy array to torch tensor."""
    return torch.from_numpy(X).float().to(device)


def torch_to_numpy(X: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy array."""
    return X.detach().cpu().numpy()

