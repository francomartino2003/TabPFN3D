"""
Post-processing transformations for synthetic datasets.

This module applies realistic transformations to the generated data:
- Warping: Non-linear distortions
- Quantization: Binning continuous values
- Missing values: MCAR (Missing Completely At Random)
- Normalization: Scale to realistic ranges
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
import numpy as np

try:
    from .config import DatasetConfig
except ImportError:
    from config import DatasetConfig


@dataclass
class PostProcessingResult:
    """
    Result of post-processing.
    
    Attributes:
        X: Transformed feature matrix
        y: Transformed target vector
        applied_transforms: Record of which transforms were applied
        column_info: Information about each column after transformation
    """
    X: np.ndarray
    y: np.ndarray
    applied_transforms: Dict[str, bool]
    column_info: List[Dict[str, Any]]


class Warper:
    """
    Applies non-linear warping to break simple relationships.
    
    Per paper: "For some datasets, we use the Kumaraswamy feature warping, 
    introducing nonlinear distortions to features"
    
    Warping functions:
    - Kumaraswamy (primary method from paper)
    - Power transforms
    - Smooth monotonic distortions
    """
    
    def __init__(self, intensity: float, rng: np.random.Generator):
        """
        Initialize warper.
        
        Args:
            intensity: How strong the warping should be (0 = none, 2 = strong)
            rng: Random number generator
        """
        self.intensity = intensity
        self.rng = rng
    
    def kumaraswamy_warp(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        """
        Apply Kumaraswamy CDF warping.
        
        The Kumaraswamy distribution CDF is: F(x; a, b) = 1 - (1 - x^a)^b
        This is a flexible family of distributions on [0, 1].
        
        Args:
            x: Values normalized to [0, 1]
            a: First shape parameter (> 0)
            b: Second shape parameter (> 0)
            
        Returns:
            Warped values in [0, 1]
        """
        # Clip to avoid numerical issues
        x = np.clip(x, 1e-10, 1 - 1e-10)
        return 1 - np.power(1 - np.power(x, a), b)
    
    def warp_column(self, x: np.ndarray) -> np.ndarray:
        """
        Apply warping to a single column.
        
        Args:
            x: Column values
            
        Returns:
            Warped values
        """
        # Choose warping type - Kumaraswamy is primary (per paper)
        warp_type = self.rng.choice([
            'kumaraswamy', 'kumaraswamy', 'kumaraswamy',  # Higher probability
            'power', 'sinh', 'rank_based'
        ])
        
        # Normalize to [0, 1] first
        x_min, x_max = x.min(), x.max()
        if x_max - x_min > 1e-10:
            x_norm = (x - x_min) / (x_max - x_min)
        else:
            return x
        
        if warp_type == 'kumaraswamy':
            # Kumaraswamy warping (per paper)
            # Sample shape parameters - intensity affects the deviation from identity
            a = self.rng.uniform(0.5, 2.0 + self.intensity)
            b = self.rng.uniform(0.5, 2.0 + self.intensity)
            x_warped = self.kumaraswamy_warp(x_norm, a, b)
            
        elif warp_type == 'power':
            # Power transform
            power = self.rng.uniform(0.5, 2.0)
            if self.rng.random() < 0.5:
                power = 1 / power
            x_warped = np.power(x_norm + 1e-10, power)
            
        elif warp_type == 'sinh':
            # Sinh-arcsinh transform (flexible skewness/kurtosis)
            skew = self.rng.uniform(-0.5, 0.5) * self.intensity
            tail = self.rng.uniform(0.8, 1.2)
            x_centered = (x_norm - 0.5) * 4
            x_warped = np.sinh(tail * np.arcsinh(x_centered) - skew)
            x_warped = (x_warped - x_warped.min()) / (x_warped.max() - x_warped.min() + 1e-10)
            
        else:  # rank_based
            # Rank-based warping
            ranks = x.argsort().argsort()
            x_warped = ranks / (len(x) - 1 + 1e-10)
            # Add small noise to break ties
            x_warped += self.rng.normal(0, 0.01 * self.intensity, size=x_warped.shape)
            x_warped = np.clip(x_warped, 0, 1)
        
        # Scale back to original range (approximately)
        return x_warped * (x_max - x_min) + x_min
    
    def warp(self, X: np.ndarray, columns: Optional[List[int]] = None) -> np.ndarray:
        """
        Apply warping to selected columns.
        
        Args:
            X: Feature matrix
            columns: Which columns to warp (None = random selection)
            
        Returns:
            Warped feature matrix
        """
        X_warped = X.copy()
        
        if columns is None:
            # Randomly select columns to warp
            n_to_warp = self.rng.binomial(X.shape[1], 0.5)
            columns = self.rng.choice(X.shape[1], size=n_to_warp, replace=False)
        
        for col in columns:
            X_warped[:, col] = self.warp_column(X[:, col])
        
        return X_warped


class Quantizer:
    """
    Discretizes continuous values into bins.
    
    Creates more realistic data by simulating measurement precision
    or naturally binned data.
    """
    
    def __init__(self, n_bins_range: Tuple[int, int], rng: np.random.Generator):
        """
        Initialize quantizer.
        
        Args:
            n_bins_range: Range for number of bins
            rng: Random number generator
        """
        self.n_bins_range = n_bins_range
        self.rng = rng
    
    def quantize_column(self, x: np.ndarray, n_bins: Optional[int] = None) -> np.ndarray:
        """
        Quantize a single column.
        
        Args:
            x: Column values
            n_bins: Number of bins (random if None)
            
        Returns:
            Quantized values
        """
        if n_bins is None:
            low, high = self.n_bins_range
            if low >= high:
                n_bins = low
            else:
                n_bins = self.rng.integers(low, high)
        
        # Handle infinite and extreme values
        x = np.clip(x, -1e10, 1e10)
        x = np.nan_to_num(x, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Compute bin edges
        x_min, x_max = x.min(), x.max()
        
        # Check for valid range
        if not np.isfinite(x_min) or not np.isfinite(x_max):
            return x
        
        if x_max - x_min < 1e-10:
            return x
        
        # Ensure range is not too large (would cause overflow)
        if x_max - x_min > 1e15:
            # Normalize to reasonable range
            x = (x - np.mean(x)) / (np.std(x) + 1e-10)
            x_min, x_max = x.min(), x.max()
        
        # Choose binning strategy
        strategy = self.rng.choice(['uniform', 'quantile', 'custom'])
        
        if strategy == 'uniform':
            edges = np.linspace(x_min, x_max, n_bins + 1)
        elif strategy == 'quantile':
            edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
            edges = np.unique(edges)  # Remove duplicates
        else:  # custom
            # Random edge positions
            inner_edges = np.sort(self.rng.uniform(x_min, x_max, n_bins - 1))
            edges = np.concatenate([[x_min], inner_edges, [x_max]])
        
        # Assign to bins
        bin_indices = np.digitize(x, edges[1:-1])
        
        # Choose output representation
        output_type = self.rng.choice(['center', 'left', 'index'])
        
        if output_type == 'center':
            # Use bin centers
            centers = (edges[:-1] + edges[1:]) / 2
            return centers[bin_indices]
        elif output_type == 'left':
            # Use left edges
            return edges[bin_indices]
        else:
            # Use bin indices
            return bin_indices.astype(float)
    
    def quantize(
        self, 
        X: np.ndarray, 
        columns: Optional[List[int]] = None,
        prob_per_column: float = 0.5
    ) -> np.ndarray:
        """
        Apply quantization to selected columns.
        
        Args:
            X: Feature matrix
            columns: Which columns to quantize (None = random selection)
            prob_per_column: Probability of quantizing each column
            
        Returns:
            Quantized feature matrix
        """
        X_quantized = X.copy()
        
        if columns is None:
            # Randomly select columns
            columns = [i for i in range(X.shape[1]) if self.rng.random() < prob_per_column]
        
        for col in columns:
            X_quantized[:, col] = self.quantize_column(X[:, col])
        
        return X_quantized


class MissingValueInjector:
    """
    Injects missing values (MCAR - Missing Completely At Random).
    
    Returns a masked array or replaces with NaN.
    """
    
    def __init__(self, missing_rate: float, rng: np.random.Generator):
        """
        Initialize injector.
        
        Args:
            missing_rate: Probability of each cell being missing
            rng: Random number generator
        """
        self.missing_rate = missing_rate
        self.rng = rng
    
    def inject(
        self, 
        X: np.ndarray, 
        uniform_rate: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject missing values into features.
        
        Args:
            X: Feature matrix
            uniform_rate: If True, use same rate for all columns
            
        Returns:
            Tuple of (X with NaN for missing, boolean mask of missing)
        """
        X_missing = X.copy().astype(float)
        
        if uniform_rate:
            # Same rate for all cells
            mask = self.rng.random(X.shape) < self.missing_rate
        else:
            # Different rate per column
            mask = np.zeros(X.shape, dtype=bool)
            for col in range(X.shape[1]):
                # Sample rate for this column
                col_rate = self.rng.uniform(0, 2 * self.missing_rate)
                col_rate = min(col_rate, 0.5)  # Cap at 50%
                mask[:, col] = self.rng.random(X.shape[0]) < col_rate
        
        X_missing[mask] = np.nan
        
        return X_missing, mask


class Normalizer:
    """
    Normalizes features to realistic ranges.
    """
    
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
    
    def normalize(self, X: np.ndarray, method: str = 'mixed') -> np.ndarray:
        """
        Normalize features.
        
        Args:
            X: Feature matrix
            method: Normalization method ('standard', 'minmax', 'mixed')
            
        Returns:
            Normalized features
        """
        X_norm = X.copy()
        
        for col in range(X.shape[1]):
            x = X[:, col]
            
            if method == 'mixed':
                col_method = self.rng.choice(['standard', 'minmax', 'none'])
            else:
                col_method = method
            
            if col_method == 'standard':
                mean, std = np.nanmean(x), np.nanstd(x)
                if std > 1e-10:
                    X_norm[:, col] = (x - mean) / std
                    
            elif col_method == 'minmax':
                x_min, x_max = np.nanmin(x), np.nanmax(x)
                if x_max - x_min > 1e-10:
                    X_norm[:, col] = (x - x_min) / (x_max - x_min)
            # else: no normalization
        
        return X_norm


class PostProcessor:
    """
    Main post-processor that applies all transformations.
    """
    
    def __init__(self, config: DatasetConfig, rng: Optional[np.random.Generator] = None):
        """
        Initialize post-processor.
        
        Args:
            config: Dataset configuration
            rng: Random number generator
        """
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng(config.seed)
        
        # Initialize sub-processors
        self.warper = Warper(config.warping_intensity, self.rng)
        # Ensure valid range for quantizer (low < high)
        min_bins = 2
        max_bins = max(min_bins + 1, config.n_quantization_bins)
        self.quantizer = Quantizer(
            (min_bins, max_bins),
            self.rng
        )
        self.missing_injector = MissingValueInjector(config.missing_rate, self.rng)
        self.normalizer = Normalizer(self.rng)
    
    def process(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        feature_types: Optional[Dict[int, str]] = None
    ) -> PostProcessingResult:
        """
        Apply all post-processing steps.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_types: Dict mapping column index to type
            
        Returns:
            PostProcessingResult with transformed data
        """
        applied = {}
        column_info = [{'original_col': i} for i in range(X.shape[1])]
        
        X_processed = X.copy()
        y_processed = y.copy()
        
        # Identify continuous columns
        if feature_types is None:
            continuous_cols = list(range(X.shape[1]))
        else:
            continuous_cols = [i for i, t in feature_types.items() if t == 'continuous']
        
        # Step 1: Warping (only on continuous features)
        if self.config.apply_warping and continuous_cols:
            # Warp a subset of continuous columns
            n_to_warp = max(1, int(len(continuous_cols) * self.rng.uniform(0.3, 0.7)))
            cols_to_warp = self.rng.choice(continuous_cols, size=n_to_warp, replace=False)
            X_processed = self.warper.warp(X_processed, cols_to_warp.tolist())
            applied['warping'] = True
            for col in cols_to_warp:
                column_info[col]['warped'] = True
        else:
            applied['warping'] = False
        
        # Step 2: Quantization (only on continuous features)
        if self.config.apply_quantization and continuous_cols:
            # Quantize a subset of continuous columns
            n_to_quantize = max(1, int(len(continuous_cols) * self.rng.uniform(0.2, 0.5)))
            cols_to_quantize = self.rng.choice(continuous_cols, size=n_to_quantize, replace=False)
            X_processed = self.quantizer.quantize(
                X_processed, 
                cols_to_quantize.tolist()
            )
            applied['quantization'] = True
            for col in cols_to_quantize:
                column_info[col]['quantized'] = True
        else:
            applied['quantization'] = False
        
        # Step 3: Missing values
        if self.config.apply_missing:
            X_processed, missing_mask = self.missing_injector.inject(X_processed)
            applied['missing_values'] = True
            for col in range(X.shape[1]):
                column_info[col]['missing_rate'] = missing_mask[:, col].mean()
        else:
            applied['missing_values'] = False
        
        # Step 4: Optional normalization (don't apply to target)
        # This is left configurable - by default we don't normalize
        # as the paper doesn't explicitly require it
        applied['normalization'] = False
        
        return PostProcessingResult(
            X=X_processed,
            y=y_processed,
            applied_transforms=applied,
            column_info=column_info
        )
    
    def process_minimal(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply minimal processing (just returns copies).
        
        Useful for debugging or when you want to see raw output.
        """
        return X.copy(), y.copy()

