"""
Data Loading for Temporal Encoder Training.

This module provides data loaders for:
1. Synthetic 3D datasets (from 03_synthetic_generator_3D)
2. Real time series classification datasets (from 01_real_data)

The key design follows TabPFN's training paradigm:
- Each batch contains multiple complete datasets
- Each dataset has its own train/test split
- Loss is computed only on test samples
"""
import sys
from pathlib import Path
from typing import Iterator, List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

# Setup paths
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
_SYNTH_3D_PATH = _PROJECT_ROOT / "03_synthetic_generator_3D"

# Import DataConfig from our local config module
try:
    from .training_config import DataConfig
except ImportError:
    # Direct execution - import from same directory
    import importlib.util
    spec = importlib.util.spec_from_file_location("local_config", _THIS_DIR / "training_config.py")
    local_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(local_config)
    DataConfig = local_config.DataConfig


@dataclass
class DatasetSample:
    """
    A single dataset sample for training/evaluation.
    
    Each sample represents a complete classification task with train/test split.
    """
    X_train: np.ndarray  # (n_train, n_features, n_timesteps)
    y_train: np.ndarray  # (n_train,)
    X_test: np.ndarray   # (n_test, n_features, n_timesteps)
    y_test: np.ndarray   # (n_test,)
    n_classes: int
    metadata: Dict[str, Any]
    
    @property
    def X_full(self) -> np.ndarray:
        """Concatenate train and test X."""
        return np.concatenate([self.X_train, self.X_test], axis=0)
    
    @property
    def y_full(self) -> np.ndarray:
        """Concatenate train and test y."""
        return np.concatenate([self.y_train, self.y_test], axis=0)
    
    @property
    def n_train(self) -> int:
        return len(self.y_train)
    
    @property
    def n_test(self) -> int:
        return len(self.y_test)
    
    @property
    def n_samples(self) -> int:
        return self.n_train + self.n_test
    
    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]
    
    @property
    def n_timesteps(self) -> int:
        return self.X_train.shape[2]


class SyntheticDataLoader:
    """
    Data loader for synthetic 3D time series datasets.
    
    Uses the generator from 03_synthetic_generator_3D to create
    datasets on-the-fly during training.
    """
    
    def __init__(
        self,
        config: DataConfig,
        seed: Optional[int] = None
    ):
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Import generator from 03_synthetic_generator_3D
        # Since we renamed our config.py to training_config.py, there's no conflict
        try:
            if str(_SYNTH_3D_PATH) not in sys.path:
                sys.path.insert(0, str(_SYNTH_3D_PATH))
            
            from generator import SyntheticDatasetGenerator3D
            from config import PriorConfig3D
            
            # Create prior config with training-appropriate settings
            prior = PriorConfig3D(
                n_samples_range=config.n_samples_range,
                n_features_range=config.n_features_range,
                n_timesteps_range=config.n_timesteps_range,
                max_classes=config.max_classes,
                prob_univariate=config.prob_univariate,
                train_ratio_range=config.train_ratio_range,
            )
            
            self.generator = SyntheticDatasetGenerator3D(prior=prior, seed=seed)
            self.prior = prior
            
        except Exception as e:
            raise ImportError(
                f"Failed to import synthetic generator: {e}. "
                f"Make sure {_SYNTH_3D_PATH} contains the generator."
            )
    
    def generate_one(self) -> DatasetSample:
        """Generate a single dataset sample."""
        dataset = self.generator.generate()
        
        # Split into train/test
        n_samples = dataset.n_samples
        train_ratio = self.rng.uniform(*self.config.train_ratio_range)
        n_train = max(2, int(n_samples * train_ratio))
        n_train = min(n_train, n_samples - 2)  # Ensure at least 2 test samples
        
        # Shuffle and split
        indices = self.rng.permutation(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        X_train = dataset.X[train_idx]
        y_train = dataset.y[train_idx]
        X_test = dataset.X[test_idx]
        y_test = dataset.y[test_idx]
        
        return DatasetSample(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_classes=dataset.n_classes,
            metadata={
                "source": "synthetic",
                "config": dataset.config.to_dict() if hasattr(dataset.config, 'to_dict') else {},
                "feature_names": dataset.feature_names,
            }
        )
    
    def generate_batch(self, batch_size: int) -> List[DatasetSample]:
        """Generate a batch of dataset samples."""
        return [self.generate_one() for _ in range(batch_size)]
    
    def __iter__(self) -> Iterator[DatasetSample]:
        """Infinite iterator over generated datasets."""
        while True:
            yield self.generate_one()
    
    def generate_fixed_validation_set(
        self, 
        n_datasets: int,
        seed: int
    ) -> List[DatasetSample]:
        """
        Generate a fixed validation set with a specific seed.
        
        This ensures reproducible validation across training runs.
        """
        # Save current state
        old_rng = self.rng
        old_generator_seed = self.generator.rng
        
        # Set fixed seed
        self.rng = np.random.default_rng(seed)
        self.generator.rng = np.random.default_rng(seed)
        
        # Generate
        val_set = [self.generate_one() for _ in range(n_datasets)]
        
        # Restore state
        self.rng = old_rng
        self.generator.rng = old_generator_seed
        
        return val_set


class RealDataLoader:
    """
    Data loader for real time series classification datasets.
    
    Loads datasets from the 01_real_data directory.
    """
    
    def __init__(
        self,
        data_path: str,
        seed: Optional[int] = None
    ):
        self.data_path = Path(data_path)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Load available datasets
        self.datasets = self._discover_datasets()
    
    def _discover_datasets(self) -> List[str]:
        """Discover available dataset files."""
        datasets = []
        
        # Look for pickle or numpy files
        if self.data_path.exists():
            for ext in ['*.pkl', '*.npz', '*.npy']:
                datasets.extend([p.stem for p in self.data_path.glob(ext)])
        
        return list(set(datasets))
    
    def load_dataset(self, name: str) -> Optional[DatasetSample]:
        """Load a specific dataset by name."""
        import pickle
        
        # Try different file formats
        pkl_path = self.data_path / f"{name}.pkl"
        npz_path = self.data_path / f"{name}.npz"
        
        try:
            if pkl_path.exists():
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                X = data.get('X', data.get('data'))
                y = data.get('y', data.get('labels', data.get('target')))
                
            elif npz_path.exists():
                data = np.load(npz_path)
                X = data['X']
                y = data['y']
            else:
                return None
            
            # Ensure 3D format
            if X.ndim == 2:
                # Assume (n_samples, n_timesteps) -> (n_samples, 1, n_timesteps)
                X = X[:, np.newaxis, :]
            
            # Split into train/test
            n_samples = len(y)
            n_train = int(n_samples * 0.7)
            
            indices = self.rng.permutation(n_samples)
            train_idx = indices[:n_train]
            test_idx = indices[n_train:]
            
            return DatasetSample(
                X_train=X[train_idx],
                y_train=y[train_idx],
                X_test=X[test_idx],
                y_test=y[test_idx],
                n_classes=len(np.unique(y)),
                metadata={"source": "real", "name": name}
            )
            
        except Exception as e:
            print(f"Warning: Failed to load dataset {name}: {e}")
            return None
    
    def load_all(self) -> List[DatasetSample]:
        """Load all available datasets."""
        samples = []
        for name in self.datasets:
            sample = self.load_dataset(name)
            if sample is not None:
                samples.append(sample)
        return samples
    
    def __len__(self) -> int:
        return len(self.datasets)


class MixedDataLoader:
    """
    Combined loader for both synthetic and real data.
    
    Useful for evaluation where we want to track performance
    on both synthetic and real datasets.
    """
    
    def __init__(
        self,
        config: DataConfig,
        seed: Optional[int] = None
    ):
        self.config = config
        self.seed = seed
        
        self.synthetic_loader = SyntheticDataLoader(config, seed)
        self.real_loader = RealDataLoader(config.real_data_path, seed)
    
    def get_synthetic_batch(self, batch_size: int) -> List[DatasetSample]:
        """Get batch of synthetic datasets."""
        return self.synthetic_loader.generate_batch(batch_size)
    
    def get_real_datasets(self) -> List[DatasetSample]:
        """Get all real datasets."""
        return self.real_loader.load_all()
    
    def get_fixed_validation_set(
        self,
        n_synthetic: int,
        seed: int
    ) -> Tuple[List[DatasetSample], List[DatasetSample]]:
        """
        Get fixed validation sets for both synthetic and real data.
        
        Returns:
            (synthetic_val, real_val)
        """
        synthetic_val = self.synthetic_loader.generate_fixed_validation_set(n_synthetic, seed)
        real_val = self.real_loader.load_all()
        
        return synthetic_val, real_val


def collate_datasets(
    samples: List[DatasetSample]
) -> Dict[str, Any]:
    """
    Collate multiple dataset samples into a batch.
    
    Note: Since datasets may have different sizes, we don't stack them.
    Instead, we return them as a list with metadata.
    """
    return {
        "samples": samples,
        "n_datasets": len(samples),
        "max_n_samples": max(s.n_samples for s in samples),
        "max_n_features": max(s.n_features for s in samples),
        "max_n_timesteps": max(s.n_timesteps for s in samples),
        "n_classes_list": [s.n_classes for s in samples],
    }

