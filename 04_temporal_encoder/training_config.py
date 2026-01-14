"""
Configuration for Temporal Encoder + TabPFN training.

All hyperparameters are defined here to ensure reproducibility.
Nothing is hardcoded in the training scripts.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from pathlib import Path
import json


@dataclass
class EncoderConfig:
    """Configuration for the temporal encoder."""
    
    # Architecture - must match TabPFN emsize (TabPFN 2.5: 192, TabPFN v2: 128)
    d_model: int = 192
    """Embedding dimension. Must match TabPFN's emsize (192 for TabPFN 2.5)."""
    
    n_queries: int = 16
    """Number of latent queries per time series feature."""
    
    n_layers: int = 2
    """Number of self-attention layers after cross-attention."""
    
    n_heads: int = 8
    """Number of attention heads."""
    
    dropout: float = 0.1
    """Dropout rate."""
    
    # Positional encoding
    pos_enc_type: str = "learned"
    """Type of positional encoding: 'sinusoidal' or 'learned'."""
    
    max_timesteps: int = 1000
    """Maximum number of timesteps for positional encoding."""
    
    # Input projection
    input_proj_type: str = "linear"
    """Type of input projection: 'linear' or 'mlp'."""
    
    input_proj_hidden: Optional[int] = None
    """Hidden dimension for MLP projection. If None, uses d_model."""
    
    def __post_init__(self):
        # TabPFN 2.5 uses emsize=192, TabPFN v2 uses emsize=128
        assert self.d_model in [128, 192], (
            f"d_model must be 128 (TabPFN v2) or 192 (TabPFN 2.5), got {self.d_model}"
        )
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Optimization
    lr: float = 2e-4
    """Learning rate."""
    
    weight_decay: float = 0.01
    """Weight decay for AdamW."""
    
    batch_datasets: int = 1
    """Number of complete datasets processed per forward pass.
    Keep at 1 for limited VRAM."""
    
    accumulation_steps: int = 64
    """Accumulate gradients over this many datasets before updating weights.
    Effective batch size = batch_datasets × accumulation_steps.
    With batch_datasets=1 and accumulation_steps=64, you get batch=64 effect."""
    
    n_steps: int = 10000
    """Total number of training steps."""
    
    eval_every: int = 100
    """Evaluate every N steps."""
    
    warmup_steps: int = 500
    """Number of warmup steps for learning rate scheduler."""
    
    grad_clip: float = 1.0
    """Gradient clipping max norm."""
    
    # Loss
    loss_type: str = "ce"
    """Loss type: 'ce' for cross-entropy."""
    
    # Evaluation
    val_synth_seed: int = 42
    """Fixed seed for synthetic validation set (reproducibility)."""
    
    val_synth_size: int = 50
    """Number of datasets in synthetic validation set."""
    
    val_real_size: int = 20
    """Number of real datasets in validation set."""
    
    eval_real_datasets: bool = False
    """If True, evaluate on real datasets during training. If False, skip real dataset evaluation."""
    
    # Checkpointing
    save_every: int = 500
    """Save checkpoint every N steps."""
    
    checkpoint_dir: str = "checkpoints"
    """Directory for saving checkpoints."""
    
    update_plot_every: int = 100
    """Update training plot every N steps."""
    
    # Logging
    log_dir: str = "logs"
    """Directory for logs."""
    
    wandb_project: Optional[str] = None
    """Wandb project name. If None, no wandb logging."""
    
    wandb_run_name: Optional[str] = None
    """Wandb run name."""
    
    # Random seed
    seed: int = 42
    """Random seed for reproducibility."""


@dataclass 
class DataConfig:
    """
    Configuration for data loading.
    
    These parameters are passed to PriorConfig3D when generating synthetic datasets.
    Note the mapping between DataConfig and PriorConfig3D parameter names:
    
    DataConfig              -> PriorConfig3D
    ─────────────────────────────────────────
    n_samples_range         -> n_samples_range
    n_features_range        -> n_features_range (for Beta distribution of multivariate)
    n_timesteps_range       -> t_subseq_range (the 't' in n×m×t output shape)
    max_classes             -> max_classes
    prob_univariate         -> prob_univariate
    train_ratio_range       -> train_ratio_range
    """
    
    # Synthetic generator settings - reduced for 8GB VRAM
    # TabPFN was designed for small datasets, so we limit sizes
    n_samples_range: Tuple[int, int] = (50, 500)
    """Range of samples per dataset (n in n×m×t). Max ~500 for 8GB VRAM."""
    
    n_features_range: Tuple[int, int] = (2, 8)
    """Range of features for multivariate series (m in n×m×t when not univariate).
    Maps to PriorConfig3D.n_features_range which controls Beta distribution."""
    
    n_timesteps_range: Tuple[int, int] = (20, 150)
    """Range of timesteps per sample (t in n×m×t).
    Maps to PriorConfig3D.t_subseq_range - the subsequence length."""
    
    max_classes: int = 10
    """Maximum number of classes for classification."""
    
    prob_univariate: float = 0.4
    """Probability of generating univariate time series (m=1)."""
    
    # Train/val split for each dataset
    train_ratio_range: Tuple[float, float] = (0.6, 0.8)
    """Range for train/test split ratio within each dataset."""
    
    # Real data
    real_data_path: str = "01_real_data/AEON/data"
    """Path to real datasets (relative to project root)."""
    
    # Number of workers for data loading
    num_workers: int = 0
    """Number of data loading workers (0 for main process)."""


@dataclass
class Preprocessing3DConfig:
    """Configuration for 3D preprocessing."""
    
    # Normalization
    normalize_type: str = "z_norm"
    """Normalization type: 'z_norm', 'quantile', 'minmax', 'none'."""
    
    normalize_per_feature: bool = True
    """If True, normalize each feature independently across time."""
    
    normalize_per_sample: bool = False
    """If True, normalize each sample independently."""
    
    # Missing values
    handle_missing: bool = True
    """Whether to handle missing values."""
    
    missing_imputation: str = "forward_fill"
    """Imputation method: 'forward_fill', 'backward_fill', 'mean', 'interpolate', 'zero'."""
    
    add_missing_flags: bool = True
    """If True, add binary features indicating missing values."""
    
    # Outliers
    clip_outliers: bool = True
    """If True, clip outliers beyond clip_std standard deviations."""
    
    clip_std: float = 5.0
    """Number of standard deviations for outlier clipping."""
    
    # Feature removal
    remove_constant_features: bool = True
    """If True, remove features with (near-)zero variance."""
    
    constant_threshold_std: float = 1e-6
    """Features with std < this threshold are considered constant."""
    
    constant_threshold_range: float = 1e-6
    """Features with (max-min) < this threshold are considered constant."""
    
    constant_threshold_unique_ratio: float = 0.01
    """Features where n_unique_values/n_total < this ratio are considered nearly constant.
    E.g., 0.01 means if a feature has fewer than 1% unique values, it's nearly constant."""


@dataclass
class FullConfig:
    """Complete configuration for the project."""
    
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: Preprocessing3DConfig = field(default_factory=Preprocessing3DConfig)
    
    # Device
    device: str = "cuda"
    """Device: 'cuda' or 'cpu'."""
    
    # TabPFN model path
    tabpfn_model_path: Optional[str] = None
    """Path to TabPFN model. If None, uses default from cache."""
    
    # Experiment name
    experiment_name: str = "temporal_encoder_v1"
    """Name for this experiment."""
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = {
            "encoder": self.encoder.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
            "preprocessing": self.preprocessing.__dict__,
            "device": self.device,
            "tabpfn_model_path": self.tabpfn_model_path,
            "experiment_name": self.experiment_name,
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'FullConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(
            encoder=EncoderConfig(**config_dict["encoder"]),
            training=TrainingConfig(**config_dict["training"]),
            data=DataConfig(**config_dict["data"]),
            preprocessing=Preprocessing3DConfig(**config_dict["preprocessing"]),
            device=config_dict["device"],
            tabpfn_model_path=config_dict["tabpfn_model_path"],
            experiment_name=config_dict["experiment_name"],
        )
    
    def __post_init__(self):
        """Validate configuration."""
        # Encoder d_model must match TabPFN emsize (192 for TabPFN 2.5, 128 for TabPFN v2)
        assert self.encoder.d_model in [128, 192], (
            f"encoder.d_model must be 128 (TabPFN v2) or 192 (TabPFN 2.5), got {self.encoder.d_model}"
        )


# Convenience function to create default config
def get_default_config() -> FullConfig:
    """Get default configuration."""
    config = FullConfig()
    # Optimized for GPU with 47GB+ memory (e.g., L40S)
    config.training.batch_datasets = 8
    config.training.accumulation_steps = 8
    # Effective batch size = 8 × 8 = 64 datasets
    return config


# Quick configs for different scenarios
def get_debug_config() -> FullConfig:
    """Get configuration for quick debugging."""
    config = FullConfig()
    config.training.n_steps = 100
    config.training.eval_every = 20
    config.training.batch_datasets = 2
    config.training.val_synth_size = 10
    config.data.n_samples_range = (50, 200)
    config.data.n_timesteps_range = (20, 100)
    return config


def get_small_config() -> FullConfig:
    """Get configuration for small-scale training."""
    config = FullConfig()
    config.training.n_steps = 1000
    config.training.eval_every = 100
    config.training.batch_datasets = 4
    return config

