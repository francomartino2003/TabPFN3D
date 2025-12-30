"""
Main Model: Temporal Encoder + Frozen TabPFN.

This module combines:
1. Preprocessor3D - for data normalization and missing value handling
2. TemporalEncoder - to transform (n, m, t) -> (n, m*K, d)
3. FrozenTabPFN - for in-context learning

The training only updates the TemporalEncoder; TabPFN is frozen.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
import numpy as np

try:
    from .training_config import FullConfig, EncoderConfig
    from .encoder import TemporalEncoder
    from .preprocessing_3d import Preprocessor3D, numpy_to_torch
    from .tabpfn_wrapper import FrozenTabPFN
except ImportError:
    from training_config import FullConfig, EncoderConfig
    from encoder import TemporalEncoder
    from preprocessing_3d import Preprocessor3D, numpy_to_torch
    from tabpfn_wrapper import FrozenTabPFN


class TemporalTabPFN(nn.Module):
    """
    Complete model combining temporal encoder with frozen TabPFN.
    
    Architecture:
        Input: (n_samples, n_features, n_timesteps)
          |
          v
        [Preprocessor3D] - normalization, missing handling (not learnable)
          |
          v
        [TemporalEncoder] - (n, m, t) -> (n, m*K, d) (LEARNABLE)
          |
          v
        [FrozenTabPFN] - in-context learning (FROZEN)
          |
          v
        Output: (n_test, n_classes) - predictions for test samples
    
    Training:
        - Only TemporalEncoder parameters are updated
        - Loss is computed on test samples only (TabPFN style)
    """
    
    def __init__(self, config: FullConfig):
        super().__init__()
        self.config = config
        
        # Preprocessor (not a nn.Module, used in numpy)
        self.preprocessor = Preprocessor3D(config.preprocessing)
        
        # Temporal encoder (LEARNABLE)
        self.encoder = TemporalEncoder(config.encoder)
        
        # Frozen TabPFN (FROZEN)
        self.tabpfn = FrozenTabPFN(
            model_path=config.tabpfn_model_path,
            device=config.device
        )
        
        # Verify dimensions match
        assert config.encoder.d_model == self.tabpfn.emsize, (
            f"Encoder d_model ({config.encoder.d_model}) must match "
            f"TabPFN emsize ({self.tabpfn.emsize})"
        )
        
        self.device = config.device
        
        # Move encoder to device
        self.encoder = self.encoder.to(config.device)
    
    def forward(
        self,
        X: torch.Tensor,
        y_train: torch.Tensor,
        n_train: int,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete model.
        
        Flow:
            X (n_samples, n_features, n_timesteps)
                ↓ Temporal Encoder
            embeddings (n_samples, n_features * n_queries, d_model=128)
                ↓ Inject into TabPFN (replaces TabPFN's X encoder)
            TabPFN: y_encoder → add_embeddings → transformer → decoder
                ↓
            logits (n_test, n_classes)
        
        Args:
            X: (n_samples, n_features, n_timesteps) - full dataset
            y_train: (n_train,) - training labels
            n_train: number of training samples
            return_embeddings: if True, also return intermediate embeddings
        
        Returns:
            dict with:
                - "logits": (n_test, n_classes) - raw predictions
                - "probs": (n_test, n_classes) - softmax probabilities
                - "embeddings": (n_samples, n_encoded_features, d) if return_embeddings
        """
        n_samples = X.shape[0]
        n_test = n_samples - n_train
        
        # 1. Encode with temporal encoder
        # (n_samples, n_features, n_timesteps) -> (n_samples, n_features * n_queries, d_model)
        embeddings = self.encoder(X)
        # embeddings shape: (n_samples, n_encoded_features, 128)
        
        # 2. Forward through frozen TabPFN using embedded X
        # This injects our embeddings directly where TabPFN expects encoder output
        logits = self.tabpfn.forward_with_embedded_x(embeddings, y_train, n_train)
        # logits: (n_test, n_classes)
        
        # 3. Compute probabilities
        probs = F.softmax(logits, dim=-1)
        
        result = {
            "logits": logits,
            "probs": probs,
        }
        
        if return_embeddings:
            result["embeddings"] = embeddings
        
        return result
    
    def compute_loss(
        self,
        X: torch.Tensor,
        y_train: torch.Tensor,
        y_test: torch.Tensor,
        n_train: int
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss on test samples.
        
        This follows TabPFN's training paradigm where:
        - Training samples have visible labels
        - Test samples have hidden labels
        - Loss is computed only on test predictions
        
        Args:
            X: (n_samples, n_features, n_timesteps)
            y_train: (n_train,) - visible labels
            y_test: (n_test,) - hidden labels for loss
            n_train: number of training samples
        
        Returns:
            dict with "loss", "accuracy", "logits", "probs"
        """
        # Forward pass
        output = self.forward(X, y_train, n_train)
        
        # Cross-entropy loss on test samples
        loss = F.cross_entropy(output["logits"], y_test.long())
        
        # Accuracy
        predictions = output["probs"].argmax(dim=-1)
        accuracy = (predictions == y_test.long()).float().mean()
        
        return {
            "loss": loss,
            "accuracy": accuracy,
            "logits": output["logits"],
            "probs": output["probs"],
        }
    
    def predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on test data given training data.
        
        This is the inference interface for a fitted model.
        
        Args:
            X_train: (n_train, n_features, n_timesteps)
            y_train: (n_train,)
            X_test: (n_test, n_features, n_timesteps)
        
        Returns:
            (predictions, probabilities)
        """
        self.eval()
        
        with torch.no_grad():
            # Concatenate train and test
            X_full = np.concatenate([X_train, X_test], axis=0)
            n_train = len(y_train)
            
            # Preprocess
            X_full = self.preprocessor.fit_transform(X_full)
            
            # Convert to torch
            X_tensor = numpy_to_torch(X_full, self.device)
            y_tensor = torch.from_numpy(y_train).to(self.device)
            
            # Forward
            output = self.forward(X_tensor, y_tensor, n_train)
            
            # Get predictions
            probs = output["probs"].cpu().numpy()
            preds = probs.argmax(axis=-1)
        
        return preds, probs
    
    def get_trainable_parameters(self):
        """Get only the trainable parameters (encoder)."""
        return self.encoder.parameters()
    
    def get_num_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
    
    def get_num_frozen_params(self) -> int:
        """Count frozen parameters."""
        return sum(p.numel() for p in self.tabpfn.parameters())


def create_model(config: Optional[FullConfig] = None) -> TemporalTabPFN:
    """Create model with given or default config."""
    if config is None:
        config = FullConfig()
    return TemporalTabPFN(config)


class LossComputer:
    """
    Utility class for computing loss over a batch of datasets.
    
    Handles the complexity of:
    - Different dataset sizes within a batch
    - Preprocessing each dataset independently
    - Accumulating losses across datasets
    """
    
    def __init__(self, model: TemporalTabPFN, config: FullConfig):
        self.model = model
        self.config = config
    
    def compute_batch_loss(
        self,
        samples: List[Any]  # List[DatasetSample]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute average loss over a batch of datasets.
        
        Args:
            samples: List of DatasetSample objects
        
        Returns:
            dict with "loss", "accuracy" averaged over batch
        """
        total_loss = 0.0
        total_accuracy = 0.0
        n_datasets = len(samples)
        
        for sample in samples:
            # Preprocess this dataset
            X_full = sample.X_full
            preprocessor = Preprocessor3D(self.config.preprocessing)
            X_proc = preprocessor.fit_transform(X_full)
            
            # Convert to tensors
            X_tensor = numpy_to_torch(X_proc, self.config.device)
            y_train = torch.from_numpy(sample.y_train).to(self.config.device)
            y_test = torch.from_numpy(sample.y_test).to(self.config.device)
            
            # Compute loss for this dataset
            output = self.model.compute_loss(
                X_tensor,
                y_train,
                y_test,
                sample.n_train
            )
            
            total_loss += output["loss"]
            total_accuracy += output["accuracy"]
        
        return {
            "loss": total_loss / n_datasets,
            "accuracy": total_accuracy / n_datasets,
        }

