"""
TabPFN Wrapper for Temporal Encoder Integration.

This module provides a wrapper around the frozen TabPFN model that:
1. Loads the pre-trained TabPFN model
2. Freezes all parameters
3. Provides a forward method that accepts pre-encoded embeddings

The key insight is that we bypass TabPFN's input encoder and inject
our temporal encoder's output directly into the transformer layers.
"""
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn

# Add TabPFN to path
TABPFN_PATH = Path(__file__).resolve().parent.parent / "00_TabPFN" / "src"
if str(TABPFN_PATH) not in sys.path:
    sys.path.insert(0, str(TABPFN_PATH))


class FrozenTabPFN(nn.Module):
    """
    Wrapper for frozen TabPFN model.
    
    This class:
    1. Loads a pre-trained TabPFN model
    2. Freezes all its parameters
    3. Provides forward methods that work with pre-encoded embeddings
    
    The model expects embeddings of shape (seq_len, batch_size, n_features, emsize)
    where emsize=128 for the default TabPFN classifier.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        cache_trainset_representation: bool = False
    ):
        super().__init__()
        self.device = device
        self.model_path = model_path
        
        # Load TabPFN model
        self.model, self.config, self.n_out = self._load_model(
            model_path, 
            cache_trainset_representation
        )
        
        # Move to device and freeze
        self.model = self.model.to(device)
        self._freeze_parameters()
        
        # Store important config values
        self.emsize = self.config.emsize  # 128
        self.nlayers = self.config.nlayers  # 12
        self.features_per_group = self.config.features_per_group  # 1
    
    def _load_model(
        self, 
        model_path: Optional[str],
        cache_trainset_representation: bool
    ) -> Tuple[nn.Module, Any, int]:
        """Load TabPFN model from checkpoint."""
        try:
            from tabpfn.model_loading import load_model, get_cache_dir
            
            if model_path is None:
                # Use default model from cache
                cache_dir = get_cache_dir()
                model_files = list(cache_dir.glob("*classifier*.ckpt"))
                if not model_files:
                    raise FileNotFoundError(
                        f"No TabPFN classifier model found in {cache_dir}. "
                        "Please run: python -c 'from tabpfn import TabPFNClassifier; TabPFNClassifier()'"
                    )
                model_path = str(model_files[0])
            
            model, criterion, config, inference_config = load_model(
                path=model_path,
                cache_trainset_representation=cache_trainset_representation
            )
            
            # Get n_out from decoder
            n_out = model.n_out if hasattr(model, 'n_out') else 10
            
            return model, config, n_out
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import TabPFN: {e}. "
                "Make sure TabPFN is installed and 00_TabPFN/src is in the path."
            )
    
    def _freeze_parameters(self):
        """Freeze all TabPFN parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
    
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        single_eval_pos: Optional[int] = None
    ) -> torch.Tensor:
        """
        Standard TabPFN forward pass.
        
        Note: We don't use @torch.no_grad() here because we need gradients 
        to flow THROUGH TabPFN to the encoder, even though TabPFN params
        are frozen (requires_grad=False).
        
        Args:
            x: (seq_len, batch_size, n_features) - raw features
            y: (n_train, batch_size) - training labels
            single_eval_pos: Position from which to evaluate (default: n_train)
        
        Returns:
            (n_test, batch_size, n_classes) - predictions
        """
        x_dict = {"main": x}
        y_dict = {"main": y}
        
        output = self.model(
            x_dict,
            y_dict,
            only_return_standard_out=True
        )
        
        return output
    
    def forward_with_embedded_x(
        self,
        embedded_x: torch.Tensor,
        y_train: torch.Tensor,
        n_train: int
    ) -> torch.Tensor:
        """
        Forward pass using pre-computed X embeddings.
        
        This REPLACES TabPFN's X encoder with our temporal encoder output.
        The flow is:
            embedded_x (from temporal encoder)
                ↓
            y_encoder (TabPFN's)
                ↓
            add_embeddings (positional info)
                ↓
            concatenate x and y
                ↓
            transformer_encoder
                ↓
            decoder
        
        Args:
            embedded_x: (n_samples, n_features, emsize) - from temporal encoder
                       This replaces TabPFN's encoder output
            y_train: (n_train,) - training labels (0 to n_classes-1)
            n_train: Number of training samples
        
        Returns:
            (n_test, n_classes) - predictions for test samples
        """
        import einops
        
        n_samples, n_features, emsize = embedded_x.shape
        n_test = n_samples - n_train
        batch_size = 1  # We process one dataset at a time
        
        # Reshape embedded_x to TabPFN format: (batch, seq, features, emsize)
        # From (n_samples, n_features, emsize) to (1, n_samples, n_features, emsize)
        embedded_x = embedded_x.unsqueeze(0)  # (1, n_samples, n_features, emsize)
        
        # Prepare y with NaN masking for test samples
        # TabPFN expects y with shape (seq_len, batch, 1)
        y_full = torch.full(
            (n_samples, batch_size, 1),
            float('nan'),
            device=embedded_x.device,
            dtype=torch.float32
        )
        y_full[:n_train, :, 0] = y_train.float().unsqueeze(1)
        
        # Transpose y to (batch, seq, 1) for y_encoder dict format
        y_dict = {"main": y_full.transpose(0, 1)}  # (batch, seq, 1)
        
        # Encode y using TabPFN's y_encoder
        # y_encoder expects dict with "main" key
        embedded_y = self.model.y_encoder(
            y_dict,
            single_eval_pos=n_train,
            cache_trainset_representation=False,
        )  # Returns (batch, seq, emsize) - already correct format!
        
        # Add positional embeddings if available
        if hasattr(self.model, 'add_embeddings'):
            embedded_x, embedded_y = self.model.add_embeddings(
                embedded_x,  # (batch, seq, features, emsize)
                embedded_y,  # (batch, seq, emsize)
                data_dags=None,
                num_features=n_features,
                seq_len=n_samples,
                cache_embeddings=False,
                use_cached_embeddings=False,
            )
        
        # Concatenate: (batch, seq, features, emsize) + (batch, seq, 1, emsize)
        embedded_input = torch.cat(
            (embedded_x, embedded_y.unsqueeze(2)), 
            dim=2
        )  # (batch, seq, features+1, emsize)
        
        # Pass through transformer encoder
        encoder_out = self.model.transformer_encoder(
            embedded_input,
            single_eval_pos=n_train,
            cache_trainset_representation=False,
            recompute_layer=False,
            save_peak_mem_factor=None,
        )  # (batch, seq, features+1, emsize)
        
        # Extract test predictions from the y position (last in feature dim)
        # Take only test samples and the y-embedding position
        test_encoder_out = encoder_out[:, n_train:, -1, :]  # (batch, n_test, emsize)
        test_encoder_out = test_encoder_out.transpose(0, 1)  # (n_test, batch, emsize)
        
        # Decode
        output = self.model.decoder_dict["standard"](test_encoder_out)
        # output: (n_test, batch, n_classes)
        
        # Remove batch dim
        return output.squeeze(1)  # (n_test, n_classes)
    
    def get_y_encoder(self) -> nn.Module:
        """Get reference to TabPFN's y encoder."""
        return self.model.y_encoder
    
    def get_transformer_encoder(self) -> nn.Module:
        """Get reference to TabPFN's transformer encoder."""
        return self.model.transformer_encoder
    
    def get_decoder(self) -> nn.Module:
        """Get reference to TabPFN's decoder."""
        return self.model.decoder_dict["standard"]


class TabPFNInterface(nn.Module):
    """
    Clean interface for using TabPFN with temporal encoder.
    
    This class handles the complexity of:
    1. Reshaping encoder output to TabPFN format
    2. Managing train/test splits
    3. Computing loss only on test samples
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda"
    ):
        super().__init__()
        self.frozen_tabpfn = FrozenTabPFN(model_path, device)
        self.device = device
        self.emsize = self.frozen_tabpfn.emsize
    
    def forward(
        self,
        embeddings: torch.Tensor,
        y_train: torch.Tensor,
        n_train: int
    ) -> torch.Tensor:
        """
        Forward pass with pre-computed embeddings.
        
        Args:
            embeddings: (n_samples, n_encoded_features, emsize) from temporal encoder
            y_train: (n_train,) training labels
            n_train: Number of training samples
        
        Returns:
            (n_test, n_classes) predictions for test samples
        """
        n_samples, n_features, emsize = embeddings.shape
        n_test = n_samples - n_train
        
        # Reshape for TabPFN: add batch dimension
        # (n_samples, n_features, emsize) -> (1, n_samples, n_features, emsize)
        embedded_x = embeddings.unsqueeze(0)
        
        # Prepare y
        y_full = torch.full((n_samples,), float('nan'), device=self.device)
        y_full[:n_train] = y_train.float()
        y_full = y_full.unsqueeze(1)  # (n_samples, 1)
        
        # Forward through frozen TabPFN
        output = self.frozen_tabpfn.forward_with_embeddings(
            embedded_x,
            y_full,
            single_eval_pos=n_train
        )
        
        # output: (n_test, 1, n_classes) -> (n_test, n_classes)
        return output.squeeze(1)


def load_frozen_tabpfn(
    model_path: Optional[str] = None,
    device: str = "cuda"
) -> FrozenTabPFN:
    """Convenience function to load frozen TabPFN."""
    return FrozenTabPFN(model_path, device)

