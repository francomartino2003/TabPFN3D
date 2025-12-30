"""
Temporal Encoder: transforms (n, m, t) -> (n, m×K, d)

This module implements a Perceiver-style encoder for time series that:
1. Projects each timestep to d_model dimensions
2. Adds positional encoding for temporal order
3. Uses cross-attention with K learnable queries to compress t timesteps
4. Optionally applies self-attention between the K latent queries

The output is compatible with TabPFN's expected input format.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

try:
    from .training_config import EncoderConfig
except ImportError:
    from training_config import EncoderConfig


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal sequences.
    
    Uses the classic sin/cos formulation from "Attention Is All You Need".
    """
    
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learnable positional encoding.
    
    Each position has a learned embedding vector.
    """
    
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        return self.dropout(x)


class TemporalCrossAttention(nn.Module):
    """
    Cross-attention: K learnable queries attend to T timesteps.
    
    This is the core component that compresses a variable-length time series
    into a fixed number of latent representations.
    
    Input: (batch, timesteps, d_model)
    Output: (batch, n_queries, d_model)
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_queries: int, 
        n_heads: int = 8, 
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.n_queries = n_queries
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Learnable queries - initialized with small random values
        self.queries = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.02)
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Cross-attention from learnable queries to input timesteps.
        
        Args:
            x: (batch, timesteps, d_model) - the time series embeddings
            mask: (batch, timesteps) - optional mask for padded positions
        Returns:
            (batch, n_queries, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Expand queries to batch size
        q = self.queries.expand(batch_size, -1, -1)  # (batch, n_queries, d_model)
        q = self.q_proj(q)
        
        # Keys and values from input sequence
        k = self.k_proj(x)  # (batch, timesteps, d_model)
        v = self.v_proj(x)  # (batch, timesteps, d_model)
        
        # Reshape for multi-head attention
        # (batch, seq, d_model) -> (batch, n_heads, seq, d_k)
        q = q.view(batch_size, self.n_queries, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores: (batch, n_heads, n_queries, timesteps)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided (for variable-length sequences)
        if mask is not None:
            # mask: (batch, timesteps) -> (batch, 1, 1, timesteps)
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # (batch, n_heads, n_queries, d_k)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, self.n_queries, self.d_model)
        out = self.out_proj(out)
        
        # Add residual from queries and normalize
        out = self.norm(out + self.queries.expand(batch_size, -1, -1))
        
        return out


class TransformerBlock(nn.Module):
    """
    Standard transformer block with self-attention and FFN.
    
    Used for self-attention between the latent queries.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class TemporalEncoder(nn.Module):
    """
    Encodes time series to fixed-size embeddings compatible with TabPFN.
    
    For each time series feature (n, t):
    1. Project each timestep to d_model dimensions
    2. Add positional encoding for temporal order
    3. Cross-attention with K queries compresses t -> K
    4. Optional self-attention between K latent tokens
    
    Input: (batch, n_features, n_timesteps)
    Output: (batch, n_features * n_queries, d_model)
    
    Example:
        With n_features=3, n_queries=16, d_model=128:
        Input: (32, 3, 100) -> Output: (32, 48, 128)
    """
    
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_queries = config.n_queries
        
        # Input projection: project each scalar timestep to d_model
        if config.input_proj_type == "linear":
            self.input_proj = nn.Linear(1, config.d_model)
        else:  # mlp
            hidden = config.input_proj_hidden or config.d_model
            self.input_proj = nn.Sequential(
                nn.Linear(1, hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden, config.d_model)
            )
        
        # Positional encoding
        if config.pos_enc_type == "sinusoidal":
            self.pos_enc = SinusoidalPositionalEncoding(
                config.d_model, 
                config.max_timesteps,
                config.dropout
            )
        else:  # learned
            self.pos_enc = LearnedPositionalEncoding(
                config.d_model, 
                config.max_timesteps,
                config.dropout
            )
        
        # Cross-attention: queries attend to timesteps
        self.cross_attn = TemporalCrossAttention(
            config.d_model, 
            config.n_queries, 
            config.n_heads, 
            config.dropout
        )
        
        # Self-attention layers between latent queries
        self.self_attn_layers = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(config.d_model)
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode 3D time series to TabPFN-compatible embeddings.
        
        Args:
            x: (batch, n_features, n_timesteps) - the time series data
            mask: (batch, n_features, n_timesteps) - optional mask for missing values
        Returns:
            (batch, n_features * n_queries, d_model)
        """
        batch_size, n_features, n_timesteps = x.shape
        
        # Process each feature independently
        # Reshape: (batch, n_features, n_timesteps) -> (batch * n_features, n_timesteps, 1)
        x_flat = x.view(batch_size * n_features, n_timesteps, 1)
        
        # Handle mask if provided
        if mask is not None:
            mask_flat = mask.view(batch_size * n_features, n_timesteps)
        else:
            mask_flat = None
        
        # Project to d_model: (batch * n_features, n_timesteps, d_model)
        x_proj = self.input_proj(x_flat)
        
        # Add positional encoding
        x_proj = self.pos_enc(x_proj)
        
        # Cross-attention: (batch * n_features, n_timesteps, d_model) -> 
        #                  (batch * n_features, n_queries, d_model)
        x_queries = self.cross_attn(x_proj, mask_flat)
        
        # Self-attention between queries
        for layer in self.self_attn_layers:
            x_queries = layer(x_queries)
        
        # Final normalization
        x_queries = self.final_norm(x_queries)
        
        # Reshape: (batch * n_features, n_queries, d_model) -> 
        #          (batch, n_features * n_queries, d_model)
        output = x_queries.view(batch_size, n_features * self.n_queries, self.d_model)
        
        return output
    
    def get_num_output_features(self, n_input_features: int) -> int:
        """Calculate number of output features for given input features."""
        return n_input_features * self.n_queries


# Convenience function
def create_temporal_encoder(config: Optional[EncoderConfig] = None) -> TemporalEncoder:
    """Create a temporal encoder with given or default config."""
    if config is None:
        config = EncoderConfig()
    return TemporalEncoder(config)

