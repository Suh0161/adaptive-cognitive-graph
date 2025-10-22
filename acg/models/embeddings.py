"""
Token embedding and position encoding layers.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for position encoding.
    
    Applies rotary embeddings to query and key tensors in attention.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 256000, base: float = 10000.0):
        """
        Initialize RoPE.
        
        Args:
            dim: Dimension per attention head
            max_seq_len: Maximum sequence length
            base: Base for frequency computation
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute cos and sin for max sequence length
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: Optional[int] = None) -> tuple:
        """
        Apply rotary embeddings to query and key tensors.
        
        Args:
            q: Query tensor (batch, n_heads, seq_len, head_dim)
            k: Key tensor (batch, n_heads, seq_len, head_dim)
            seq_len: Sequence length (if None, use q.size(2))
            
        Returns:
            Tuple of rotated (q, k) tensors
        """
        if seq_len is None:
            seq_len = q.size(2)
        
        # Get cached cos and sin
        cos = self.cos_cached[:seq_len, :].to(q.dtype)
        sin = self.sin_cached[:seq_len, :].to(q.dtype)
        
        # Reshape for broadcasting: (seq_len, head_dim) -> (1, 1, seq_len, head_dim)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        # Apply rotation
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        
        return q_embed, k_embed


class TokenEmbedding(nn.Module):
    """
    Token embedding layer with optional position encoding.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int = 256000,
        dropout: float = 0.1
    ):
        """
        Initialize token embedding.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed token IDs.
        
        Args:
            token_ids: (batch, seq_len) - Input token IDs
            
        Returns:
            embeddings: (batch, seq_len, d_model) - Token embeddings
        """
        # Embed tokens
        embeddings = self.token_embedding(token_ids)
        
        # Scale embeddings
        embeddings = embeddings * math.sqrt(self.d_model)
        
        # Apply dropout
        embeddings = self.dropout(embeddings)
        
        return embeddings
