"""
Multi-head attention and feed-forward network modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .embeddings import RotaryPositionEmbedding


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with RoPE position encoding.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 256000,
        dropout: float = 0.1
    ):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Rotary position embedding
        self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through multi-head attention.
        
        Args:
            x: (batch, seq_len, d_model) - Input tensor
            attention_mask: (batch, seq_len, seq_len) - Optional attention mask
            
        Returns:
            output: (batch, seq_len, d_model) - Attention output
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to (batch, n_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary position embeddings
        q, k = self.rope(q, k, seq_len)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v)
        
        # Reshape back to (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output


class SwiGLU(nn.Module):
    """
    SwiGLU activation function: Swish-Gated Linear Unit.
    
    SwiGLU(x) = Swish(W1 * x) ⊙ (W2 * x)
    where Swish(x) = x * sigmoid(x)
    """
    
    def __init__(self, d_model: int, d_ff: int):
        """
        Initialize SwiGLU.
        
        Args:
            d_model: Input dimension
            d_ff: Feed-forward dimension
        """
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU activation.
        
        Args:
            x: (batch, seq_len, d_model) - Input tensor
            
        Returns:
            output: (batch, seq_len, d_ff) - Activated tensor
        """
        return F.silu(self.w1(x)) * self.w2(x)


class FeedForward(nn.Module):
    """
    Feed-forward network with SwiGLU activation.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        """
        Initialize feed-forward network.
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.swiglu = SwiGLU(d_model, d_ff)
        self.out_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feed-forward network.
        
        Args:
            x: (batch, seq_len, d_model) - Input tensor
            
        Returns:
            output: (batch, seq_len, d_model) - FFN output
        """
        x = self.swiglu(x)
        x = self.out_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with multi-head attention and SwiGLU FFN.
    
    Uses pre-normalization (LayerNorm before attention/FFN).
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int = 256000,
        dropout: float = 0.1
    ):
        """
        Initialize transformer block.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        
        # Pre-normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Attention and FFN
        self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: (batch, seq_len, d_model) - Input tensor
            attention_mask: Optional attention mask
            
        Returns:
            output: (batch, seq_len, d_model) - Block output
        """
        # Pre-norm attention with residual
        residual = x
        x = self.norm1(x)
        x = self.attention(x, attention_mask)
        x = self.dropout(x)
        x = residual + x
        
        # Pre-norm FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x
        
        return x
