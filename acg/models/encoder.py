"""
Semantic encoder with hybrid Transformer + SSM architecture.
"""

import torch
import torch.nn as nn
from typing import Optional
from .base import BaseModule
from ..config import ACGConfig
from .embeddings import TokenEmbedding
from .attention import TransformerBlock
from .ssm import SSMBlock
from ..utils import validate_token_ids


class SemanticEncoder(BaseModule):
    """
    Hybrid Transformer + SSM encoder for long-context processing.

    Combines Transformer blocks for attention-based processing with SSM blocks
    for linear-time long-context modeling.
    """

    def __init__(self, config: ACGConfig):
        """
        Initialize semantic encoder.

        Args:
            config: ACG configuration
        """
        super().__init__(config)

        # Token embedding
        self.embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )

        # Build hybrid encoder layers
        self.layers = nn.ModuleList()
        d_ff = config.d_model * config.ffn_mult

        for layer_idx in range(config.n_layers):
            # Determine if this layer should be SSM or Transformer
            if (layer_idx + 1) % config.ssm_every == 0:
                # Insert SSM block
                layer = SSMBlock(
                    d_model=config.d_model,
                    d_state=16,
                    d_conv=4,
                    expand_factor=2,
                    dropout=config.dropout,
                )
            else:
                # Insert Transformer block
                layer = TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=d_ff,
                    max_seq_len=config.max_seq_len,
                    dropout=config.dropout,
                )

            self.layers.append(layer)

        # Final layer normalization
        self.final_norm = nn.LayerNorm(config.d_model)

        # Gradient checkpointing flag
        self.use_gradient_checkpointing = config.use_gradient_checkpointing

    def forward(
        self, token_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through encoder.

        Args:
            token_ids: (batch, seq_len) - Input token IDs
            attention_mask: (batch, seq_len, seq_len) - Optional attention mask

        Returns:
            hidden_states: (batch, seq_len, d_model) - Contextual representations
        """
        # Validate inputs
        validate_token_ids(token_ids, self.config.vocab_size, self.config.max_seq_len)

        # Embed tokens
        hidden_states = self.embedding(token_ids)

        # Process through hybrid layers
        for layer in self.layers:
            if self.use_gradient_checkpointing and self.training:
                # Use gradient checkpointing
                hidden_states = torch.utils.checkpoint.checkpoint(
                    self._forward_layer,
                    layer,
                    hidden_states,
                    attention_mask,
                    use_reentrant=False,
                )
            else:
                hidden_states = self._forward_layer(
                    layer, hidden_states, attention_mask
                )

        # Final normalization
        hidden_states = self.final_norm(hidden_states)

        return hidden_states

    def _forward_layer(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass through a single layer.

        Args:
            layer: Layer module (TransformerBlock or SSMBlock)
            hidden_states: Input hidden states
            attention_mask: Optional attention mask

        Returns:
            Output hidden states
        """
        if isinstance(layer, TransformerBlock):
            return layer(hidden_states, attention_mask)
        else:
            # SSM block doesn't use attention mask
            return layer(hidden_states)
