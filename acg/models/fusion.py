"""
Output fusion layer for combining expert outputs into final predictions.

Implements weighted fusion with verifier weights, residual connections,
and final projection to vocabulary.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from .base import BaseModule
from ..config import ACGConfig


class OutputFusionLayer(BaseModule):
    """
    Fuses expert outputs into final predictions using learned weighted combination.
    
    Implements:
    - Per-expert projection layers 
    - Weighted fusion with verifier weights 
    - Residual connection from encoder 
    - Layer normalization and dropout
    - Final projection to vocabulary
    """
    
    def __init__(self, config: ACGConfig):
        """
        Initialize output fusion layer.
        
        Args:
            config: ACG configuration
        """
        super().__init__(config)
        
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size
        self.n_experts = config.n_experts
        self.dropout_rate = config.dropout
        self.use_residual = True  # Always use residual connection 
        
        # Per-expert projection layers 
        # Each expert output gets its own learned projection
        self.expert_projections = nn.ModuleList([
            nn.Linear(config.d_model, config.d_model, bias=False)
            for _ in range(config.n_experts)
        ])
        
        # Layer normalization 
        self.layer_norm = nn.LayerNorm(config.d_model)
        
        # Dropout 
        self.dropout = nn.Dropout(config.dropout)
        
        # Final projection to vocabulary 
        self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with appropriate scaling."""
        # Initialize expert projections
        for proj in self.expert_projections:
            nn.init.normal_(proj.weight, mean=0.0, std=0.02)
        
        # Initialize output projection
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.02)
    
    def _project_expert_outputs(
        self,
        expert_outputs: List[torch.Tensor],
        expert_ids: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Apply per-expert projections to expert outputs.
        
        Args:
            expert_outputs: List of (batch, seq_len, d_model) tensors
            expert_ids: (batch, k) - Expert IDs for each output
            
        Returns:
            List of projected expert outputs
        """
        projected_outputs = []
        
        for i, expert_out in enumerate(expert_outputs):
            # Get the expert ID for this output
            # Note: expert_ids has shape (batch, k), we need the i-th expert for each batch
            expert_id = expert_ids[0, i].item()  # Assuming same experts across batch
            
            # Apply the corresponding expert projection
            projected = self.expert_projections[expert_id](expert_out)
            projected_outputs.append(projected)
        
        return projected_outputs
    
    def _fuse_expert_outputs(
        self,
        projected_outputs: List[torch.Tensor],
        verifier_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse projected expert outputs using verifier weights.
        
        Implements weighted fusion:
        - Compute softmax-normalized weights
        - Calculate weighted sum of projected expert outputs
        
        Args:
            projected_outputs: List of (batch, seq_len, d_model) tensors
            verifier_weights: (batch, k) - Weights for each expert
            
        Returns:
            fused: (batch, seq_len, d_model) - Fused representation
        """
        batch_size = projected_outputs[0].size(0)
        seq_len = projected_outputs[0].size(1)
        k = len(projected_outputs)
        
        # Stack expert outputs: (k, batch, seq_len, d_model)
        stacked_outputs = torch.stack(projected_outputs, dim=0)
        
        # Normalize verifier weights with softmax 
        # verifier_weights: (batch, k)
        normalized_weights = F.softmax(verifier_weights, dim=1)
        
        # Reshape weights for broadcasting: (k, batch, 1, 1)
        weights = normalized_weights.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)
        
        # Compute weighted sum 
        # stacked_outputs: (k, batch, seq_len, d_model)
        # weights: (k, batch, 1, 1)
        # Result: (batch, seq_len, d_model)
        fused = (stacked_outputs * weights).sum(dim=0)
        
        return fused
    
    def forward(
        self,
        expert_outputs: List[torch.Tensor],
        verifier_weights: torch.Tensor,
        encoder_output: torch.Tensor,
        expert_ids: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Fuse expert outputs with learned weighting.
        
        Args:
            expert_outputs: List of (batch, seq_len, d_model) tensors
            verifier_weights: (batch, k) - Weights for each expert
            encoder_output: (batch, seq_len, d_model) - Residual connection
            expert_ids: (batch, k) - Expert IDs (optional, defaults to 0..k-1)
            
        Returns:
            logits: (batch, seq_len, vocab_size) - Final predictions
        """
        # Validate inputs
        self._validate_inputs(expert_outputs, verifier_weights, encoder_output)
        
        batch_size = encoder_output.size(0)
        seq_len = encoder_output.size(1)
        k = len(expert_outputs)
        
        # Default expert IDs if not provided
        if expert_ids is None:
            expert_ids = torch.arange(k, device=encoder_output.device).unsqueeze(0).expand(batch_size, k)
        
        # Step 1: Project each expert output 
        projected_outputs = self._project_expert_outputs(expert_outputs, expert_ids)
        
        # Step 2: Fuse with verifier weights 
        fused = self._fuse_expert_outputs(projected_outputs, verifier_weights)
        
        # Step 3: Add residual connection from encoder 
        if self.use_residual:
            fused = fused + encoder_output
        
        # Step 4: Apply layer normalization 
        fused = self.layer_norm(fused)
        
        # Step 5: Apply dropout 
        fused = self.dropout(fused)
        
        # Step 6: Final projection to vocabulary 
        logits = self.output_projection(fused)
        
        return logits
    
    def _validate_inputs(
        self,
        expert_outputs: List[torch.Tensor],
        verifier_weights: torch.Tensor,
        encoder_output: torch.Tensor
    ):
        """
        Validate input tensors.
        
        Args:
            expert_outputs: List of expert output tensors
            verifier_weights: Verifier weights
            encoder_output: Encoder output
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not expert_outputs:
            raise ValueError("expert_outputs cannot be empty")
        
        k = len(expert_outputs)
        batch_size = encoder_output.size(0)
        seq_len = encoder_output.size(1)
        
        # Check expert outputs
        for i, expert_out in enumerate(expert_outputs):
            if expert_out.dim() != 3:
                raise ValueError(f"Expert output {i} must be 3D, got {expert_out.dim()}D")
            
            if expert_out.size(0) != batch_size:
                raise ValueError(
                    f"Expert output {i} batch size {expert_out.size(0)} "
                    f"doesn't match encoder {batch_size}"
                )
            
            if expert_out.size(1) != seq_len:
                raise ValueError(
                    f"Expert output {i} seq_len {expert_out.size(1)} "
                    f"doesn't match encoder {seq_len}"
                )
            
            if expert_out.size(2) != self.d_model:
                raise ValueError(
                    f"Expert output {i} d_model {expert_out.size(2)} "
                    f"doesn't match config {self.d_model}"
                )
            
            # Check for NaN/inf
            if torch.isnan(expert_out).any():
                raise ValueError(f"Expert output {i} contains NaN")
            if torch.isinf(expert_out).any():
                raise ValueError(f"Expert output {i} contains inf")
        
        # Check verifier weights
        if verifier_weights.dim() != 2:
            raise ValueError(f"verifier_weights must be 2D, got {verifier_weights.dim()}D")
        
        if verifier_weights.size(0) != batch_size:
            raise ValueError(
                f"verifier_weights batch size {verifier_weights.size(0)} "
                f"doesn't match encoder {batch_size}"
            )
        
        if verifier_weights.size(1) != k:
            raise ValueError(
                f"verifier_weights size {verifier_weights.size(1)} "
                f"doesn't match number of experts {k}"
            )
        
        # Check encoder output
        if encoder_output.dim() != 3:
            raise ValueError(f"encoder_output must be 3D, got {encoder_output.dim()}D")
        
        if encoder_output.size(2) != self.d_model:
            raise ValueError(
                f"encoder_output d_model {encoder_output.size(2)} "
                f"doesn't match config {self.d_model}"
            )
