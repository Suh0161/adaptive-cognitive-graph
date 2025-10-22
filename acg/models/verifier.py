"""
Verifier expert for quality assessment and correction.

Implements:
- Confidence scoring via MLP
- Threshold-based correction decision
- Correction block for refinement
- BCE loss for verifier training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .base import BaseModule
from .attention import TransformerBlock
from ..config import ACGConfig


class VerifierMLP(nn.Module):
    """
    Multi-layer perceptron for confidence scoring.
    
    Takes mean-pooled sequence and outputs confidence score in [0, 1].
    """
    
    def __init__(self, d_model: int, hidden_dim: int, n_layers: int):
        """
        Initialize verifier MLP.
        
        Args:
            d_model: Input dimension
            hidden_dim: Hidden layer dimension
            n_layers: Number of hidden layers (2-3)
        """
        super().__init__()
        
        assert 2 <= n_layers <= 3, "n_layers must be between 2 and 3"
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(d_model, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))
        
        # Hidden layers
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        # Output layer (to scalar)
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, pooled_states: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence score.
        
        Args:
            pooled_states: (batch, d_model) - Mean-pooled sequence
            
        Returns:
            confidence: (batch,) - Confidence scores in [0, 1]
        """
        logits = self.mlp(pooled_states)  # (batch, 1)
        confidence = torch.sigmoid(logits.squeeze(-1))  # (batch,)
        return confidence


class CorrectionBlock(nn.Module):
    """
    Correction block for refining low-confidence outputs.
    
    Can be either a Transformer block or MLP.
    """
    
    def __init__(
        self,
        config: ACGConfig,
        correction_type: str = "transformer"
    ):
        """
        Initialize correction block.
        
        Args:
            config: ACG configuration
            correction_type: "transformer" or "mlp"
        """
        super().__init__()
        
        self.correction_type = correction_type
        self.d_model = config.d_model
        
        if correction_type == "transformer":
            # Single transformer block for correction
            self.correction = TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.get_ffn_dim(),
                max_seq_len=config.max_seq_len,
                dropout=config.dropout
            )
        elif correction_type == "mlp":
            # MLP with residual connection
            hidden_dim = config.verifier_hidden
            self.correction = nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden_dim, config.d_model),
                nn.Dropout(config.dropout)
            )
        else:
            raise ValueError(f"Unknown correction_type: {correction_type}")
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply correction to hidden states.
        
        Args:
            hidden_states: (batch, seq_len, d_model)
            
        Returns:
            corrected_states: (batch, seq_len, d_model)
        """
        if self.correction_type == "transformer":
            # Transformer block has built-in residual
            return self.correction(hidden_states)
        else:
            # MLP with explicit residual
            return hidden_states + self.correction(hidden_states)


class VerifierExpert(BaseModule):
    """
    Quality assessment and correction module.
    
    Assesses semantic coherence and logical validity of merged expert outputs.
    Provides confidence scores and optional correction for low-confidence outputs.
    
    """
    
    def __init__(self, config: ACGConfig):
        """
        Initialize verifier expert.
        
        Args:
            config: ACG configuration
        """
        super().__init__(config)
        
        self.d_model = config.d_model
        self.threshold = config.verifier_threshold
        self.loss_weight = config.verify_loss_weight
        
        # Verifier MLP for confidence scoring 
        self.verifier_mlp = VerifierMLP(
            d_model=config.d_model,
            hidden_dim=config.verifier_hidden,
            n_layers=config.verifier_layers
        )
        
        # Correction block for refinement 
        self.correction_block = CorrectionBlock(
            config=config,
            correction_type=config.correction_type
        )
    
    def forward(
        self,
        merged_states: torch.Tensor,
        apply_correction: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Verify and optionally correct merged expert outputs.
        
        Args:
            merged_states: (batch, seq_len, d_model) - Merged expert outputs
            apply_correction: If True, apply correction for low confidence
            
        Returns:
            output_states: (batch, seq_len, d_model) - Verified/corrected states
            confidence: (batch,) - Confidence scores in [0, 1]
        """
        # Validate input
        self.validate_input(merged_states, expected_shape=(None, None, self.d_model))
        
        batch_size, seq_len, _ = merged_states.shape
        
        # Mean pooling over sequence 
        pooled_states = merged_states.mean(dim=1)  # (batch, d_model)
        
        # Compute confidence score 
        confidence = self.verifier_mlp(pooled_states)  # (batch,)
        
        # Apply correction if confidence below threshold
        if apply_correction:
            # Create mask for low-confidence samples
            low_confidence_mask = confidence < self.threshold  # (batch,)
            
            if low_confidence_mask.any():
                # Apply correction to low-confidence samples
                corrected_states = self.correction_block(merged_states)
                
                # Mix original and corrected based on mask
                # Expand mask to match tensor dimensions
                mask_expanded = low_confidence_mask.view(batch_size, 1, 1)
                output_states = torch.where(
                    mask_expanded,
                    corrected_states,
                    merged_states
                )
            else:
                output_states = merged_states
        else:
            output_states = merged_states
        
        return output_states, confidence
    
    def compute_verification_loss(
        self,
        confidence: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute binary cross-entropy loss for verifier training.
        
        Args:
            confidence: (batch,) - Predicted confidence scores
            labels: (batch,) - Ground truth labels (1 for correct, 0 for corrupted)
            
        Returns:
            loss: Scalar verification loss
            
        """
        # BCE loss (Req 5.5)
        bce_loss = F.binary_cross_entropy(confidence, labels.float())
        
        # Apply loss weight (Req 5.6)
        weighted_loss = self.loss_weight * bce_loss
        
        return weighted_loss
    
    def get_correction_rate(self, confidence: torch.Tensor) -> float:
        """
        Get percentage of samples that would be corrected.
        
        Args:
            confidence: (batch,) - Confidence scores
            
        Returns:
            Correction rate as percentage
        """
        low_confidence = (confidence < self.threshold).float()
        return low_confidence.mean().item() * 100.0
    
    def set_threshold(self, threshold: float):
        """
        Update confidence threshold.
        
        Args:
            threshold: New threshold value (must be in [0.3, 0.7])
        """
        if not 0.3 <= threshold <= 0.7:
            raise ValueError(f"Threshold must be in [0.3, 0.7], got {threshold}")
        self.threshold = threshold
