"""
Intent router for phase and expert selection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Literal
from .base import BaseModule
from ..config import ACGConfig
from ..utils import RoutingMap
import math


class SequencePooling(nn.Module):
    """
    Pooling layer for sequence aggregation.

    Supports mean pooling and attention-based pooling.
    """

    def __init__(
        self, d_model: int, pooling_type: Literal["mean", "attention"] = "mean"
    ):
        """
        Initialize pooling layer.

        Args:
            d_model: Model dimension
            pooling_type: Type of pooling ("mean" or "attention")
        """
        super().__init__()
        self.pooling_type = pooling_type

        if pooling_type == "attention":
            # Learnable attention weights
            self.attention_weights = nn.Linear(d_model, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Pool sequence into single vector.

        Args:
            hidden_states: (batch, seq_len, d_model) - Input sequences

        Returns:
            pooled: (batch, d_model) - Pooled representation
        """
        if self.pooling_type == "mean":
            # Simple mean pooling
            pooled = hidden_states.mean(dim=1)
        else:
            # Attention-based pooling
            # Compute attention scores
            attn_scores = self.attention_weights(hidden_states)  # (batch, seq_len, 1)
            attn_weights = F.softmax(attn_scores, dim=1)  # (batch, seq_len, 1)

            # Weighted sum
            pooled = (hidden_states * attn_weights).sum(dim=1)  # (batch, d_model)

        return pooled


class IntentRouter(BaseModule):
    """
    Router for phase and expert selection.

    Determines cognitive phase and selects top-k experts based on input semantics.
    """

    def __init__(self, config: ACGConfig):
        """
        Initialize intent router.

        Args:
            config: ACG configuration
        """
        super().__init__(config)

        # Pooling layer
        self.pooling = SequencePooling(config.d_model, config.pooling)

        # Phase classifier
        self.phase_classifier = nn.Linear(config.d_model, config.n_phases, bias=False)

        # Expert scorer
        self.expert_scorer = nn.Linear(config.d_model, config.n_experts, bias=False)

        # Phase gate vectors (learned modulation for each phase)
        self.phase_gates = nn.Parameter(torch.ones(config.n_phases, config.n_experts))

        # Initialize weights
        nn.init.normal_(self.phase_classifier.weight, std=0.02)
        nn.init.normal_(self.expert_scorer.weight, std=0.02)
        nn.init.ones_(self.phase_gates)

    def forward(self, hidden_states: torch.Tensor) -> RoutingMap:
        """
        Route inputs to appropriate experts.

        Args:
            hidden_states: (batch, seq_len, d_model) - Encoder outputs

        Returns:
            routing_map: RoutingMap with phase_id, expert_ids, expert_gates, etc.
        """
        batch_size = hidden_states.size(0)

        # Pool sequence
        pooled = self.pooling(hidden_states)  # (batch, d_model)

        # Compute phase logits and probabilities
        phase_logits = self.phase_classifier(pooled)  # (batch, n_phases)
        phase_probs = F.softmax(phase_logits, dim=-1)  # (batch, n_phases)

        # Select phase (argmax for inference, soft for training)
        if self.training:
            # Gumbel-softmax for differentiable sampling
            phase_id = F.gumbel_softmax(phase_logits, tau=1.0, hard=True).argmax(dim=-1)
        else:
            phase_id = phase_probs.argmax(dim=-1)  # (batch,)

        # Compute expert logits
        expert_logits = self.expert_scorer(pooled)  # (batch, n_experts)
        expert_probs = F.softmax(expert_logits, dim=-1)  # (batch, n_experts)

        # Apply phase gating
        # Get phase gates for selected phases
        phase_gate = self.phase_gates[phase_id]  # (batch, n_experts)
        gated_expert_probs = expert_probs * phase_gate  # (batch, n_experts)

        # Renormalize
        gated_expert_probs = gated_expert_probs / (
            gated_expert_probs.sum(dim=-1, keepdim=True) + 1e-10
        )

        # Select top-k experts
        k = self.config.active_experts
        expert_gates, expert_ids = torch.topk(gated_expert_probs, k, dim=-1)

        # Normalize gates to sum to 1
        expert_gates = expert_gates / (expert_gates.sum(dim=-1, keepdim=True) + 1e-10)

        # Create routing map
        routing_map = RoutingMap(
            phase_id=phase_id,
            expert_ids=expert_ids,
            expert_gates=expert_gates,
            phase_probs=phase_probs,
            expert_probs=expert_probs,
        )

        return routing_map

    def compute_balance_loss(self, expert_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy-based balance loss to encourage expert diversity.

        Args:
            expert_probs: (batch, n_experts) - Expert probabilities

        Returns:
            balance_loss: Scalar tensor
        """
        # Average probabilities across batch
        avg_probs = expert_probs.mean(dim=0)  # (n_experts,)

        # Compute entropy
        entropy = -(avg_probs * torch.log(avg_probs + 1e-10)).sum()

        # Maximum entropy (uniform distribution)
        max_entropy = math.log(self.config.n_experts)

        # Balance loss: 1 - (entropy / max_entropy)
        # Lower is better (higher entropy = better balance)
        balance_loss = 1.0 - (entropy / max_entropy)

        return balance_loss
