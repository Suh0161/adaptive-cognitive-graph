"""
Routing data structures and utilities.
"""

import torch
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RoutingMap:
    """
    Data structure for routing information.
    
    Contains phase selection, expert selection, and gating information.
    """
    
    phase_id: torch.Tensor  # (batch,) - Selected phase indices
    expert_ids: torch.Tensor  # (batch, k) - Selected expert indices
    expert_gates: torch.Tensor  # (batch, k) - Gate values for experts
    phase_probs: torch.Tensor  # (batch, n_phases) - Phase probabilities
    expert_probs: torch.Tensor  # (batch, n_experts) - Expert probabilities
    
    def get_active_experts(self, batch_idx: int) -> List[int]:
        """
        Get list of active expert IDs for a specific batch item.
        
        Args:
            batch_idx: Index of batch item
            
        Returns:
            List of expert IDs
        """
        return self.expert_ids[batch_idx].tolist()
    
    def get_batch_size(self) -> int:
        """Get batch size."""
        return self.phase_id.size(0)
    
    def get_num_active_experts(self) -> int:
        """Get number of active experts per sample."""
        return self.expert_ids.size(1)
    
    def to(self, device: torch.device) -> "RoutingMap":
        """Move all tensors to specified device."""
        return RoutingMap(
            phase_id=self.phase_id.to(device),
            expert_ids=self.expert_ids.to(device),
            expert_gates=self.expert_gates.to(device),
            phase_probs=self.phase_probs.to(device),
            expert_probs=self.expert_probs.to(device),
        )
    
    def detach(self) -> "RoutingMap":
        """Detach all tensors from computation graph."""
        return RoutingMap(
            phase_id=self.phase_id.detach(),
            expert_ids=self.expert_ids.detach(),
            expert_gates=self.expert_gates.detach(),
            phase_probs=self.phase_probs.detach(),
            expert_probs=self.expert_probs.detach(),
        )
