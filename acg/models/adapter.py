"""
Memory adapter for continual learning using LoRA.

Implements low-rank adaptation with EMA-based weight updates for per-expert
continual learning without catastrophic forgetting.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from .base import BaseModule
from ..config import ACGConfig


class MemoryAdapter(BaseModule):
    """
    LoRA-based continual learning adapter.
    
    Implements low-rank adaptation (LoRA) with exponential moving average (EMA)
    weight updates for efficient continual learning. The adapter maintains
    approximately 1% of base model parameters.

    """
    
    def __init__(self, config: ACGConfig, target_dim: int):
        """
        Initialize LoRA adapter.
        
        Args:
            config: ACG configuration
            target_dim: Dimension of target module (typically d_model)
        """
        super().__init__(config)
        self.target_dim = target_dim
        self.rank = config.lora_rank
        self.alpha = config.lora_alpha
        self.ema_decay = config.ema_decay
        self.scaling = self.alpha / self.rank
        
        # Low-rank matrices 
        # A: down-projection (d_model -> rank)
        # B: up-projection (rank -> d_model)
        self.lora_A = nn.Parameter(torch.empty(target_dim, self.rank))
        self.lora_B = nn.Parameter(torch.empty(self.rank, target_dim))
        
        # Weight delta for EMA tracking (Req 4.4)
        self.register_buffer('weight_delta', torch.zeros(target_dim, target_dim))
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """
        Initialize LoRA parameters.
        
        A is initialized with Gaussian distribution.
        B is initialized with zeros so initial adaptation is zero.
        

        """
        # Initialize A with Gaussian (Kaiming uniform)
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        
        # Initialize B with zeros
        nn.init.zeros_(self.lora_B)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply adapter to hidden states.
        
        Computes: output = hidden_states @ (I + scaling * B @ A)
        
        Args:
            hidden_states: (batch, seq_len, target_dim) - Input hidden states
            
        Returns:
            adapted_states: (batch, seq_len, target_dim) - Adapted hidden states
        
 
        """
        # Validate input
        self.validate_input(hidden_states, expected_shape=(None, None, self.target_dim))
        
        # Compute low-rank adaptation: ΔW = B @ A
        # Apply scaling: alpha / rank 
        # hidden_states @ A -> (batch, seq_len, rank)
        # result @ B -> (batch, seq_len, target_dim)
        adapted = hidden_states @ self.lora_A @ self.lora_B * self.scaling
        
        # Add to original hidden states
        output = hidden_states + adapted
        
        return output
    
    def compute_effective_weights(self) -> torch.Tensor:
        """
        Compute effective weight matrix as base plus delta.
        
        Returns:
            effective_weights: (target_dim, target_dim) - Effective weight matrix
        
        Requirement 4.5
        """
        # Compute LoRA contribution: scaling * B @ A
        lora_weights = self.scaling * (self.lora_B.T @ self.lora_A.T)
        
        # Effective weights = I + weight_delta + lora_weights
        # For simplicity, we return the adaptation component
        effective_weights = self.weight_delta + lora_weights
        
        return effective_weights
    
    def update_weights(self, gradients: Optional[Dict[str, torch.Tensor]] = None) -> None:
        """
        Update adapter weights using EMA.
        
        Implements: delta_W = alpha * grad_W + (1 - alpha) * delta_W_previous
        
        Args:
            gradients: Dictionary of gradient tensors (optional, uses current grads if None)
        

        """
        # Compute gradient of LoRA weights
        if gradients is not None:
            # Use provided gradients
            grad_A = gradients.get('lora_A', torch.zeros_like(self.lora_A))
            grad_B = gradients.get('lora_B', torch.zeros_like(self.lora_B))
        else:
            # Use current gradients
            grad_A = self.lora_A.grad if self.lora_A.grad is not None else torch.zeros_like(self.lora_A)
            grad_B = self.lora_B.grad if self.lora_B.grad is not None else torch.zeros_like(self.lora_B)
        
        # Compute weight gradient: grad_W = scaling * B @ A_grad + scaling * B_grad @ A
        grad_W = self.scaling * (grad_B.T @ self.lora_A.T + self.lora_B.T @ grad_A.T)
        
        # EMA update : delta_W = alpha * grad_W + (1 - alpha) * delta_W
        # Note: Using ema_decay as (1 - alpha) for smoothing
        self.weight_delta = self.ema_decay * self.weight_delta + (1 - self.ema_decay) * grad_W
    
    def get_adaptation_magnitude(self) -> float:
        """
        Get L2 norm of current adaptation.
        
        Returns:
            Magnitude of weight delta
        """
        return torch.norm(self.weight_delta).item()
    
    def reset_adaptation(self):
        """Reset weight delta to zero."""
        self.weight_delta.zero_()
    
    def get_lora_params(self) -> int:
        """
        Get number of LoRA parameters.
        
        Returns:
            Number of parameters in LoRA matrices
        """
        return self.lora_A.numel() + self.lora_B.numel()


class AdaptedLinear(nn.Module):
    """
    Linear layer with LoRA adapter attached.
    
    Wraps a base linear layer and applies LoRA adaptation.
    Used to attach adapters to target modules in experts.
    
    Requirement 4.1
    """
    
    def __init__(self, base_linear: nn.Linear, config: ACGConfig):
        """
        Initialize adapted linear layer.
        
        Args:
            base_linear: Base linear layer to adapt
            config: ACG configuration
        """
        super().__init__()
        self.base_linear = base_linear
        self.adapter = MemoryAdapter(config, base_linear.out_features)
        
        # Freeze base weights
        for param in self.base_linear.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with adaptation.
        
        Args:
            x: (batch, seq_len, in_features) - Input tensor
            
        Returns:
            output: (batch, seq_len, out_features) - Adapted output
        """
        # Apply base linear transformation
        base_output = self.base_linear(x)
        
        # Apply adapter
        adapted_output = self.adapter(base_output)
        
        return adapted_output
    
    def update_adapter_weights(self):
        """Update adapter weights using EMA."""
        self.adapter.update_weights()


def attach_adapters_to_expert(expert_block, config: ACGConfig) -> Dict[str, MemoryAdapter]:
    """
    Attach LoRA adapters to target modules in an expert block.
    
    Attaches adapters to:
    - Query projections (q_proj) in attention layers
    - Value projections (v_proj) in attention layers  
    - FFN projections (w1, w2 in SwiGLU) in feed-forward layers
    
    Args:
        expert_block: ExpertBlock instance to attach adapters to
        config: ACG configuration
        
    Returns:
        adapters: Dictionary mapping module names to MemoryAdapter instances
    
    """
    adapters = {}
    
    # Iterate through transformer layers in expert
    for layer_idx, layer in enumerate(expert_block.layers):
        # Attach adapters to attention projections
        if hasattr(layer, 'attention'):
            attention = layer.attention
            
            # Query projection adapter
            if hasattr(attention, 'q_proj'):
                adapter_name = f'layer_{layer_idx}_q_proj'
                adapter = MemoryAdapter(config, attention.q_proj.out_features)
                adapters[adapter_name] = adapter
            
            # Value projection adapter
            if hasattr(attention, 'v_proj'):
                adapter_name = f'layer_{layer_idx}_v_proj'
                adapter = MemoryAdapter(config, attention.v_proj.out_features)
                adapters[adapter_name] = adapter
        
        # Attach adapters to FFN projections
        if hasattr(layer, 'ffn'):
            ffn = layer.ffn
            
            # SwiGLU w1 projection adapter
            if hasattr(ffn, 'swiglu') and hasattr(ffn.swiglu, 'w1'):
                adapter_name = f'layer_{layer_idx}_ffn_w1'
                adapter = MemoryAdapter(config, ffn.swiglu.w1.out_features)
                adapters[adapter_name] = adapter
            
            # SwiGLU w2 projection adapter
            if hasattr(ffn, 'swiglu') and hasattr(ffn.swiglu, 'w2'):
                adapter_name = f'layer_{layer_idx}_ffn_w2'
                adapter = MemoryAdapter(config, ffn.swiglu.w2.out_features)
                adapters[adapter_name] = adapter
    
    return adapters


def apply_adapters_to_expert_output(
    expert_output: torch.Tensor,
    adapters: Dict[str, MemoryAdapter]
) -> torch.Tensor:
    """
    Apply all adapters to expert output.
    
    This is a simplified approach that applies adaptation to the final output.
    For full integration, adapters should be applied within each layer.
    
    Args:
        expert_output: (batch, seq_len, d_model) - Expert output
        adapters: Dictionary of adapters
        
    Returns:
        adapted_output: (batch, seq_len, d_model) - Adapted output
    """
    # For now, we apply a single adapter to the output
    # In a full implementation, adapters would be integrated into each layer
    if adapters:
        # Use the first adapter as a representative
        adapter = next(iter(adapters.values()))
        return adapter(expert_output)
    return expert_output
