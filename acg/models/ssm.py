"""
State Space Model (SSM) block for linear-time sequence processing.

Implements a Mamba-style selective state space model with gating.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model with input-dependent dynamics.
    
    Implements the core SSM computation with selective gating mechanism.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2
    ):
        """
        Initialize selective SSM.
        
        Args:
            d_model: Model dimension
            d_state: State dimension
            d_conv: Convolution kernel size
            expand_factor: Expansion factor for inner dimension
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand_factor
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner
        )
        
        # SSM parameters (input-dependent)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        
        # State space matrices
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize SSM parameters."""
        # Initialize A_log with negative values for stability
        nn.init.uniform_(self.A_log, -math.log(self.d_state), -1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through selective SSM.
        
        Args:
            x: (batch, seq_len, d_model) - Input tensor
            
        Returns:
            output: (batch, seq_len, d_model) - SSM output
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection and split
        x_and_res = self.in_proj(x)  # (batch, seq_len, d_inner * 2)
        x_inner, res = x_and_res.chunk(2, dim=-1)  # Each (batch, seq_len, d_inner)
        
        # Apply convolution (need to transpose for Conv1d)
        x_conv = x_inner.transpose(1, 2)  # (batch, d_inner, seq_len)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # Trim padding
        x_conv = x_conv.transpose(1, 2)  # (batch, seq_len, d_inner)
        
        # Activation
        x_conv = F.silu(x_conv)
        
        # Compute input-dependent SSM parameters
        ssm_params = self.x_proj(x_conv)  # (batch, seq_len, d_state * 2 + 1)
        
        # Split into B, C, and delta
        delta, B, C = torch.split(
            ssm_params,
            [1, self.d_state, self.d_state],
            dim=-1
        )
        
        # delta: (batch, seq_len, 1)
        # B: (batch, seq_len, d_state)
        # C: (batch, seq_len, d_state)
        
        # Softplus for delta (ensures positive)
        delta = F.softplus(delta)
        
        # Compute A from A_log
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        
        # Selective scan (simplified version)
        y = self.selective_scan(x_conv, delta, A, B, C)
        
        # Skip connection with D
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)
        
        # Gating with residual
        y = y * F.silu(res)
        
        # Output projection
        output = self.out_proj(y)
        
        return output
    
    def selective_scan(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform selective scan operation.
        
        This is a simplified implementation. For production, use optimized
        CUDA kernels or the official Mamba implementation.
        
        Args:
            x: (batch, seq_len, d_inner) - Input
            delta: (batch, seq_len, 1) - Step size
            A: (d_inner, d_state) - State transition matrix
            B: (batch, seq_len, d_state) - Input matrix
            C: (batch, seq_len, d_state) - Output matrix
            
        Returns:
            y: (batch, seq_len, d_inner) - Scan output
        """
        batch_size, seq_len, d_inner = x.shape
        d_state = A.size(1)
        
        # Discretize A and B
        # A_bar = exp(delta * A)
        # B_bar = delta * B
        delta = delta.squeeze(-1)  # (batch, seq_len)
        
        # Expand A for batch and sequence
        A_expanded = A.unsqueeze(0).unsqueeze(0)  # (1, 1, d_inner, d_state)
        delta_A = delta.unsqueeze(-1).unsqueeze(-1) * A_expanded  # (batch, seq_len, d_inner, d_state)
        A_bar = torch.exp(delta_A)
        
        # B_bar
        delta_B = delta.unsqueeze(-1) * B  # (batch, seq_len, d_state)
        B_bar = delta_B.unsqueeze(2)  # (batch, seq_len, 1, d_state)
        
        # Initialize state
        h = torch.zeros(batch_size, d_inner, d_state, device=x.device, dtype=x.dtype)
        
        # Scan through sequence
        outputs = []
        for t in range(seq_len):
            # Update state: h = A_bar * h + B_bar * x
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)
            
            # Compute output: y = C * h
            y_t = torch.sum(C[:, t].unsqueeze(1) * h, dim=-1)  # (batch, d_inner)
            outputs.append(y_t)
        
        # Stack outputs
        y = torch.stack(outputs, dim=1)  # (batch, seq_len, d_inner)
        
        return y


class SSMBlock(nn.Module):
    """
    Complete SSM block with normalization and residual connection.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize SSM block.
        
        Args:
            d_model: Model dimension
            d_state: State dimension
            d_conv: Convolution kernel size
            expand_factor: Expansion factor
            dropout: Dropout rate
        """
        super().__init__()
        
        # Pre-normalization
        self.norm = nn.LayerNorm(d_model)
        
        # SSM layer
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand_factor)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SSM block.
        
        Args:
            x: (batch, seq_len, d_model) - Input tensor
            
        Returns:
            output: (batch, seq_len, d_model) - SSM block output
        """
        # Pre-norm with residual
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.dropout(x)
        x = residual + x
        
        return x
