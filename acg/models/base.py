"""
Base module interface for all ACG components.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from ..config import ACGConfig


class BaseModule(nn.Module, ABC):
    """
    Abstract base class for all ACG modules.

    Provides common functionality and interface for all components.
    """

    def __init__(self, config: ACGConfig):
        """
        Initialize base module.

        Args:
            config: ACG configuration object
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """
        Forward pass - must be implemented by subclasses.
        """
        pass

    def get_num_params(self, trainable_only: bool = True) -> int:
        """
        Get number of parameters in module.

        Args:
            trainable_only: If True, count only trainable parameters

        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def reset_parameters(self):
        """
        Reset module parameters to initial values.

        Can be overridden by subclasses for custom initialization.
        """
        for module in self.modules():
            if hasattr(module, "reset_parameters") and module != self:
                module.reset_parameters()

    def validate_input(self, x: torch.Tensor, expected_shape: Optional[tuple] = None):
        """
        Validate input tensor.

        Args:
            x: Input tensor
            expected_shape: Expected shape (use None for variable dimensions)

        Raises:
            ValueError: If input is invalid
        """
        if torch.isnan(x).any():
            raise ValueError("Input contains NaN values")

        if torch.isinf(x).any():
            raise ValueError("Input contains infinite values")

        if expected_shape is not None:
            if len(x.shape) != len(expected_shape):
                raise ValueError(
                    f"Expected {len(expected_shape)}D tensor, got {len(x.shape)}D"
                )

            for i, (actual, expected) in enumerate(zip(x.shape, expected_shape)):
                if expected is not None and actual != expected:
                    raise ValueError(
                        f"Dimension {i}: expected {expected}, got {actual}"
                    )

    def get_device(self) -> torch.device:
        """Get device of module parameters."""
        return next(self.parameters()).device

    def get_dtype(self) -> torch.dtype:
        """Get dtype of module parameters."""
        return next(self.parameters()).dtype
