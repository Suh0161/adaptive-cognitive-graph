"""
Input validation utilities.
"""

import torch
from typing import Optional, Tuple


def check_nan_inf(tensor: torch.Tensor, name: str = "tensor") -> None:
    """
    Check tensor for NaN or infinite values.
    
    Args:
        tensor: Tensor to check
        name: Name of tensor for error messages
        
    Raises:
        ValueError: If NaN or inf values found
    """
    if torch.isnan(tensor).any():
        raise ValueError(f"{name} contains NaN values")
    
    if torch.isinf(tensor).any():
        raise ValueError(f"{name} contains infinite values")


def validate_tensor(
    tensor: torch.Tensor,
    expected_shape: Optional[Tuple[Optional[int], ...]] = None,
    expected_dtype: Optional[torch.dtype] = None,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    name: str = "tensor"
) -> None:
    """
    Comprehensive tensor validation.
    
    Args:
        tensor: Tensor to validate
        expected_shape: Expected shape (None for variable dimensions)
        expected_dtype: Expected data type
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name of tensor for error messages
        
    Raises:
        ValueError: If validation fails
    """
    # Check for NaN/inf
    check_nan_inf(tensor, name)
    
    # Check shape
    if expected_shape is not None:
        if len(tensor.shape) != len(expected_shape):
            raise ValueError(
                f"{name}: expected {len(expected_shape)}D tensor, "
                f"got {len(tensor.shape)}D"
            )
        
        for i, (actual, expected) in enumerate(zip(tensor.shape, expected_shape)):
            if expected is not None and actual != expected:
                raise ValueError(
                    f"{name} dimension {i}: expected {expected}, got {actual}"
                )
    
    # Check dtype
    if expected_dtype is not None and tensor.dtype != expected_dtype:
        raise ValueError(
            f"{name}: expected dtype {expected_dtype}, got {tensor.dtype}"
        )
    
    # Check value range
    if min_val is not None and tensor.min() < min_val:
        raise ValueError(
            f"{name}: minimum value {tensor.min()} is below {min_val}"
        )
    
    if max_val is not None and tensor.max() > max_val:
        raise ValueError(
            f"{name}: maximum value {tensor.max()} is above {max_val}"
        )


def validate_token_ids(
    token_ids: torch.Tensor,
    vocab_size: int,
    max_seq_len: int
) -> None:
    """
    Validate input token IDs.
    
    Args:
        token_ids: (batch, seq_len) token ID tensor
        vocab_size: Vocabulary size
        max_seq_len: Maximum sequence length
        
    Raises:
        ValueError: If validation fails
    """
    if token_ids.dim() != 2:
        raise ValueError(
            f"token_ids must be 2D (batch, seq_len), got {token_ids.dim()}D"
        )
    
    batch_size, seq_len = token_ids.shape
    
    if seq_len > max_seq_len:
        raise ValueError(
            f"Sequence length {seq_len} exceeds maximum {max_seq_len}"
        )
    
    if token_ids.min() < 0:
        raise ValueError(f"token_ids contains negative values: {token_ids.min()}")
    
    if token_ids.max() >= vocab_size:
        raise ValueError(
            f"token_ids contains values >= vocab_size: "
            f"{token_ids.max()} >= {vocab_size}"
        )
