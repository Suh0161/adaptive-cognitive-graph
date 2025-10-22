"""
Training utilities for ACG model.

Implements loss computation, optimizer setup, and gradient clipping.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from .config import ACGConfig


def compute_language_modeling_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    padding_mask: Optional[torch.Tensor] = None,
    ignore_index: int = -100
) -> torch.Tensor:
    """
    Compute cross-entropy loss for language modeling.
    
    Args:
        logits: (batch, seq_len, vocab_size) - Model predictions
        targets: (batch, seq_len) - Target token IDs
        padding_mask: (batch, seq_len) - Optional mask (1 for valid, 0 for padding)
        ignore_index: Token ID to ignore in loss computation
        
    Returns:
        loss: Scalar tensor with language modeling loss

    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # Reshape for cross-entropy
    logits_flat = logits.view(-1, vocab_size)  # (batch * seq_len, vocab_size)
    targets_flat = targets.view(-1)  # (batch * seq_len,)
    
    # Compute cross-entropy loss
    if padding_mask is not None:
        # Apply mask by setting padding positions to ignore_index
        mask_flat = padding_mask.view(-1)  # (batch * seq_len,)
        targets_masked = targets_flat.clone()
        targets_masked[mask_flat == 0] = ignore_index
        loss = F.cross_entropy(logits_flat, targets_masked, ignore_index=ignore_index)
    else:
        loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=ignore_index)
    
    return loss


def compute_routing_balance_loss(expert_probs: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy-based routing balance loss.
    
    Encourages uniform expert utilization by maximizing entropy
    of expert selection distribution.
    
    Args:
        expert_probs: (batch, n_experts) - Expert selection probabilities
        
    Returns:
        loss: Scalar tensor with balance loss (0 = perfect balance, 1 = worst)

    """
    # Compute average probability for each expert across batch
    avg_probs = expert_probs.mean(dim=0)  # (n_experts,)
    
    # Compute entropy
    entropy = -(avg_probs * torch.log(avg_probs + 1e-10)).sum()
    
    # Maximum entropy (uniform distribution)
    n_experts = expert_probs.size(1)
    max_entropy = torch.log(torch.tensor(n_experts, dtype=expert_probs.dtype, device=expert_probs.device))
    
    # Balance loss: 1 - normalized entropy (0 = perfect balance)
    balance_loss = 1.0 - (entropy / max_entropy)
    
    return balance_loss


def compute_verification_loss(
    confidence: torch.Tensor,
    correctness_labels: torch.Tensor
) -> torch.Tensor:
    """
    Compute binary cross-entropy loss for verifier training.
    
    Args:
        confidence: (batch,) - Verifier confidence scores [0, 1]
        correctness_labels: (batch,) - Binary labels (1 = correct, 0 = incorrect)
        
    Returns:
        loss: Scalar tensor with verification loss

    """
    # Binary cross-entropy loss
    loss = F.binary_cross_entropy(confidence, correctness_labels.float())
    
    return loss


def compute_phase_classification_loss(
    phase_logits: torch.Tensor,
    phase_labels: torch.Tensor
) -> torch.Tensor:
    """
    Compute cross-entropy loss for phase classification.
    
    Args:
        phase_logits: (batch, n_phases) - Phase classification logits
        phase_labels: (batch,) - Ground truth phase IDs
        
    Returns:
        loss: Scalar tensor with phase classification loss

    """
    loss = F.cross_entropy(phase_logits, phase_labels)
    
    return loss



def compute_total_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    confidence: torch.Tensor,
    expert_probs: torch.Tensor,
    config: ACGConfig,
    padding_mask: Optional[torch.Tensor] = None,
    correctness_labels: Optional[torch.Tensor] = None,
    phase_logits: Optional[torch.Tensor] = None,
    phase_labels: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute total training loss with all components.
    
    Combines language modeling loss, routing balance loss, verification loss,
    and optional phase classification loss with configurable weights.
    
    Args:
        logits: (batch, seq_len, vocab_size) - Model predictions
        targets: (batch, seq_len) - Target token IDs
        confidence: (batch,) - Verifier confidence scores
        expert_probs: (batch, n_experts) - Expert selection probabilities
        config: Model configuration with loss weights
        padding_mask: (batch, seq_len) - Optional padding mask
        correctness_labels: (batch,) - Optional correctness labels for verifier
        phase_logits: (batch, n_phases) - Optional phase classification logits
        phase_labels: (batch,) - Optional ground truth phase labels
        
    Returns:
        total_loss: Scalar tensor with weighted sum of all losses
        loss_dict: Dictionary with individual loss components

    """
    loss_dict = {}
    
    # 1. Language modeling loss 
    lm_loss = compute_language_modeling_loss(logits, targets, padding_mask)
    loss_dict['lm_loss'] = lm_loss
    
    # 2. Routing balance loss 
    balance_loss = compute_routing_balance_loss(expert_probs)
    loss_dict['balance_loss'] = balance_loss
    
    if correctness_labels is not None:
        verify_loss = compute_verification_loss(confidence, correctness_labels)
        loss_dict['verify_loss'] = verify_loss
    else:
        # If no labels provided, use zero loss
        verify_loss = torch.tensor(0.0, device=logits.device)
        loss_dict['verify_loss'] = verify_loss
    
    # 4. Phase classification loss 
    if phase_logits is not None and phase_labels is not None:
        phase_loss = compute_phase_classification_loss(phase_logits, phase_labels)
        loss_dict['phase_loss'] = phase_loss
    else:
        phase_loss = torch.tensor(0.0, device=logits.device)
        loss_dict['phase_loss'] = phase_loss
    
    # Compute weighted total loss
    total_loss = (
        lm_loss +
        config.balance_loss_weight * balance_loss +
        config.verify_loss_weight * verify_loss +
        config.phase_loss_weight * phase_loss
    )
    
    loss_dict['total_loss'] = total_loss
    
    return total_loss, loss_dict



def clip_gradients(
    model: nn.Module,
    max_norm: float
) -> float:
    """
    Clip gradients by global norm.
    
    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm
        
    Returns:
        total_norm: Total gradient norm before clipping
        
    """
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm
    )
    
    return float(total_norm)


def setup_optimizer(
    model: nn.Module,
    config: ACGConfig
) -> torch.optim.Optimizer:
    """
    Set up optimizer with configuration.
    
    Args:
        model: PyTorch model
        config: Model configuration
        
    Returns:
        optimizer: Configured optimizer
        
    """
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
    elif config.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    return optimizer


class ACGTrainer:
    """
    Training wrapper for ACG model.
    
    Handles forward pass, loss computation, backward pass, and optimization.
    Supports mixed precision training with fp16/bf16.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: ACGConfig,
        use_mixed_precision: bool = True,
        mixed_precision_dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: ACG model
            config: Model configuration
            use_mixed_precision: Whether to use mixed precision training
            mixed_precision_dtype: Dtype for mixed precision (None = auto-detect)
        """
        self.model = model
        self.config = config
        self.optimizer = setup_optimizer(model, config)
        self.step_count = 0
        
        # Setup mixed precision
        self.use_mixed_precision = use_mixed_precision
        if mixed_precision_dtype is None:
            # Auto-detect based on config
            if hasattr(config, 'mixed_precision'):
                if config.mixed_precision == 'bf16':
                    mixed_precision_dtype = torch.bfloat16
                elif config.mixed_precision == 'fp16':
                    mixed_precision_dtype = torch.float16
                else:
                    mixed_precision_dtype = torch.float32
                    self.use_mixed_precision = False
            else:
                mixed_precision_dtype = torch.bfloat16
        
        self.mixed_precision_dtype = mixed_precision_dtype
        
        # Create gradient scaler for fp16
        self.use_scaler = (
            self.use_mixed_precision and 
            mixed_precision_dtype == torch.float16
        )
        if self.use_scaler:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def training_step(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        correctness_labels: Optional[torch.Tensor] = None,
        phase_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Perform single training step with mixed precision support.
        
        Args:
            token_ids: (batch, seq_len) - Input token IDs
            targets: (batch, seq_len) - Target token IDs
            padding_mask: (batch, seq_len) - Optional padding mask
            correctness_labels: (batch,) - Optional correctness labels
            phase_labels: (batch,) - Optional phase labels
            
        Returns:
            metrics: Dictionary with loss values and gradient norm
            
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(
            enabled=self.use_mixed_precision,
            dtype=self.mixed_precision_dtype
        ):
            # Forward pass with routing info
            logits, confidence, routing_map = self.model(
                token_ids,
                return_routing_info=True
            )
            
            # Compute loss
            total_loss, loss_dict = compute_total_loss(
                logits=logits,
                targets=targets,
                confidence=confidence,
                expert_probs=routing_map.expert_probs,
                config=self.config,
                padding_mask=padding_mask,
                correctness_labels=correctness_labels,
                phase_logits=routing_map.phase_probs if phase_labels is not None else None,
                phase_labels=phase_labels
            )
        
        # Backward pass with gradient scaling for fp16
        if self.use_scaler:
            self.scaler.scale(total_loss).backward()
            
            # Unscale before gradient clipping
            self.scaler.unscale_(self.optimizer)
            grad_norm = clip_gradients(self.model, self.config.grad_clip)
            
            # Optimizer step with scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard backward pass
            total_loss.backward()
            grad_norm = clip_gradients(self.model, self.config.grad_clip)
            self.optimizer.step()
        
        # Update memory adapters if enabled
        if hasattr(self.model, 'update_adapter_weights'):
            self.model.update_adapter_weights()
        
        self.step_count += 1
        
        # Return metrics
        metrics = {
            'total_loss': total_loss.item(),
            'lm_loss': loss_dict['lm_loss'].item(),
            'balance_loss': loss_dict['balance_loss'].item(),
            'verify_loss': loss_dict['verify_loss'].item(),
            'phase_loss': loss_dict['phase_loss'].item(),
            'grad_norm': grad_norm
        }
        
        return metrics
