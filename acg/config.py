"""
Configuration dataclass for ACG model with comprehensive validation.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ACGConfig:
    """
    Configuration for Adaptive Cognitive Graph model.
    
    All parameters include validation ranges.
    """
    
    # Model architecture 
    d_model: int = 2048
    n_heads: int = 32
    vocab_size: int = 50257
    
    # Context and layers 
    max_seq_len: int = 256000
    n_layers: int = 8
    
    # Encoder configuration 
    ssm_every: int = 2
    ffn_mult: int = 4
    activation: Literal["swiglu", "gelu", "relu"] = "swiglu"
    dropout: float = 0.1
    
    # Router configuration 
    n_experts: int = 64
    active_experts: int = 4
    n_phases: int = 4
    pooling: Literal["mean", "attention"] = "mean"
    
    # Expert configuration
    expert_layers: int = 4
    enable_cross_edges: bool = True
    dag_topology: Literal["phase_grouped", "fully_connected", "chain"] = "phase_grouped"
    weight_tying: bool = False
    expert_parallel: bool = False  # Enable expert-parallel processing across GPUs
    
    # Memory adapter configuration 
    use_memory_adapter: bool = True
    lora_rank: int = 16
    lora_alpha: float = 0.2
    ema_decay: float = 0.9
    
    # Verifier configuration 
    verifier_hidden: int = 512
    verifier_layers: int = 2
    verifier_threshold: float = 0.5
    correction_type: Literal["transformer", "mlp"] = "transformer"
    
    # Training loss weights 
    balance_loss_weight: float = 0.01
    verify_loss_weight: float = 0.05
    phase_loss_weight: float = 0.02
    grad_clip: float = 1.0
    
    # Optimization
    learning_rate: float = 3e-4
    optimizer: Literal["adamw", "adam"] = "adamw"
    batch_size: int = 64
    
    # Deployment
    use_gradient_checkpointing: bool = False
    mixed_precision: Literal["fp32", "fp16", "bf16"] = "bf16"
    
    def __post_init__(self):
        """Validate configuration parameters."""
        self._validate()
    
    def _validate(self):
        """Validate all configuration parameters."""
        
        # Model dimension between 512 and 4096
        if not 512 <= self.d_model <= 4096:
            raise ValueError(f"d_model must be between 512 and 4096, got {self.d_model}")
        
        # Number of experts between 8 and 64
        if not 8 <= self.n_experts <= 64:
            raise ValueError(f"n_experts must be between 8 and 64, got {self.n_experts}")
        
        # Active experts between 2 and 16
        if not 2 <= self.active_experts <= 16:
            raise ValueError(f"active_experts must be between 2 and 16, got {self.active_experts}")
        
        # Active experts cannot exceed total experts
        if self.active_experts > self.n_experts:
            raise ValueError(f"active_experts ({self.active_experts}) cannot exceed n_experts ({self.n_experts})")
        
        # Number of phases between 2 and 8
        if not 2 <= self.n_phases <= 8:
            raise ValueError(f"n_phases must be between 2 and 8, got {self.n_phases}")
        
        # Context length between 512 and 256000
        if not 512 <= self.max_seq_len <= 256000:
            raise ValueError(f"max_seq_len must be between 512 and 256000, got {self.max_seq_len}")
        
        # Encoder layers between 4 and 16
        if not 4 <= self.n_layers <= 16:
            raise ValueError(f"n_layers must be between 4 and 16, got {self.n_layers}")
        
        # SSM insertion interval between 1 and 4
        if not 1 <= self.ssm_every <= 4:
            raise ValueError(f"ssm_every must be between 1 and 4, got {self.ssm_every}")
        
        # FFN multiplier between 2 and 8
        if not 2 <= self.ffn_mult <= 8:
            raise ValueError(f"ffn_mult must be between 2 and 8, got {self.ffn_mult}")
        
        # Attention heads must be divisible by model dimension
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        
        # Dropout between 0.0 and 0.3
        if not 0.0 <= self.dropout <= 0.3:
            raise ValueError(f"dropout must be between 0.0 and 0.3, got {self.dropout}")
        
        # LoRA rank between 4 and 64
        if not 4 <= self.lora_rank <= 64:
            raise ValueError(f"lora_rank must be between 4 and 64, got {self.lora_rank}")
        
        # Verifier hidden dimension between 256 and 2048
        if not 256 <= self.verifier_hidden <= 2048:
            raise ValueError(f"verifier_hidden must be between 256 and 2048, got {self.verifier_hidden}")
        
        #Verifier threshold between 0.3 and 0.7
        if not 0.3 <= self.verifier_threshold <= 0.7:
            raise ValueError(f"verifier_threshold must be between 0.3 and 0.7, got {self.verifier_threshold}")
        
        # Additional validations for loss weights 
        if not 0.001 <= self.balance_loss_weight <= 0.1:
            raise ValueError(f"balance_loss_weight must be between 0.001 and 0.1, got {self.balance_loss_weight}")
        
        if not 0.01 <= self.verify_loss_weight <= 0.1:
            raise ValueError(f"verify_loss_weight must be between 0.01 and 0.1, got {self.verify_loss_weight}")
        
        if not 0.01 <= self.phase_loss_weight <= 0.1:
            raise ValueError(f"phase_loss_weight must be between 0.01 and 0.1, got {self.phase_loss_weight}")
        
        # Gradient clipping validation
        if not 0.5 <= self.grad_clip <= 2.0:
            raise ValueError(f"grad_clip must be between 0.5 and 2.0, got {self.grad_clip}")
        
        # Learning rate validation
        if not 1e-5 <= self.learning_rate <= 1e-3:
            raise ValueError(f"learning_rate must be between 1e-5 and 1e-3, got {self.learning_rate}")
        
        # EMA decay validation 
        if not 0.85 <= self.ema_decay <= 0.95:
            raise ValueError(f"ema_decay must be between 0.85 and 0.95, got {self.ema_decay}")
        
        # LoRA alpha validation
        if not 0.1 <= self.lora_alpha <= 0.5:
            raise ValueError(f"lora_alpha must be between 0.1 and 0.5, got {self.lora_alpha}")
    
    def get_head_dim(self) -> int:
        """Get dimension per attention head."""
        return self.d_model // self.n_heads
    
    def get_ffn_dim(self) -> int:
        """Get feed-forward network hidden dimension."""
        return self.d_model * self.ffn_mult
    
    def estimate_active_params(self, total_params: int) -> int:
        """
        Estimate active parameters per token.
        
        Args:
            total_params: Total model parameters
            
        Returns:
            Estimated active parameters per forward pass
        """
        # Active params = encoder + (active_experts / n_experts) * expert_params + router + verifier + fusion
        # Rough estimate: active_experts / n_experts of total expert parameters
        expert_ratio = self.active_experts / self.n_experts
        return int(total_params * expert_ratio)
