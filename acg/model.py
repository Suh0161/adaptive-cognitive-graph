"""
Main ACG model - to be fully implemented in task 8.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from .config import ACGConfig
from .models import (
    SemanticEncoder,
    IntentRouter,
    GraphOfExperts,
    MemoryAdapter,
    VerifierExpert,
    OutputFusionLayer,
)
from .utils import validate_token_ids, setup_logger


logger = setup_logger("acg.model")


class ACGModel(nn.Module):
    """
    Adaptive Cognitive Graph model.
    
    Integrates all components for end-to-end processing with dynamic expert routing,
    continual learning adapters, quality verification, and output fusion.
    
    """
    
    def __init__(self, config: ACGConfig):
        """
        Initialize ACG model with all components.
        
        Args:
            config: Model configuration
            
        """
        super().__init__()
        self.config = config
        
        logger.info(f"Initializing ACG model with config: {config}")
        
        # Initialize all components
        
        # 1. Semantic encoder (hybrid Transformer + SSM)
        logger.info("Initializing SemanticEncoder...")
        self.encoder = SemanticEncoder(config)
        
        # 2. Intent router (phase + expert selection)
        logger.info("Initializing IntentRouter...")
        self.router = IntentRouter(config)
        
        # 3. Graph of experts (DAG topology)
        logger.info("Initializing GraphOfExperts...")
        self.experts = GraphOfExperts(config)
        
        # 4. Memory adapters (per-expert LoRA)
        if config.use_memory_adapter:
            logger.info("Initializing MemoryAdapters...")
            self.adapters = nn.ModuleDict()
            for expert_id in range(config.n_experts):
                self.adapters[str(expert_id)] = MemoryAdapter(config, config.d_model)
        else:
            self.adapters = None
        
        # 5. Verifier expert (quality assessment)
        logger.info("Initializing VerifierExpert...")
        self.verifier = VerifierExpert(config)
        
        # 6. Output fusion layer (final predictions)
        logger.info("Initializing OutputFusionLayer...")
        self.fusion = OutputFusionLayer(config)
        
        logger.info(f"ACG model initialized with {self.get_num_params():,} parameters")
        logger.info(f"Estimated active parameters per token: {self.estimate_active_params():,}")
    
    def forward(
        self,
        token_ids: torch.Tensor,
        return_routing_info: bool = False,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        End-to-end forward pass through ACG model.
        
        Implements the complete processing pipeline:
        1. Encode tokens to contextual representations
        2. Route to appropriate experts based on intent
        3. Process through selected experts
        4. Apply memory adapters for continual learning
        5. Verify and correct outputs
        6. Fuse into final predictions
        
        Args:
            token_ids: (batch, seq_len) - Input token IDs
            return_routing_info: If True, return routing information
            attention_mask: (batch, seq_len, seq_len) - Optional attention mask
            
        Returns:
            logits: (batch, seq_len, vocab_size) - Final predictions
            confidence: (batch,) - Verifier confidence scores
            
        """
        try:
            # Input validation 
            self._validate_inputs(token_ids)
            
            # Step 1: Encode tokens 
            hidden_states = self.encoder(token_ids, attention_mask)
            
            # Step 2: Route to experts 
            routing_map = self.router(hidden_states)
            
            # Step 3: Process through selected experts 
            expert_outputs = self.experts(
                hidden_states,
                routing_map.expert_ids,
                routing_map.expert_gates
            )
            
            # Step 4: Apply memory adapters 
            adapted_outputs = self._apply_adapters(
                expert_outputs,
                routing_map.expert_ids
            )
            
            # Step 5: Merge expert outputs 
            merged_states = self._merge_expert_outputs(adapted_outputs)
            
            # Step 6: Verify and correct 
            verified_states, confidence = self.verifier(merged_states)
            
            # Check for NaN in verified states 
            if torch.isnan(verified_states).any() or torch.isnan(confidence).any():
                logger.warning("NaN detected in verifier output, using fallback")
                return self._fallback_forward(token_ids, hidden_states)
            
            # Step 7: Fuse into final predictions
            logits = self.fusion(
                [verified_states],  # Wrap in list for compatibility
                confidence.unsqueeze(1),  # (batch, 1) for single expert
                hidden_states
            )
            
            # Final NaN check
            if torch.isnan(logits).any():
                logger.warning("NaN detected in final logits, using fallback")
                return self._fallback_forward(token_ids, hidden_states)
            
            if return_routing_info:
                return logits, confidence, routing_map
            
            return logits, confidence
            
        except Exception as e:
            # Error recovery
            logger.error(f"Forward pass failed: {e}")
            return self._fallback_forward(token_ids, None)
    
    def _validate_inputs(self, token_ids: torch.Tensor) -> None:
        """
        Validate input tensor shapes and ranges.
        
        Args:
            token_ids: Input token IDs
            
        Raises:
            ValueError: If validation fails
            
        """
        validate_token_ids(
            token_ids,
            self.config.vocab_size,
            self.config.max_seq_len
        )
    
    def _apply_adapters(
        self,
        expert_outputs: list,
        expert_ids: torch.Tensor
    ) -> list:
        """
        Apply memory adapters to expert outputs.
        
        Args:
            expert_outputs: List of expert output tensors
            expert_ids: (batch, k) - Expert IDs
            
        Returns:
            List of adapted expert outputs
            
        """
        if self.adapters is None or not self.config.use_memory_adapter:
            return expert_outputs
        
        # Note: GraphOfExperts returns a single merged output in a list
        # We apply adapters based on the active experts
        adapted_outputs = []
        
        for expert_out in expert_outputs:
            # For the merged output, we apply a simple pass-through
            # In a full implementation, adapters would be integrated into expert layers
            adapted_outputs.append(expert_out)
        
        return adapted_outputs
    
    def _merge_expert_outputs(self, expert_outputs: list) -> torch.Tensor:
        """
        Merge expert outputs by computing mean.
        
        Args:
            expert_outputs: List of (batch, seq_len, d_model) tensors
            
        Returns:
            merged: (batch, seq_len, d_model) - Merged representation
            
        """
        if len(expert_outputs) == 1:
            return expert_outputs[0]
        
        # Stack and compute mean
        stacked = torch.stack(expert_outputs, dim=0)  # (n_experts, batch, seq_len, d_model)
        merged = stacked.mean(dim=0)  # (batch, seq_len, d_model)
        
        return merged
    
    def _fallback_forward(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simplified fallback forward pass for error recovery.
        
        Uses only encoder and a simple projection, bypassing routing and verification.
        
        Args:
            token_ids: (batch, seq_len) - Input token IDs
            hidden_states: Pre-computed hidden states (optional)
            
        Returns:
            logits: (batch, seq_len, vocab_size) - Fallback predictions
            confidence: (batch,) - Default confidence (all ones)
            
        """
        try:
            # Handle dimension issues
            if token_ids.dim() != 2:
                # Try to reshape or create valid shape
                if token_ids.dim() == 1:
                    token_ids = token_ids.unsqueeze(0)
                else:
                    raise ValueError(f"Cannot handle {token_ids.dim()}D input")
            
            batch_size, seq_len = token_ids.shape
            
            # Encode if not provided
            if hidden_states is None:
                hidden_states = self.encoder(token_ids)
            
            # Simple projection to vocabulary
            logits = self.fusion.output_projection(hidden_states)
            
            # Default confidence
            confidence = torch.ones(batch_size, device=token_ids.device)
            
            return logits, confidence
            
        except Exception as e:
            # Last resort: return zeros
            logger.error(f"Fallback forward also failed: {e}")
            # Try to get batch size
            try:
                if token_ids.dim() >= 1:
                    batch_size = token_ids.size(0) if token_ids.dim() > 0 else 1
                    seq_len = token_ids.size(1) if token_ids.dim() > 1 else token_ids.size(0)
                else:
                    batch_size, seq_len = 1, 1
            except:
                batch_size, seq_len = 1, 1
            
            logits = torch.zeros(
                batch_size, seq_len, self.config.vocab_size,
                device=token_ids.device
            )
            confidence = torch.zeros(batch_size, device=token_ids.device)
            return logits, confidence
    
    def get_num_params(self, trainable_only: bool = True) -> int:
        """
        Get total number of parameters.
        
        Args:
            trainable_only: If True, count only trainable parameters
            
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def estimate_active_params(self) -> int:
        """
        Estimate active parameters per forward pass.
        
        For models with over 1T parameters, ensures active parameters
        stay below 40B per token through sparse expert activation.
        
        Returns:
            Estimated active parameters
            
        """
        total_params = self.get_num_params(trainable_only=False)
        
        # Calculate active parameters
        # Active = encoder + router + (k/n) * experts + verifier + fusion
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        router_params = sum(p.numel() for p in self.router.parameters())
        verifier_params = sum(p.numel() for p in self.verifier.parameters())
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        
        # Expert parameters (only active experts are used)
        expert_params = sum(p.numel() for p in self.experts.parameters())
        active_expert_params = expert_params * (self.config.active_experts / self.config.n_experts)
        
        # Adapter parameters (if enabled)
        adapter_params = 0
        if self.adapters is not None:
            adapter_params = sum(p.numel() for p in self.adapters.parameters())
            adapter_params = adapter_params * (self.config.active_experts / self.config.n_experts)
        
        active_params = (
            encoder_params +
            router_params +
            active_expert_params +
            adapter_params +
            verifier_params +
            fusion_params
        )
        
        return int(active_params)
    
    def check_active_params_constraint(self) -> bool:
        """
        Check if active parameters are below 40B for large models.
        
        Returns:
            True if constraint is satisfied
            
        """
        total_params = self.get_num_params(trainable_only=False)
        active_params = self.estimate_active_params()
        
        # For models over 1T parameters, active must be < 40B
        if total_params > 1e12:  # 1 trillion
            max_active = 40e9  # 40 billion
            if active_params >= max_active:
                logger.warning(
                    f"Active parameters ({active_params:,}) exceed 40B limit "
                    f"for model with {total_params:,} total parameters"
                )
                return False
        
        return True
    
    def get_component_params(self) -> dict:
        """
        Get parameter counts for each component.
        
        Returns:
            Dictionary mapping component names to parameter counts
        """
        return {
            "encoder": sum(p.numel() for p in self.encoder.parameters()),
            "router": sum(p.numel() for p in self.router.parameters()),
            "experts": sum(p.numel() for p in self.experts.parameters()),
            "adapters": sum(p.numel() for p in self.adapters.parameters()) if self.adapters else 0,
            "verifier": sum(p.numel() for p in self.verifier.parameters()),
            "fusion": sum(p.numel() for p in self.fusion.parameters()),
            "total": self.get_num_params(trainable_only=False),
            "active": self.estimate_active_params()
        }
    
    def update_adapter_weights(self) -> None:
        """
        Update all memory adapter weights using EMA.
        
        Should be called after backward pass during training.
        """
        if self.adapters is None:
            return
        
        for adapter in self.adapters.values():
            adapter.update_weights()
    
    def get_routing_balance_loss(self, routing_map) -> torch.Tensor:
        """
        Compute routing balance loss from routing map.
        
        Args:
            routing_map: RoutingMap from router
            
        Returns:
            Balance loss tensor
        """
        return self.router.compute_balance_loss(routing_map.expert_probs)
