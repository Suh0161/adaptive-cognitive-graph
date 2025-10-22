"""
Graph of experts with DAG topology.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Literal, Tuple
from .base import BaseModule
from ..config import ACGConfig
from .attention import TransformerBlock


class ExpertBlock(BaseModule):
    """
    Individual expert module (mini-transformer).
    
    Each expert is a specialized transformer with 2-6 layers.
    """
    
    def __init__(self, config: ACGConfig, expert_id: int):
        """
        Initialize expert block.
        
        Args:
            config: ACG configuration
            expert_id: Unique identifier for this expert
        """
        super().__init__(config)
        self.expert_id = expert_id
        
        # Build expert layers
        self.layers = nn.ModuleList()
        d_ff = config.d_model * config.ffn_mult
        
        for _ in range(config.expert_layers):
            layer = TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=d_ff,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout
            )
            self.layers.append(layer)
        
        # Final normalization
        self.norm = nn.LayerNorm(config.d_model)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        neighbor_contribution: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process inputs through expert.
        
        Args:
            hidden_states: (batch, seq_len, d_model) - Input states
            neighbor_contribution: (batch, seq_len, d_model) - Optional neighbor info
            
        Returns:
            output: (batch, seq_len, d_model) - Expert output
        """
        # Add neighbor contribution if provided
        if neighbor_contribution is not None:
            hidden_states = hidden_states + neighbor_contribution
        
        # Process through expert layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        return hidden_states


class GraphOfExperts(BaseModule):
    """
    DAG of expert modules with cross-edge communication.
    
    Supports multiple topology types, optional weight tying, and expert parallelism.
    """
    
    def __init__(self, config: ACGConfig):
        """
        Initialize graph of experts.
        
        Args:
            config: ACG configuration
        """
        super().__init__(config)
        
        # Create expert blocks
        self.experts = nn.ModuleList([
            ExpertBlock(config, expert_id=i)
            for i in range(config.n_experts)
        ])
        
        # Build DAG topology
        self.adjacency = self._build_dag_topology(
            config.n_experts,
            config.n_phases,
            config.dag_topology
        )
        
        # Cross-edge communication weights
        if config.enable_cross_edges:
            self.edge_weights = nn.ModuleDict()
            for expert_id, neighbors in self.adjacency.items():
                if neighbors:
                    # Create edge weight matrix for this expert
                    self.edge_weights[str(expert_id)] = nn.Linear(
                        config.d_model,
                        config.d_model,
                        bias=False
                    )
        
        # Weight tying (share parameters within phase groups)
        if config.weight_tying:
            self._apply_weight_tying(config.n_phases)
        
        # Expert parallelism setup
        self.expert_parallel = config.expert_parallel
        self.expert_devices = None
        if self.expert_parallel and torch.cuda.device_count() > 1:
            self._setup_expert_parallelism()
    
    def _build_dag_topology(
        self,
        n_experts: int,
        n_phases: int,
        topology: Literal["phase_grouped", "fully_connected", "chain"]
    ) -> Dict[int, List[int]]:
        """
        Build DAG adjacency list.
        
        Args:
            n_experts: Number of experts
            n_phases: Number of cognitive phases
            topology: Topology type
            
        Returns:
            adjacency: Dict mapping expert_id to list of neighbor expert_ids
        """
        adjacency = {i: [] for i in range(n_experts)}
        
        if topology == "phase_grouped":
            # Group experts by phase, connect within groups
            experts_per_phase = n_experts // n_phases
            for phase in range(n_phases):
                start_idx = phase * experts_per_phase
                end_idx = start_idx + experts_per_phase
                
                # Connect experts within same phase
                for i in range(start_idx, end_idx):
                    for j in range(start_idx, end_idx):
                        if i != j:
                            adjacency[i].append(j)
        
        elif topology == "fully_connected":
            # All experts can communicate
            for i in range(n_experts):
                adjacency[i] = [j for j in range(n_experts) if j != i]
        
        elif topology == "chain":
            # Sequential chain of experts
            for i in range(n_experts - 1):
                adjacency[i].append(i + 1)
        
        return adjacency
    
    def _apply_weight_tying(self, n_phases: int):
        """
        Apply weight tying within phase groups.
        
        Shares parameters across experts in the same cognitive phase,
        reducing total parameter count while maintaining phase specialization.
        
        Args:
            n_phases: Number of phases
        """
        experts_per_phase = len(self.experts) // n_phases
        
        for phase in range(n_phases):
            start_idx = phase * experts_per_phase
            if start_idx + 1 >= len(self.experts):
                continue
            
            # Use first expert in phase as reference
            reference_expert = self.experts[start_idx]
            
            # Tie weights for other experts in same phase
            for i in range(start_idx + 1, min(start_idx + experts_per_phase, len(self.experts))):
                for ref_layer, expert_layer in zip(reference_expert.layers, self.experts[i].layers):
                    # Share attention and FFN weights
                    expert_layer.attention = ref_layer.attention
                    expert_layer.ffn = ref_layer.ffn
    
    def _setup_expert_parallelism(self):
        """
        Distribute experts across available GPUs for parallel execution.
        
        Implements expert-parallel processing by assigning each expert to a specific GPU.
        This enables processing multiple experts simultaneously across devices.
        """
        n_devices = torch.cuda.device_count()
        self.expert_devices = {}
        
        # Distribute experts evenly across GPUs
        for expert_id in range(len(self.experts)):
            device_id = expert_id % n_devices
            device = torch.device(f'cuda:{device_id}')
            
            # Move expert to assigned device
            self.experts[expert_id] = self.experts[expert_id].to(device)
            self.expert_devices[expert_id] = device
            
            # Move edge weights if they exist
            if self.config.enable_cross_edges and str(expert_id) in self.edge_weights:
                self.edge_weights[str(expert_id)] = self.edge_weights[str(expert_id)].to(device)
    
    def _get_expert_device(self, expert_id: int) -> torch.device:
        """
        Get the device where an expert is located.
        
        Args:
            expert_id: Expert identifier
            
        Returns:
            Device where expert resides
        """
        if self.expert_devices is not None:
            return self.expert_devices[expert_id]
        return next(self.experts[expert_id].parameters()).device
    
    def _compute_neighbor_contribution(
        self,
        expert_id: int,
        expert_outputs: Dict[int, torch.Tensor],
        target_device: torch.device
    ) -> Optional[torch.Tensor]:
        """
        Compute aggregated contribution from neighbor experts.
        
        Handles cross-device communication when experts are distributed across GPUs.
        
        Args:
            expert_id: Current expert ID
            expert_outputs: Dict of already computed expert outputs
            target_device: Device where the result should be placed
            
        Returns:
            neighbor_contribution: (batch, seq_len, d_model) or None
        """
        if not self.config.enable_cross_edges:
            return None
        
        neighbors = self.adjacency[expert_id]
        if not neighbors:
            return None
        
        # Get edge weight matrix
        edge_weight_key = str(expert_id)
        if edge_weight_key not in self.edge_weights:
            return None
        edge_weight = self.edge_weights[edge_weight_key]
        
        # Aggregate neighbor outputs
        contributions = []
        for neighbor_id in neighbors:
            if neighbor_id in expert_outputs:
                neighbor_out = expert_outputs[neighbor_id]
                
                # Move to target device if needed (cross-GPU communication)
                if neighbor_out.device != target_device:
                    neighbor_out = neighbor_out.to(target_device)
                
                # Apply edge weight
                weighted_out = edge_weight(neighbor_out)
                contributions.append(weighted_out)
        
        if not contributions:
            return None
        
        # Average contributions
        neighbor_contribution = torch.stack(contributions).mean(dim=0)
        
        return neighbor_contribution
    
    def _process_expert_parallel(
        self,
        hidden_states: torch.Tensor,
        unique_experts: List[int],
        expert_output_cache: Dict[int, torch.Tensor]
    ) -> Dict[int, torch.Tensor]:
        """
        Process experts in parallel across multiple GPUs.
        
        Args:
            hidden_states: (batch, seq_len, d_model) - Input states
            unique_experts: List of expert IDs to process
            expert_output_cache: Cache for storing outputs
            
        Returns:
            Updated expert_output_cache with computed outputs
        """
        # Process experts in parallel using threading for async GPU execution
        import concurrent.futures
        
        def process_single_expert(expert_id: int) -> Tuple[int, torch.Tensor]:
            """Process a single expert on its assigned device."""
            expert_device = self._get_expert_device(expert_id)
            
            # Move input to expert's device
            hidden_states_device = hidden_states.to(expert_device)
            
            # Compute neighbor contribution
            neighbor_contrib = self._compute_neighbor_contribution(
                expert_id,
                expert_output_cache,
                expert_device
            )
            
            # Process through expert
            with torch.cuda.device(expert_device):
                output = self.experts[expert_id](hidden_states_device, neighbor_contrib)
            
            return expert_id, output
        
        # Use ThreadPoolExecutor for parallel GPU execution
        # CUDA operations are async, so threads work well here
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(unique_experts)) as executor:
            futures = [executor.submit(process_single_expert, eid) for eid in unique_experts]
            
            for future in concurrent.futures.as_completed(futures):
                expert_id, output = future.result()
                expert_output_cache[expert_id] = output
        
        return expert_output_cache
    
    def _process_expert_sequential(
        self,
        hidden_states: torch.Tensor,
        unique_experts: List[int],
        expert_output_cache: Dict[int, torch.Tensor]
    ) -> Dict[int, torch.Tensor]:
        """
        Process experts sequentially (single GPU or CPU).
        
        Args:
            hidden_states: (batch, seq_len, d_model) - Input states
            unique_experts: List of expert IDs to process
            expert_output_cache: Cache for storing outputs
            
        Returns:
            Updated expert_output_cache with computed outputs
        """
        device = hidden_states.device
        
        for expert_id in unique_experts:
            # Compute neighbor contribution
            neighbor_contrib = self._compute_neighbor_contribution(
                expert_id,
                expert_output_cache,
                device
            )
            
            # Process through expert
            output = self.experts[expert_id](hidden_states, neighbor_contrib)
            expert_output_cache[expert_id] = output
        
        return expert_output_cache
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        expert_gates: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Process through selected experts with optional parallel execution.
        
        Supports both sequential and parallel expert processing. When expert_parallel
        is enabled and multiple GPUs are available, experts are processed in parallel
        across devices for improved throughput.
        
        Args:
            hidden_states: (batch, seq_len, d_model) - Input states
            expert_ids: (batch, k) - Selected expert indices
            expert_gates: (batch, k) - Gate values for experts
            
        Returns:
            expert_outputs: List of (batch, seq_len, d_model) tensors
        """
        batch_size, seq_len, d_model = hidden_states.shape
        k = expert_ids.size(1)
        original_device = hidden_states.device
        
        # Collect unique expert IDs across batch
        unique_experts = torch.unique(expert_ids).tolist()
        
        # Process experts (parallel or sequential)
        expert_output_cache = {}
        if self.expert_parallel and self.expert_devices is not None:
            expert_output_cache = self._process_expert_parallel(
                hidden_states,
                unique_experts,
                expert_output_cache
            )
        else:
            expert_output_cache = self._process_expert_sequential(
                hidden_states,
                unique_experts,
                expert_output_cache
            )
        
        # Gather outputs for each batch item
        expert_outputs = []
        for b in range(batch_size):
            batch_outputs = []
            for i in range(k):
                expert_id = expert_ids[b, i].item()
                output = expert_output_cache[expert_id]
                
                # Move output back to original device if needed
                if output.device != original_device:
                    output = output.to(original_device)
                
                # Extract batch item
                output = output[b:b+1]  # Keep batch dim
                batch_outputs.append(output)
            
            # Stack and weight by gates
            stacked = torch.cat(batch_outputs, dim=0)  # (k, seq_len, d_model)
            gates = expert_gates[b].view(k, 1, 1)  # (k, 1, 1)
            weighted = (stacked * gates).sum(dim=0, keepdim=True)  # (1, seq_len, d_model)
            expert_outputs.append(weighted)
        
        # Concatenate batch
        expert_outputs = torch.cat(expert_outputs, dim=0)  # (batch, seq_len, d_model)
        
        return [expert_outputs]  # Return as list for compatibility
