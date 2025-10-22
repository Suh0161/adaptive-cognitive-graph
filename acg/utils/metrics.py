"""
Diagnostic metrics and monitoring for ACG model.

Implements comprehensive metrics for tracking model behavior during training:
- Routing entropy for expert selection diversity
- Expert utilization tracking
- Verifier confidence monitoring
- Memory adapter drift measurement
- FLOPs efficiency calculation

"""

import torch
import math
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class RoutingEntropyMetric:
    """
    Compute entropy of expert selection distribution.
    
    Tracks routing diversity to ensure balanced expert utilization.
    Higher entropy indicates more diverse expert selection.

    """
    
    def __init__(self):
        """Initialize routing entropy metric."""
        self.expert_probs_buffer = []
    
    def update(self, expert_probs: torch.Tensor) -> None:
        """
        Add expert probabilities to buffer.
        
        Args:
            expert_probs: (batch, n_experts) - Expert selection probabilities
        """
        self.expert_probs_buffer.append(expert_probs.detach().cpu())
    
    def compute(self) -> float:
        """
        Compute average routing entropy over buffered batches.
        
        Returns:
            entropy: Average entropy of expert selection distribution
        """
        if not self.expert_probs_buffer:
            return 0.0
        
        # Concatenate all batches
        all_probs = torch.cat(self.expert_probs_buffer, dim=0)  # (total_samples, n_experts)
        
        # Compute average probability for each expert
        avg_probs = all_probs.mean(dim=0)  # (n_experts,)
        
        # Compute entropy: H = -sum(p * log(p))
        entropy = -(avg_probs * torch.log(avg_probs + 1e-10)).sum()
        
        return float(entropy.item())
    
    def compute_normalized(self) -> float:
        """
        Compute normalized routing entropy (0 to 1).
        
        Returns:
            normalized_entropy: Entropy normalized by maximum possible entropy
        """
        if not self.expert_probs_buffer:
            return 0.0
        
        entropy = self.compute()
        
        # Maximum entropy for uniform distribution
        n_experts = self.expert_probs_buffer[0].size(1)
        max_entropy = math.log(n_experts)
        
        # Normalize to [0, 1]
        normalized = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return float(normalized)
    
    def reset(self) -> None:
        """Clear buffer for next logging interval."""
        self.expert_probs_buffer.clear()


class ExpertUtilizationTracker:
    """
    Track percentage of experts activated over time.
    
    Monitors per-expert activation frequency to identify underutilized experts.

    """
    
    def __init__(self, n_experts: int):
        """
        Initialize expert utilization tracker.
        
        Args:
            n_experts: Total number of experts in the model
        """
        self.n_experts = n_experts
        self.expert_counts = torch.zeros(n_experts, dtype=torch.long)
        self.total_steps = 0
    
    def update(self, expert_ids: torch.Tensor) -> None:
        """
        Update expert activation counts.
        
        Args:
            expert_ids: (batch, k) - Selected expert indices
        """
        # Flatten and count unique experts
        flat_ids = expert_ids.flatten().cpu()
        
        for expert_id in flat_ids:
            if 0 <= expert_id < self.n_experts:
                self.expert_counts[expert_id] += 1
        
        self.total_steps += 1
    
    def compute_utilization_percentage(self) -> float:
        """
        Compute percentage of experts activated.
        
        Returns:
            utilization: Percentage of experts used at least once (0-100)

        """
        if self.total_steps == 0:
            return 0.0
        
        # Count experts with at least one activation
        active_experts = (self.expert_counts > 0).sum().item()
        
        utilization = 100.0 * active_experts / self.n_experts
        
        return float(utilization)
    
    def get_per_expert_frequency(self) -> Dict[int, float]:
        """
        Get activation frequency for each expert.
        
        Returns:
            frequencies: Dictionary mapping expert ID to activation frequency

        """
        if self.total_steps == 0:
            return {i: 0.0 for i in range(self.n_experts)}
        
        frequencies = {}
        for expert_id in range(self.n_experts):
            freq = float(self.expert_counts[expert_id].item()) / self.total_steps
            frequencies[expert_id] = freq
        
        return frequencies
    
    def get_underutilized_experts(self, threshold: float = 0.01) -> List[int]:
        """
        Get list of underutilized experts.
        
        Args:
            threshold: Minimum frequency threshold (default 1%)
            
        Returns:
            expert_ids: List of expert IDs below threshold
        """
        frequencies = self.get_per_expert_frequency()
        underutilized = [
            expert_id for expert_id, freq in frequencies.items()
            if freq < threshold
        ]
        return underutilized
    
    def reset(self) -> None:
        """Reset counters for next interval."""
        self.expert_counts.zero_()
        self.total_steps = 0
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get comprehensive utilization statistics.
        
        Returns:
            stats: Dictionary with utilization metrics
        """
        frequencies = self.get_per_expert_frequency()
        freq_values = list(frequencies.values())
        
        return {
            'utilization_pct': self.compute_utilization_percentage(),
            'mean_frequency': sum(freq_values) / len(freq_values) if freq_values else 0.0,
            'max_frequency': max(freq_values) if freq_values else 0.0,
            'min_frequency': min(freq_values) if freq_values else 0.0,
            'total_steps': float(self.total_steps)
        }


class VerifierConfidenceMonitor:
    """
    Monitor verifier confidence scores.
    
    Tracks mean confidence and distribution to assess verifier behavior.

    """
    
    def __init__(self):
        """Initialize verifier confidence monitor."""
        self.confidence_buffer = []
    
    def update(self, confidence: torch.Tensor) -> None:
        """
        Add confidence scores to buffer.
        
        Args:
            confidence: (batch,) - Verifier confidence scores [0, 1]
        """
        self.confidence_buffer.append(confidence.detach().cpu())
    
    def compute_mean(self) -> float:
        """
        Compute mean confidence score.
        
        Returns:
            mean_confidence: Average confidence across all samples
            
        """
        if not self.confidence_buffer:
            return 0.0
        
        all_confidence = torch.cat(self.confidence_buffer, dim=0)
        mean_conf = all_confidence.mean()
        
        return float(mean_conf.item())
    
    def compute_distribution(self) -> Dict[str, float]:
        """
        Compute confidence distribution statistics.
        
        Returns:
            stats: Dictionary with distribution metrics
        """
        if not self.confidence_buffer:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0
            }
        
        all_confidence = torch.cat(self.confidence_buffer, dim=0)
        
        return {
            'mean': float(all_confidence.mean().item()),
            'std': float(all_confidence.std().item()),
            'min': float(all_confidence.min().item()),
            'max': float(all_confidence.max().item()),
            'median': float(all_confidence.median().item())
        }
    
    def compute_low_confidence_rate(self, threshold: float = 0.5) -> float:
        """
        Compute percentage of samples with low confidence.
        
        Args:
            threshold: Confidence threshold
            
        Returns:
            rate: Percentage of samples below threshold (0-100)
        """
        if not self.confidence_buffer:
            return 0.0
        
        all_confidence = torch.cat(self.confidence_buffer, dim=0)
        low_conf_count = (all_confidence < threshold).sum().item()
        total_count = all_confidence.numel()
        
        rate = 100.0 * low_conf_count / total_count
        
        return float(rate)
    
    def reset(self) -> None:
        """Clear buffer for next logging interval."""
        self.confidence_buffer.clear()


class MemoryAdapterDriftMeasure:
    """
    Measure drift in memory adapter weights.
    
    Tracks L2 norm difference in adapter weights between checkpoints
    to monitor continual learning behavior.
    
    """
    
    def __init__(self):
        """Initialize memory adapter drift measure."""
        self.previous_weights = {}
        self.drift_history = []
    
    def save_checkpoint(self, adapters: torch.nn.ModuleDict) -> None:
        """
        Save current adapter weights as checkpoint.
        
        Args:
            adapters: ModuleDict of MemoryAdapter instances
        """
        self.previous_weights = {}
        
        for expert_id, adapter in adapters.items():
            # Save weight delta (the adapted part)
            if hasattr(adapter, 'weight_delta'):
                self.previous_weights[expert_id] = adapter.weight_delta.detach().clone()
    
    def compute_drift(self, adapters: torch.nn.ModuleDict) -> float:
        """
        Compute L2 norm difference from previous checkpoint.
        
        Args:
            adapters: ModuleDict of MemoryAdapter instances
            
        Returns:
            drift: Average L2 norm difference across all adapters
            
        """
        if not self.previous_weights:
            # No previous checkpoint, save current and return 0
            self.save_checkpoint(adapters)
            return 0.0
        
        total_drift = 0.0
        count = 0
        
        for expert_id, adapter in adapters.items():
            if expert_id not in self.previous_weights:
                continue
            
            if not hasattr(adapter, 'weight_delta'):
                continue
            
            # Compute L2 norm of difference
            prev_weights = self.previous_weights[expert_id]
            curr_weights = adapter.weight_delta
            
            diff = curr_weights - prev_weights
            l2_norm = torch.norm(diff, p=2).item()
            
            total_drift += l2_norm
            count += 1
        
        avg_drift = total_drift / count if count > 0 else 0.0
        
        # Save current as new checkpoint
        self.save_checkpoint(adapters)
        
        # Record in history
        self.drift_history.append(avg_drift)
        
        return float(avg_drift)
    
    def get_drift_history(self) -> List[float]:
        """
        Get historical drift measurements.
        
        Returns:
            history: List of drift values over time
        """
        return self.drift_history.copy()
    
    def get_drift_stats(self) -> Dict[str, float]:
        """
        Get drift statistics.
        
        Returns:
            stats: Dictionary with drift metrics
        """
        if not self.drift_history:
            return {
                'mean': 0.0,
                'max': 0.0,
                'min': 0.0,
                'current': 0.0
            }
        
        return {
            'mean': sum(self.drift_history) / len(self.drift_history),
            'max': max(self.drift_history),
            'min': min(self.drift_history),
            'current': self.drift_history[-1] if self.drift_history else 0.0
        }
    
    def reset(self) -> None:
        """Reset drift tracking."""
        self.previous_weights.clear()
        self.drift_history.clear()


class FLOPsEfficiencyCalculator:
    """
    Calculate FLOPs efficiency compared to dense baseline.
    
    Measures active compute per forward pass and compares to
    equivalent dense model to quantify efficiency gains.
    
    """
    
    def __init__(
        self,
        d_model: int,
        n_experts: int,
        active_experts: int,
        expert_layers: int,
        seq_len: int
    ):
        """
        Initialize FLOPs calculator.
        
        Args:
            d_model: Model dimension
            n_experts: Total number of experts
            active_experts: Number of active experts per token
            expert_layers: Number of layers per expert
            seq_len: Sequence length
        """
        self.d_model = d_model
        self.n_experts = n_experts
        self.active_experts = active_experts
        self.expert_layers = expert_layers
        self.seq_len = seq_len
    
    def compute_sparse_flops(self) -> float:
        """
        Compute FLOPs for sparse expert activation.
        
        Returns:
            flops: Estimated FLOPs for sparse model
            
        """
        # FLOPs per expert layer (simplified)
        # Attention: 4 * d_model^2 * seq_len (Q, K, V, O projections)
        # FFN: 8 * d_model^2 * seq_len (up and down projections with 4x expansion)
        flops_per_layer = (4 + 8) * self.d_model ** 2 * self.seq_len
        
        # Total FLOPs for active experts
        sparse_flops = self.active_experts * self.expert_layers * flops_per_layer
        
        return float(sparse_flops)
    
    def compute_dense_flops(self) -> float:
        """
        Compute FLOPs for equivalent dense model.
        
        Returns:
            flops: Estimated FLOPs for dense baseline
        """
        # Dense model processes all experts
        flops_per_layer = (4 + 8) * self.d_model ** 2 * self.seq_len
        
        # Total FLOPs for all experts
        dense_flops = self.n_experts * self.expert_layers * flops_per_layer
        
        return float(dense_flops)
    
    def compute_efficiency_ratio(self) -> float:
        """
        Compute efficiency ratio (sparse / dense).
        
        Lower ratio indicates better efficiency.
        
        Returns:
            ratio: FLOPs ratio (sparse / dense)

        """
        sparse_flops = self.compute_sparse_flops()
        dense_flops = self.compute_dense_flops()
        
        ratio = sparse_flops / dense_flops if dense_flops > 0 else 1.0
        
        return float(ratio)
    
    def compute_speedup(self) -> float:
        """
        Compute theoretical speedup over dense baseline.
        
        Returns:
            speedup: Speedup factor (dense / sparse)
        """
        ratio = self.compute_efficiency_ratio()
        speedup = 1.0 / ratio if ratio > 0 else 1.0
        
        return float(speedup)
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get comprehensive FLOPs statistics.
        
        Returns:
            stats: Dictionary with FLOPs metrics
        """
        return {
            'sparse_flops': self.compute_sparse_flops(),
            'dense_flops': self.compute_dense_flops(),
            'efficiency_ratio': self.compute_efficiency_ratio(),
            'speedup': self.compute_speedup()
        }


class DiagnosticMetrics:
    """
    Comprehensive diagnostic metrics tracker.
    
    Aggregates all metrics with configurable logging intervals.

    """
    
    def __init__(
        self,
        n_experts: int,
        d_model: int,
        active_experts: int,
        expert_layers: int,
        seq_len: int,
        log_interval: int = 100
    ):
        """
        Initialize diagnostic metrics.
        
        Args:
            n_experts: Total number of experts
            d_model: Model dimension
            active_experts: Number of active experts
            expert_layers: Number of layers per expert
            seq_len: Typical sequence length
            log_interval: Steps between logging (10-1000)
            
        """
        if not 10 <= log_interval <= 1000:
            raise ValueError(f"log_interval must be between 10 and 1000, got {log_interval}")
        
        self.log_interval = log_interval
        self.step_count = 0
        
        # Initialize all metric trackers
        self.routing_entropy = RoutingEntropyMetric()
        self.expert_utilization = ExpertUtilizationTracker(n_experts)
        self.verifier_confidence = VerifierConfidenceMonitor()
        self.adapter_drift = MemoryAdapterDriftMeasure()
        self.flops_calculator = FLOPsEfficiencyCalculator(
            d_model, n_experts, active_experts, expert_layers, seq_len
        )
    
    def update(
        self,
        expert_probs: torch.Tensor,
        expert_ids: torch.Tensor,
        confidence: torch.Tensor
    ) -> None:
        """
        Update all metrics with current batch.
        
        Args:
            expert_probs: (batch, n_experts) - Expert selection probabilities
            expert_ids: (batch, k) - Selected expert indices
            confidence: (batch,) - Verifier confidence scores
        """
        self.routing_entropy.update(expert_probs)
        self.expert_utilization.update(expert_ids)
        self.verifier_confidence.update(confidence)
        self.step_count += 1
    
    def should_log(self) -> bool:
        """
        Check if current step should trigger logging.
        
        Returns:
            should_log: True if step count is multiple of log_interval
            
        """
        return self.step_count % self.log_interval == 0
    
    def compute_all(
        self,
        adapters: Optional[torch.nn.ModuleDict] = None
    ) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Args:
            adapters: Optional ModuleDict of adapters for drift measurement
            
        Returns:
            metrics: Dictionary with all metric values
        """
        metrics = {
            # Routing metrics
            'routing_entropy': self.routing_entropy.compute(),
            'routing_entropy_normalized': self.routing_entropy.compute_normalized(),
            
            # Expert utilization
            'expert_utilization_pct': self.expert_utilization.compute_utilization_percentage(),
            
            # Verifier metrics
            'verifier_confidence_mean': self.verifier_confidence.compute_mean(),
            'verifier_low_conf_rate': self.verifier_confidence.compute_low_confidence_rate(),
            
            # FLOPs efficiency
            'flops_efficiency_ratio': self.flops_calculator.compute_efficiency_ratio(),
            'flops_speedup': self.flops_calculator.compute_speedup(),
        }
        
        # Add adapter drift if available
        if adapters is not None:
            metrics['adapter_drift'] = self.adapter_drift.compute_drift(adapters)
        
        # Add detailed stats
        metrics.update({
            f'expert_util_{k}': v
            for k, v in self.expert_utilization.get_stats().items()
        })
        
        metrics.update({
            f'verifier_{k}': v
            for k, v in self.verifier_confidence.compute_distribution().items()
        })
        
        return metrics
    
    def reset_interval_metrics(self) -> None:
        """Reset metrics that accumulate over logging interval."""
        self.routing_entropy.reset()
        self.verifier_confidence.reset()
        # Note: expert_utilization is not reset to track over 1000 steps
    
    def reset_all(self) -> None:
        """Reset all metrics."""
        self.routing_entropy.reset()
        self.expert_utilization.reset()
        self.verifier_confidence.reset()
        self.adapter_drift.reset()
        self.step_count = 0
