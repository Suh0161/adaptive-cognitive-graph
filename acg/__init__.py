"""
Adaptive Cognitive Graph (ACG) v1.1

A scalable, reasoning-centric neural architecture that unifies Mixture-of-Experts 
efficiency, dynamic compute allocation, long-context modeling, and built-in self-verification.
"""

__version__ = "1.1.0"

from .config import ACGConfig
from .model import ACGModel
from .training import (
    compute_language_modeling_loss,
    compute_routing_balance_loss,
    compute_verification_loss,
    compute_phase_classification_loss,
    compute_total_loss,
    clip_gradients,
    setup_optimizer,
    ACGTrainer
)

__all__ = [
    "ACGConfig",
    "ACGModel",
    "compute_language_modeling_loss",
    "compute_routing_balance_loss",
    "compute_verification_loss",
    "compute_phase_classification_loss",
    "compute_total_loss",
    "clip_gradients",
    "setup_optimizer",
    "ACGTrainer"
]
