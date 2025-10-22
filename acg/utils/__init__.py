"""
Utility functions and helpers for ACG.
"""

from .routing import RoutingMap
from .validation import validate_tensor, check_nan_inf, validate_token_ids
from .logging import setup_logger
from .metrics import (
    RoutingEntropyMetric,
    ExpertUtilizationTracker,
    VerifierConfidenceMonitor,
    MemoryAdapterDriftMeasure,
    FLOPsEfficiencyCalculator,
    DiagnosticMetrics,
)

__all__ = [
    "RoutingMap",
    "validate_tensor",
    "check_nan_inf",
    "validate_token_ids",
    "setup_logger",
    "RoutingEntropyMetric",
    "ExpertUtilizationTracker",
    "VerifierConfidenceMonitor",
    "MemoryAdapterDriftMeasure",
    "FLOPsEfficiencyCalculator",
    "DiagnosticMetrics",
]
