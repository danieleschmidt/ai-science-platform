"""AI Science Platform - Core Package"""

from .algorithms.discovery import DiscoveryEngine
from .experiments.runner import ExperimentRunner

# Use simple models for compatibility (no PyTorch dependencies)
from .models.simple import SimpleModel as BaseModel
from .models.simple import SimpleDiscoveryModel as DiscoveryModel

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"

__all__ = [
    "BaseModel",
    "DiscoveryEngine", 
    "ExperimentRunner",
    "DiscoveryModel",
]