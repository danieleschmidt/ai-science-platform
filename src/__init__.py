"""AI Science Platform - Core Package"""

from .models.base import BaseModel
from .algorithms.discovery import DiscoveryEngine
from .experiments.runner import ExperimentRunner

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"

__all__ = [
    "BaseModel",
    "DiscoveryEngine", 
    "ExperimentRunner",
]