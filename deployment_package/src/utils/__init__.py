"""Utility functions and helpers"""

from .data_utils import generate_sample_data, validate_data

# Optional matplotlib-dependent imports
try:
    from .visualization import plot_discovery_results
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    def plot_discovery_results(*args, **kwargs):
        print("Visualization not available - matplotlib not installed")
        return None

__all__ = [
    "generate_sample_data",
    "validate_data", 
    "plot_discovery_results",
    "VISUALIZATION_AVAILABLE"
]