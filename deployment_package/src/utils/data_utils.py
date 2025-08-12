"""Data utility functions for scientific computing"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def generate_sample_data(size: int = 1000, 
                        data_type: str = "normal",
                        seed: Optional[int] = None,
                        **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sample data for testing and experimentation
    
    Args:
        size: Number of samples to generate
        data_type: Type of data ('normal', 'exponential', 'sine', 'polynomial')
        seed: Random seed for reproducibility
        **kwargs: Additional parameters for specific data types
    
    Returns:
        Tuple of (features, targets)
    """
    if seed is not None:
        np.random.seed(seed)
    
    if data_type == "normal":
        mean = kwargs.get("mean", 0.0)
        std = kwargs.get("std", 1.0)
        features = np.random.normal(mean, std, size)
        targets = features + np.random.normal(0, 0.1 * std, size)
        
    elif data_type == "exponential":
        scale = kwargs.get("scale", 1.0)
        features = np.random.exponential(scale, size)
        targets = np.log(features + 1) + np.random.normal(0, 0.1, size)
        
    elif data_type == "sine":
        frequency = kwargs.get("frequency", 1.0)
        amplitude = kwargs.get("amplitude", 1.0)
        noise = kwargs.get("noise", 0.1)
        
        x = np.linspace(0, 4 * np.pi, size)
        features = x
        targets = amplitude * np.sin(frequency * x) + np.random.normal(0, noise, size)
        
    elif data_type == "polynomial":
        degree = kwargs.get("degree", 2)
        noise = kwargs.get("noise", 0.1)
        
        x = np.linspace(-2, 2, size)
        coeffs = np.random.randn(degree + 1)
        targets = sum(coeff * x**i for i, coeff in enumerate(coeffs))
        targets += np.random.normal(0, noise, size)
        features = x
        
    else:
        raise ValueError(f"Unknown data_type: {data_type}")
    
    logger.info(f"Generated {data_type} data with {size} samples")
    return features.reshape(-1, 1), targets


def validate_data(data: np.ndarray, 
                 targets: Optional[np.ndarray] = None,
                 min_samples: int = 10) -> Dict[str, Any]:
    """Validate data quality and characteristics
    
    Args:
        data: Input features
        targets: Target values (optional)
        min_samples: Minimum required samples
    
    Returns:
        Validation report dictionary
    """
    report = {
        "valid": True,
        "issues": [],
        "statistics": {},
        "recommendations": []
    }
    
    # Basic shape validation
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    n_samples, n_features = data.shape
    report["statistics"]["n_samples"] = n_samples
    report["statistics"]["n_features"] = n_features
    
    # Check minimum samples
    if n_samples < min_samples:
        report["valid"] = False
        report["issues"].append(f"Insufficient samples: {n_samples} < {min_samples}")
    
    # Check for NaN or infinite values
    if np.any(np.isnan(data)):
        report["valid"] = False
        report["issues"].append("Data contains NaN values")
    
    if np.any(np.isinf(data)):
        report["valid"] = False  
        report["issues"].append("Data contains infinite values")
    
    # Statistical checks
    for i in range(n_features):
        feature_data = data[:, i]
        feature_stats = {
            "mean": float(np.mean(feature_data)),
            "std": float(np.std(feature_data)),
            "min": float(np.min(feature_data)),
            "max": float(np.max(feature_data)),
            "unique_values": len(np.unique(feature_data))
        }
        
        # Check for constant features
        if feature_stats["std"] == 0:
            report["issues"].append(f"Feature {i} is constant")
            report["recommendations"].append(f"Consider removing feature {i}")
        
        # Check for low variance
        if feature_stats["std"] < 1e-6:
            report["issues"].append(f"Feature {i} has very low variance")
        
        report["statistics"][f"feature_{i}"] = feature_stats
    
    # Target validation if provided
    if targets is not None:
        if len(targets) != n_samples:
            report["valid"] = False
            report["issues"].append(f"Targets length ({len(targets)}) != samples ({n_samples})")
        else:
            target_stats = {
                "mean": float(np.mean(targets)),
                "std": float(np.std(targets)), 
                "min": float(np.min(targets)),
                "max": float(np.max(targets))
            }
            report["statistics"]["targets"] = target_stats
            
            if np.any(np.isnan(targets)):
                report["valid"] = False
                report["issues"].append("Targets contain NaN values")
            
            if np.any(np.isinf(targets)):
                report["valid"] = False
                report["issues"].append("Targets contain infinite values")
    
    # Data distribution analysis
    data_flat = data.flatten()
    skewness = _calculate_skewness(data_flat)
    kurtosis = _calculate_kurtosis(data_flat)
    
    report["statistics"]["distribution"] = {
        "skewness": float(skewness),
        "kurtosis": float(kurtosis)
    }
    
    if abs(skewness) > 2:
        report["recommendations"].append("Data is highly skewed, consider transformation")
    
    if abs(kurtosis) > 3:
        report["recommendations"].append("Data has heavy tails, consider outlier detection")
    
    logger.info(f"Data validation complete: {report['valid']}, {len(report['issues'])} issues found")
    return report


def _calculate_skewness(data: np.ndarray) -> float:
    """Calculate skewness of data"""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    return np.mean(((data - mean) / std) ** 3)


def _calculate_kurtosis(data: np.ndarray) -> float:
    """Calculate kurtosis of data"""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    return np.mean(((data - mean) / std) ** 4) - 3