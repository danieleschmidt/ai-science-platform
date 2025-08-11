"""Comprehensive validation utilities with security measures"""

import re
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


class SecurityError(Exception):
    """Custom exception for security-related validation failures"""
    pass


class ValidationMixin:
    """Mixin class providing common validation methods"""
    
    def validate_positive_int(self, value: Any, field_name: str) -> int:
        """Validate that a value is a positive integer"""
        if not isinstance(value, int):
            raise ValidationError(f"{field_name} must be an integer, got {type(value).__name__}")
        if value <= 0:
            raise ValidationError(f"{field_name} must be positive, got {value}")
        return value
    
    def validate_probability(self, value: Any, field_name: str, 
                           min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Validate that a value is a valid probability"""
        if not isinstance(value, (int, float)):
            raise ValidationError(f"{field_name} must be numeric, got {type(value).__name__}")
        if not (min_val <= value <= max_val):
            raise ValidationError(f"{field_name} must be between {min_val} and {max_val}, got {value}")
        return float(value)


def validate_input_data(data: Any, 
                       expected_type: type,
                       min_length: Optional[int] = None,
                       max_length: Optional[int] = None,
                       allow_none: bool = False,
                       field_name: str = "data") -> bool:
    """Validate input data with comprehensive checks
    
    Args:
        data: Data to validate
        expected_type: Expected type of the data
        min_length: Minimum length for sequences
        max_length: Maximum length for sequences  
        allow_none: Whether None values are allowed
        field_name: Name of the field for error messages
    
    Returns:
        True if validation passes
        
    Raises:
        ValidationError: If validation fails
        SecurityError: If security checks fail
    """
    # Check for None values
    if data is None:
        if allow_none:
            return True
        raise ValidationError(f"{field_name} cannot be None")
    
    # Type checking
    if not isinstance(data, expected_type):
        raise ValidationError(
            f"{field_name} must be of type {expected_type.__name__}, got {type(data).__name__}"
        )
    
    # Length checks for sequences
    if hasattr(data, '__len__'):
        data_len = len(data)
        
        if min_length is not None and data_len < min_length:
            raise ValidationError(
                f"{field_name} length {data_len} is below minimum {min_length}"
            )
        
        if max_length is not None and data_len > max_length:
            raise SecurityError(
                f"{field_name} length {data_len} exceeds maximum {max_length} (potential DoS)"
            )
    
    # Security checks for strings
    if isinstance(data, str):
        _validate_string_security(data, field_name)
    
    # Security checks for file paths
    if isinstance(data, (str, Path)) and ('/' in str(data) or '\\' in str(data)):
        _validate_file_path_security(str(data), field_name)
    
    logger.debug(f"Validation passed for {field_name}")
    return True


def _validate_string_security(text: str, field_name: str) -> None:
    """Validate string for security issues"""
    # Check for potential injection attacks
    dangerous_patterns = [
        r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',  # Script tags
        r'javascript\s*:',  # JavaScript URLs
        r'on\w+\s*=',  # Event handlers
        r'eval\s*\(',  # eval calls
        r'exec\s*\(',  # exec calls
        r'\bselect\b.*\bfrom\b',  # SQL injection patterns
        r'\bunion\b.*\bselect\b',  # SQL union
        r'(\.\./){2,}',  # Directory traversal
        r'\.\.[\\/]',  # Directory traversal
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            raise SecurityError(f"Potentially malicious content detected in {field_name}")
    
    # Check for excessive length (potential DoS)
    if len(text) > 100000:  # 100KB limit
        raise SecurityError(f"{field_name} exceeds safe string length limit")


def _validate_file_path_security(filepath: str, field_name: str) -> None:
    """Validate file path for security issues"""
    # Normalize path
    normalized_path = Path(filepath).resolve()
    
    # Check for directory traversal
    if '..' in filepath:
        raise SecurityError(f"Directory traversal detected in {field_name}: {filepath}")
    
    # Check for absolute paths outside allowed directories
    allowed_dirs = [Path.cwd(), Path("/tmp"), Path("./logs"), Path("./experiment_results")]
    
    is_allowed = False
    for allowed_dir in allowed_dirs:
        try:
            normalized_path.relative_to(allowed_dir.resolve())
            is_allowed = True
            break
        except ValueError:
            continue
    
    if not is_allowed:
        raise SecurityError(f"File path outside allowed directories: {filepath}")


def validate_experiment_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate experiment configuration with security checks
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Sanitized configuration dictionary
        
    Raises:
        ValidationError: If configuration is invalid
        SecurityError: If security checks fail
    """
    required_fields = ['name', 'description', 'parameters', 'metrics_to_track']
    
    # Check required fields
    for field in required_fields:
        if field not in config:
            raise ValidationError(f"Required field '{field}' missing from config")
    
    # Validate each field
    validate_input_data(config['name'], str, min_length=1, max_length=100, field_name='name')
    validate_input_data(config['description'], str, min_length=1, max_length=1000, field_name='description')
    validate_input_data(config['parameters'], dict, field_name='parameters')
    validate_input_data(config['metrics_to_track'], list, min_length=1, max_length=50, field_name='metrics_to_track')
    
    # Validate parameters dictionary
    if len(config['parameters']) > 100:
        raise SecurityError("Too many parameters (potential DoS)")
    
    for key, value in config['parameters'].items():
        validate_input_data(key, str, min_length=1, max_length=50, field_name=f'parameter key: {key}')
        
        # Validate parameter values
        if isinstance(value, str):
            validate_input_data(value, str, max_length=1000, field_name=f'parameter value: {key}')
        elif isinstance(value, (int, float)):
            if abs(value) > 1e10:
                raise ValidationError(f"Parameter value {key} is too large: {value}")
    
    # Validate metrics list
    for metric in config['metrics_to_track']:
        validate_input_data(metric, str, min_length=1, max_length=50, field_name=f'metric: {metric}')
    
    # Validate optional fields
    if 'num_runs' in config:
        num_runs = config['num_runs']
        if not isinstance(num_runs, int) or num_runs < 1 or num_runs > 1000:
            raise ValidationError(f"num_runs must be an integer between 1 and 1000, got {num_runs}")
    
    if 'seed' in config and config['seed'] is not None:
        seed = config['seed']
        if not isinstance(seed, int) or seed < 0 or seed > 2**32:
            raise ValidationError(f"seed must be an integer between 0 and 2^32, got {seed}")
    
    logger.info(f"Experiment configuration validated: {config['name']}")
    return config


def validate_discovery_parameters(threshold: float,
                                context: str = "",
                                data_shape: Optional[Tuple[int, ...]] = None) -> bool:
    """Validate discovery engine parameters
    
    Args:
        threshold: Discovery confidence threshold
        context: Context string for discovery
        data_shape: Shape of input data
        
    Returns:
        True if validation passes
        
    Raises:
        ValidationError: If parameters are invalid
    """
    # Validate threshold
    if not isinstance(threshold, (int, float)):
        raise ValidationError(f"Threshold must be numeric, got {type(threshold).__name__}")
    
    if not (0.0 <= threshold <= 1.0):
        raise ValidationError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
    
    # Validate context string
    if context:
        validate_input_data(context, str, max_length=500, field_name='context')
    
    # Validate data shape
    if data_shape is not None:
        if not isinstance(data_shape, tuple):
            raise ValidationError(f"Data shape must be a tuple, got {type(data_shape).__name__}")
        
        if len(data_shape) == 0:
            raise ValidationError("Data shape cannot be empty")
        
        for dim in data_shape:
            if not isinstance(dim, int) or dim <= 0:
                raise ValidationError(f"Invalid dimension in data shape: {dim}")
        
        # Check for reasonable data sizes (prevent memory exhaustion)
        total_elements = 1
        for dim in data_shape:
            total_elements *= dim
        
        if total_elements > 10_000_000:  # 10M elements max
            raise SecurityError(f"Data size too large: {total_elements} elements")
    
    logger.debug("Discovery parameters validated successfully")
    return True


def sanitize_output(data: Any, max_string_length: int = 1000) -> Any:
    """Sanitize output data for safe display/logging
    
    Args:
        data: Data to sanitize
        max_string_length: Maximum length for strings
        
    Returns:
        Sanitized data
    """
    if isinstance(data, str):
        # Truncate long strings
        if len(data) > max_string_length:
            return data[:max_string_length] + "... [TRUNCATED]"
        
        # Remove potential HTML/script content
        sanitized = re.sub(r'<[^>]*>', '', data)
        return sanitized
    
    elif isinstance(data, dict):
        return {key: sanitize_output(value, max_string_length) for key, value in data.items()}
    
    elif isinstance(data, list):
        # Limit list length
        if len(data) > 100:
            return [sanitize_output(item, max_string_length) for item in data[:100]] + ["... [TRUNCATED]"]
        return [sanitize_output(item, max_string_length) for item in data]
    
    elif isinstance(data, (int, float, bool)) or data is None:
        return data
    
    else:
        # Convert other types to string representation
        return str(data)[:max_string_length]


def validate_numeric_array(data: List[Union[int, float]], 
                         field_name: str = "data",
                         min_value: Optional[float] = None,
                         max_value: Optional[float] = None) -> bool:
    """Validate numeric array data
    
    Args:
        data: List of numeric values
        field_name: Name of the field for error messages
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        True if validation passes
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(data, (list, tuple)):
        raise ValidationError(f"{field_name} must be a list or tuple")
    
    if len(data) == 0:
        raise ValidationError(f"{field_name} cannot be empty")
    
    for i, value in enumerate(data):
        if not isinstance(value, (int, float)):
            raise ValidationError(f"{field_name}[{i}] must be numeric, got {type(value).__name__}")
        
        # Check for special float values
        if isinstance(value, float):
            if str(value).lower() in ['nan', 'inf', '-inf']:
                raise ValidationError(f"{field_name}[{i}] contains invalid value: {value}")
        
        # Range checks
        if min_value is not None and value < min_value:
            raise ValidationError(f"{field_name}[{i}] below minimum {min_value}: {value}")
        
        if max_value is not None and value > max_value:
            raise ValidationError(f"{field_name}[{i}] above maximum {max_value}: {value}")
    
    logger.debug(f"Numeric array validation passed for {field_name}")
    return True