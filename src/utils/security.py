"""Security and validation utilities for the AI Science Platform"""

import numpy as np
import hashlib
import secrets
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Security configuration settings"""
    max_file_size_mb: int = 100
    allowed_file_extensions: List[str] = None
    enable_data_validation: bool = True
    enable_input_sanitization: bool = True
    enable_audit_logging: bool = True
    max_array_size: int = 10_000_000  # 10M elements
    max_memory_usage_mb: int = 1000
    
    def __post_init__(self):
        if self.allowed_file_extensions is None:
            self.allowed_file_extensions = ['.npy', '.csv', '.json', '.txt']  # Removed .pkl for security


class SecurityValidator:
    """Security validation and sanitization for scientific data"""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.audit_log = []
        
    def validate_array_input(self, data: np.ndarray, name: str = "data") -> Tuple[bool, str]:
        """Validate numpy array for security and safety
        
        Args:
            data: Input numpy array
            name: Name for logging purposes
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check if input is actually a numpy array
            if not isinstance(data, np.ndarray):
                return False, f"{name} must be a numpy array, got {type(data)}"
            
            # Check array size limits
            if data.size > self.config.max_array_size:
                return False, f"{name} size {data.size} exceeds maximum {self.config.max_array_size}"
            
            # Check memory usage estimate
            memory_mb = data.nbytes / (1024 * 1024)
            if memory_mb > self.config.max_memory_usage_mb:
                return False, f"{name} memory usage {memory_mb:.1f}MB exceeds limit {self.config.max_memory_usage_mb}MB"
            
            # Check for malicious values
            if np.any(np.isinf(data)):
                if not self.config.enable_data_validation:
                    logger.warning(f"{name} contains infinite values")
                else:
                    return False, f"{name} contains infinite values"
            
            if np.any(np.isnan(data)):
                if not self.config.enable_data_validation:
                    logger.warning(f"{name} contains NaN values")
                else:
                    return False, f"{name} contains NaN values"
            
            # Check for extreme values that might indicate an attack
            if data.dtype in [np.float32, np.float64]:
                max_val = np.max(np.abs(data))
                if max_val > 1e10:
                    logger.warning(f"{name} contains extremely large values (max: {max_val})")
            
            self._audit_log(f"Array validation passed for {name}: shape={data.shape}, dtype={data.dtype}")
            return True, ""
            
        except Exception as e:
            error_msg = f"Array validation failed for {name}: {str(e)}"
            self._audit_log(error_msg, level="ERROR")
            return False, error_msg
    
    def sanitize_string_input(self, input_str: str, max_length: int = 1000) -> str:
        """Sanitize string input for security
        
        Args:
            input_str: Input string
            max_length: Maximum allowed length
            
        Returns:
            Sanitized string
        """
        if not isinstance(input_str, str):
            input_str = str(input_str)
        
        # Truncate if too long
        if len(input_str) > max_length:
            input_str = input_str[:max_length]
            logger.warning(f"String truncated to {max_length} characters")
        
        # Remove potentially dangerous characters
        if self.config.enable_input_sanitization:
            # Remove null bytes and control characters
            input_str = ''.join(char for char in input_str 
                               if ord(char) >= 32 or char in ['\n', '\r', '\t'])
            
            # Remove potentially dangerous patterns
            dangerous_patterns = ['<script', 'javascript:', 'data:', 'vbscript:', 'onload=']
            for pattern in dangerous_patterns:
                input_str = input_str.replace(pattern, '')
        
        return input_str
    
    def validate_file_path(self, file_path: Union[str, Path]) -> Tuple[bool, str]:
        """Validate file path for security
        
        Args:
            file_path: Path to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            path = Path(file_path)
            
            # Check for path traversal attempts
            if '..' in str(path):
                return False, "Path traversal detected"
            
            # Check file extension if file exists or has extension
            if path.suffix and path.suffix.lower() not in self.config.allowed_file_extensions:
                return False, f"File extension {path.suffix} not allowed"
            
            # Check file size if file exists
            if path.exists() and path.is_file():
                size_mb = path.stat().st_size / (1024 * 1024)
                if size_mb > self.config.max_file_size_mb:
                    return False, f"File size {size_mb:.1f}MB exceeds limit {self.config.max_file_size_mb}MB"
            
            self._audit_log(f"File path validation passed: {file_path}")
            return True, ""
            
        except Exception as e:
            error_msg = f"File path validation failed: {str(e)}"
            self._audit_log(error_msg, level="ERROR")
            return False, error_msg
    
    def validate_model_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate model parameters for security
        
        Args:
            params: Dictionary of model parameters
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check parameter types and ranges
            safe_numeric_types = (int, float, np.integer, np.floating)
            
            for key, value in params.items():
                # Sanitize key
                if not isinstance(key, str) or len(key) > 100:
                    return False, f"Invalid parameter key: {key}"
                
                # Check value types
                if isinstance(value, (list, tuple)):
                    if len(value) > 1000:
                        return False, f"Parameter {key} has too many values"
                    for item in value:
                        if not isinstance(item, safe_numeric_types):
                            return False, f"Parameter {key} contains unsafe value: {item}"
                
                elif isinstance(value, dict):
                    if len(value) > 100:
                        return False, f"Parameter {key} has too many keys"
                
                elif isinstance(value, str):
                    if len(value) > 1000:
                        return False, f"Parameter {key} string too long"
                
                elif isinstance(value, safe_numeric_types):
                    # Check for reasonable numeric ranges
                    if abs(float(value)) > 1e10:
                        return False, f"Parameter {key} value too large: {value}"
                
                else:
                    return False, f"Parameter {key} has unsupported type: {type(value)}"
            
            self._audit_log(f"Model parameter validation passed for {len(params)} parameters")
            return True, ""
            
        except Exception as e:
            error_msg = f"Model parameter validation failed: {str(e)}"
            self._audit_log(error_msg, level="ERROR")
            return False, error_msg
    
    def generate_secure_filename(self, base_name: str, extension: str = ".json") -> str:
        """Generate a secure filename with timestamp and random component
        
        Args:
            base_name: Base name for the file
            extension: File extension
            
        Returns:
            Secure filename
        """
        # Sanitize base name
        safe_base = self.sanitize_string_input(base_name, max_length=50)
        safe_base = ''.join(c for c in safe_base if c.isalnum() or c in ['_', '-'])
        
        # Add timestamp and random component
        timestamp = int(time.time())
        random_suffix = secrets.token_hex(4)
        
        return f"{safe_base}_{timestamp}_{random_suffix}{extension}"
    
    def compute_data_hash(self, data: np.ndarray) -> str:
        """Compute secure hash of data for integrity verification
        
        Args:
            data: Input data array
            
        Returns:
            SHA-256 hash of the data
        """
        data_bytes = data.tobytes()
        return hashlib.sha256(data_bytes).hexdigest()
    
    def verify_data_integrity(self, data: np.ndarray, expected_hash: str) -> bool:
        """Verify data integrity using hash comparison
        
        Args:
            data: Data to verify
            expected_hash: Expected hash value
            
        Returns:
            True if data integrity is verified
        """
        actual_hash = self.compute_data_hash(data)
        is_valid = actual_hash == expected_hash
        
        if is_valid:
            self._audit_log("Data integrity verification passed")
        else:
            self._audit_log("Data integrity verification FAILED", level="ERROR")
        
        return is_valid
    
    def safe_pickle_load(self, file_path: Union[str, Path]) -> Tuple[Any, bool, str]:
        """DEPRECATED: Use safe_json_load instead for security
        
        Args:
            file_path: Path to pickle file
            
        Returns:
            Tuple of (data, success, error_message)
        """
        logger.warning("safe_pickle_load is deprecated due to security concerns. Use safe_json_load instead.")
        return None, False, "Pickle loading is disabled for security reasons. Use JSON format instead."
    
    def safe_json_load(self, file_path: Union[str, Path]) -> Tuple[Any, bool, str]:
        """Safely load JSON file with validation
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Tuple of (data, success, error_message)
        """
        try:
            # Validate file path first
            is_valid, error = self.validate_file_path(file_path)
            if not is_valid:
                return None, False, error
            
            path = Path(file_path)
            if not path.exists():
                return None, False, f"File does not exist: {file_path}"
            
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self._audit_log(f"Safely loaded JSON file: {file_path}")
            return data, True, ""
            
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in file {file_path}: {str(e)}"
            self._audit_log(error_msg, level="ERROR")
            return None, False, error_msg
        except Exception as e:
            error_msg = f"Failed to load JSON file {file_path}: {str(e)}"
            self._audit_log(error_msg, level="ERROR")
            return None, False, error_msg
    
    def _audit_log(self, message: str, level: str = "INFO"):
        """Add entry to audit log"""
        if self.config.enable_audit_logging:
            entry = {
                'timestamp': time.time(),
                'level': level,
                'message': message
            }
            self.audit_log.append(entry)
            
            # Log to standard logger as well
            if level == "ERROR":
                logger.error(message)
            elif level == "WARNING":
                logger.warning(message)
            else:
                logger.info(message)
    
    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get the current audit log"""
        return self.audit_log.copy()
    
    def clear_audit_log(self):
        """Clear the audit log"""
        self.audit_log.clear()


# Global security validator instance
_global_validator = None

def get_security_validator() -> SecurityValidator:
    """Get global security validator instance"""
    global _global_validator
    if _global_validator is None:
        _global_validator = SecurityValidator()
    return _global_validator


def validate_input(data: np.ndarray, name: str = "data") -> np.ndarray:
    """Convenience function to validate and return array input
    
    Args:
        data: Input array
        name: Name for logging
        
    Returns:
        Validated array
        
    Raises:
        ValueError: If validation fails
    """
    validator = get_security_validator()
    is_valid, error = validator.validate_array_input(data, name)
    
    if not is_valid:
        raise ValueError(f"Input validation failed: {error}")
    
    return data


def sanitize_string(input_str: str, max_length: int = 1000) -> str:
    """Convenience function to sanitize string input
    
    Args:
        input_str: Input string
        max_length: Maximum length
        
    Returns:
        Sanitized string
    """
    validator = get_security_validator()
    return validator.sanitize_string_input(input_str, max_length)