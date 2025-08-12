"""
Robust Framework - Enhanced Error Handling, Logging, and Security
Generation 2: MAKE IT ROBUST
"""

import logging
import functools
import time
import traceback
import psutil
import hashlib
import json
from typing import Any, Dict, List, Optional, Callable, Union
from pathlib import Path
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
import contextlib


@dataclass
class SecurityConfig:
    """Security configuration for the platform"""
    max_memory_mb: int = 1024
    max_execution_time_seconds: int = 300
    allowed_file_extensions: List[str] = None
    max_file_size_mb: int = 100
    enable_audit_logging: bool = True
    rate_limit_requests_per_minute: int = 100
    
    def __post_init__(self):
        if self.allowed_file_extensions is None:
            self.allowed_file_extensions = ['.py', '.json', '.csv', '.txt', '.md']


@dataclass
class MonitoringMetrics:
    """System monitoring metrics"""
    cpu_usage: float
    memory_usage_mb: float
    execution_time: float
    error_count: int
    success_count: int
    timestamp: datetime


class SecurityError(Exception):
    """Security-related exceptions"""
    pass


class ResourceExhaustionError(Exception):
    """Resource exhaustion exceptions"""
    pass


class RobustLogger:
    """Enhanced logging with security and performance monitoring"""
    
    def __init__(self, name: str, log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_format = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_format = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s'
            )
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)
        
        # Security audit handler
        self.audit_logs = []
        self.performance_metrics = []
    
    def info(self, message: str, **kwargs):
        """Log info message with context"""
        self._log_with_context('info', message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context"""
        self._log_with_context('warning', message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with context"""
        self._log_with_context('error', message, **kwargs)
    
    def security_audit(self, event: str, details: Dict[str, Any]):
        """Log security-related events"""
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'event': event,
            'details': details,
            'thread_id': threading.current_thread().ident
        }
        self.audit_logs.append(audit_entry)
        self.logger.warning(f"SECURITY_AUDIT: {event} | {details}")
    
    def performance_metric(self, metric: MonitoringMetrics):
        """Log performance metrics"""
        self.performance_metrics.append(metric)
        self.logger.info(
            f"PERFORMANCE: CPU={metric.cpu_usage:.1f}% | "
            f"Memory={metric.memory_usage_mb:.1f}MB | "
            f"Time={metric.execution_time:.3f}s | "
            f"Errors={metric.error_count} | "
            f"Success={metric.success_count}"
        )
    
    def _log_with_context(self, level: str, message: str, **kwargs):
        """Log message with additional context"""
        context = {
            'thread_id': threading.current_thread().ident,
            'process_id': psutil.Process().pid,
            'memory_mb': psutil.Process().memory_info().rss / 1024 / 1024,
            **kwargs
        }
        
        formatted_message = f"{message} | Context: {context}"
        getattr(self.logger, level)(formatted_message)


class ResourceMonitor:
    """Monitor system resources and enforce limits"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
    def check_resources(self):
        """Check if resource limits are exceeded"""
        current_time = time.time()
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Check execution time
        if current_time - self.start_time > self.config.max_execution_time_seconds:
            raise ResourceExhaustionError(
                f"Execution time exceeded {self.config.max_execution_time_seconds}s"
            )
        
        # Check memory usage
        if current_memory > self.config.max_memory_mb:
            raise ResourceExhaustionError(
                f"Memory usage exceeded {self.config.max_memory_mb}MB (current: {current_memory:.1f}MB)"
            )
    
    def get_metrics(self) -> MonitoringMetrics:
        """Get current resource metrics"""
        current_time = time.time()
        process = psutil.Process()
        
        return MonitoringMetrics(
            cpu_usage=process.cpu_percent(),
            memory_usage_mb=process.memory_info().rss / 1024 / 1024,
            execution_time=current_time - self.start_time,
            error_count=0,  # Will be updated by error tracking
            success_count=0,  # Will be updated by success tracking
            timestamp=datetime.now()
        )


class InputValidator:
    """Comprehensive input validation with security checks"""
    
    @staticmethod
    def validate_string(value: Any, max_length: int = 1000, field_name: str = "input") -> str:
        """Validate string input with security checks"""
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be a string, got {type(value).__name__}")
        
        if len(value) > max_length:
            raise SecurityError(f"{field_name} exceeds maximum length {max_length}")
        
        # Check for potential injection attacks
        dangerous_patterns = [
            '<script', 'javascript:', r'on\w+\s*=', r'eval\s*\(', r'exec\s*\(',
            r'\bselect\b.*\bfrom\b', r'\bunion\b.*\bselect\b', r'(\.\./){2,}'
        ]
        
        import re
        for pattern in dangerous_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                raise SecurityError(f"Potentially malicious content detected in {field_name}")
        
        return value
    
    @staticmethod
    def validate_numeric(value: Any, min_val: Optional[float] = None, 
                        max_val: Optional[float] = None, field_name: str = "input") -> float:
        """Validate numeric input with range checks"""
        if not isinstance(value, (int, float)):
            raise ValueError(f"{field_name} must be numeric, got {type(value).__name__}")
        
        # Check for special values
        if isinstance(value, float) and (value != value or abs(value) == float('inf')):
            raise ValueError(f"{field_name} contains invalid float value: {value}")
        
        if min_val is not None and value < min_val:
            raise ValueError(f"{field_name} must be >= {min_val}, got {value}")
        
        if max_val is not None and value > max_val:
            raise ValueError(f"{field_name} must be <= {max_val}, got {value}")
        
        return float(value)
    
    @staticmethod
    def validate_file_path(path: Union[str, Path], field_name: str = "file_path") -> Path:
        """Validate file path for security issues"""
        path_obj = Path(path)
        
        # Check for directory traversal
        if '..' in str(path):
            raise SecurityError(f"Directory traversal detected in {field_name}: {path}")
        
        # Check file extension
        config = SecurityConfig()
        if path_obj.suffix not in config.allowed_file_extensions:
            raise SecurityError(f"File extension not allowed: {path_obj.suffix}")
        
        # Check if path exists and is readable
        if path_obj.exists():
            if not path_obj.is_file():
                raise ValueError(f"{field_name} must be a file, not directory")
            
            # Check file size
            size_mb = path_obj.stat().st_size / 1024 / 1024
            if size_mb > config.max_file_size_mb:
                raise SecurityError(f"File size exceeds limit: {size_mb:.1f}MB > {config.max_file_size_mb}MB")
        
        return path_obj


class CircuitBreaker:
    """Circuit breaker pattern for resilient operations"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == 'OPEN':
            if self.last_failure_time and \
               time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'HALF_OPEN'
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise e
    
    def on_success(self):
        """Handle successful operation"""
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def on_failure(self):
        """Handle failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'


class RetryMechanism:
    """Exponential backoff retry mechanism"""
    
    @staticmethod
    def retry(max_attempts: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        """Decorator for retrying functions with exponential backoff"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        
                        if attempt == max_attempts - 1:
                            raise e
                        
                        # Exponential backoff
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        time.sleep(delay)
                
                raise last_exception
            return wrapper
        return decorator


def robust_execution(max_retries: int = 3, 
                    timeout_seconds: Optional[int] = None,
                    enable_circuit_breaker: bool = True,
                    log_performance: bool = True):
    """
    Comprehensive robust execution decorator
    
    Features:
    - Retry mechanism with exponential backoff
    - Resource monitoring and limits
    - Circuit breaker pattern
    - Performance logging
    - Security validation
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Initialize components
            logger = RobustLogger(f"robust_execution.{func.__name__}")
            config = SecurityConfig()
            if timeout_seconds:
                config.max_execution_time_seconds = timeout_seconds
            
            monitor = ResourceMonitor(config)
            circuit_breaker = CircuitBreaker() if enable_circuit_breaker else None
            
            start_time = time.time()
            error_count = 0
            
            # Validate inputs
            try:
                validated_args, validated_kwargs = _validate_function_inputs(
                    func, args, kwargs, logger
                )
            except Exception as e:
                logger.security_audit("input_validation_failed", {
                    "function": func.__name__,
                    "error": str(e)
                })
                raise
            
            # Execute with retries
            for attempt in range(max_retries):
                try:
                    # Check resources before execution
                    monitor.check_resources()
                    
                    # Execute function
                    if circuit_breaker:
                        result = circuit_breaker.call(func, *validated_args, **validated_kwargs)
                    else:
                        result = func(*validated_args, **validated_kwargs)
                    
                    # Log success
                    execution_time = time.time() - start_time
                    
                    if log_performance:
                        metrics = monitor.get_metrics()
                        metrics.execution_time = execution_time
                        metrics.error_count = error_count
                        metrics.success_count = 1
                        logger.performance_metric(metrics)
                    
                    logger.info(
                        f"Function executed successfully",
                        function=func.__name__,
                        attempt=attempt + 1,
                        execution_time=execution_time
                    )
                    
                    return result
                    
                except (ResourceExhaustionError, SecurityError) as e:
                    # Don't retry for security/resource issues
                    logger.security_audit("execution_failed", {
                        "function": func.__name__,
                        "error_type": type(e).__name__,
                        "error": str(e),
                        "attempt": attempt + 1
                    })
                    raise
                    
                except Exception as e:
                    error_count += 1
                    execution_time = time.time() - start_time
                    
                    logger.error(
                        f"Function execution failed",
                        function=func.__name__,
                        attempt=attempt + 1,
                        error=str(e),
                        error_type=type(e).__name__,
                        execution_time=execution_time
                    )
                    
                    if attempt == max_retries - 1:
                        # Final attempt failed
                        if log_performance:
                            metrics = monitor.get_metrics()
                            metrics.execution_time = execution_time
                            metrics.error_count = error_count
                            metrics.success_count = 0
                            logger.performance_metric(metrics)
                        
                        logger.security_audit("execution_failed_final", {
                            "function": func.__name__,
                            "total_attempts": max_retries,
                            "final_error": str(e),
                            "traceback": traceback.format_exc()
                        })
                        raise
                    
                    # Wait before retry (exponential backoff)
                    wait_time = min(1.0 * (2 ** attempt), 30.0)
                    time.sleep(wait_time)
            
        return wrapper
    return decorator


def _validate_function_inputs(func: Callable, args: tuple, kwargs: dict, 
                             logger: RobustLogger) -> tuple:
    """Validate function inputs for security and type safety"""
    validated_args = []
    validated_kwargs = {}
    
    # Validate positional arguments
    for i, arg in enumerate(args):
        try:
            validated_arg = _validate_single_input(arg, f"arg_{i}")
            validated_args.append(validated_arg)
        except Exception as e:
            logger.security_audit("argument_validation_failed", {
                "function": func.__name__,
                "argument_index": i,
                "argument_type": type(arg).__name__,
                "error": str(e)
            })
            raise ValueError(f"Invalid argument {i}: {str(e)}")
    
    # Validate keyword arguments
    for key, value in kwargs.items():
        try:
            # Validate key
            validated_key = InputValidator.validate_string(key, 100, f"kwarg_key_{key}")
            
            # Validate value
            validated_value = _validate_single_input(value, f"kwarg_{key}")
            validated_kwargs[validated_key] = validated_value
            
        except Exception as e:
            logger.security_audit("kwarg_validation_failed", {
                "function": func.__name__,
                "kwarg_name": key,
                "kwarg_type": type(value).__name__,
                "error": str(e)
            })
            raise ValueError(f"Invalid keyword argument {key}: {str(e)}")
    
    return tuple(validated_args), validated_kwargs


def _validate_single_input(value: Any, field_name: str) -> Any:
    """Validate a single input value"""
    import numpy as np
    
    # Handle different types
    if isinstance(value, str):
        return InputValidator.validate_string(value, field_name=field_name)
    
    elif isinstance(value, (int, float)):
        return InputValidator.validate_numeric(value, field_name=field_name)
    
    elif isinstance(value, (Path, str)) and ('/' in str(value) or '\\' in str(value)):
        # Potential file path
        return InputValidator.validate_file_path(value, field_name)
    
    elif isinstance(value, np.ndarray):
        # Validate numpy arrays
        if value.size > 10_000_000:  # 10M elements
            raise SecurityError(f"Array too large: {value.size} elements")
        
        if not np.isfinite(value).all():
            raise ValueError(f"Array contains non-finite values")
        
        return value
    
    elif isinstance(value, (list, tuple)):
        # Validate sequences
        if len(value) > 100_000:
            raise SecurityError(f"Sequence too long: {len(value)} elements")
        
        validated_items = []
        for i, item in enumerate(value):
            validated_item = _validate_single_input(item, f"{field_name}[{i}]")
            validated_items.append(validated_item)
        
        return type(value)(validated_items)
    
    elif isinstance(value, dict):
        # Validate dictionaries
        if len(value) > 10_000:
            raise SecurityError(f"Dictionary too large: {len(value)} items")
        
        validated_dict = {}
        for key, val in value.items():
            validated_key = _validate_single_input(key, f"{field_name}_key")
            validated_val = _validate_single_input(val, f"{field_name}[{key}]")
            validated_dict[validated_key] = validated_val
        
        return validated_dict
    
    else:
        # For other types, perform basic checks
        if hasattr(value, '__sizeof__'):
            size_bytes = value.__sizeof__()
            if size_bytes > 100_000_000:  # 100MB
                raise SecurityError(f"Object too large: {size_bytes} bytes")
        
        return value


@contextlib.contextmanager
def secure_operation(operation_name: str, max_time: int = 300):
    """Context manager for secure operations with monitoring"""
    logger = RobustLogger(f"secure_operation.{operation_name}")
    config = SecurityConfig()
    config.max_execution_time_seconds = max_time
    
    monitor = ResourceMonitor(config)
    start_time = time.time()
    
    try:
        logger.info(f"Starting secure operation: {operation_name}")
        yield monitor
        
        execution_time = time.time() - start_time
        logger.info(
            f"Secure operation completed successfully",
            operation=operation_name,
            execution_time=execution_time
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(
            f"Secure operation failed",
            operation=operation_name,
            error=str(e),
            error_type=type(e).__name__,
            execution_time=execution_time
        )
        
        logger.security_audit("secure_operation_failed", {
            "operation": operation_name,
            "error": str(e),
            "execution_time": execution_time,
            "traceback": traceback.format_exc()
        })
        
        raise


class HealthChecker:
    """System health monitoring and diagnostics"""
    
    def __init__(self):
        self.logger = RobustLogger("health_checker")
        self.checks = []
    
    def add_check(self, name: str, check_func: Callable[[], bool], critical: bool = False):
        """Add a health check"""
        self.checks.append({
            'name': name,
            'function': check_func,
            'critical': critical
        })
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {
            'overall_status': 'HEALTHY',
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'critical_failures': [],
            'warnings': []
        }
        
        for check in self.checks:
            try:
                with secure_operation(f"health_check_{check['name']}", max_time=30):
                    status = check['function']()
                    
                results['checks'][check['name']] = {
                    'status': 'PASS' if status else 'FAIL',
                    'critical': check['critical']
                }
                
                if not status:
                    if check['critical']:
                        results['critical_failures'].append(check['name'])
                        results['overall_status'] = 'CRITICAL'
                    else:
                        results['warnings'].append(check['name'])
                        if results['overall_status'] == 'HEALTHY':
                            results['overall_status'] = 'WARNING'
                
            except Exception as e:
                self.logger.error(f"Health check failed: {check['name']}", error=str(e))
                
                results['checks'][check['name']] = {
                    'status': 'ERROR',
                    'error': str(e),
                    'critical': check['critical']
                }
                
                if check['critical']:
                    results['critical_failures'].append(check['name'])
                    results['overall_status'] = 'CRITICAL'
        
        self.logger.info(f"Health check completed", overall_status=results['overall_status'])
        return results


# Built-in health checks
def check_memory_usage() -> bool:
    """Check if memory usage is within acceptable limits"""
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb < 2048  # 2GB limit


def check_cpu_usage() -> bool:
    """Check if CPU usage is within acceptable limits"""
    cpu_percent = psutil.cpu_percent(interval=1)
    return cpu_percent < 90  # 90% limit


def check_disk_space() -> bool:
    """Check if disk space is sufficient"""
    disk_usage = psutil.disk_usage('/')
    free_percent = (disk_usage.free / disk_usage.total) * 100
    return free_percent > 10  # 10% free space required


# Initialize default health checker
default_health_checker = HealthChecker()
default_health_checker.add_check("memory_usage", check_memory_usage, critical=True)
default_health_checker.add_check("cpu_usage", check_cpu_usage, critical=False)
default_health_checker.add_check("disk_space", check_disk_space, critical=True)