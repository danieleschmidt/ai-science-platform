"""Comprehensive error handling utilities for the AI Science Platform"""

import logging
import traceback
import functools
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class ErrorContext:
    """Context information for error handling"""
    function_name: str
    module_name: str
    timestamp: float
    parameters: Dict[str, Any]
    error_type: str
    error_message: str
    stack_trace: str
    suggested_fix: Optional[str] = None


class PlatformError(Exception):
    """Base exception class for AI Science Platform"""
    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(message)
        self.context = context


class DataValidationError(PlatformError):
    """Exception raised for data validation errors"""
    pass


class ModelError(PlatformError):
    """Exception raised for model-related errors"""
    pass


class DiscoveryError(PlatformError):
    """Exception raised for discovery engine errors"""
    pass


class ExperimentError(PlatformError):
    """Exception raised for experiment runner errors"""
    pass


class VisualizationError(PlatformError):
    """Exception raised for visualization errors"""
    pass


class ErrorHandler:
    """Comprehensive error handling and recovery system"""
    
    def __init__(self, log_errors: bool = True, raise_on_error: bool = True):
        self.log_errors = log_errors
        self.raise_on_error = raise_on_error
        self.error_history = []
        self.recovery_strategies = {}
        self.logger = logging.getLogger(__name__)
        
        # Register default recovery strategies
        self._register_default_recovery_strategies()
    
    def handle_error(self, 
                    func_name: str, 
                    module_name: str, 
                    error: Exception, 
                    parameters: Dict[str, Any] = None,
                    recovery_strategy: str = None) -> Tuple[bool, Any]:
        """Handle an error with optional recovery
        
        Args:
            func_name: Name of function where error occurred
            module_name: Name of module
            error: The exception that occurred
            parameters: Function parameters when error occurred
            recovery_strategy: Name of recovery strategy to try
            
        Returns:
            Tuple of (recovered, result)
        """
        # Create error context
        context = ErrorContext(
            function_name=func_name,
            module_name=module_name,
            timestamp=time.time(),
            parameters=parameters or {},
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            suggested_fix=self._suggest_fix(error, func_name)
        )
        
        # Log error
        if self.log_errors:
            self.logger.error(f"Error in {module_name}.{func_name}: {str(error)}")
            self.logger.debug(f"Stack trace: {context.stack_trace}")
        
        # Store in error history
        self.error_history.append(context)
        
        # Attempt recovery if strategy provided
        if recovery_strategy and recovery_strategy in self.recovery_strategies:
            try:
                result = self.recovery_strategies[recovery_strategy](context, error)
                self.logger.info(f"Successfully recovered using strategy: {recovery_strategy}")
                return True, result
            except Exception as recovery_error:
                self.logger.error(f"Recovery strategy {recovery_strategy} failed: {str(recovery_error)}")
        
        # Raise error if configured to do so
        if self.raise_on_error:
            if isinstance(error, PlatformError):
                raise error
            else:
                # Wrap in appropriate platform error
                platform_error = self._wrap_error(error, func_name)
                platform_error.context = context
                raise platform_error
        
        return False, None
    
    def register_recovery_strategy(self, name: str, strategy: Callable):
        """Register a recovery strategy
        
        Args:
            name: Name of the strategy
            strategy: Function that takes (context, error) and returns result
        """
        self.recovery_strategies[name] = strategy
    
    def _register_default_recovery_strategies(self):
        """Register default recovery strategies"""
        
        def fallback_to_default_data(context: ErrorContext, error: Exception):
            """Generate fallback default data"""
            if 'size' in context.parameters:
                size = context.parameters['size']
                return np.random.normal(0, 1, size), np.random.normal(0, 1, size)
            else:
                return np.array([0.0]), np.array([0.0])
        
        def retry_with_reduced_complexity(context: ErrorContext, error: Exception):
            """Retry operation with reduced complexity"""
            params = context.parameters.copy()
            
            # Reduce common complexity parameters
            if 'degree' in params and params['degree'] > 1:
                params['degree'] = max(1, params['degree'] - 1)
            if 'n_iterations' in params:
                params['n_iterations'] = max(1, params['n_iterations'] // 2)
            if 'threshold' in params:
                params['threshold'] = min(0.9, params['threshold'] + 0.1)
            
            return params
        
        def graceful_degradation(context: ErrorContext, error: Exception):
            """Provide gracefully degraded results"""
            return {
                'status': 'degraded',
                'error': str(error),
                'partial_results': True,
                'timestamp': context.timestamp
            }
        
        self.register_recovery_strategy('fallback_data', fallback_to_default_data)
        self.register_recovery_strategy('reduce_complexity', retry_with_reduced_complexity)
        self.register_recovery_strategy('graceful_degradation', graceful_degradation)
    
    def _suggest_fix(self, error: Exception, func_name: str) -> Optional[str]:
        """Suggest potential fixes for common errors"""
        error_str = str(error).lower()
        
        if 'nan' in error_str:
            return "Check for NaN values in input data. Use np.isnan() to detect and handle them."
        
        elif 'inf' in error_str or 'infinite' in error_str:
            return "Check for infinite values in input data. Use np.isinf() to detect and handle them."
        
        elif 'singular matrix' in error_str or 'linalg' in error_str:
            return "Matrix is singular or ill-conditioned. Try adding regularization or using pseudo-inverse."
        
        elif 'memory' in error_str:
            return "Insufficient memory. Try reducing data size or processing in batches."
        
        elif 'shape' in error_str or 'dimension' in error_str:
            return "Array shape mismatch. Check input dimensions and reshape if necessary."
        
        elif 'index' in error_str or 'bounds' in error_str:
            return "Index out of bounds. Check array sizes and index ranges."
        
        elif 'convergence' in error_str:
            return "Algorithm failed to converge. Try increasing iterations or adjusting tolerance."
        
        elif 'file' in error_str and 'not found' in error_str:
            return "File not found. Check file path and permissions."
        
        else:
            return f"Consider checking input parameters for {func_name}() and consulting documentation."
    
    def _wrap_error(self, error: Exception, func_name: str) -> PlatformError:
        """Wrap exception in appropriate platform error"""
        if any(keyword in func_name.lower() for keyword in ['data', 'validate', 'generate']):
            return DataValidationError(str(error))
        elif any(keyword in func_name.lower() for keyword in ['model', 'train', 'predict', 'fit']):
            return ModelError(str(error))
        elif any(keyword in func_name.lower() for keyword in ['discover', 'hypothesis', 'engine']):
            return DiscoveryError(str(error))
        elif any(keyword in func_name.lower() for keyword in ['experiment', 'run', 'analyze']):
            return ExperimentError(str(error))
        elif any(keyword in func_name.lower() for keyword in ['plot', 'visualiz', 'chart', 'graph']):
            return VisualizationError(str(error))
        else:
            return PlatformError(str(error))
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors"""
        if not self.error_history:
            return {'total_errors': 0, 'error_types': {}, 'common_issues': []}
        
        error_types = {}
        common_issues = []
        
        for context in self.error_history:
            error_type = context.error_type
            error_types[error_type] = error_types.get(error_type, 0) + 1
            
            if context.suggested_fix and context.suggested_fix not in common_issues:
                common_issues.append(context.suggested_fix)
        
        return {
            'total_errors': len(self.error_history),
            'error_types': error_types,
            'common_issues': common_issues[:5],  # Top 5
            'recent_errors': [
                {
                    'function': ctx.function_name,
                    'error': ctx.error_message,
                    'timestamp': ctx.timestamp
                } for ctx in self.error_history[-5:]  # Last 5
            ]
        }
    
    def clear_error_history(self):
        """Clear error history"""
        self.error_history.clear()


def robust_execution(recovery_strategy: str = None, 
                    error_handler: ErrorHandler = None):
    """Decorator for robust function execution with error handling
    
    Args:
        recovery_strategy: Name of recovery strategy to use
        error_handler: Custom error handler instance
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            handler = error_handler or get_default_error_handler()
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Extract function parameters for context
                import inspect
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                
                # Convert numpy arrays to descriptions to avoid logging large arrays
                safe_params = {}
                for k, v in bound_args.arguments.items():
                    if isinstance(v, np.ndarray):
                        safe_params[k] = f"array(shape={v.shape}, dtype={v.dtype})"
                    elif hasattr(v, '__len__') and len(v) > 100:
                        safe_params[k] = f"large_object(type={type(v).__name__}, len={len(v)})"
                    else:
                        safe_params[k] = str(v)[:100]  # Truncate long strings
                
                recovered, result = handler.handle_error(
                    func_name=func.__name__,
                    module_name=func.__module__,
                    error=e,
                    parameters=safe_params,
                    recovery_strategy=recovery_strategy
                )
                
                if recovered:
                    return result
                
                # If we get here, error was not handled (handler configured not to raise)
                return None
                
        return wrapper
    return decorator


def safe_array_operation(func: Callable) -> Callable:
    """Decorator for safe numpy array operations"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Validate array inputs
        for arg in args:
            if isinstance(arg, np.ndarray):
                if np.any(np.isnan(arg)):
                    raise DataValidationError(f"Input array contains NaN values in {func.__name__}")
                if np.any(np.isinf(arg)):
                    raise DataValidationError(f"Input array contains infinite values in {func.__name__}")
        
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                if np.any(np.isnan(v)):
                    raise DataValidationError(f"Parameter '{k}' contains NaN values in {func.__name__}")
                if np.any(np.isinf(v)):
                    raise DataValidationError(f"Parameter '{k}' contains infinite values in {func.__name__}")
        
        return func(*args, **kwargs)
    
    return wrapper


# Global error handler
_global_error_handler = None

def get_default_error_handler() -> ErrorHandler:
    """Get the default global error handler"""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler


def configure_error_handling(log_errors: bool = True, raise_on_error: bool = True):
    """Configure global error handling behavior"""
    global _global_error_handler
    _global_error_handler = ErrorHandler(log_errors=log_errors, raise_on_error=raise_on_error)


class ValidationMixin:
    """Mixin class to add validation methods to other classes"""
    
    def validate_positive_number(self, value: Union[int, float], name: str) -> Union[int, float]:
        """Validate that a number is positive"""
        if not isinstance(value, (int, float)):
            raise DataValidationError(f"{name} must be a number, got {type(value)}")
        if value <= 0:
            raise DataValidationError(f"{name} must be positive, got {value}")
        return value
    
    def validate_probability(self, value: float, name: str) -> float:
        """Validate that a value is a valid probability [0, 1]"""
        if not isinstance(value, (int, float)):
            raise DataValidationError(f"{name} must be a number, got {type(value)}")
        if not 0 <= value <= 1:
            raise DataValidationError(f"{name} must be between 0 and 1, got {value}")
        return float(value)
    
    def validate_array_shape(self, array: np.ndarray, expected_shape: Tuple, name: str) -> np.ndarray:
        """Validate array has expected shape"""
        if not isinstance(array, np.ndarray):
            raise DataValidationError(f"{name} must be numpy array, got {type(array)}")
        
        if array.shape != expected_shape:
            raise DataValidationError(f"{name} shape {array.shape} != expected {expected_shape}")
        
        return array