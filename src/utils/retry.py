"""Retry mechanisms with exponential backoff and jitter"""

import time
import random
import functools
import logging
from typing import Any, Callable, List, Optional, Type, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_max: float = 1.0  # seconds
    retryable_exceptions: tuple = (Exception,)
    non_retryable_exceptions: tuple = ()


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted"""
    def __init__(self, attempts: int, last_exception: Exception):
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(f"Retry exhausted after {attempts} attempts. Last error: {last_exception}")


class RetryHandler:
    """Handles retry logic with exponential backoff"""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.attempt_history: List[dict] = []
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        self.attempt_history.clear()
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                
                # Record successful attempt
                self.attempt_history.append({
                    "attempt": attempt,
                    "success": True,
                    "duration": time.time() - start_time,
                    "exception": None
                })
                
                if attempt > 1:
                    logger.info(f"Function succeeded on attempt {attempt}/{self.config.max_attempts}")
                
                return result
                
            except Exception as e:
                last_exception = e
                execution_time = time.time() - start_time
                
                # Record failed attempt
                self.attempt_history.append({
                    "attempt": attempt,
                    "success": False,
                    "duration": execution_time,
                    "exception": str(e),
                    "exception_type": type(e).__name__
                })
                
                # Check if exception is non-retryable
                if self._is_non_retryable(e):
                    logger.warning(f"Non-retryable exception on attempt {attempt}: {e}")
                    raise e
                
                # Check if exception is retryable
                if not self._is_retryable(e):
                    logger.warning(f"Non-retryable exception type on attempt {attempt}: {type(e).__name__}")
                    raise e
                
                # If this is the last attempt, raise the exception
                if attempt == self.config.max_attempts:
                    logger.error(f"All {self.config.max_attempts} retry attempts exhausted")
                    raise RetryExhaustedError(attempt, e)
                
                # Calculate delay for next attempt
                delay = self._calculate_delay(attempt)
                logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {delay:.2f}s...")
                time.sleep(delay)
        
        # This should never be reached, but just in case
        raise RetryExhaustedError(self.config.max_attempts, last_exception)
    
    def _is_retryable(self, exception: Exception) -> bool:
        """Check if exception is retryable"""
        return isinstance(exception, self.config.retryable_exceptions)
    
    def _is_non_retryable(self, exception: Exception) -> bool:
        """Check if exception is explicitly non-retryable"""
        return isinstance(exception, self.config.non_retryable_exceptions)
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the next attempt"""
        # Exponential backoff
        delay = min(
            self.config.base_delay * (self.config.exponential_base ** (attempt - 1)),
            self.config.max_delay
        )
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            jitter = random.uniform(0, self.config.jitter_max)
            delay += jitter
        
        return delay
    
    def get_attempt_history(self) -> List[dict]:
        """Get history of all attempts"""
        return self.attempt_history.copy()


def retry(max_attempts: int = 3,
          base_delay: float = 1.0,
          max_delay: float = 60.0,
          exponential_base: float = 2.0,
          jitter: bool = True,
          jitter_max: float = 1.0,
          retryable_exceptions: tuple = (Exception,),
          non_retryable_exceptions: tuple = ()) -> Callable:
    """Decorator to add retry logic to a function"""
    
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        jitter_max=jitter_max,
        retryable_exceptions=retryable_exceptions,
        non_retryable_exceptions=non_retryable_exceptions
    )
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            handler = RetryHandler(config)
            return handler.execute(func, *args, **kwargs)
        
        return wrapper
    return decorator


def retry_with_circuit_breaker(name: str,
                              max_attempts: int = 3,
                              base_delay: float = 1.0,
                              failure_threshold: int = 5,
                              recovery_timeout: float = 60.0) -> Callable:
    """Combined retry and circuit breaker decorator"""
    from .circuit_breaker import circuit_breaker, CircuitBreakerError
    
    def decorator(func: Callable) -> Callable:
        # Apply circuit breaker first, then retry
        circuit_protected = circuit_breaker(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=Exception
        )(func)
        
        # Apply retry logic, but don't retry circuit breaker errors
        retry_protected = retry(
            max_attempts=max_attempts,
            base_delay=base_delay,
            non_retryable_exceptions=(CircuitBreakerError,)
        )(circuit_protected)
        
        return retry_protected
    
    return decorator


# Convenience functions for common retry patterns

def retry_on_connection_error(max_attempts: int = 5, base_delay: float = 2.0) -> Callable:
    """Retry decorator specifically for connection-related errors"""
    return retry(
        max_attempts=max_attempts,
        base_delay=base_delay,
        retryable_exceptions=(ConnectionError, TimeoutError, OSError),
        non_retryable_exceptions=(ValueError, TypeError)
    )


def retry_on_temporary_error(max_attempts: int = 3, base_delay: float = 1.0) -> Callable:
    """Retry decorator for temporary errors"""
    from .error_handling import DataValidationError, ModelError
    
    return retry(
        max_attempts=max_attempts,
        base_delay=base_delay,
        retryable_exceptions=(RuntimeError, IOError, ConnectionError),
        non_retryable_exceptions=(DataValidationError, ModelError, ValueError, TypeError)
    )


def execute_with_retry(func: Callable, 
                      config: Optional[RetryConfig] = None,
                      *args, **kwargs) -> Any:
    """Execute a function with retry logic without using decorator"""
    if config is None:
        config = RetryConfig()
    
    handler = RetryHandler(config)
    return handler.execute(func, *args, **kwargs)