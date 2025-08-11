"""Circuit breaker pattern for resilient service calls"""

import time
import threading
import functools
from typing import Any, Callable, Dict, Optional
from enum import Enum
import logging
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    expected_exception: type = Exception
    success_threshold: int = 1  # Successful calls needed in half-open to close


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    """Circuit breaker implementation for resilient operations"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.recent_calls = deque(maxlen=100)  # Track recent call outcomes
        self._lock = threading.Lock()
        
        logger.info(f"Circuit breaker initialized: threshold={config.failure_threshold}, timeout={config.recovery_timeout}s")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker"""
        with self._lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time < self.config.recovery_timeout:
                    self._record_call(False, "circuit_open")
                    raise CircuitBreakerError("Circuit breaker is OPEN")
                else:
                    # Time to try half-open
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
        
        # Execute the function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.config.expected_exception as e:
            self._on_failure(e)
            raise
    
    def _on_success(self) -> None:
        """Handle successful call"""
        with self._lock:
            self._record_call(True, "success")
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    logger.info("Circuit breaker CLOSED after successful recovery")
            
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                if self.failure_count > 0:
                    self.failure_count = max(0, self.failure_count - 1)
    
    def _on_failure(self, error: Exception) -> None:
        """Handle failed call"""
        with self._lock:
            self._record_call(False, str(error))
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state in [CircuitState.CLOSED, CircuitState.HALF_OPEN]:
                if self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitState.OPEN
                    logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")
    
    def _record_call(self, success: bool, outcome: str) -> None:
        """Record call outcome for monitoring"""
        self.recent_calls.append({
            "timestamp": time.time(),
            "success": success,
            "outcome": outcome,
            "state": self.state.value
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        with self._lock:
            recent_successes = sum(1 for call in self.recent_calls if call["success"])
            recent_failures = len(self.recent_calls) - recent_successes
            
            success_rate = recent_successes / len(self.recent_calls) if self.recent_calls else 0.0
            
            return {
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_failure_time": self.last_failure_time,
                "time_since_last_failure": time.time() - self.last_failure_time,
                "recent_calls": len(self.recent_calls),
                "recent_success_rate": success_rate,
                "recent_successes": recent_successes,
                "recent_failures": recent_failures
            }
    
    def force_open(self) -> None:
        """Force circuit breaker to open state"""
        with self._lock:
            self.state = CircuitState.OPEN
            self.last_failure_time = time.time()
            logger.warning("Circuit breaker forced OPEN")
    
    def force_close(self) -> None:
        """Force circuit breaker to closed state"""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            logger.info("Circuit breaker forced CLOSED")


class CircuitBreakerRegistry:
    """Registry to manage multiple circuit breakers"""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()
    
    def get_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create a circuit breaker"""
        with self._lock:
            if name not in self._breakers:
                if config is None:
                    config = CircuitBreakerConfig()
                self._breakers[name] = CircuitBreaker(config)
                logger.info(f"Created circuit breaker: {name}")
            return self._breakers[name]
    
    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers"""
        with self._lock:
            return {name: breaker.get_statistics() 
                   for name, breaker in self._breakers.items()}
    
    def reset_all(self) -> None:
        """Reset all circuit breakers to closed state"""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.force_close()
            logger.info("All circuit breakers reset")


# Global registry
_registry = CircuitBreakerRegistry()


def circuit_breaker(name: str, 
                   failure_threshold: int = 5,
                   recovery_timeout: float = 60.0,
                   expected_exception: type = Exception) -> Callable:
    """Decorator to add circuit breaker to a function"""
    
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        expected_exception=expected_exception
    )
    
    def decorator(func: Callable) -> Callable:
        breaker = _registry.get_breaker(name, config)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        
        return wrapper
    return decorator


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get circuit breaker instance"""
    return _registry.get_breaker(name, config)


def get_all_circuit_breaker_stats() -> Dict[str, Dict[str, Any]]:
    """Get all circuit breaker statistics"""
    return _registry.get_all_statistics()