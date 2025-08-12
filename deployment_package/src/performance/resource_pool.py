"""Resource pooling and connection management for scalability"""

import time
import logging
import threading
from typing import Any, Optional, Callable, Dict, List, Generic, TypeVar
from dataclasses import dataclass
from queue import Queue, Empty, Full
from contextlib import contextmanager
import weakref
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class ResourceMetrics:
    """Metrics for resource pool monitoring"""
    total_created: int = 0
    active_resources: int = 0
    idle_resources: int = 0
    checked_out: int = 0
    wait_time_total: float = 0.0
    avg_wait_time: float = 0.0
    pool_hits: int = 0
    pool_misses: int = 0


class PooledResource(Generic[T]):
    """Wrapper for pooled resources with lifecycle tracking"""
    
    def __init__(self, resource: T, resource_id: str, created_at: float):
        self.resource = resource
        self.resource_id = resource_id
        self.created_at = created_at
        self.last_used = created_at
        self.use_count = 0
        self.is_healthy = True
        self._lock = threading.Lock()
    
    def mark_used(self) -> None:
        """Mark resource as used"""
        with self._lock:
            self.last_used = time.time()
            self.use_count += 1
    
    def mark_unhealthy(self) -> None:
        """Mark resource as unhealthy"""
        with self._lock:
            self.is_healthy = False
    
    def age_seconds(self) -> float:
        """Get resource age in seconds"""
        return time.time() - self.created_at
    
    def idle_time_seconds(self) -> float:
        """Get idle time in seconds"""
        return time.time() - self.last_used
    
    def __enter__(self):
        self.mark_used()
        return self.resource
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.mark_unhealthy()


class ResourceFactory(ABC, Generic[T]):
    """Abstract factory for creating pooled resources"""
    
    @abstractmethod
    def create_resource(self) -> T:
        """Create a new resource instance"""
        pass
    
    @abstractmethod
    def is_resource_healthy(self, resource: T) -> bool:
        """Check if resource is healthy and can be reused"""
        pass
    
    @abstractmethod
    def destroy_resource(self, resource: T) -> None:
        """Clean up resource when removing from pool"""
        pass
    
    def validate_resource(self, resource: T) -> bool:
        """Validate resource before use (optional override)"""
        return self.is_resource_healthy(resource)


class ResourcePool(Generic[T]):
    """Generic resource pool with advanced lifecycle management"""
    
    def __init__(self,
                 factory: ResourceFactory[T],
                 min_size: int = 1,
                 max_size: int = 10,
                 max_idle_time: int = 300,  # 5 minutes
                 max_age: int = 3600,       # 1 hour
                 health_check_interval: int = 60,
                 checkout_timeout: float = 30.0):
        
        self.factory = factory
        self.min_size = min_size
        self.max_size = max_size
        self.max_idle_time = max_idle_time
        self.max_age = max_age
        self.health_check_interval = health_check_interval
        self.checkout_timeout = checkout_timeout
        
        self._pool = Queue(maxsize=max_size)
        self._checked_out = set()
        self._metrics = ResourceMetrics()
        self._lock = threading.RLock()
        self._shutdown = False
        
        # Start background maintenance
        self._maintenance_thread = threading.Thread(target=self._maintenance_loop, daemon=True)
        self._maintenance_thread.start()
        
        # Initialize minimum pool size
        self._initialize_pool()
        
        logger.info(f"Resource pool initialized: min={min_size}, max={max_size}")
    
    def _initialize_pool(self) -> None:
        """Initialize pool with minimum number of resources"""
        for _ in range(self.min_size):
            try:
                resource = self._create_pooled_resource()
                self._pool.put_nowait(resource)
            except Exception as e:
                logger.error(f"Failed to initialize resource in pool: {e}")
    
    def _create_pooled_resource(self) -> PooledResource[T]:
        """Create a new pooled resource"""
        with self._lock:
            resource_id = f"resource_{self._metrics.total_created}"
            raw_resource = self.factory.create_resource()
            pooled_resource = PooledResource(raw_resource, resource_id, time.time())
            
            self._metrics.total_created += 1
            self._metrics.idle_resources += 1
            
            logger.debug(f"Created new pooled resource: {resource_id}")
            return pooled_resource
    
    @contextmanager
    def get_resource(self):
        """Get resource from pool with automatic return"""
        resource = self.checkout()
        try:
            yield resource.resource
        finally:
            self.checkin(resource)
    
    def checkout(self) -> PooledResource[T]:
        """Check out a resource from the pool"""
        start_time = time.time()
        
        try:
            # Try to get resource from pool
            try:
                pooled_resource = self._pool.get(timeout=self.checkout_timeout)
                with self._lock:
                    self._metrics.idle_resources -= 1
                    self._metrics.checked_out += 1
                    self._metrics.pool_hits += 1
                
            except Empty:
                # Pool is empty, try to create new resource
                if self._get_total_resources() < self.max_size:
                    pooled_resource = self._create_pooled_resource()
                    with self._lock:
                        self._metrics.idle_resources -= 1
                        self._metrics.checked_out += 1
                        self._metrics.pool_misses += 1
                else:
                    raise RuntimeError(f"Resource pool exhausted (max_size={self.max_size})")
            
            # Validate resource health
            if not self._is_resource_valid(pooled_resource):
                self._destroy_pooled_resource(pooled_resource)
                # Recursively try again
                return self.checkout()
            
            # Track resource
            with self._lock:
                self._checked_out.add(pooled_resource)
                self._metrics.active_resources += 1
            
            pooled_resource.mark_used()
            
            # Update metrics
            wait_time = time.time() - start_time
            with self._lock:
                self._metrics.wait_time_total += wait_time
                self._update_avg_wait_time()
            
            logger.debug(f"Checked out resource: {pooled_resource.resource_id}")
            return pooled_resource
        
        except Exception as e:
            logger.error(f"Failed to checkout resource: {e}")
            raise
    
    def checkin(self, pooled_resource: PooledResource[T]) -> None:
        """Return a resource to the pool"""
        try:
            with self._lock:
                if pooled_resource in self._checked_out:
                    self._checked_out.remove(pooled_resource)
                    self._metrics.active_resources -= 1
                    self._metrics.checked_out -= 1
                else:
                    logger.warning(f"Resource not in checked out set: {pooled_resource.resource_id}")
                    return
            
            # Check if resource is still healthy
            if (pooled_resource.is_healthy and 
                self._is_resource_valid(pooled_resource) and
                not self._should_retire_resource(pooled_resource)):
                
                try:
                    self._pool.put_nowait(pooled_resource)
                    with self._lock:
                        self._metrics.idle_resources += 1
                    
                    logger.debug(f"Returned resource to pool: {pooled_resource.resource_id}")
                
                except Full:
                    # Pool is full, destroy resource
                    logger.debug(f"Pool full, destroying resource: {pooled_resource.resource_id}")
                    self._destroy_pooled_resource(pooled_resource)
            
            else:
                # Resource is unhealthy or expired, destroy it
                logger.debug(f"Destroying unhealthy/expired resource: {pooled_resource.resource_id}")
                self._destroy_pooled_resource(pooled_resource)
        
        except Exception as e:
            logger.error(f"Failed to checkin resource: {e}")
            self._destroy_pooled_resource(pooled_resource)
    
    def _is_resource_valid(self, pooled_resource: PooledResource[T]) -> bool:
        """Check if pooled resource is valid for use"""
        try:
            return (pooled_resource.is_healthy and 
                   self.factory.validate_resource(pooled_resource.resource))
        except Exception as e:
            logger.warning(f"Resource validation failed: {e}")
            return False
    
    def _should_retire_resource(self, pooled_resource: PooledResource[T]) -> bool:
        """Check if resource should be retired due to age or idle time"""
        return (pooled_resource.age_seconds() > self.max_age or
                pooled_resource.idle_time_seconds() > self.max_idle_time)
    
    def _destroy_pooled_resource(self, pooled_resource: PooledResource[T]) -> None:
        """Safely destroy a pooled resource"""
        try:
            self.factory.destroy_resource(pooled_resource.resource)
            with self._lock:
                if self._metrics.idle_resources > 0:
                    self._metrics.idle_resources -= 1
            
            logger.debug(f"Destroyed resource: {pooled_resource.resource_id}")
        
        except Exception as e:
            logger.error(f"Failed to destroy resource {pooled_resource.resource_id}: {e}")
    
    def _get_total_resources(self) -> int:
        """Get total number of resources (idle + active)"""
        with self._lock:
            return self._metrics.idle_resources + self._metrics.active_resources
    
    def _update_avg_wait_time(self) -> None:
        """Update average wait time"""
        total_checkouts = self._metrics.pool_hits + self._metrics.pool_misses
        if total_checkouts > 0:
            self._metrics.avg_wait_time = self._metrics.wait_time_total / total_checkouts
    
    def _maintenance_loop(self) -> None:
        """Background maintenance loop"""
        while not self._shutdown:
            try:
                self._perform_maintenance()
                time.sleep(self.health_check_interval)
            
            except Exception as e:
                logger.error(f"Maintenance loop error: {e}")
                time.sleep(self.health_check_interval)
    
    def _perform_maintenance(self) -> None:
        """Perform pool maintenance tasks"""
        # Check idle resources for expiration
        expired_resources = []
        
        # Temporarily drain pool to check resources
        temp_resources = []
        while True:
            try:
                resource = self._pool.get_nowait()
                if self._should_retire_resource(resource):
                    expired_resources.append(resource)
                else:
                    temp_resources.append(resource)
            
            except Empty:
                break
        
        # Return non-expired resources
        for resource in temp_resources:
            try:
                self._pool.put_nowait(resource)
            except Full:
                expired_resources.append(resource)
        
        # Destroy expired resources
        for resource in expired_resources:
            self._destroy_pooled_resource(resource)
        
        # Ensure minimum pool size
        current_total = self._get_total_resources()
        if current_total < self.min_size:
            needed = self.min_size - current_total
            for _ in range(needed):
                try:
                    resource = self._create_pooled_resource()
                    self._pool.put_nowait(resource)
                except Exception as e:
                    logger.error(f"Failed to maintain minimum pool size: {e}")
                    break
        
        if expired_resources:
            logger.info(f"Pool maintenance: removed {len(expired_resources)} expired resources")
    
    def get_metrics(self) -> ResourceMetrics:
        """Get current pool metrics"""
        with self._lock:
            return ResourceMetrics(
                total_created=self._metrics.total_created,
                active_resources=self._metrics.active_resources,
                idle_resources=self._metrics.idle_resources,
                checked_out=self._metrics.checked_out,
                wait_time_total=self._metrics.wait_time_total,
                avg_wait_time=self._metrics.avg_wait_time,
                pool_hits=self._metrics.pool_hits,
                pool_misses=self._metrics.pool_misses
            )
    
    def resize(self, new_min_size: int, new_max_size: int) -> None:
        """Dynamically resize the pool"""
        with self._lock:
            if new_max_size < new_min_size:
                raise ValueError("Max size cannot be less than min size")
            
            old_min, old_max = self.min_size, self.max_size
            self.min_size = new_min_size
            self.max_size = new_max_size
            
            logger.info(f"Pool resized: {old_min}-{old_max} -> {new_min_size}-{new_max_size}")
    
    def shutdown(self) -> None:
        """Shutdown the resource pool"""
        logger.info("Shutting down resource pool")
        self._shutdown = True
        
        # Wait for maintenance thread
        if self._maintenance_thread.is_alive():
            self._maintenance_thread.join(timeout=5)
        
        # Destroy all resources
        while True:
            try:
                resource = self._pool.get_nowait()
                self._destroy_pooled_resource(resource)
            except Empty:
                break
        
        # Destroy checked out resources (when they're returned)
        # Note: In production, you'd want to track these and force cleanup
        
        logger.info("Resource pool shutdown complete")


class DiscoveryEngineFactory(ResourceFactory):
    """Factory for creating discovery engine instances"""
    
    def __init__(self, discovery_threshold: float = 0.7):
        self.discovery_threshold = discovery_threshold
    
    def create_resource(self):
        """Create a new discovery engine"""
        from ..algorithms.discovery import DiscoveryEngine
        return DiscoveryEngine(discovery_threshold=self.discovery_threshold)
    
    def is_resource_healthy(self, resource) -> bool:
        """Check if discovery engine is healthy"""
        try:
            # Simple health check - try to create a hypothesis
            import numpy as np
            test_data = np.random.randn(10, 1)
            resource.generate_hypothesis(test_data, "health_check")
            return True
        except Exception:
            return False
    
    def destroy_resource(self, resource) -> None:
        """Clean up discovery engine"""
        # Discovery engines don't need special cleanup
        pass


class ModelFactory(ResourceFactory):
    """Factory for creating model instances"""
    
    def __init__(self, model_class: type, model_config: Dict[str, Any] = None):
        self.model_class = model_class
        self.model_config = model_config or {}
    
    def create_resource(self):
        """Create a new model instance"""
        return self.model_class(**self.model_config)
    
    def is_resource_healthy(self, resource) -> bool:
        """Check if model is healthy"""
        try:
            # Check if model has required attributes/methods
            return hasattr(resource, 'predict') and hasattr(resource, 'train')
        except Exception:
            return False
    
    def destroy_resource(self, resource) -> None:
        """Clean up model"""
        if hasattr(resource, 'cleanup'):
            try:
                resource.cleanup()
            except Exception as e:
                logger.warning(f"Model cleanup failed: {e}")


# Convenience classes
class DiscoveryPool(ResourcePool):
    """Specialized resource pool for discovery engines"""
    
    def __init__(self, 
                 discovery_threshold: float = 0.7,
                 min_size: int = 2,
                 max_size: int = 10,
                 **kwargs):
        
        factory = DiscoveryEngineFactory(discovery_threshold)
        super().__init__(factory, min_size, max_size, **kwargs)


class ModelPool(ResourcePool):
    """Specialized resource pool for ML models"""
    
    def __init__(self,
                 model_class: type,
                 model_config: Dict[str, Any] = None,
                 min_size: int = 1,
                 max_size: int = 5,
                 **kwargs):
        
        factory = ModelFactory(model_class, model_config)
        super().__init__(factory, min_size, max_size, **kwargs)


# Global pools registry
_resource_pools: Dict[str, ResourcePool] = {}
_pools_lock = threading.Lock()


def get_resource_pool(pool_name: str) -> Optional[ResourcePool]:
    """Get named resource pool"""
    with _pools_lock:
        return _resource_pools.get(pool_name)


def register_resource_pool(pool_name: str, pool: ResourcePool) -> None:
    """Register a named resource pool"""
    with _pools_lock:
        _resource_pools[pool_name] = pool
        logger.info(f"Registered resource pool: {pool_name}")


def shutdown_all_pools() -> None:
    """Shutdown all registered resource pools"""
    with _pools_lock:
        for name, pool in _resource_pools.items():
            try:
                pool.shutdown()
                logger.info(f"Shutdown resource pool: {name}")
            except Exception as e:
                logger.error(f"Failed to shutdown pool {name}: {e}")
        
        _resource_pools.clear()


# Auto-scaling pool manager
class AutoScalingPoolManager:
    """Manages multiple pools with auto-scaling capabilities"""
    
    def __init__(self, scale_up_threshold: float = 0.8, scale_down_threshold: float = 0.3):
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self._pools: Dict[str, ResourcePool] = {}
        self._scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self._shutdown = False
        self._scaling_thread.start()
    
    def add_pool(self, name: str, pool: ResourcePool) -> None:
        """Add pool to auto-scaling management"""
        self._pools[name] = pool
        logger.info(f"Added pool to auto-scaling: {name}")
    
    def _scaling_loop(self) -> None:
        """Auto-scaling monitoring loop"""
        while not self._shutdown:
            try:
                for name, pool in self._pools.items():
                    self._check_and_scale(name, pool)
                
                time.sleep(30)  # Check every 30 seconds
            
            except Exception as e:
                logger.error(f"Auto-scaling loop error: {e}")
                time.sleep(30)
    
    def _check_and_scale(self, name: str, pool: ResourcePool) -> None:
        """Check if pool needs scaling"""
        try:
            metrics = pool.get_metrics()
            total_resources = metrics.active_resources + metrics.idle_resources
            
            if total_resources == 0:
                return
            
            utilization = metrics.active_resources / total_resources
            
            # Scale up if utilization is high
            if (utilization > self.scale_up_threshold and 
                total_resources < pool.max_size):
                
                new_max = min(pool.max_size, total_resources + 2)
                pool.resize(pool.min_size, new_max)
                logger.info(f"Scaled up pool {name}: utilization={utilization:.2f}")
            
            # Scale down if utilization is low
            elif (utilization < self.scale_down_threshold and 
                  total_resources > pool.min_size):
                
                new_max = max(pool.min_size, total_resources - 1)
                pool.resize(pool.min_size, new_max)
                logger.info(f"Scaled down pool {name}: utilization={utilization:.2f}")
        
        except Exception as e:
            logger.error(f"Failed to scale pool {name}: {e}")
    
    def shutdown(self) -> None:
        """Shutdown auto-scaling manager"""
        self._shutdown = True
        if self._scaling_thread.is_alive():
            self._scaling_thread.join(timeout=5)