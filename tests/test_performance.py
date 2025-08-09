"""Tests for performance modules"""

import pytest
import numpy as np
import time
import threading
import tempfile
from unittest.mock import Mock, patch

from src.performance.caching import LRUCache, CacheManager, cached_function, clear_all_caches
from src.performance.parallel import ParallelProcessor, parallel_discovery
from src.performance.profiling import PerformanceMonitor, ProfileManager, profile_function
from src.performance.resource_pool import ResourcePool, DiscoveryPool, ResourceFactory


class TestLRUCache:
    
    def test_initialization(self):
        """Test LRU cache initialization"""
        cache = LRUCache(max_size=5, ttl=10)
        assert cache.max_size == 5
        assert cache.ttl == 10
        assert cache._hits == 0
        assert cache._misses == 0
    
    def test_basic_operations(self):
        """Test basic cache operations"""
        cache = LRUCache(max_size=3, ttl=60)
        
        # Test miss
        found, value = cache.get("key1")
        assert not found
        assert value is None
        assert cache._misses == 1
        
        # Test put and hit
        cache.put("key1", "value1")
        found, value = cache.get("key1")
        assert found
        assert value == "value1"
        assert cache._hits == 1
    
    def test_size_limit(self):
        """Test cache size limit enforcement"""
        cache = LRUCache(max_size=2, ttl=60)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")  # Should evict key1
        
        found, _ = cache.get("key1")
        assert not found  # key1 should be evicted
        
        found, value = cache.get("key2")
        assert found and value == "value2"
        
        found, value = cache.get("key3")
        assert found and value == "value3"
    
    def test_ttl_expiration(self):
        """Test TTL expiration"""
        cache = LRUCache(max_size=5, ttl=0.1)  # 100ms TTL
        
        cache.put("key1", "value1")
        found, value = cache.get("key1")
        assert found and value == "value1"
        
        time.sleep(0.2)  # Wait for expiration
        
        found, _ = cache.get("key1")
        assert not found  # Should be expired
    
    def test_lru_ordering(self):
        """Test LRU ordering"""
        cache = LRUCache(max_size=2, ttl=60)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Access key1 to make it more recent
        cache.get("key1")
        
        # Add key3, should evict key2 (less recently used)
        cache.put("key3", "value3")
        
        found, _ = cache.get("key2")
        assert not found  # key2 should be evicted
        
        found, value = cache.get("key1")
        assert found and value == "value1"
    
    def test_stats(self):
        """Test cache statistics"""
        cache = LRUCache(max_size=3, ttl=60)
        
        cache.put("key1", "value1")
        cache.get("key1")  # hit
        cache.get("key2")  # miss
        
        stats = cache.stats()
        assert stats["size"] == 1
        assert stats["max_size"] == 3
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


class TestCacheManager:
    
    def test_initialization(self):
        """Test cache manager initialization"""
        cache_mgr = CacheManager(memory_cache_size=10, memory_ttl=30)
        assert cache_mgr.memory_cache.max_size == 10
        assert cache_mgr.memory_cache.ttl == 30
        assert cache_mgr._enabled == True
    
    def test_caching_workflow(self):
        """Test complete caching workflow"""
        cache_mgr = CacheManager()
        
        # Test cache miss
        found, value = cache_mgr.get("test_func", (1, 2), {"param": "value"})
        assert not found
        
        # Store result
        cache_mgr.put("test_func", (1, 2), {"param": "value"}, "result")
        
        # Test cache hit
        found, value = cache_mgr.get("test_func", (1, 2), {"param": "value"})
        assert found
        assert value == "result"
    
    def test_numpy_array_caching(self):
        """Test caching with numpy arrays"""
        cache_mgr = CacheManager()
        
        data = np.array([1, 2, 3, 4, 5])
        cache_mgr.put("array_func", (data,), {}, "array_result")
        
        found, value = cache_mgr.get("array_func", (data,), {})
        assert found
        assert value == "array_result"
    
    def test_cache_disabled(self):
        """Test cache when disabled"""
        cache_mgr = CacheManager()
        cache_mgr.disable()
        
        found, value = cache_mgr.get("func", (), {})
        assert not found
        
        cache_mgr.put("func", (), {}, "result")
        found, value = cache_mgr.get("func", (), {})
        assert not found  # Should still be disabled


class TestCachedFunction:
    
    def test_cached_function_decorator(self):
        """Test cached function decorator"""
        call_count = 0
        
        @cached_function()
        def expensive_function(x, y=10):
            nonlocal call_count
            call_count += 1
            return x * y + call_count
        
        # First call
        result1 = expensive_function(5, y=20)
        assert call_count == 1
        
        # Second call with same args (should use cache)
        result2 = expensive_function(5, y=20)
        assert call_count == 1  # Should not increment
        assert result1 == result2
        
        # Third call with different args
        result3 = expensive_function(10, y=20)
        assert call_count == 2  # Should increment
    
    def test_cached_function_disabled(self):
        """Test cached function with cache disabled"""
        call_count = 0
        
        @cached_function(cache_disabled=True)
        def function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        function(5)
        function(5)  # Same args, but cache disabled
        
        assert call_count == 2  # Should call both times


class TestParallelProcessor:
    
    def test_initialization(self):
        """Test parallel processor initialization"""
        processor = ParallelProcessor(max_workers=4, use_processes=False)
        assert processor.max_workers == 4
        assert processor.use_processes == False
    
    def test_parallel_map_simple(self):
        """Test simple parallel map operation"""
        processor = ParallelProcessor(max_workers=2)
        
        def square(x):
            return x ** 2
        
        data = [1, 2, 3, 4, 5]
        results = processor.map(square, data)
        
        assert results == [1, 4, 9, 16, 25]
    
    def test_parallel_map_with_progress(self):
        """Test parallel map with progress callback"""
        processor = ParallelProcessor(max_workers=2)
        progress_calls = []
        
        def progress_callback(completed, total):
            progress_calls.append((completed, total))
        
        def identity(x):
            return x
        
        data = [1, 2, 3]
        processor.map(identity, data, progress_callback=progress_callback)
        
        assert len(progress_calls) == 3
        assert progress_calls[-1] == (3, 3)  # Final call should be (completed, total)
    
    def test_parallel_starmap(self):
        """Test parallel starmap operation"""
        processor = ParallelProcessor(max_workers=2)
        
        def add(x, y):
            return x + y
        
        data = [(1, 2), (3, 4), (5, 6)]
        results = processor.starmap(add, data)
        
        assert results == [3, 7, 11]
    
    def test_parallel_error_handling(self):
        """Test parallel processing with errors"""
        processor = ParallelProcessor(max_workers=2)
        
        def failing_function(x):
            if x == 3:
                raise ValueError("Test error")
            return x * 2
        
        data = [1, 2, 3, 4]
        results = processor.map(failing_function, data)
        
        # Should have None for the failed task
        expected = [2, 4, None, 8]
        assert results == expected
    
    def test_single_item_optimization(self):
        """Test optimization for single item processing"""
        processor = ParallelProcessor(max_workers=4)
        
        def identity(x):
            return x
        
        data = [42]
        results = processor.map(identity, data)
        
        assert results == [42]
    
    def test_empty_data(self):
        """Test parallel processing with empty data"""
        processor = ParallelProcessor(max_workers=2)
        
        def identity(x):
            return x
        
        results = processor.map(identity, [])
        assert results == []
    
    def test_get_stats(self):
        """Test getting processing statistics"""
        processor = ParallelProcessor(max_workers=2)
        
        def identity(x):
            return x
        
        processor.map(identity, [1, 2, 3])
        stats = processor.get_stats()
        
        assert "tasks_completed" in stats
        assert "total_execution_time" in stats
        assert stats["max_workers"] == 2


class TestPerformanceMonitor:
    
    def test_basic_monitoring(self):
        """Test basic performance monitoring"""
        monitor = PerformanceMonitor()
        
        monitor.start()
        time.sleep(0.1)  # Small delay
        metrics = monitor.stop()
        
        assert metrics.execution_time >= 0.1
        assert metrics.cpu_percent >= 0
        assert metrics.memory_mb > 0
    
    def test_context_manager(self):
        """Test performance monitor as context manager"""
        monitor = PerformanceMonitor()
        
        with monitor.measure():
            time.sleep(0.05)
        
        measurements = monitor.get_measurements()
        assert len(measurements) == 1
        assert measurements[0].execution_time >= 0.05
    
    def test_multiple_measurements(self):
        """Test multiple measurements"""
        monitor = PerformanceMonitor()
        
        for _ in range(3):
            with monitor.measure():
                time.sleep(0.01)
        
        measurements = monitor.get_measurements()
        assert len(measurements) == 3
        
        monitor.clear_measurements()
        assert len(monitor.get_measurements()) == 0


class MockResource:
    """Mock resource for testing resource pool"""
    
    def __init__(self, resource_id):
        self.resource_id = resource_id
        self.is_healthy = True
    
    def do_work(self):
        return f"work_done_{self.resource_id}"
    
    def mark_unhealthy(self):
        self.is_healthy = False


class MockResourceFactory(ResourceFactory):
    """Mock factory for testing resource pool"""
    
    def __init__(self):
        self.created_count = 0
        self.destroyed_count = 0
    
    def create_resource(self):
        self.created_count += 1
        return MockResource(f"resource_{self.created_count}")
    
    def is_resource_healthy(self, resource):
        return resource.is_healthy
    
    def destroy_resource(self, resource):
        self.destroyed_count += 1
    
    def validate_resource(self, resource):
        return self.is_resource_healthy(resource)


class TestResourcePool:
    
    def test_pool_initialization(self):
        """Test resource pool initialization"""
        factory = MockResourceFactory()
        pool = ResourcePool(factory, min_size=2, max_size=5)
        
        time.sleep(0.1)  # Allow initialization
        
        metrics = pool.get_metrics()
        assert metrics.total_created >= 2  # Should create min_size resources
        assert metrics.idle_resources >= 2
    
    def test_checkout_checkin(self):
        """Test basic checkout/checkin workflow"""
        factory = MockResourceFactory()
        pool = ResourcePool(factory, min_size=1, max_size=3)
        
        time.sleep(0.1)  # Allow initialization
        
        # Checkout resource
        resource = pool.checkout()
        assert resource is not None
        assert hasattr(resource.resource, 'resource_id')
        
        metrics = pool.get_metrics()
        assert metrics.checked_out == 1
        assert metrics.active_resources == 1
        
        # Checkin resource
        pool.checkin(resource)
        
        metrics = pool.get_metrics()
        assert metrics.checked_out == 0
        assert metrics.idle_resources >= 1
    
    def test_context_manager(self):
        """Test resource pool context manager"""
        factory = MockResourceFactory()
        pool = ResourcePool(factory, min_size=1, max_size=3)
        
        time.sleep(0.1)  # Allow initialization
        
        with pool.get_resource() as resource:
            result = resource.do_work()
            assert "work_done" in result
        
        # Resource should be automatically returned
        metrics = pool.get_metrics()
        assert metrics.checked_out == 0
    
    def test_pool_exhaustion(self):
        """Test pool exhaustion handling"""
        factory = MockResourceFactory()
        pool = ResourcePool(factory, min_size=1, max_size=2, checkout_timeout=0.1)
        
        time.sleep(0.1)  # Allow initialization
        
        # Checkout all resources
        resource1 = pool.checkout()
        resource2 = pool.checkout()
        
        # Try to checkout one more (should fail)
        with pytest.raises(RuntimeError, match="Resource pool exhausted"):
            pool.checkout()
        
        # Return resources
        pool.checkin(resource1)
        pool.checkin(resource2)
    
    def test_unhealthy_resource_handling(self):
        """Test handling of unhealthy resources"""
        factory = MockResourceFactory()
        pool = ResourcePool(factory, min_size=1, max_size=3)
        
        time.sleep(0.1)  # Allow initialization
        
        resource = pool.checkout()
        resource.resource.mark_unhealthy()
        resource.mark_unhealthy()
        
        pool.checkin(resource)
        
        # Resource should be destroyed, not returned to pool
        assert factory.destroyed_count >= 1
    
    def test_pool_resize(self):
        """Test dynamic pool resizing"""
        factory = MockResourceFactory()
        pool = ResourcePool(factory, min_size=1, max_size=2)
        
        time.sleep(0.1)  # Allow initialization
        
        pool.resize(2, 5)  # Increase sizes
        
        assert pool.min_size == 2
        assert pool.max_size == 5
    
    def test_pool_shutdown(self):
        """Test pool shutdown"""
        factory = MockResourceFactory()
        pool = ResourcePool(factory, min_size=2, max_size=3)
        
        time.sleep(0.1)  # Allow initialization
        
        pool.shutdown()
        
        # Should destroy all resources
        assert factory.destroyed_count >= 2


class TestDiscoveryPool:
    
    def test_discovery_pool_creation(self):
        """Test specialized discovery pool creation"""
        pool = DiscoveryPool(discovery_threshold=0.8, min_size=1, max_size=3)
        
        time.sleep(0.1)  # Allow initialization
        
        with pool.get_resource() as discovery_engine:
            assert hasattr(discovery_engine, 'discover')
            assert hasattr(discovery_engine, 'generate_hypothesis')
            assert discovery_engine.discovery_threshold == 0.8
    
    def test_discovery_pool_functionality(self):
        """Test discovery pool functionality"""
        pool = DiscoveryPool(min_size=1, max_size=2)
        
        time.sleep(0.1)  # Allow initialization
        
        with pool.get_resource() as engine:
            # Test basic discovery functionality
            data = np.random.randn(50, 2)
            targets = np.random.randn(50)
            
            discoveries = engine.discover(data, targets, "test_context")
            assert isinstance(discoveries, list)
        
        pool.shutdown()


# Integration tests
class TestPerformanceIntegration:
    
    def test_cached_parallel_processing(self):
        """Test integration of caching with parallel processing"""
        call_count = 0
        
        @cached_function()
        def expensive_computation(x):
            nonlocal call_count
            call_count += 1
            time.sleep(0.01)  # Simulate work
            return x ** 2
        
        processor = ParallelProcessor(max_workers=2)
        
        # First run - should compute all values
        data = [1, 2, 3, 4, 1, 2]  # Some duplicates
        results1 = processor.map(expensive_computation, data)
        
        # Second run - should use cache for duplicates
        results2 = processor.map(expensive_computation, data)
        
        assert results1 == results2
        assert call_count == 4  # Only unique values should be computed
    
    def test_profiled_resource_pool(self):
        """Test integration of profiling with resource pool"""
        @profile_function(detailed=False)
        def use_resource_pool():
            factory = MockResourceFactory()
            pool = ResourcePool(factory, min_size=1, max_size=2)
            
            time.sleep(0.05)  # Allow initialization
            
            with pool.get_resource() as resource:
                return resource.do_work()
        
        result = use_resource_pool()
        assert "work_done" in result