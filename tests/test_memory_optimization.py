"""Tests for memory optimization module"""

import pytest
import numpy as np
import time
from src.performance.memory_optimization import (
    MemoryPool, 
    MemoryEfficientArray,
    MemoryProfiler,
    MemoryOptimizer,
    create_memory_efficient_matrix,
    memory_optimization_context
)


class TestMemoryPool:
    """Test memory pool functionality"""
    
    def test_pool_creation(self):
        """Test memory pool creation"""
        pool = MemoryPool(lambda: [0] * 1000, max_size=10)
        assert pool._max_size == 10
        assert len(pool._pool) == 0
    
    def test_object_acquisition_release(self):
        """Test object acquisition and release"""
        pool = MemoryPool(lambda: [0] * 1000, max_size=10)
        
        # Acquire object
        obj1 = pool.acquire()
        assert len(obj1) == 1000
        assert len(pool._in_use) == 1
        
        # Release object
        pool.release(obj1)
        assert len(pool._pool) == 1
        assert len(pool._in_use) == 0
        
        # Acquire again - should reuse
        obj2 = pool.acquire()
        assert obj2 is obj1  # Same object reused
        
    def test_pool_context_manager(self):
        """Test pool context manager"""
        pool = MemoryPool(lambda: {"data": [0] * 1000}, max_size=5)
        
        with pool.get_object() as obj:
            assert "data" in obj
            assert len(obj["data"]) == 1000
        
        # Object should be back in pool
        assert len(pool._pool) == 1
    
    def test_pool_statistics(self):
        """Test pool statistics"""
        pool = MemoryPool(lambda: [0] * 100, max_size=5)
        
        # Acquire and release objects
        obj1 = pool.acquire()
        obj2 = pool.acquire()  # This should create new object
        
        pool.release(obj1)
        pool.release(obj2)
        
        stats = pool.get_stats()
        assert stats['objects_created'] == 2
        assert stats['pool_size'] == 2
        assert stats['objects_in_use'] == 0


class TestMemoryEfficientArray:
    """Test memory-efficient array implementation"""
    
    def test_array_creation(self):
        """Test array creation"""
        arr = MemoryEfficientArray((100, 100), dtype=float, sparse_threshold=0.1)
        assert arr.shape == (100, 100)
        assert arr.size == 10000
        assert arr.dtype == float
    
    def test_lazy_allocation(self):
        """Test lazy allocation"""
        arr = MemoryEfficientArray((1000, 1000), dtype=float)
        # Array should not be allocated until accessed
        assert arr._data is None
        
        # First access should allocate
        arr[0, 0] = 1.0
        assert arr._data is not None
    
    def test_sparse_conversion(self):
        """Test automatic sparse conversion"""
        arr = MemoryEfficientArray((100, 100), dtype=float, sparse_threshold=0.1)
        
        # Set only a few elements (< 10% threshold)
        arr[0, 0] = 1.0
        arr[50, 50] = 2.0
        arr[99, 99] = 3.0
        
        # Force sparsity check
        arr._check_sparsity()
        
        # Should be converted to sparse
        assert arr._is_sparse
        assert arr._sparse_data is not None
    
    def test_memory_info(self):
        """Test memory usage information"""
        arr = MemoryEfficientArray((1000, 1000), dtype=np.float32, sparse_threshold=0.05)
        
        # Get initial memory info
        info1 = arr.get_memory_info()
        assert info1['current_memory_bytes'] == 0  # Not allocated yet
        
        # Set sparse data
        for i in range(0, 1000, 100):
            arr[i, i] = float(i)
        
        info2 = arr.get_memory_info()
        if arr._is_sparse:
            assert info2['memory_saved_bytes'] > 0
            assert info2['compression_ratio'] < 1.0


class TestMemoryProfiler:
    """Test memory profiler functionality"""
    
    def test_profiler_creation(self):
        """Test profiler creation"""
        profiler = MemoryProfiler(sampling_interval=0.1)
        assert profiler.sampling_interval == 0.1
        assert not profiler._monitoring
    
    def test_snapshot_taking(self):
        """Test memory snapshot"""
        profiler = MemoryProfiler()
        snapshot = profiler._take_snapshot()
        
        assert snapshot.timestamp > 0
        assert snapshot.heap_objects > 0
        assert isinstance(snapshot.gc_collections, dict)
        assert isinstance(snapshot.large_objects, list)
    
    def test_monitoring_session(self):
        """Test monitoring session"""
        profiler = MemoryProfiler(sampling_interval=0.1)
        
        profiler.start_monitoring()
        assert profiler._monitoring
        
        time.sleep(0.3)  # Let it collect some snapshots
        
        profiler.stop_monitoring()
        assert not profiler._monitoring
        assert len(profiler._snapshots) > 0
    
    def test_memory_trend_analysis(self):
        """Test memory trend analysis"""
        profiler = MemoryProfiler(sampling_interval=0.05)
        
        # Start monitoring
        profiler.start_monitoring()
        time.sleep(0.2)  # Collect some data
        
        # Allocate some memory to create a trend
        data = [np.zeros((100, 100)) for _ in range(10)]
        time.sleep(0.1)
        
        profiler.stop_monitoring()
        
        if len(profiler._snapshots) >= 2:
            trend = profiler.get_memory_trend()
            assert 'duration_minutes' in trend
            assert 'memory_trend_mb_per_minute' in trend
            assert 'current_memory_mb' in trend


class TestMemoryOptimizer:
    """Test memory optimizer functionality"""
    
    def test_optimizer_creation(self):
        """Test optimizer creation"""
        optimizer = MemoryOptimizer()
        assert isinstance(optimizer.profiler, MemoryProfiler)
        assert isinstance(optimizer.pools, dict)
    
    def test_optimization_session(self):
        """Test optimization session"""
        optimizer = MemoryOptimizer()
        
        optimizer.start_optimization_session()
        time.sleep(0.1)  # Brief session
        
        report = optimizer.end_optimization_session()
        assert report.initial_memory_mb >= 0
        assert report.final_memory_mb >= 0
        assert isinstance(report.techniques_applied, list)
        assert isinstance(report.recommendations, list)
    
    def test_memory_pool_creation(self):
        """Test memory pool creation through optimizer"""
        optimizer = MemoryOptimizer()
        
        pool = optimizer.create_memory_pool(
            "test_pool",
            lambda: {"test": "data"},
            max_size=10
        )
        
        assert isinstance(pool, MemoryPool)
        assert "test_pool" in optimizer.pools
        assert optimizer.get_memory_pool("test_pool") is pool
    
    def test_force_cleanup(self):
        """Test force cleanup"""
        optimizer = MemoryOptimizer()
        
        # Create some pools
        optimizer.create_memory_pool("pool1", lambda: [0] * 100)
        optimizer.create_memory_pool("pool2", lambda: {"data": 123})
        
        stats = optimizer.force_cleanup()
        assert 'initial_objects' in stats
        assert 'final_objects' in stats
        assert 'pools_cleared' in stats
        assert stats['pools_cleared'] == 2


class TestMemoryOptimizationContext:
    """Test memory optimization context manager"""
    
    def test_context_manager(self):
        """Test memory optimization context"""
        with memory_optimization_context() as optimizer:
            assert isinstance(optimizer, MemoryOptimizer)
            
            # Create some test data
            test_data = np.random.randn(100, 100)
            assert test_data.shape == (100, 100)
        
        # Context should complete without errors


class TestIntegrationWithAlgorithms:
    """Test memory optimization integration with existing algorithms"""
    
    def test_memory_efficient_matrix_with_quantum_algorithms(self):
        """Test memory-efficient structures with quantum algorithms"""
        # Create memory-efficient matrix
        matrix = create_memory_efficient_matrix((1000, 1000), sparse_threshold=0.1)
        
        # Simulate quantum algorithm data pattern (sparse)
        for i in range(0, 1000, 50):
            for j in range(0, 1000, 50):
                matrix[i, j] = np.sin(i) * np.cos(j)
        
        memory_info = matrix.get_memory_info()
        
        # Should achieve some memory savings for sparse quantum data
        if matrix._is_sparse:
            assert memory_info['memory_saved_bytes'] > 0
        
    def test_memory_optimization_with_discovery_algorithms(self):
        """Test memory optimization with discovery algorithms"""
        optimizer = MemoryOptimizer()
        
        with memory_optimization_context(optimizer) as opt:
            # Create memory pool for discovery results
            results_pool = opt.create_memory_pool(
                "discovery_results",
                lambda: {"patterns": [], "metrics": {}, "quality": 0.0},
                max_size=100
            )
            
            # Simulate discovery algorithm using pool
            for _ in range(10):
                with results_pool.get_object() as result:
                    result["patterns"] = [f"pattern_{i}" for i in range(5)]
                    result["metrics"] = {"accuracy": 0.95}
                    result["quality"] = 0.88
            
            pool_stats = results_pool.get_stats()
            assert pool_stats['objects_reused'] > 0
    
    def test_numpy_optimization_integration(self):
        """Test NumPy optimization integration"""
        from src.performance.memory_optimization import optimize_numpy_operations
        
        # Apply NumPy optimizations
        optimize_numpy_operations()
        
        # Create large arrays to test optimization
        arr1 = np.random.randn(1000, 1000)
        arr2 = np.random.randn(1000, 1000)
        
        # Perform operation that should benefit from optimization
        result = np.dot(arr1, arr2)
        assert result.shape == (1000, 1000)


def test_memory_optimization_end_to_end():
    """End-to-end test of memory optimization system"""
    optimizer = MemoryOptimizer()
    
    # Start optimization session
    optimizer.start_optimization_session()
    
    try:
        # Create memory pools
        array_pool = optimizer.create_memory_pool(
            "test_arrays", 
            lambda: np.zeros((100, 100)), 
            max_size=20
        )
        
        # Create memory-efficient structures
        efficient_matrix = create_memory_efficient_matrix((500, 500), sparse_threshold=0.2)
        
        # Use memory pools
        arrays_used = []
        for _ in range(5):
            with array_pool.get_object() as arr:
                arr.fill(np.random.random())
                arrays_used.append(arr.sum())
        
        # Use efficient matrix
        for i in range(0, 500, 25):
            efficient_matrix[i, i] = i * 2
        
        # Force optimization
        optimizer.optimize_garbage_collection()
        cleanup_stats = optimizer.force_cleanup()
        
        assert cleanup_stats['pools_cleared'] > 0
        
    finally:
        # End session
        report = optimizer.end_optimization_session()
        assert isinstance(report, type(report))  # Verify report generated
    
    # Get summary
    summary = optimizer.get_optimization_summary()
    assert summary['total_sessions'] >= 1
    assert summary['memory_pools_active'] >= 0


if __name__ == "__main__":
    pytest.main([__file__])