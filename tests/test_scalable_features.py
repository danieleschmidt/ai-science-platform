"""Comprehensive tests for scalable features"""

import pytest
import asyncio
import numpy as np
import time
from unittest.mock import Mock, patch

from src.performance.async_processing import AsyncTaskQueue, DiscoveryTaskProcessor
from src.performance.auto_scaling import AutoScaler, ScalingThresholds, LoadBalancer
from src.utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from src.utils.retry import RetryHandler, RetryConfig
from src.utils.backup import BackupManager


class TestAsyncProcessing:
    """Test async processing capabilities"""
    
    @pytest.mark.asyncio
    async def test_task_queue_basic_functionality(self):
        """Test basic task queue operations"""
        queue = AsyncTaskQueue(max_concurrent_tasks=2)
        await queue.start_workers()
        
        def test_func(x, y):
            return x + y
        
        # Submit task
        task_id = await queue.submit_task("test_1", test_func, 5, 10)
        assert task_id == "test_1"
        
        # Get result
        result = await queue.get_task_result(task_id, timeout=5.0)
        assert result == 15
        
        # Check stats
        stats = queue.get_stats()
        assert stats["total_completed"] == 1
        assert stats["total_failed"] == 0
        
        await queue.shutdown()
    
    @pytest.mark.asyncio
    async def test_task_queue_priority_ordering(self):
        """Test priority-based task ordering"""
        queue = AsyncTaskQueue(max_concurrent_tasks=1, enable_priorities=True)
        await queue.start_workers()
        
        results = []
        def collector(x):
            time.sleep(0.1)
            results.append(x)
            return x
        
        # Submit tasks with different priorities
        await queue.submit_task("low", collector, "low", priority=1)
        await queue.submit_task("high", collector, "high", priority=5)
        await queue.submit_task("medium", collector, "medium", priority=3)
        
        # Wait for completion
        await asyncio.sleep(0.5)
        
        # High priority should execute first
        assert results[0] == "low"  # Already started
        assert results[1] == "high"  # Highest priority
        assert results[2] == "medium"  # Medium priority
        
        await queue.shutdown()
    
    @pytest.mark.asyncio
    async def test_task_queue_error_handling(self):
        """Test error handling and retry logic"""
        queue = AsyncTaskQueue(max_concurrent_tasks=2)
        await queue.start_workers()
        
        def failing_func():
            raise ValueError("Test error")
        
        # Submit failing task
        task_id = await queue.submit_task("failing_task", failing_func, max_retries=2)
        
        # Should raise the error after retries
        with pytest.raises(ValueError):
            await queue.get_task_result(task_id, timeout=10.0)
        
        await queue.shutdown()
    
    @pytest.mark.asyncio
    async def test_discovery_task_processor(self):
        """Test specialized discovery task processor"""
        processor = DiscoveryTaskProcessor(max_concurrent=2)
        await processor.start()
        
        # Create test data batches
        data_batches = [
            np.random.randn(50, 3),
            np.random.randn(40, 3),
            np.random.randn(60, 3)
        ]
        
        discovery_params = {"discovery_threshold": 0.5}
        
        # Process batches
        results = await processor.process_discovery_batch(data_batches, discovery_params)
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, list)  # List of discoveries
        
        await processor.shutdown()


class TestAutoScaling:
    """Test auto-scaling functionality"""
    
    def test_scaling_thresholds(self):
        """Test scaling threshold configuration"""
        thresholds = ScalingThresholds(
            cpu_high=85.0,
            cpu_low=25.0,
            max_workers=10
        )
        
        assert thresholds.cpu_high == 85.0
        assert thresholds.cpu_low == 25.0
        assert thresholds.max_workers == 10
    
    def test_autoscaler_initialization(self):
        """Test autoscaler initialization"""
        thresholds = ScalingThresholds(min_workers=2, max_workers=8)
        scaler = AutoScaler(thresholds, monitoring_interval=30.0)
        
        assert scaler.current_workers == 2
        assert scaler.thresholds.max_workers == 8
        assert scaler.monitoring_interval == 30.0
    
    def test_autoscaler_manual_scaling(self):
        """Test manual scaling operations"""
        scaler = AutoScaler()
        
        # Mock scaling callbacks
        scale_up_calls = []
        scale_down_calls = []
        
        def mock_scale_up(workers):
            scale_up_calls.append(workers)
        
        def mock_scale_down(workers):
            scale_down_calls.append(workers)
        
        scaler.set_scaling_callbacks(mock_scale_up, mock_scale_down)
        
        # Test scaling up
        success = scaler.force_scale(5)
        assert success
        assert scaler.current_workers == 5
        assert len(scale_up_calls) == 1
        
        # Test scaling down
        success = scaler.force_scale(2)
        assert success
        assert scaler.current_workers == 2
        assert len(scale_down_calls) == 1
        
        # Test invalid scaling
        success = scaler.force_scale(50)  # Above max
        assert not success
    
    def test_load_balancer(self):
        """Test load balancing functionality"""
        balancer = LoadBalancer()
        
        # Register workers
        balancer.register_worker("worker_1", lambda: "result_1")
        balancer.register_worker("worker_2", lambda: "result_2")
        
        # Test worker selection strategies
        worker_rr = balancer.select_worker(strategy="round_robin")
        assert worker_rr in ["worker_1", "worker_2"]
        
        worker_ll = balancer.select_worker(strategy="least_loaded")
        assert worker_ll in ["worker_1", "worker_2"]
        
        # Update worker stats
        balancer.update_worker_stats("worker_1", active_tasks=5, avg_response_time=2.0)
        
        # Least loaded should prefer worker_2 now
        worker_ll = balancer.select_worker(strategy="least_loaded")
        assert worker_ll == "worker_2"


class TestCircuitBreaker:
    """Test circuit breaker functionality"""
    
    def test_circuit_breaker_basic_operation(self):
        """Test basic circuit breaker operation"""
        config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=1.0)
        breaker = CircuitBreaker(config)
        
        def successful_func():
            return "success"
        
        # Should work normally
        result = breaker.call(successful_func)
        assert result == "success"
        
        stats = breaker.get_statistics()
        assert stats["state"] == "closed"
    
    def test_circuit_breaker_failure_handling(self):
        """Test circuit breaker failure handling"""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.1)
        breaker = CircuitBreaker(config)
        
        def failing_func():
            raise ValueError("Test failure")
        
        # First failure
        with pytest.raises(ValueError):
            breaker.call(failing_func)
        
        # Second failure - should open circuit
        with pytest.raises(ValueError):
            breaker.call(failing_func)
        
        stats = breaker.get_statistics()
        assert stats["state"] == "open"
        assert stats["failure_count"] == 2
        
        # Circuit should reject calls now
        from src.utils.circuit_breaker import CircuitBreakerError
        with pytest.raises(CircuitBreakerError):
            breaker.call(failing_func)
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery"""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.1)
        breaker = CircuitBreaker(config)
        
        def failing_then_succeeding():
            if not hasattr(failing_then_succeeding, 'called'):
                failing_then_succeeding.called = True
                raise ValueError("Initial failure")
            return "recovered"
        
        # Trigger failure to open circuit
        with pytest.raises(ValueError):
            breaker.call(failing_then_succeeding)
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Should transition to half-open and succeed
        result = breaker.call(failing_then_succeeding)
        assert result == "recovered"
        
        stats = breaker.get_statistics()
        assert stats["state"] == "closed"


class TestRetryMechanism:
    """Test retry mechanism functionality"""
    
    def test_retry_handler_success(self):
        """Test retry handler with successful function"""
        config = RetryConfig(max_attempts=3, base_delay=0.1)
        handler = RetryHandler(config)
        
        def successful_func(x):
            return x * 2
        
        result = handler.execute(successful_func, 5)
        assert result == 10
        
        history = handler.get_attempt_history()
        assert len(history) == 1
        assert history[0]["success"] is True
    
    def test_retry_handler_eventual_success(self):
        """Test retry handler with eventual success"""
        config = RetryConfig(max_attempts=3, base_delay=0.01)
        handler = RetryHandler(config)
        
        call_count = 0
        def eventually_successful():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Temporary failure")
            return "success"
        
        result = handler.execute(eventually_successful)
        assert result == "success"
        
        history = handler.get_attempt_history()
        assert len(history) == 3
        assert history[0]["success"] is False
        assert history[1]["success"] is False
        assert history[2]["success"] is True
    
    def test_retry_handler_exhaustion(self):
        """Test retry exhaustion"""
        config = RetryConfig(max_attempts=2, base_delay=0.01)
        handler = RetryHandler(config)
        
        def always_failing():
            raise RuntimeError("Persistent failure")
        
        from src.utils.retry import RetryExhaustedError
        with pytest.raises(RetryExhaustedError):
            handler.execute(always_failing)
        
        history = handler.get_attempt_history()
        assert len(history) == 2
        assert all(not h["success"] for h in history)


class TestBackupSystem:
    """Test backup and recovery system"""
    
    def test_backup_manager_initialization(self):
        """Test backup manager initialization"""
        manager = BackupManager("/tmp/test_backups")
        
        assert manager.backup_root.exists()
        assert isinstance(manager.backups, dict)
    
    def test_backup_single_file(self):
        """Test backing up a single file"""
        import tempfile
        import os
        
        manager = BackupManager("/tmp/test_backups")
        
        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("Test content for backup")
            test_file = f.name
        
        try:
            # Create backup
            backup_id = manager.create_backup(test_file, "test_backup", compression=False)
            
            assert backup_id in manager.backups
            backup_info = manager.get_backup_info(backup_id)
            assert backup_info.status == "completed"
            assert backup_info.file_count == 1
            
            # Verify backup
            assert manager.verify_backup(backup_id)
            
        finally:
            os.unlink(test_file)
    
    def test_backup_restore(self):
        """Test backup restore functionality"""
        import tempfile
        import os
        
        manager = BackupManager("/tmp/test_backups")
        
        original_content = "Original backup content"
        
        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(original_content)
            test_file = f.name
        
        try:
            # Create backup
            backup_id = manager.create_backup(test_file, "restore_test", compression=False)
            
            # Delete original
            os.unlink(test_file)
            
            # Restore backup
            success = manager.restore_backup(backup_id, test_file)
            assert success
            assert os.path.exists(test_file)
            
            # Verify content
            with open(test_file, 'r') as f:
                restored_content = f.read()
            assert restored_content == original_content
            
        finally:
            if os.path.exists(test_file):
                os.unlink(test_file)


class TestIntegration:
    """Integration tests for combined functionality"""
    
    @pytest.mark.asyncio
    async def test_scalable_discovery_integration(self):
        """Test integration of scalable discovery with all features"""
        # This test would require the full scalable discovery engine
        # For now, test basic async + discovery integration
        
        from src.algorithms.discovery import DiscoveryEngine
        
        queue = AsyncTaskQueue(max_concurrent_tasks=2)
        await queue.start_workers()
        
        def discovery_task(data, threshold):
            engine = DiscoveryEngine(discovery_threshold=threshold)
            return engine.discover(data, context="integration_test")
        
        # Submit discovery tasks
        test_data = np.random.randn(100, 4)
        task_id = await queue.submit_task("discovery_integration", discovery_task, test_data, 0.6)
        
        # Get results
        discoveries = await queue.get_task_result(task_id, timeout=30.0)
        assert isinstance(discoveries, list)
        
        await queue.shutdown()
    
    def test_error_handling_integration(self):
        """Test integration of circuit breaker + retry"""
        from src.utils.circuit_breaker import circuit_breaker
        from src.utils.retry import retry
        
        call_count = 0
        
        @circuit_breaker("test_integration", failure_threshold=3)
        @retry(max_attempts=2, base_delay=0.01)
        def integrated_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("Test failure")
            return "success"
        
        # Should succeed after retry
        result = integrated_function()
        assert result == "success"
        assert call_count == 2


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    @pytest.mark.asyncio
    async def test_async_queue_performance(self):
        """Benchmark async queue performance"""
        queue = AsyncTaskQueue(max_concurrent_tasks=4)
        await queue.start_workers()
        
        def cpu_task(n):
            # Simple CPU-bound task
            return sum(i*i for i in range(n))
        
        # Submit many tasks
        task_count = 20
        start_time = time.time()
        
        task_ids = []
        for i in range(task_count):
            task_id = await queue.submit_task(f"perf_task_{i}", cpu_task, 1000)
            task_ids.append(task_id)
        
        # Collect results
        results = []
        for task_id in task_ids:
            result = await queue.get_task_result(task_id, timeout=30.0)
            results.append(result)
        
        total_time = time.time() - start_time
        
        assert len(results) == task_count
        assert total_time < 10.0  # Should complete within 10 seconds
        
        # Check throughput
        throughput = task_count / total_time
        assert throughput > 2.0  # At least 2 tasks per second
        
        await queue.shutdown()
    
    def test_cache_performance(self):
        """Test caching system performance"""
        from src.performance.caching import LRUCache
        
        cache = LRUCache(max_size=1000, ttl=300)
        
        # Fill cache
        start_time = time.time()
        for i in range(500):
            cache.put(f"key_{i}", f"value_{i}")
        
        fill_time = time.time() - start_time
        assert fill_time < 1.0  # Should fill cache quickly
        
        # Test retrieval performance
        start_time = time.time()
        hit_count = 0
        for i in range(1000):
            key = f"key_{i % 500}"  # 50% hit rate expected
            hit, value = cache.get(key)
            if hit:
                hit_count += 1
        
        retrieval_time = time.time() - start_time
        assert retrieval_time < 0.5  # Should be fast
        assert hit_count > 400  # Good hit rate


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])