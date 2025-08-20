"""Tests for multi-GPU acceleration module"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
from src.performance.multi_gpu_acceleration import (
    GPUDevice, GPUTask, GPUTaskResult, MultiGPUConfig,
    WorkloadDistributionStrategy, GPUTaskPriority,
    GPULoadBalancer, MultiGPUTaskScheduler, MultiGPUManager,
    GPUStream, GPUStreamPool
)


class TestGPUDevice:
    """Test GPU device representation"""
    
    def test_device_creation(self):
        """Test GPU device creation"""
        device = GPUDevice(
            device_id=0,
            name="Test GPU",
            compute_capability=(7, 5),
            total_memory=8000000000,
            available_memory=6000000000,
            utilization=0.5,
            temperature=65.0,
            power_usage=150.0
        )
        
        assert device.device_id == 0
        assert device.name == "Test GPU"
        assert device.compute_capability == (7, 5)
        assert device.memory_utilization == 0.25  # (8000-6000)/8000
        assert device.is_available == True
        assert device.current_load == 0.0


class TestGPUTask:
    """Test GPU task representation"""
    
    def test_task_creation(self):
        """Test GPU task creation"""
        def test_function(x, y):
            return x + y
        
        task = GPUTask(
            task_id="test_task",
            function=test_function,
            args=(1, 2),
            kwargs={},
            priority=GPUTaskPriority.HIGH,
            memory_requirement=1000000
        )
        
        assert task.task_id == "test_task"
        assert task.function == test_function
        assert task.args == (1, 2)
        assert task.priority == GPUTaskPriority.HIGH
        assert task.memory_requirement == 1000000
        assert task.max_retries == 3
    
    def test_task_auto_id_generation(self):
        """Test automatic task ID generation"""
        task = GPUTask(
            task_id="",
            function=lambda x: x,
            args=(1,),
            kwargs={}
        )
        
        assert task.task_id.startswith("task_")
        assert len(task.task_id) > 5


class TestGPULoadBalancer:
    """Test GPU load balancer"""
    
    def setup_method(self):
        """Setup test devices"""
        self.devices = [
            GPUDevice(0, "GPU0", (7, 5), 8000000000, 6000000000, 0.2, 60.0, 120.0),
            GPUDevice(1, "GPU1", (8, 0), 12000000000, 10000000000, 0.1, 55.0, 100.0),
            GPUDevice(2, "GPU2", (6, 1), 4000000000, 3000000000, 0.5, 70.0, 180.0)
        ]
        # Set current loads for testing
        self.devices[0].current_load = 0.5  # Medium load
        self.devices[1].current_load = 0.1  # Low load
        self.devices[2].current_load = 0.8  # High load
    
    def test_round_robin_selection(self):
        """Test round-robin device selection"""
        balancer = GPULoadBalancer(self.devices, WorkloadDistributionStrategy.ROUND_ROBIN)
        
        task = GPUTask("test", lambda x: x, (1,), {})
        
        # Should cycle through devices
        device_ids = []
        for _ in range(6):
            device_id = balancer.select_device(task)
            device_ids.append(device_id)
        
        # Should see pattern: 0, 1, 2, 0, 1, 2
        assert device_ids[:3] == [0, 1, 2]
        assert device_ids[3:6] == [0, 1, 2]
    
    def test_load_balanced_selection(self):
        """Test load-balanced device selection"""
        balancer = GPULoadBalancer(self.devices, WorkloadDistributionStrategy.LOAD_BALANCED)
        
        task = GPUTask("test", lambda x: x, (1,), {})
        
        # Should prefer device with lowest load (GPU1 with 0.1)
        device_id = balancer.select_device(task)
        assert device_id == 1
    
    def test_compute_capability_selection(self):
        """Test compute capability-based selection"""
        balancer = GPULoadBalancer(self.devices, WorkloadDistributionStrategy.COMPUTE_CAPABILITY)
        
        high_priority_task = GPUTask("test", lambda x: x, (1,), {}, priority=GPUTaskPriority.HIGH)
        normal_priority_task = GPUTask("test", lambda x: x, (1,), {}, priority=GPUTaskPriority.NORMAL)
        
        # High priority should prefer highest compute capability (GPU1 with 8.0)
        device_id = balancer.select_device(high_priority_task)
        assert device_id == 1
    
    def test_memory_based_selection(self):
        """Test memory-based device selection"""
        balancer = GPULoadBalancer(self.devices, WorkloadDistributionStrategy.MEMORY_BASED)
        
        # Task requiring 11GB memory
        large_memory_task = GPUTask("test", lambda x: x, (1,), {}, memory_requirement=11000000000)
        
        # Should select GPU1 (only one with enough memory)
        device_id = balancer.select_device(large_memory_task)
        assert device_id == 1
        
        # Small memory task should use load balancing
        small_memory_task = GPUTask("test", lambda x: x, (1,), {}, memory_requirement=1000000)
        device_id = balancer.select_device(small_memory_task)
        assert device_id == 1  # Still lowest load
    
    def test_dynamic_adaptive_selection(self):
        """Test dynamic adaptive selection"""
        balancer = GPULoadBalancer(self.devices, WorkloadDistributionStrategy.DYNAMIC_ADAPTIVE)
        
        task = GPUTask("test", lambda x: x, (1,), {})
        
        # Should select a device (exact choice depends on scoring algorithm)
        device_id = balancer.select_device(task)
        assert device_id in [0, 1, 2]
    
    def test_preferred_device(self):
        """Test preferred device selection"""
        balancer = GPULoadBalancer(self.devices, WorkloadDistributionStrategy.ROUND_ROBIN)
        
        task = GPUTask("test", lambda x: x, (1,), {}, preferred_device=2)
        
        # Should use preferred device if available
        device_id = balancer.select_device(task)
        assert device_id == 2
    
    def test_unavailable_device_handling(self):
        """Test handling of unavailable devices"""
        # Mark one device as unavailable
        self.devices[1].is_available = False
        
        balancer = GPULoadBalancer(self.devices, WorkloadDistributionStrategy.ROUND_ROBIN)
        task = GPUTask("test", lambda x: x, (1,), {})
        
        # Should only select from available devices
        device_ids = set()
        for _ in range(10):
            device_id = balancer.select_device(task)
            device_ids.add(device_id)
        
        assert 1 not in device_ids  # Unavailable device should not be selected
        assert device_ids.issubset({0, 2})  # Only available devices
    
    def test_load_update(self):
        """Test device load updating"""
        balancer = GPULoadBalancer(self.devices, WorkloadDistributionStrategy.LOAD_BALANCED)
        
        initial_load = balancer.devices[0].current_load
        
        # Update load
        balancer.update_device_load(0, 0.3)
        
        assert balancer.devices[0].current_load == initial_load + 0.3
        
        # Test bounds
        balancer.update_device_load(0, 1.0)  # Should cap at 1.0
        assert balancer.devices[0].current_load == 1.0
        
        balancer.update_device_load(0, -2.0)  # Should floor at 0.0
        assert balancer.devices[0].current_load == 0.0


class TestGPUStreamPool:
    """Test GPU stream pool"""
    
    def test_stream_pool_creation(self):
        """Test stream pool creation"""
        pool = GPUStreamPool(device_id=0, pool_size=4)
        
        assert pool.device_id == 0
        assert pool.pool_size == 4
        assert pool.streams.qsize() == 4
    
    def test_stream_acquisition_release(self):
        """Test stream acquisition and release"""
        pool = GPUStreamPool(device_id=0, pool_size=2)
        
        # Acquire streams
        stream1 = pool.acquire_stream()
        stream2 = pool.acquire_stream()
        
        assert pool.get_active_count() == 2
        assert pool.streams.qsize() == 0
        
        # Release streams
        pool.release_stream(stream1)
        pool.release_stream(stream2)
        
        assert pool.get_active_count() == 0
        assert pool.streams.qsize() == 2


class TestMultiGPUTaskScheduler:
    """Test multi-GPU task scheduler"""
    
    def setup_method(self):
        """Setup test environment"""
        self.devices = [
            GPUDevice(0, "GPU0", (7, 5), 8000000000, 6000000000, 0.2, 60.0, 120.0),
            GPUDevice(1, "GPU1", (8, 0), 12000000000, 10000000000, 0.1, 55.0, 100.0)
        ]
        self.config = MultiGPUConfig(
            workload_strategy=WorkloadDistributionStrategy.LOAD_BALANCED,
            task_queue_size=100
        )
    
    def test_scheduler_creation(self):
        """Test scheduler creation"""
        scheduler = MultiGPUTaskScheduler(self.devices, self.config)
        
        assert len(scheduler.devices) == 2
        assert len(scheduler.stream_pools) == 2
        assert scheduler.config == self.config
        assert not scheduler.is_running
    
    def test_scheduler_start_stop(self):
        """Test scheduler start/stop"""
        scheduler = MultiGPUTaskScheduler(self.devices, self.config)
        
        scheduler.start()
        assert scheduler.is_running
        
        scheduler.stop()
        assert not scheduler.is_running
    
    @patch('src.performance.multi_gpu_acceleration.CUPY_AVAILABLE', False)
    def test_task_submission_cpu_fallback(self):
        """Test task submission with CPU fallback"""
        scheduler = MultiGPUTaskScheduler(self.devices, self.config)
        scheduler.start()
        
        try:
            # Submit simple task
            def add_numbers(a, b):
                return a + b
            
            task = GPUTask("test_add", add_numbers, (2, 3), {})
            task_id = scheduler.submit_task(task)
            
            assert task_id == "test_add"
            assert task_id in scheduler.pending_tasks
            
            # Wait for completion
            result = scheduler.wait_for_task(task_id, timeout=5.0)
            
            assert result.success
            assert result.result == 5
            assert result.task_id == task_id
            
        finally:
            scheduler.stop()
    
    def test_task_priority_handling(self):
        """Test task priority handling"""
        scheduler = MultiGPUTaskScheduler(self.devices, self.config)
        
        # Submit tasks with different priorities
        high_task = GPUTask("high", lambda: "high", (), {}, priority=GPUTaskPriority.HIGH)
        low_task = GPUTask("low", lambda: "low", (), {}, priority=GPUTaskPriority.LOW)
        
        # Submit low priority first
        scheduler.submit_task(low_task)
        scheduler.submit_task(high_task)
        
        # High priority should be processed first
        _, _, first_task = scheduler.task_queue.get()
        assert first_task.priority == GPUTaskPriority.HIGH
    
    def test_scheduler_statistics(self):
        """Test scheduler statistics"""
        scheduler = MultiGPUTaskScheduler(self.devices, self.config)
        
        stats = scheduler.get_scheduler_stats()
        
        assert 'tasks_submitted' in stats
        assert 'tasks_completed' in stats
        assert 'tasks_failed' in stats
        assert 'pending_tasks' in stats
        assert 'active_tasks' in stats
        assert 'queue_size' in stats


class TestMultiGPUManager:
    """Test multi-GPU manager"""
    
    @patch('src.performance.multi_gpu_acceleration.CUPY_AVAILABLE', False)
    def test_manager_initialization_no_gpu(self):
        """Test manager initialization without GPU"""
        config = MultiGPUConfig()
        manager = MultiGPUManager(config)
        
        assert len(manager.devices) == 0
        assert not manager.is_initialized
    
    @patch('src.performance.multi_gpu_acceleration.CUPY_AVAILABLE', True)
    @patch('src.performance.multi_gpu_acceleration.cp')
    def test_manager_initialization_with_mock_gpu(self, mock_cp):
        """Test manager initialization with mocked GPU"""
        # Mock CuPy functions
        mock_cp.cuda.runtime.getDeviceCount.return_value = 2
        mock_cp.cuda.runtime.getDeviceProperties.return_value = {
            'name': b'Mock GPU',
            'major': 7,
            'minor': 5,
            'totalGlobalMem': 8000000000
        }
        mock_cp.cuda.runtime.memGetInfo.return_value = (6000000000, 8000000000)
        mock_cp.cuda.Device.return_value.__enter__ = Mock()
        mock_cp.cuda.Device.return_value.__exit__ = Mock()
        
        config = MultiGPUConfig(max_devices=2)
        manager = MultiGPUManager(config)
        
        assert len(manager.devices) == 2
        assert manager.is_initialized
    
    def test_system_status(self):
        """Test system status reporting"""
        # Create manager with mock devices
        config = MultiGPUConfig()
        manager = MultiGPUManager(config)
        
        # Manually add mock devices for testing
        manager.devices = [
            GPUDevice(0, "Test GPU 0", (7, 5), 8000000000, 6000000000, 0.2, 60.0, 120.0),
            GPUDevice(1, "Test GPU 1", (8, 0), 12000000000, 10000000000, 0.1, 55.0, 100.0)
        ]
        
        status = manager.get_system_status()
        
        assert 'devices' in status
        assert 'configuration' in status
        assert 'capabilities' in status
        assert len(status['devices']) == 2
        
        # Check device information
        device0 = status['devices'][0]
        assert device0['device_id'] == 0
        assert device0['name'] == "Test GPU 0"
        assert device0['compute_capability'] == (7, 5)


class TestIntegrationScenarios:
    """Integration tests for multi-GPU scenarios"""
    
    def test_parallel_computation_simulation(self):
        """Test parallel computation simulation"""
        # Create mock devices
        devices = [
            GPUDevice(0, "GPU0", (7, 5), 8000000000, 6000000000, 0.0, 60.0, 120.0),
            GPUDevice(1, "GPU1", (8, 0), 12000000000, 10000000000, 0.0, 55.0, 100.0)
        ]
        
        config = MultiGPUConfig(workload_strategy=WorkloadDistributionStrategy.LOAD_BALANCED)
        scheduler = MultiGPUTaskScheduler(devices, config)
        scheduler.start()
        
        try:
            # Submit multiple tasks
            def compute_sum(data):
                return sum(data)
            
            task_ids = []
            for i in range(10):
                task = GPUTask(f"task_{i}", compute_sum, ([i, i+1, i+2],), {})
                task_id = scheduler.submit_task(task)
                task_ids.append(task_id)
            
            # Wait for all tasks to complete
            results = []
            for task_id in task_ids:
                result = scheduler.wait_for_task(task_id, timeout=10.0)
                assert result.success
                results.append(result.result)
            
            # Verify results
            expected_results = [sum([i, i+1, i+2]) for i in range(10)]
            assert results == expected_results
            
        finally:
            scheduler.stop()
    
    def test_load_balancing_behavior(self):
        """Test load balancing behavior"""
        devices = [
            GPUDevice(0, "GPU0", (7, 5), 8000000000, 6000000000, 0.8, 60.0, 120.0),  # High load
            GPUDevice(1, "GPU1", (8, 0), 12000000000, 10000000000, 0.1, 55.0, 100.0)   # Low load
        ]
        # Set current loads
        devices[0].current_load = 0.8  # High load
        devices[1].current_load = 0.1  # Low load
        
        balancer = GPULoadBalancer(devices, WorkloadDistributionStrategy.LOAD_BALANCED)
        
        # Multiple task selections should prefer low-load device
        selected_devices = []
        for _ in range(10):
            task = GPUTask("test", lambda x: x, (1,), {})
            device_id = balancer.select_device(task)
            selected_devices.append(device_id)
        
        # Should mostly select GPU1 (device 1) due to lower load
        gpu1_selections = selected_devices.count(1)
        assert gpu1_selections >= 8  # At least 80% should go to less loaded device
    
    def test_memory_constraint_handling(self):
        """Test memory constraint handling"""
        devices = [
            GPUDevice(0, "GPU0", (7, 5), 4000000000, 1000000000, 0.2, 60.0, 120.0),    # 1GB available
            GPUDevice(1, "GPU1", (8, 0), 12000000000, 8000000000, 0.1, 55.0, 100.0)   # 8GB available
        ]
        # Set current loads
        devices[0].current_load = 0.2  # Higher load
        devices[1].current_load = 0.1  # Lower load
        
        balancer = GPULoadBalancer(devices, WorkloadDistributionStrategy.MEMORY_BASED)
        
        # Task requiring 5GB should go to GPU1
        large_task = GPUTask("large", lambda x: x, (1,), {}, memory_requirement=5000000000)
        device_id = balancer.select_device(large_task)
        assert device_id == 1
        
        # Small task can go to either, but should prefer less loaded (GPU1)
        small_task = GPUTask("small", lambda x: x, (1,), {}, memory_requirement=500000000)
        device_id = balancer.select_device(small_task)
        assert device_id == 1  # Should still prefer GPU1 due to lower load


def test_multi_gpu_configuration():
    """Test multi-GPU configuration"""
    config = MultiGPUConfig(
        max_devices=4,
        workload_strategy=WorkloadDistributionStrategy.DYNAMIC_ADAPTIVE,
        enable_peer_to_peer=True,
        memory_pool_fraction=0.9,
        stream_pool_size=8
    )
    
    assert config.max_devices == 4
    assert config.workload_strategy == WorkloadDistributionStrategy.DYNAMIC_ADAPTIVE
    assert config.enable_peer_to_peer == True
    assert config.memory_pool_fraction == 0.9
    assert config.stream_pool_size == 8


if __name__ == "__main__":
    pytest.main([__file__])