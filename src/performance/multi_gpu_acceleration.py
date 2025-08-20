"""
Advanced Multi-GPU Acceleration Framework

Provides comprehensive multi-GPU support for parallel processing including:
- Multi-GPU workload distribution and load balancing
- GPU cluster management and coordination
- Asynchronous execution and stream management
- Memory optimization across multiple devices
- Advanced scheduling and resource allocation
- Fault tolerance and automatic recovery
"""

import time
import logging
import threading
import queue
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor, Future
import weakref

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    MPI = None

from .gpu_acceleration import GPUConfig, GPUPerformanceMetrics
from ..utils.secure_random import ScientificRandomGenerator

logger = logging.getLogger(__name__)


class WorkloadDistributionStrategy(Enum):
    """Strategies for distributing workload across GPUs"""
    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    COMPUTE_CAPABILITY = "compute_capability"
    MEMORY_BASED = "memory_based"
    DYNAMIC_ADAPTIVE = "dynamic_adaptive"


class GPUTaskPriority(Enum):
    """Task priority levels for GPU scheduling"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class GPUDevice:
    """GPU device information and state"""
    device_id: int
    name: str
    compute_capability: Tuple[int, int]
    total_memory: int
    available_memory: int
    utilization: float
    temperature: float
    power_usage: float
    is_available: bool = True
    current_load: float = 0.0
    active_streams: int = 0
    
    def __post_init__(self):
        self.memory_utilization = 1.0 - (self.available_memory / max(self.total_memory, 1))


@dataclass
class GPUTask:
    """GPU computation task"""
    task_id: str
    function: Callable
    args: Tuple
    kwargs: Dict[str, Any]
    priority: GPUTaskPriority = GPUTaskPriority.NORMAL
    estimated_duration: float = 1.0
    memory_requirement: int = 0
    preferred_device: Optional[int] = None
    callback: Optional[Callable] = None
    dependencies: List[str] = field(default_factory=list)
    max_retries: int = 3
    
    def __post_init__(self):
        if not self.task_id:
            self.task_id = f"task_{int(time.time() * 1000000)}"


@dataclass
class GPUTaskResult:
    """Result of GPU task execution"""
    task_id: str
    device_id: int
    result: Any
    execution_time: float
    memory_used: int
    success: bool
    error_message: Optional[str] = None
    performance_metrics: Optional[GPUPerformanceMetrics] = None


@dataclass
class MultiGPUConfig:
    """Configuration for multi-GPU system"""
    max_devices: int = -1  # -1 means use all available
    workload_strategy: WorkloadDistributionStrategy = WorkloadDistributionStrategy.DYNAMIC_ADAPTIVE
    enable_peer_to_peer: bool = True
    memory_pool_fraction: float = 0.8
    stream_pool_size: int = 4
    task_queue_size: int = 1000
    load_balancing_threshold: float = 0.8
    enable_profiling: bool = True
    enable_fault_tolerance: bool = True
    cluster_communication: bool = False  # For multi-node clusters


class GPUStream:
    """GPU stream wrapper for asynchronous execution"""
    
    def __init__(self, device_id: int):
        self.device_id = device_id
        self.stream = None
        self.is_active = False
        self.creation_time = time.time()
        
        if CUPY_AVAILABLE:
            with cp.cuda.Device(device_id):
                self.stream = cp.cuda.Stream()
    
    def __enter__(self):
        if self.stream:
            self.stream.__enter__()
            self.is_active = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.stream:
            self.stream.__exit__(exc_type, exc_val, exc_tb)
            self.is_active = False
    
    def synchronize(self):
        """Synchronize stream execution"""
        if self.stream:
            self.stream.synchronize()


class GPUStreamPool:
    """Pool of GPU streams for efficient resource management"""
    
    def __init__(self, device_id: int, pool_size: int = 4):
        self.device_id = device_id
        self.pool_size = pool_size
        self.streams = queue.Queue()
        self.active_streams = set()
        self.lock = threading.Lock()
        
        # Initialize stream pool
        for _ in range(pool_size):
            stream = GPUStream(device_id)
            self.streams.put(stream)
    
    def acquire_stream(self) -> GPUStream:
        """Acquire stream from pool"""
        try:
            stream = self.streams.get_nowait()
            with self.lock:
                self.active_streams.add(stream)
            return stream
        except queue.Empty:
            # Create temporary stream if pool is exhausted
            stream = GPUStream(self.device_id)
            with self.lock:
                self.active_streams.add(stream)
            return stream
    
    def release_stream(self, stream: GPUStream):
        """Release stream back to pool"""
        with self.lock:
            if stream in self.active_streams:
                self.active_streams.remove(stream)
        
        if len(self.active_streams) < self.pool_size:
            try:
                self.streams.put_nowait(stream)
            except queue.Full:
                pass  # Pool is full, let stream be garbage collected
    
    def get_active_count(self) -> int:
        """Get number of active streams"""
        with self.lock:
            return len(self.active_streams)


class GPULoadBalancer:
    """Advanced load balancer for multi-GPU workload distribution"""
    
    def __init__(self, devices: List[GPUDevice], strategy: WorkloadDistributionStrategy):
        self.devices = {device.device_id: device for device in devices}
        self.strategy = strategy
        self.task_history = {}
        self.performance_history = {}
        self.round_robin_index = 0
        self.lock = threading.Lock()
    
    def select_device(self, task: GPUTask) -> int:
        """Select optimal device for task execution"""
        
        if task.preferred_device is not None and task.preferred_device in self.devices:
            device = self.devices[task.preferred_device]
            if device.is_available and self._can_accommodate_task(device, task):
                return task.preferred_device
        
        available_devices = [d for d in self.devices.values() if d.is_available]
        
        if not available_devices:
            raise RuntimeError("No available GPU devices")
        
        if self.strategy == WorkloadDistributionStrategy.ROUND_ROBIN:
            return self._round_robin_selection(available_devices)
        
        elif self.strategy == WorkloadDistributionStrategy.LOAD_BALANCED:
            return self._load_balanced_selection(available_devices, task)
        
        elif self.strategy == WorkloadDistributionStrategy.COMPUTE_CAPABILITY:
            return self._compute_capability_selection(available_devices, task)
        
        elif self.strategy == WorkloadDistributionStrategy.MEMORY_BASED:
            return self._memory_based_selection(available_devices, task)
        
        elif self.strategy == WorkloadDistributionStrategy.DYNAMIC_ADAPTIVE:
            return self._dynamic_adaptive_selection(available_devices, task)
        
        else:
            return available_devices[0].device_id
    
    def _round_robin_selection(self, devices: List[GPUDevice]) -> int:
        """Simple round-robin device selection"""
        with self.lock:
            device = devices[self.round_robin_index % len(devices)]
            self.round_robin_index += 1
            return device.device_id
    
    def _load_balanced_selection(self, devices: List[GPUDevice], task: GPUTask) -> int:
        """Select device based on current load"""
        # Sort by current load (ascending)
        devices_by_load = sorted(devices, key=lambda d: d.current_load)
        
        # Select device that can accommodate the task
        for device in devices_by_load:
            if self._can_accommodate_task(device, task):
                return device.device_id
        
        # Fallback to least loaded device
        return devices_by_load[0].device_id
    
    def _compute_capability_selection(self, devices: List[GPUDevice], task: GPUTask) -> int:
        """Select device based on compute capability"""
        # Sort by compute capability (descending)
        devices_by_capability = sorted(
            devices, 
            key=lambda d: (d.compute_capability[0], d.compute_capability[1]), 
            reverse=True
        )
        
        # For high-priority tasks, prefer higher compute capability
        if task.priority in [GPUTaskPriority.HIGH, GPUTaskPriority.CRITICAL]:
            for device in devices_by_capability:
                if self._can_accommodate_task(device, task):
                    return device.device_id
        
        # For normal tasks, use load balancing among capable devices
        return self._load_balanced_selection(devices_by_capability[:len(devices)//2 + 1], task)
    
    def _memory_based_selection(self, devices: List[GPUDevice], task: GPUTask) -> int:
        """Select device based on memory availability"""
        # Filter devices that can accommodate task memory requirements
        suitable_devices = [
            d for d in devices 
            if d.available_memory >= task.memory_requirement
        ]
        
        if not suitable_devices:
            # Fallback to device with most available memory
            suitable_devices = [max(devices, key=lambda d: d.available_memory)]
        
        # Among suitable devices, select by load
        return self._load_balanced_selection(suitable_devices, task)
    
    def _dynamic_adaptive_selection(self, devices: List[GPUDevice], task: GPUTask) -> int:
        """Advanced adaptive selection based on multiple factors"""
        
        scores = {}
        
        for device in devices:
            # Base score components
            load_score = 1.0 - device.current_load  # Lower load is better
            memory_score = device.available_memory / max(device.total_memory, 1)
            capability_score = (device.compute_capability[0] + device.compute_capability[1] * 0.1) / 10.0
            
            # Historical performance score
            history_score = self._get_historical_performance_score(device.device_id, task)
            
            # Temperature and power efficiency score
            efficiency_score = self._get_efficiency_score(device)
            
            # Task-specific adjustments
            task_priority_weight = {
                GPUTaskPriority.LOW: 0.5,
                GPUTaskPriority.NORMAL: 1.0,
                GPUTaskPriority.HIGH: 1.5,
                GPUTaskPriority.CRITICAL: 2.0
            }[task.priority]
            
            # Composite score
            composite_score = (
                0.3 * load_score +
                0.25 * memory_score +
                0.2 * capability_score +
                0.15 * history_score +
                0.1 * efficiency_score
            ) * task_priority_weight
            
            # Penalty for insufficient memory
            if device.available_memory < task.memory_requirement:
                composite_score *= 0.1
            
            scores[device.device_id] = composite_score
        
        # Select device with highest score
        best_device_id = max(scores.keys(), key=lambda d: scores[d])
        return best_device_id
    
    def _can_accommodate_task(self, device: GPUDevice, task: GPUTask) -> bool:
        """Check if device can accommodate the task"""
        return (
            device.is_available and
            device.available_memory >= task.memory_requirement and
            device.current_load < 0.9  # Leave some headroom
        )
    
    def _get_historical_performance_score(self, device_id: int, task: GPUTask) -> float:
        """Get historical performance score for device"""
        if device_id not in self.performance_history:
            return 0.5  # Neutral score for new device
        
        # Simple moving average of recent performance
        recent_performances = self.performance_history[device_id][-10:]
        if recent_performances:
            return min(1.0, np.mean(recent_performances))
        else:
            return 0.5
    
    def _get_efficiency_score(self, device: GPUDevice) -> float:
        """Calculate efficiency score based on temperature and power"""
        # Simplified efficiency calculation
        temp_score = max(0.0, 1.0 - (device.temperature - 50) / 50)  # Prefer cooler devices
        power_score = max(0.0, 1.0 - device.power_usage / 300)  # Prefer lower power usage
        
        return (temp_score + power_score) / 2
    
    def update_device_load(self, device_id: int, load_delta: float):
        """Update device load information"""
        if device_id in self.devices:
            with self.lock:
                self.devices[device_id].current_load = max(0.0, 
                    min(1.0, self.devices[device_id].current_load + load_delta))
    
    def record_task_performance(self, device_id: int, performance_score: float):
        """Record task performance for learning"""
        if device_id not in self.performance_history:
            self.performance_history[device_id] = []
        
        self.performance_history[device_id].append(performance_score)
        
        # Keep only recent history
        if len(self.performance_history[device_id]) > 100:
            self.performance_history[device_id] = self.performance_history[device_id][-50:]


class MultiGPUTaskScheduler:
    """Advanced task scheduler for multi-GPU systems"""
    
    def __init__(self, devices: List[GPUDevice], config: MultiGPUConfig):
        self.devices = {device.device_id: device for device in devices}
        self.config = config
        self.load_balancer = GPULoadBalancer(devices, config.workload_strategy)
        
        # Task management
        self.task_queue = queue.PriorityQueue(maxsize=config.task_queue_size)
        self.pending_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        
        # Resource management
        self.stream_pools = {
            device_id: GPUStreamPool(device_id, config.stream_pool_size)
            for device_id in self.devices.keys()
        }
        
        # Execution management
        self.executor = ThreadPoolExecutor(max_workers=len(devices) * 2)
        self.active_futures = {}
        self.is_running = False
        self.scheduler_thread = None
        
        # Performance tracking
        self.execution_stats = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_execution_time': 0.0,
            'average_queue_time': 0.0
        }
        
        self.lock = threading.Lock()
    
    def start(self):
        """Start the task scheduler"""
        if self.is_running:
            return
        
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        logger.info("Multi-GPU task scheduler started")
    
    def stop(self):
        """Stop the task scheduler"""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join()
        
        self.executor.shutdown(wait=True)
        
        logger.info("Multi-GPU task scheduler stopped")
    
    def submit_task(self, task: GPUTask) -> str:
        """Submit task for execution"""
        # Validate task
        if not task.function:
            raise ValueError("Task function cannot be None")
        
        # Check dependencies
        if task.dependencies:
            for dep_id in task.dependencies:
                if dep_id not in self.completed_tasks:
                    raise ValueError(f"Dependency {dep_id} not completed")
        
        # Add to queue with priority
        priority_value = -task.priority.value  # Negative for max priority queue
        queue_item = (priority_value, time.time(), task)
        
        try:
            self.task_queue.put_nowait(queue_item)
            with self.lock:
                self.pending_tasks[task.task_id] = task
                self.execution_stats['tasks_submitted'] += 1
            
            logger.debug(f"Task {task.task_id} submitted with priority {task.priority.name}")
            return task.task_id
            
        except queue.Full:
            raise RuntimeError("Task queue is full")
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.is_running:
            try:
                # Get next task from queue
                try:
                    priority, submit_time, task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Select device for execution
                try:
                    device_id = self.load_balancer.select_device(task)
                except RuntimeError as e:
                    logger.error(f"Failed to select device for task {task.task_id}: {e}")
                    self._handle_task_failure(task, str(e))
                    continue
                
                # Submit for execution
                future = self.executor.submit(self._execute_task, task, device_id)
                
                with self.lock:
                    self.active_futures[task.task_id] = future
                    # Update queue time
                    queue_time = time.time() - submit_time
                    self.execution_stats['average_queue_time'] = (
                        (self.execution_stats['average_queue_time'] * 
                         self.execution_stats['tasks_submitted'] + queue_time) /
                        (self.execution_stats['tasks_submitted'] + 1)
                    )
                
                # Set completion callback
                future.add_done_callback(lambda f, tid=task.task_id: self._task_completed(tid, f))
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
    
    def _execute_task(self, task: GPUTask, device_id: int) -> GPUTaskResult:
        """Execute task on specified device"""
        start_time = time.time()
        
        try:
            # Update device load
            self.load_balancer.update_device_load(device_id, 0.1)
            
            # Acquire stream
            stream_pool = self.stream_pools[device_id]
            stream = stream_pool.acquire_stream()
            
            try:
                # Set device context
                if CUPY_AVAILABLE:
                    with cp.cuda.Device(device_id):
                        with stream:
                            # Execute task function
                            result = task.function(*task.args, **task.kwargs)
                            
                            # Synchronize stream
                            stream.synchronize()
                else:
                    # CPU fallback
                    result = task.function(*task.args, **task.kwargs)
                
                execution_time = time.time() - start_time
                
                # Record performance
                performance_score = 1.0 / max(execution_time, 0.001)  # Simple performance metric
                self.load_balancer.record_task_performance(device_id, performance_score)
                
                # Create result
                task_result = GPUTaskResult(
                    task_id=task.task_id,
                    device_id=device_id,
                    result=result,
                    execution_time=execution_time,
                    memory_used=task.memory_requirement,
                    success=True
                )
                
                # Call callback if provided
                if task.callback:
                    try:
                        task.callback(task_result)
                    except Exception as e:
                        logger.warning(f"Task callback failed for {task.task_id}: {e}")
                
                return task_result
                
            finally:
                # Release stream
                stream_pool.release_stream(stream)
                
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Task {task.task_id} failed on device {device_id}: {e}")
            
            return GPUTaskResult(
                task_id=task.task_id,
                device_id=device_id,
                result=None,
                execution_time=execution_time,
                memory_used=0,
                success=False,
                error_message=str(e)
            )
        
        finally:
            # Update device load
            self.load_balancer.update_device_load(device_id, -0.1)
    
    def _task_completed(self, task_id: str, future: Future):
        """Handle task completion"""
        try:
            result = future.result()
            
            with self.lock:
                # Remove from active futures
                if task_id in self.active_futures:
                    del self.active_futures[task_id]
                
                # Remove from pending tasks
                if task_id in self.pending_tasks:
                    del self.pending_tasks[task_id]
                
                # Update statistics
                if result.success:
                    self.completed_tasks[task_id] = result
                    self.execution_stats['tasks_completed'] += 1
                    self.execution_stats['total_execution_time'] += result.execution_time
                else:
                    self.failed_tasks[task_id] = result
                    self.execution_stats['tasks_failed'] += 1
                    
                    # Retry logic
                    if task_id in self.pending_tasks:
                        task = self.pending_tasks[task_id]
                        if task.max_retries > 0:
                            task.max_retries -= 1
                            logger.info(f"Retrying task {task_id}, {task.max_retries} retries left")
                            self.submit_task(task)
            
        except Exception as e:
            logger.error(f"Error handling task completion for {task_id}: {e}")
    
    def _handle_task_failure(self, task: GPUTask, error_message: str):
        """Handle task failure"""
        result = GPUTaskResult(
            task_id=task.task_id,
            device_id=-1,
            result=None,
            execution_time=0.0,
            memory_used=0,
            success=False,
            error_message=error_message
        )
        
        with self.lock:
            self.failed_tasks[task.task_id] = result
            if task.task_id in self.pending_tasks:
                del self.pending_tasks[task.task_id]
            self.execution_stats['tasks_failed'] += 1
    
    def get_task_result(self, task_id: str) -> Optional[GPUTaskResult]:
        """Get task result by ID"""
        with self.lock:
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id]
            elif task_id in self.failed_tasks:
                return self.failed_tasks[task_id]
            else:
                return None
    
    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> GPUTaskResult:
        """Wait for task completion"""
        start_time = time.time()
        
        while True:
            result = self.get_task_result(task_id)
            if result:
                return result
            
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
            
            time.sleep(0.1)
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        with self.lock:
            stats = self.execution_stats.copy()
            stats.update({
                'pending_tasks': len(self.pending_tasks),
                'active_tasks': len(self.active_futures),
                'queue_size': self.task_queue.qsize(),
                'devices_count': len(self.devices),
                'average_execution_time': (
                    self.execution_stats['total_execution_time'] / 
                    max(self.execution_stats['tasks_completed'], 1)
                )
            })
        
        return stats


class MultiGPUManager:
    """Main multi-GPU management system"""
    
    def __init__(self, config: Optional[MultiGPUConfig] = None):
        """Initialize multi-GPU manager"""
        self.config = config or MultiGPUConfig()
        self.devices = []
        self.scheduler = None
        self.is_initialized = False
        
        # Performance tracking
        self.performance_metrics = {}
        self.cluster_communication = None
        
        # Initialize system
        self._initialize_devices()
        self._setup_peer_to_peer()
        self._initialize_scheduler()
        
        logger.info(f"MultiGPUManager initialized with {len(self.devices)} devices")
    
    def _initialize_devices(self):
        """Detect and initialize GPU devices"""
        if not CUPY_AVAILABLE:
            logger.warning("CuPy not available, multi-GPU support disabled")
            return
        
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            max_devices = self.config.max_devices if self.config.max_devices > 0 else device_count
            
            for device_id in range(min(device_count, max_devices)):
                try:
                    with cp.cuda.Device(device_id):
                        # Get device properties
                        props = cp.cuda.runtime.getDeviceProperties(device_id)
                        
                        # Get memory info
                        total_memory = props['totalGlobalMem']
                        free_memory, _ = cp.cuda.runtime.memGetInfo()
                        
                        device = GPUDevice(
                            device_id=device_id,
                            name=props['name'].decode('utf-8'),
                            compute_capability=(props['major'], props['minor']),
                            total_memory=total_memory,
                            available_memory=free_memory,
                            utilization=0.0,
                            temperature=65.0,  # Default estimate
                            power_usage=150.0  # Default estimate
                        )
                        
                        self.devices.append(device)
                        logger.info(f"Initialized GPU {device_id}: {device.name}")
                        
                except Exception as e:
                    logger.warning(f"Failed to initialize GPU {device_id}: {e}")
            
        except Exception as e:
            logger.error(f"Failed to detect GPU devices: {e}")
    
    def _setup_peer_to_peer(self):
        """Setup peer-to-peer memory access between GPUs"""
        if not self.config.enable_peer_to_peer or len(self.devices) < 2:
            return
        
        try:
            for i, device1 in enumerate(self.devices):
                for j, device2 in enumerate(self.devices):
                    if i != j:
                        try:
                            # Check if P2P access is possible
                            can_access = cp.cuda.runtime.deviceCanAccessPeer(
                                device1.device_id, device2.device_id
                            )
                            
                            if can_access:
                                # Enable P2P access
                                with cp.cuda.Device(device1.device_id):
                                    cp.cuda.runtime.deviceEnablePeerAccess(device2.device_id)
                                
                                logger.info(f"Enabled P2P access: GPU {device1.device_id} -> GPU {device2.device_id}")
                        
                        except Exception as e:
                            logger.debug(f"P2P access not available between GPU {device1.device_id} and {device2.device_id}: {e}")
        
        except Exception as e:
            logger.warning(f"Failed to setup peer-to-peer access: {e}")
    
    def _initialize_scheduler(self):
        """Initialize task scheduler"""
        if self.devices:
            self.scheduler = MultiGPUTaskScheduler(self.devices, self.config)
            self.scheduler.start()
            self.is_initialized = True
    
    def submit_computation(self, 
                         function: Callable,
                         *args,
                         priority: GPUTaskPriority = GPUTaskPriority.NORMAL,
                         memory_requirement: int = 0,
                         preferred_device: Optional[int] = None,
                         callback: Optional[Callable] = None,
                         **kwargs) -> str:
        """Submit computation for multi-GPU execution"""
        
        if not self.is_initialized:
            raise RuntimeError("Multi-GPU manager not initialized")
        
        task = GPUTask(
            task_id="",  # Will be auto-generated
            function=function,
            args=args,
            kwargs=kwargs,
            priority=priority,
            memory_requirement=memory_requirement,
            preferred_device=preferred_device,
            callback=callback
        )
        
        return self.scheduler.submit_task(task)
    
    def parallel_map(self, 
                    function: Callable,
                    data_list: List[Any],
                    chunk_size: Optional[int] = None,
                    priority: GPUTaskPriority = GPUTaskPriority.NORMAL) -> List[Any]:
        """Execute function on data list in parallel across GPUs"""
        
        if not data_list:
            return []
        
        # Determine chunk size
        if chunk_size is None:
            chunk_size = max(1, len(data_list) // (len(self.devices) * 2))
        
        # Split data into chunks
        chunks = [data_list[i:i + chunk_size] for i in range(0, len(data_list), chunk_size)]
        
        # Submit tasks
        task_ids = []
        for chunk in chunks:
            task_id = self.submit_computation(
                lambda data_chunk: [function(item) for item in data_chunk],
                chunk,
                priority=priority
            )
            task_ids.append(task_id)
        
        # Collect results
        results = []
        for task_id in task_ids:
            result = self.scheduler.wait_for_task(task_id)
            if result.success:
                results.extend(result.result)
            else:
                raise RuntimeError(f"Task {task_id} failed: {result.error_message}")
        
        return results
    
    def scatter_gather_computation(self,
                                 data: np.ndarray,
                                 function: Callable,
                                 axis: int = 0) -> np.ndarray:
        """Scatter data across GPUs, compute, and gather results"""
        
        if not self.devices:
            raise RuntimeError("No GPU devices available")
        
        # Split data across devices
        num_devices = len(self.devices)
        chunks = np.array_split(data, num_devices, axis=axis)
        
        # Submit tasks to each device
        task_ids = []
        for i, chunk in enumerate(chunks):
            device_id = self.devices[i % num_devices].device_id
            task_id = self.submit_computation(
                function,
                chunk,
                preferred_device=device_id,
                priority=GPUTaskPriority.HIGH
            )
            task_ids.append(task_id)
        
        # Gather results
        results = []
        for task_id in task_ids:
            result = self.scheduler.wait_for_task(task_id)
            if result.success:
                results.append(result.result)
            else:
                raise RuntimeError(f"Scatter-gather task failed: {result.error_message}")
        
        # Concatenate results
        return np.concatenate(results, axis=axis)
    
    def benchmark_multi_gpu(self, 
                          workload_size: int = 1000,
                          compute_intensity: str = "medium") -> Dict[str, Any]:
        """Benchmark multi-GPU performance"""
        
        logger.info(f"Starting multi-GPU benchmark with workload size {workload_size}")
        
        # Generate benchmark workload
        if compute_intensity == "light":
            compute_function = lambda x: np.sum(x ** 2)
        elif compute_intensity == "medium":
            compute_function = lambda x: np.linalg.norm(x @ x.T)
        elif compute_intensity == "heavy":
            compute_function = lambda x: np.linalg.eigvals(x @ x.T)
        else:
            compute_function = lambda x: np.mean(x)
        
        # Create test data
        test_data = [
            np.random.randn(100, 100).astype(np.float32) 
            for _ in range(workload_size)
        ]
        
        # Single GPU benchmark
        start_time = time.time()
        single_gpu_results = []
        for data in test_data[:min(100, workload_size)]:  # Limit for timing
            result = compute_function(data)
            single_gpu_results.append(result)
        single_gpu_time = time.time() - start_time
        
        # Multi-GPU benchmark
        start_time = time.time()
        multi_gpu_results = self.parallel_map(compute_function, test_data)
        multi_gpu_time = time.time() - start_time
        
        # Calculate metrics
        speedup = (single_gpu_time * len(test_data) / 100) / multi_gpu_time
        efficiency = speedup / len(self.devices)
        
        benchmark_results = {
            'workload_size': workload_size,
            'compute_intensity': compute_intensity,
            'num_devices': len(self.devices),
            'single_gpu_time': single_gpu_time,
            'multi_gpu_time': multi_gpu_time,
            'speedup': speedup,
            'efficiency': efficiency,
            'scheduler_stats': self.scheduler.get_scheduler_stats() if self.scheduler else {},
            'device_utilization': {
                device.device_id: device.current_load 
                for device in self.devices
            }
        }
        
        logger.info(f"Multi-GPU benchmark completed: {speedup:.2f}x speedup, {efficiency:.2f} efficiency")
        
        return benchmark_results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'devices': [
                {
                    'device_id': device.device_id,
                    'name': device.name,
                    'compute_capability': device.compute_capability,
                    'memory_utilization': device.memory_utilization,
                    'current_load': device.current_load,
                    'is_available': device.is_available
                }
                for device in self.devices
            ],
            'scheduler_stats': self.scheduler.get_scheduler_stats() if self.scheduler else {},
            'configuration': {
                'workload_strategy': self.config.workload_strategy.value,
                'peer_to_peer_enabled': self.config.enable_peer_to_peer,
                'memory_pool_fraction': self.config.memory_pool_fraction,
                'stream_pool_size': self.config.stream_pool_size
            },
            'capabilities': [
                f'{len(self.devices)} GPU devices available',
                f'Workload distribution: {self.config.workload_strategy.value}',
                'Peer-to-peer memory access' if self.config.enable_peer_to_peer else 'No P2P access',
                'Automatic load balancing',
                'Fault tolerance and recovery',
                'Dynamic device selection'
            ]
        }
    
    def shutdown(self):
        """Shutdown multi-GPU manager"""
        if self.scheduler:
            self.scheduler.stop()
        
        logger.info("Multi-GPU manager shutdown complete")


def run_multi_gpu_demo():
    """Demonstrate multi-GPU capabilities"""
    print("🚀 Multi-GPU Acceleration Demo")
    print("=" * 50)
    
    # Initialize multi-GPU manager
    config = MultiGPUConfig(
        workload_strategy=WorkloadDistributionStrategy.DYNAMIC_ADAPTIVE,
        enable_peer_to_peer=True,
        stream_pool_size=4
    )
    
    manager = MultiGPUManager(config)
    
    if not manager.devices:
        print("❌ No GPU devices available for demo")
        return
    
    print(f"✅ Initialized {len(manager.devices)} GPU devices")
    
    # Display system status
    status = manager.get_system_status()
    print(f"\n📊 System Status:")
    for device in status['devices']:
        print(f"   GPU {device['device_id']}: {device['name']} "
              f"(Load: {device['current_load']:.1%})")
    
    # Run benchmark
    print(f"\n🧪 Running multi-GPU benchmark...")
    
    try:
        benchmark_results = manager.benchmark_multi_gpu(
            workload_size=500,
            compute_intensity="medium"
        )
        
        print(f"✅ Benchmark Results:")
        print(f"   Speedup: {benchmark_results['speedup']:.2f}x")
        print(f"   Efficiency: {benchmark_results['efficiency']:.2f}")
        print(f"   Multi-GPU time: {benchmark_results['multi_gpu_time']:.3f}s")
        
        scheduler_stats = benchmark_results['scheduler_stats']
        print(f"   Tasks completed: {scheduler_stats.get('tasks_completed', 0)}")
        print(f"   Average execution time: {scheduler_stats.get('average_execution_time', 0):.3f}s")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
    
    # Test parallel processing
    print(f"\n⚡ Testing parallel matrix operations...")
    
    try:
        # Create test matrices
        matrices = [np.random.randn(200, 200).astype(np.float32) for _ in range(20)]
        
        # Parallel computation
        start_time = time.time()
        results = manager.parallel_map(
            lambda x: np.linalg.norm(x @ x.T),
            matrices,
            priority=GPUTaskPriority.HIGH
        )
        parallel_time = time.time() - start_time
        
        print(f"✅ Parallel computation completed:")
        print(f"   Processed {len(matrices)} matrices in {parallel_time:.3f}s")
        print(f"   Results range: {min(results):.2f} - {max(results):.2f}")
        
    except Exception as e:
        print(f"❌ Parallel computation failed: {e}")
    
    # Display final system status
    final_status = manager.get_system_status()
    print(f"\n📈 Final Device Utilization:")
    for device in final_status['devices']:
        print(f"   GPU {device['device_id']}: {device['current_load']:.1%} load")
    
    # Cleanup
    manager.shutdown()
    print(f"\n✅ Multi-GPU demo completed successfully!")


if __name__ == "__main__":
    run_multi_gpu_demo()