"""
Scalable Framework - Performance Optimization and Scaling
Generation 3: MAKE IT SCALE
"""

import asyncio
import concurrent.futures
import multiprocessing as mp
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
import numpy as np
import time
import threading
from dataclasses import dataclass
from functools import wraps, lru_cache
import queue
import gc
import weakref

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class ScalingConfig:
    """Configuration for scaling and performance optimization"""
    max_workers: int = None  # Auto-detect based on CPU count
    chunk_size: int = 1000
    memory_limit_mb: int = 2048
    enable_caching: bool = True
    cache_size: int = 1000
    enable_gpu: bool = False
    parallel_threshold: int = 100
    async_threshold: int = 50
    
    def __post_init__(self):
        if self.max_workers is None:
            self.max_workers = min(32, (mp.cpu_count() or 1) + 4)


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring"""
    execution_time: float
    memory_usage_mb: float
    cpu_usage: float
    throughput: float  # operations per second
    parallel_efficiency: float
    cache_hit_rate: float


class ResourcePool:
    """Managed resource pool for scaling operations"""
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.pool = queue.Queue(maxsize=max_size)
        self.created_count = 0
        self.lock = threading.Lock()
        
    def acquire(self):
        """Acquire a resource from the pool"""
        try:
            return self.pool.get_nowait()
        except queue.Empty:
            with self.lock:
                if self.created_count < self.max_size:
                    self.created_count += 1
                    return self._create_resource()
                else:
                    # Wait for available resource
                    return self.pool.get(timeout=30)
    
    def release(self, resource):
        """Release a resource back to the pool"""
        try:
            self.pool.put_nowait(resource)
        except queue.Full:
            # Pool is full, dispose of resource
            self._dispose_resource(resource)
    
    def _create_resource(self):
        """Create a new resource (override in subclasses)"""
        return {}
    
    def _dispose_resource(self, resource):
        """Dispose of a resource (override in subclasses)"""
        pass


class AdaptiveCache:
    """Adaptive caching system with intelligent eviction"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
        self.access_counts = {}
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key in self.cache:
                self.access_counts[key] = self.access_counts.get(key, 0) + 1
                self.access_times[key] = time.time()
                self.hit_count += 1
                return self.cache[key]
            else:
                self.miss_count += 1
                return None
    
    def put(self, key: str, value: Any):
        """Put value in cache with intelligent eviction"""
        with self.lock:
            if len(self.cache) >= self.max_size:
                self._evict_items()
            
            self.cache[key] = value
            self.access_counts[key] = 1
            self.access_times[key] = time.time()
    
    def _evict_items(self):
        """Evict items using LFU + LRU hybrid strategy"""
        if not self.cache:
            return
        
        # Calculate eviction scores (lower = more likely to evict)
        current_time = time.time()
        scores = {}
        
        for key in self.cache:
            frequency = self.access_counts.get(key, 1)
            recency = current_time - self.access_times.get(key, current_time)
            
            # Hybrid score: frequency * recency_weight
            score = frequency * max(0.1, 1.0 / (recency + 1))
            scores[key] = score
        
        # Remove 25% of items with lowest scores
        num_to_remove = max(1, len(self.cache) // 4)
        items_to_remove = sorted(scores.keys(), key=lambda k: scores[k])[:num_to_remove]
        
        for key in items_to_remove:
            self.cache.pop(key, None)
            self.access_counts.pop(key, None)
            self.access_times.pop(key, None)
    
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total_accesses = self.hit_count + self.miss_count
        return self.hit_count / total_accesses if total_accesses > 0 else 0.0
    
    def clear(self):
        """Clear the cache"""
        with self.lock:
            self.cache.clear()
            self.access_counts.clear()
            self.access_times.clear()


class BatchProcessor:
    """Efficient batch processing with dynamic sizing"""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.adaptive_cache = AdaptiveCache(config.cache_size)
        
    def process_batch(self, data: List[Any], processor_func: Callable,
                     batch_size: Optional[int] = None) -> List[Any]:
        """Process data in optimized batches"""
        
        if not data:
            return []
        
        # Determine optimal batch size
        if batch_size is None:
            batch_size = self._calculate_optimal_batch_size(len(data))
        
        results = []
        
        # Process in batches
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            
            # Check cache for batch results
            batch_key = self._generate_batch_key(batch, processor_func.__name__)
            cached_result = self.adaptive_cache.get(batch_key)
            
            if cached_result is not None:
                results.extend(cached_result)
            else:
                # Process batch
                batch_result = processor_func(batch)
                results.extend(batch_result)
                
                # Cache result
                if self.config.enable_caching:
                    self.adaptive_cache.put(batch_key, batch_result)
        
        return results
    
    def _calculate_optimal_batch_size(self, total_size: int) -> int:
        """Calculate optimal batch size based on system resources"""
        
        # Base batch size
        base_size = self.config.chunk_size
        
        # Adjust based on available memory
        if PSUTIL_AVAILABLE:
            memory = psutil.virtual_memory()
            available_mb = memory.available / 1024 / 1024
            
            # Scale batch size based on available memory
            memory_factor = min(2.0, available_mb / 1024)  # Cap at 2x for 1GB+ available
            base_size = int(base_size * memory_factor)
        
        # Ensure reasonable bounds
        min_size = max(1, total_size // 100)  # At least 1% of total
        max_size = max(min_size, total_size // 4)  # At most 25% of total
        
        return max(min_size, min(base_size, max_size))
    
    def _generate_batch_key(self, batch: List[Any], func_name: str) -> str:
        """Generate cache key for batch"""
        # Simple hash-based key (in production, use more sophisticated hashing)
        batch_repr = str(len(batch)) + str(type(batch[0]).__name__ if batch else "empty")
        return f"{func_name}_{hash(batch_repr) % 1000000}"


class ParallelExecutor:
    """High-performance parallel execution engine"""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers)
        self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=config.max_workers)
        self.resource_pool = ResourcePool(config.max_workers)
        
    def execute_parallel(self, tasks: List[Callable], use_processes: bool = False) -> List[Any]:
        """Execute tasks in parallel"""
        
        if len(tasks) < self.config.parallel_threshold:
            # Sequential execution for small task sets
            return [task() for task in tasks]
        
        executor = self.process_pool if use_processes else self.thread_pool
        
        try:
            futures = [executor.submit(task) for task in tasks]
            results = []
            
            for future in concurrent.futures.as_completed(futures, timeout=300):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append(f"Error: {str(e)}")
            
            return results
            
        except concurrent.futures.TimeoutError:
            return ["Timeout"] * len(tasks)
    
    def map_parallel(self, func: Callable, data: List[Any], 
                    chunk_size: Optional[int] = None, use_processes: bool = False) -> List[Any]:
        """Parallel map operation with chunking"""
        
        if len(data) < self.config.parallel_threshold:
            return list(map(func, data))
        
        # Determine chunk size
        if chunk_size is None:
            chunk_size = max(1, len(data) // self.config.max_workers)
        
        # Create chunks
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        # Define chunk processor
        def process_chunk(chunk):
            return [func(item) for item in chunk]
        
        # Execute chunks in parallel
        executor = self.process_pool if use_processes else self.thread_pool
        
        try:
            chunk_results = list(executor.map(process_chunk, chunks, timeout=300))
            
            # Flatten results
            results = []
            for chunk_result in chunk_results:
                results.extend(chunk_result)
            
            return results
            
        except concurrent.futures.TimeoutError:
            return ["Timeout"] * len(data)
    
    def shutdown(self):
        """Shutdown executors"""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)


class AsyncProcessor:
    """Asynchronous processing for I/O bound operations"""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_workers)
        
    async def process_async(self, coroutines: List[Callable]) -> List[Any]:
        """Process coroutines asynchronously"""
        
        if len(coroutines) < self.config.async_threshold:
            # Sequential execution for small sets
            results = []
            for coro in coroutines:
                if asyncio.iscoroutinefunction(coro):
                    result = await coro()
                else:
                    result = coro()
                results.append(result)
            return results
        
        async def limited_coro(coro):
            async with self.semaphore:
                if asyncio.iscoroutinefunction(coro):
                    return await coro()
                else:
                    return coro()
        
        tasks = [limited_coro(coro) for coro in coroutines]
        return await asyncio.gather(*tasks, return_exceptions=True)


def scalable_execution(enable_caching: bool = True,
                      enable_parallel: bool = True,
                      enable_async: bool = False,
                      chunk_size: Optional[int] = None):
    """
    Decorator for scalable execution with performance optimization
    
    Features:
    - Intelligent caching
    - Parallel execution
    - Asynchronous processing
    - Memory optimization
    - Performance monitoring
    """
    
    def decorator(func):
        # Initialize components
        config = ScalingConfig()
        if chunk_size:
            config.chunk_size = chunk_size
            
        cache = AdaptiveCache(config.cache_size) if enable_caching else None
        batch_processor = BatchProcessor(config)
        parallel_executor = ParallelExecutor(config) if enable_parallel else None
        async_processor = AsyncProcessor(config) if enable_async else None
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = _get_memory_usage()
            
            # Generate cache key
            cache_key = None
            if cache:
                cache_key = _generate_cache_key(func.__name__, args, kwargs)
                cached_result = cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # Execute function with optimization
            try:
                # Check if input is suitable for parallel processing
                if parallel_executor and _should_parallelize(args, kwargs):
                    result = _execute_with_parallelization(
                        func, args, kwargs, parallel_executor, batch_processor
                    )
                else:
                    result = func(*args, **kwargs)
                
                # Cache result
                if cache and cache_key:
                    cache.put(cache_key, result)
                
                # Log performance metrics
                execution_time = time.time() - start_time
                memory_usage = _get_memory_usage() - start_memory
                
                _log_performance_metrics(
                    func.__name__, execution_time, memory_usage,
                    cache.hit_rate() if cache else 0.0
                )
                
                return result
                
            except Exception as e:
                # Error handling with graceful degradation
                print(f"Scalable execution failed for {func.__name__}: {e}")
                # Fallback to basic execution
                return func(*args, **kwargs)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            """Async version of the wrapper"""
            start_time = time.time()
            
            # Similar caching logic for async
            cache_key = None
            if cache:
                cache_key = _generate_cache_key(func.__name__, args, kwargs)
                cached_result = cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # Execute async
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Cache and log
            if cache and cache_key:
                cache.put(cache_key, result)
            
            execution_time = time.time() - start_time
            _log_performance_metrics(func.__name__, execution_time, 0, 0)
            
            return result
        
        # Return appropriate wrapper based on function type
        if enable_async and asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


def _should_parallelize(args: tuple, kwargs: dict) -> bool:
    """Determine if function call should use parallelization"""
    
    # Check for large numpy arrays
    for arg in args:
        if isinstance(arg, np.ndarray) and arg.size > 1000:
            return True
    
    # Check for large lists
    for arg in args:
        if isinstance(arg, (list, tuple)) and len(arg) > 100:
            return True
    
    return False


def _execute_with_parallelization(func: Callable, args: tuple, kwargs: dict,
                                 executor: ParallelExecutor, 
                                 batch_processor: BatchProcessor) -> Any:
    """Execute function with parallelization strategies"""
    
    # Find the largest array/list argument for chunking
    largest_arg_idx = -1
    largest_size = 0
    
    for i, arg in enumerate(args):
        if isinstance(arg, (np.ndarray, list, tuple)):
            size = len(arg) if hasattr(arg, '__len__') else arg.size
            if size > largest_size:
                largest_size = size
                largest_arg_idx = i
    
    if largest_arg_idx == -1:
        # No suitable argument for parallelization
        return func(*args, **kwargs)
    
    # Chunk the largest argument and process in parallel
    large_arg = args[largest_arg_idx]
    chunk_size = max(1, len(large_arg) // executor.config.max_workers)
    
    # Create processing tasks
    def create_task(chunk):
        modified_args = list(args)
        modified_args[largest_arg_idx] = chunk
        return lambda: func(*tuple(modified_args), **kwargs)
    
    # Split into chunks
    if isinstance(large_arg, np.ndarray):
        chunks = np.array_split(large_arg, executor.config.max_workers)
    else:
        chunks = [large_arg[i:i + chunk_size] 
                 for i in range(0, len(large_arg), chunk_size)]
    
    # Execute chunks in parallel
    tasks = [create_task(chunk) for chunk in chunks]
    chunk_results = executor.execute_parallel(tasks)
    
    # Combine results (assuming they can be concatenated)
    if isinstance(chunk_results[0], np.ndarray):
        return np.concatenate(chunk_results)
    elif isinstance(chunk_results[0], (list, tuple)):
        combined = []
        for result in chunk_results:
            combined.extend(result)
        return combined
    else:
        return chunk_results


def _generate_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Generate cache key for function call"""
    
    # Simple hash-based approach (in production, use more sophisticated method)
    key_parts = [func_name]
    
    for arg in args:
        if isinstance(arg, np.ndarray):
            key_parts.append(f"array_{arg.shape}_{arg.dtype}")
        elif isinstance(arg, (list, tuple)):
            key_parts.append(f"seq_{len(arg)}_{type(arg).__name__}")
        else:
            key_parts.append(str(arg)[:50])  # Truncate long strings
    
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}={str(v)[:50]}")
    
    return "_".join(key_parts)


def _get_memory_usage() -> float:
    """Get current memory usage in MB"""
    if PSUTIL_AVAILABLE:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    return 0.0


def _log_performance_metrics(func_name: str, execution_time: float, 
                           memory_usage: float, cache_hit_rate: float):
    """Log performance metrics"""
    
    throughput = 1.0 / execution_time if execution_time > 0 else 0
    
    print(f"PERF[{func_name}]: "
          f"Time={execution_time:.3f}s, "
          f"Memory={memory_usage:.1f}MB, "
          f"Throughput={throughput:.1f}ops/s, "
          f"Cache={cache_hit_rate:.1%}")


class AutoScaler:
    """Automatic scaling based on workload and resources"""
    
    def __init__(self):
        self.metrics_history = []
        self.current_workers = mp.cpu_count()
        self.min_workers = 1
        self.max_workers = min(32, mp.cpu_count() * 4)
        
    def monitor_and_scale(self, current_load: float, response_time: float) -> int:
        """Monitor performance and adjust scaling"""
        
        # Record metrics
        self.metrics_history.append({
            'timestamp': time.time(),
            'load': current_load,
            'response_time': response_time,
            'workers': self.current_workers
        })
        
        # Keep only recent history
        cutoff_time = time.time() - 300  # 5 minutes
        self.metrics_history = [
            m for m in self.metrics_history 
            if m['timestamp'] > cutoff_time
        ]
        
        # Scaling decision logic
        if len(self.metrics_history) < 3:
            return self.current_workers
        
        avg_response_time = np.mean([m['response_time'] for m in self.metrics_history[-3:]])
        avg_load = np.mean([m['load'] for m in self.metrics_history[-3:]])
        
        # Scale up conditions
        if avg_response_time > 1.0 and avg_load > 0.8:
            new_workers = min(self.max_workers, int(self.current_workers * 1.5))
        
        # Scale down conditions
        elif avg_response_time < 0.3 and avg_load < 0.4:
            new_workers = max(self.min_workers, int(self.current_workers * 0.8))
        
        else:
            new_workers = self.current_workers
        
        if new_workers != self.current_workers:
            print(f"AutoScaler: Adjusting workers from {self.current_workers} to {new_workers}")
            self.current_workers = new_workers
        
        return self.current_workers


# Global instances for reuse
_global_config = ScalingConfig()
_global_cache = AdaptiveCache(_global_config.cache_size)
_global_batch_processor = BatchProcessor(_global_config)
_global_parallel_executor = ParallelExecutor(_global_config)
_global_autoscaler = AutoScaler()


def optimize_memory():
    """Optimize memory usage"""
    
    # Force garbage collection
    gc.collect()
    
    # Clear global cache if it's getting large
    if len(_global_cache.cache) > _global_config.cache_size * 0.9:
        _global_cache.clear()
        print("Memory optimization: Cache cleared")
    
    # Report memory usage
    if PSUTIL_AVAILABLE:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"Memory optimization: Current usage {memory_mb:.1f}MB")


def get_performance_stats() -> Dict[str, Any]:
    """Get comprehensive performance statistics"""
    
    stats = {
        'cache_hit_rate': _global_cache.hit_rate(),
        'cache_size': len(_global_cache.cache),
        'current_workers': _global_autoscaler.current_workers
    }
    
    if PSUTIL_AVAILABLE:
        process = psutil.Process()
        stats.update({
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'cpu_percent': process.cpu_percent(),
            'num_threads': process.num_threads()
        })
    
    return stats


# Cleanup function for graceful shutdown
def cleanup_scalable_framework():
    """Cleanup scalable framework resources"""
    try:
        _global_parallel_executor.shutdown()
        _global_cache.clear()
        print("Scalable framework cleanup completed")
    except Exception as e:
        print(f"Cleanup warning: {e}")


# Register cleanup at module level
import atexit
atexit.register(cleanup_scalable_framework)