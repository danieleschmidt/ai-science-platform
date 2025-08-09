"""Performance optimization utilities for the AI Science Platform"""

import time
import functools
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import logging
from dataclasses import dataclass
import weakref
import psutil
import gc
from collections import OrderedDict, defaultdict

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    cache_hits: int = 0
    cache_misses: int = 0
    parallel_speedup: Optional[float] = None


class PerformanceProfiler:
    """Performance profiler for tracking execution metrics"""
    
    def __init__(self):
        self.metrics_history = defaultdict(list)
        self.current_metrics = {}
    
    def start_profiling(self, operation_name: str):
        """Start profiling an operation"""
        self.current_metrics[operation_name] = {
            'start_time': time.time(),
            'start_memory': psutil.Process().memory_info().rss / 1024 / 1024,
            'start_cpu': psutil.cpu_percent()
        }
    
    def end_profiling(self, operation_name: str) -> PerformanceMetrics:
        """End profiling and return metrics"""
        if operation_name not in self.current_metrics:
            raise ValueError(f"No profiling started for operation: {operation_name}")
        
        start_data = self.current_metrics[operation_name]
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        end_cpu = psutil.cpu_percent()
        
        metrics = PerformanceMetrics(
            execution_time=end_time - start_data['start_time'],
            memory_usage_mb=max(0, end_memory - start_data['start_memory']),
            cpu_usage_percent=end_cpu
        )
        
        self.metrics_history[operation_name].append(metrics)
        del self.current_metrics[operation_name]
        
        return metrics
    
    def get_average_metrics(self, operation_name: str) -> Optional[PerformanceMetrics]:
        """Get average metrics for an operation"""
        if operation_name not in self.metrics_history:
            return None
        
        metrics_list = self.metrics_history[operation_name]
        if not metrics_list:
            return None
        
        return PerformanceMetrics(
            execution_time=np.mean([m.execution_time for m in metrics_list]),
            memory_usage_mb=np.mean([m.memory_usage_mb for m in metrics_list]),
            cpu_usage_percent=np.mean([m.cpu_usage_percent for m in metrics_list]),
            cache_hits=sum(m.cache_hits for m in metrics_list),
            cache_misses=sum(m.cache_misses for m in metrics_list)
        )


class LRUCache:
    """Thread-safe Least Recently Used cache with size limits"""
    
    def __init__(self, maxsize: int = 128, ttl: Optional[float] = None):
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache = OrderedDict()
        self.timestamps = {}
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: Any) -> Tuple[Any, bool]:
        """Get item from cache, returns (value, hit)"""
        with self.lock:
            current_time = time.time()
            
            # Check if key exists and is not expired
            if key in self.cache:
                if self.ttl is None or (current_time - self.timestamps[key]) < self.ttl:
                    # Move to end (most recently used)
                    value = self.cache.pop(key)
                    self.cache[key] = value
                    self.hits += 1
                    return value, True
                else:
                    # Expired, remove
                    del self.cache[key]
                    del self.timestamps[key]
            
            self.misses += 1
            return None, False
    
    def put(self, key: Any, value: Any):
        """Put item in cache"""
        with self.lock:
            current_time = time.time()
            
            if key in self.cache:
                # Update existing
                self.cache.pop(key)
                self.cache[key] = value
                self.timestamps[key] = current_time
            else:
                # Add new
                if len(self.cache) >= self.maxsize:
                    # Remove least recently used
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                    del self.timestamps[oldest_key]
                
                self.cache[key] = value
                self.timestamps[key] = current_time
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            self.hits = 0
            self.misses = 0
    
    def stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        with self.lock:
            return {
                'size': len(self.cache),
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
            }


class MemoryOptimizer:
    """Memory optimization utilities"""
    
    @staticmethod
    def optimize_array_dtype(arr: np.ndarray, preserve_precision: bool = True) -> np.ndarray:
        """Optimize array dtype to reduce memory usage"""
        if not isinstance(arr, np.ndarray):
            return arr
        
        original_dtype = arr.dtype
        
        # Integer optimization
        if np.issubdtype(arr.dtype, np.integer):
            if preserve_precision:
                # Find minimum integer type that can hold the data
                min_val, max_val = np.min(arr), np.max(arr)
                
                if np.int8(min_val) == min_val and np.int8(max_val) == max_val:
                    optimized = arr.astype(np.int8)
                elif np.int16(min_val) == min_val and np.int16(max_val) == max_val:
                    optimized = arr.astype(np.int16)
                elif np.int32(min_val) == min_val and np.int32(max_val) == max_val:
                    optimized = arr.astype(np.int32)
                else:
                    optimized = arr.astype(np.int64)
            else:
                optimized = arr.astype(np.int32)
        
        # Float optimization
        elif np.issubdtype(arr.dtype, np.floating):
            if preserve_precision:
                # Check if float32 precision is sufficient
                if arr.dtype == np.float64:
                    arr_f32 = arr.astype(np.float32)
                    if np.allclose(arr, arr_f32, rtol=1e-6):
                        optimized = arr_f32
                    else:
                        optimized = arr
                else:
                    optimized = arr
            else:
                optimized = arr.astype(np.float32)
        else:
            optimized = arr
        
        memory_saved = arr.nbytes - optimized.nbytes
        if memory_saved > 0:
            logger.debug(f"Optimized array dtype from {original_dtype} to {optimized.dtype}, "
                        f"saved {memory_saved} bytes ({memory_saved/arr.nbytes*100:.1f}%)")
        
        return optimized
    
    @staticmethod
    def batch_process_arrays(arrays: List[np.ndarray], 
                           process_func: Callable, 
                           batch_size: Optional[int] = None,
                           **kwargs) -> List[Any]:
        """Process arrays in batches to manage memory usage"""
        if batch_size is None:
            # Estimate batch size based on available memory
            available_memory = psutil.virtual_memory().available
            total_array_memory = sum(arr.nbytes for arr in arrays)
            
            if total_array_memory > 0:
                batch_size = max(1, int(len(arrays) * available_memory * 0.5 / total_array_memory))
            else:
                batch_size = len(arrays)
        
        results = []
        for i in range(0, len(arrays), batch_size):
            batch = arrays[i:i+batch_size]
            batch_results = [process_func(arr, **kwargs) for arr in batch]
            results.extend(batch_results)
            
            # Force garbage collection between batches
            gc.collect()
        
        return results


class ParallelProcessor:
    """Parallel processing utilities with automatic optimization"""
    
    def __init__(self, max_workers: Optional[int] = None, use_processes: bool = False):
        self.max_workers = max_workers or min(8, multiprocessing.cpu_count())
        self.use_processes = use_processes
        self.executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    
    def parallel_map(self, func: Callable, items: List[Any], 
                    chunk_size: Optional[int] = None, 
                    show_progress: bool = False) -> List[Any]:
        """Apply function to items in parallel"""
        if len(items) == 0:
            return []
        
        # Determine optimal chunk size
        if chunk_size is None:
            chunk_size = max(1, len(items) // (self.max_workers * 4))
        
        # Use sequential processing for small datasets
        if len(items) < self.max_workers * 2:
            return [func(item) for item in items]
        
        results = [None] * len(items)
        
        with self.executor_class(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {}
            for i in range(0, len(items), chunk_size):
                chunk = items[i:i + chunk_size]
                chunk_indices = list(range(i, min(i + chunk_size, len(items))))
                
                if len(chunk) == 1:
                    future = executor.submit(func, chunk[0])
                    future_to_index[future] = chunk_indices
                else:
                    future = executor.submit(self._process_chunk, func, chunk)
                    future_to_index[future] = chunk_indices
            
            # Collect results
            completed = 0
            for future in as_completed(future_to_index):
                indices = future_to_index[future]
                try:
                    chunk_results = future.result()
                    if not isinstance(chunk_results, list):
                        chunk_results = [chunk_results]
                    
                    for idx, result in zip(indices, chunk_results):
                        results[idx] = result
                    
                    completed += len(indices)
                    
                    if show_progress:
                        progress = completed / len(items) * 100
                        print(f"Progress: {progress:.1f}% ({completed}/{len(items)})", end='\r')
                        
                except Exception as e:
                    logger.error(f"Parallel processing error: {str(e)}")
                    # Fill with None for failed items
                    for idx in indices:
                        results[idx] = None
        
        if show_progress:
            print()  # New line after progress
        
        return results
    
    @staticmethod
    def _process_chunk(func: Callable, chunk: List[Any]) -> List[Any]:
        """Process a chunk of items"""
        return [func(item) for item in chunk]
    
    def parallel_reduce(self, func: Callable, items: List[Any], 
                       reduce_func: Callable, initial_value: Any = None) -> Any:
        """Parallel map-reduce operation"""
        if not items:
            return initial_value
        
        # Map phase
        mapped_results = self.parallel_map(func, items)
        
        # Reduce phase
        result = initial_value
        for mapped_result in mapped_results:
            if result is None:
                result = mapped_result
            else:
                result = reduce_func(result, mapped_result)
        
        return result


# Global instances
_global_profiler = None
_global_cache = None
_global_optimizer = None
_global_parallel_processor = None

def get_profiler() -> PerformanceProfiler:
    """Get global performance profiler"""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler

def get_cache(maxsize: int = 128, ttl: Optional[float] = None) -> LRUCache:
    """Get global cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = LRUCache(maxsize=maxsize, ttl=ttl)
    return _global_cache

def get_memory_optimizer() -> MemoryOptimizer:
    """Get global memory optimizer"""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = MemoryOptimizer()
    return _global_optimizer

def get_parallel_processor(max_workers: Optional[int] = None, 
                         use_processes: bool = False) -> ParallelProcessor:
    """Get global parallel processor"""
    global _global_parallel_processor
    if _global_parallel_processor is None:
        _global_parallel_processor = ParallelProcessor(max_workers, use_processes)
    return _global_parallel_processor


# Decorators
def cached(maxsize: int = 128, ttl: Optional[float] = None, key_func: Optional[Callable] = None):
    """Decorator for caching function results"""
    def decorator(func: Callable) -> Callable:
        cache = LRUCache(maxsize=maxsize, ttl=ttl)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Simple key generation (not suitable for complex objects)
                cache_key = str(args) + str(sorted(kwargs.items()))
            
            # Try to get from cache
            result, hit = cache.get(cache_key)
            if hit:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.put(cache_key, result)
            
            return result
        
        # Add cache management methods
        wrapper.cache_clear = cache.clear
        wrapper.cache_stats = cache.stats
        
        return wrapper
    return decorator


def profiled(operation_name: Optional[str] = None):
    """Decorator for profiling function performance"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = get_profiler()
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            profiler.start_profiling(op_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                metrics = profiler.end_profiling(op_name)
                logger.debug(f"Profile {op_name}: {metrics.execution_time:.3f}s, "
                           f"{metrics.memory_usage_mb:.1f}MB, {metrics.cpu_usage_percent:.1f}% CPU")
        
        return wrapper
    return decorator


def memory_optimized(preserve_precision: bool = True):
    """Decorator for memory optimization of array operations"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            optimizer = get_memory_optimizer()
            
            # Optimize input arrays
            optimized_args = []
            for arg in args:
                if isinstance(arg, np.ndarray):
                    optimized_args.append(optimizer.optimize_array_dtype(arg, preserve_precision))
                else:
                    optimized_args.append(arg)
            
            optimized_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, np.ndarray):
                    optimized_kwargs[k] = optimizer.optimize_array_dtype(v, preserve_precision)
                else:
                    optimized_kwargs[k] = v
            
            # Execute function with optimized inputs
            result = func(*optimized_args, **optimized_kwargs)
            
            # Optimize result if it's an array
            if isinstance(result, np.ndarray):
                result = optimizer.optimize_array_dtype(result, preserve_precision)
            elif isinstance(result, (list, tuple)) and len(result) > 0 and isinstance(result[0], np.ndarray):
                result = type(result)(optimizer.optimize_array_dtype(arr, preserve_precision) 
                                    for arr in result)
            
            return result
        
        return wrapper
    return decorator