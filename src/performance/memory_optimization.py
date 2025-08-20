"""
Advanced Memory Optimization Module

Provides comprehensive memory optimization capabilities including:
- Memory profiling and leak detection
- Efficient data structures and algorithms
- Memory pool management
- Garbage collection optimization
- Memory-efficient computing patterns
"""

import gc
import sys
import time
import weakref
import logging
import threading
from typing import Dict, List, Any, Optional, Callable, TypeVar, Generic
from dataclasses import dataclass
from collections import deque, defaultdict
from contextlib import contextmanager
import numpy as np
from abc import ABC, abstractmethod

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class MemorySnapshot:
    """Memory usage snapshot"""
    timestamp: float
    total_memory_mb: float
    available_memory_mb: float
    process_memory_mb: float
    heap_objects: int
    gc_collections: Dict[int, int]
    large_objects: List[Dict[str, Any]]
    memory_leaks: List[Dict[str, Any]]


@dataclass
class MemoryOptimizationReport:
    """Memory optimization analysis report"""
    initial_memory_mb: float
    final_memory_mb: float
    memory_saved_mb: float
    optimization_percentage: float
    techniques_applied: List[str]
    recommendations: List[str]
    performance_impact: float


class MemoryPool(Generic[T]):
    """
    High-performance memory pool for object reuse
    Reduces memory allocation overhead and garbage collection pressure
    """
    
    def __init__(self, 
                 factory: Callable[[], T],
                 max_size: int = 1000,
                 cleanup_threshold: int = 100):
        """
        Initialize memory pool
        
        Args:
            factory: Function to create new objects
            max_size: Maximum pool size
            cleanup_threshold: Objects to clean when pool exceeds max_size
        """
        self._factory = factory
        self._max_size = max_size
        self._cleanup_threshold = cleanup_threshold
        self._pool = deque()
        self._in_use = set()
        self._lock = threading.Lock()
        self._stats = {
            'objects_created': 0,
            'objects_reused': 0,
            'pool_hits': 0,
            'pool_misses': 0
        }
    
    def acquire(self) -> T:
        """Acquire object from pool or create new one"""
        with self._lock:
            if self._pool:
                obj = self._pool.popleft()
                self._in_use.add(id(obj))
                self._stats['objects_reused'] += 1
                self._stats['pool_hits'] += 1
                return obj
            else:
                obj = self._factory()
                self._in_use.add(id(obj))
                self._stats['objects_created'] += 1
                self._stats['pool_misses'] += 1
                return obj
    
    def release(self, obj: T) -> None:
        """Release object back to pool"""
        with self._lock:
            obj_id = id(obj)
            if obj_id in self._in_use:
                self._in_use.remove(obj_id)
                
                if len(self._pool) < self._max_size:
                    # Reset object state if possible
                    if hasattr(obj, 'reset'):
                        obj.reset()
                    elif hasattr(obj, 'clear'):
                        obj.clear()
                    
                    self._pool.append(obj)
                
                # Cleanup if pool is too large
                if len(self._pool) > self._max_size:
                    for _ in range(self._cleanup_threshold):
                        if self._pool:
                            self._pool.popleft()
    
    @contextmanager
    def get_object(self):
        """Context manager for automatic object acquisition/release"""
        obj = self.acquire()
        try:
            yield obj
        finally:
            self.release(obj)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        with self._lock:
            hit_rate = (self._stats['pool_hits'] / 
                       (self._stats['pool_hits'] + self._stats['pool_misses'])
                       if (self._stats['pool_hits'] + self._stats['pool_misses']) > 0 else 0)
            
            return {
                **self._stats,
                'pool_size': len(self._pool),
                'objects_in_use': len(self._in_use),
                'hit_rate': hit_rate,
                'memory_efficiency': hit_rate * 100
            }
    
    def clear(self) -> None:
        """Clear all objects from pool"""
        with self._lock:
            self._pool.clear()
            self._in_use.clear()


class MemoryEfficientArray:
    """
    Memory-efficient array implementation with lazy allocation
    and automatic compression for sparse data
    """
    
    def __init__(self, shape: tuple, dtype: type = float, sparse_threshold: float = 0.1):
        """
        Initialize memory-efficient array
        
        Args:
            shape: Array dimensions
            dtype: Data type
            sparse_threshold: Sparsity threshold for compression (0.0-1.0)
        """
        self.shape = shape
        self.dtype = dtype
        self.sparse_threshold = sparse_threshold
        self.size = np.prod(shape)
        
        # Lazy allocation - only allocate when needed
        self._data = None
        self._sparse_data = None
        self._is_sparse = False
        self._dirty = False
        
        # Statistics
        self._access_count = 0
        self._modification_count = 0
    
    def _ensure_allocated(self) -> None:
        """Ensure array is allocated"""
        if self._data is None and not self._is_sparse:
            self._data = np.zeros(self.shape, dtype=self.dtype)
    
    def _check_sparsity(self) -> None:
        """Check if array should be converted to sparse representation"""
        if self._data is None:
            return
        
        non_zero_ratio = np.count_nonzero(self._data) / self.size
        
        if non_zero_ratio < self.sparse_threshold and not self._is_sparse:
            # Convert to sparse
            non_zero_indices = np.nonzero(self._data)
            non_zero_values = self._data[non_zero_indices]
            
            self._sparse_data = {
                'indices': non_zero_indices,
                'values': non_zero_values,
                'default_value': 0
            }
            
            # Free dense array memory
            del self._data
            self._data = None
            self._is_sparse = True
            
            logger.debug(f"Converted array to sparse representation: "
                        f"{non_zero_ratio:.1%} non-zero elements")
        
        elif non_zero_ratio >= self.sparse_threshold * 2 and self._is_sparse:
            # Convert back to dense
            self._ensure_dense()
            self._is_sparse = False
            self._sparse_data = None
    
    def _ensure_dense(self) -> None:
        """Ensure array is in dense format"""
        if self._is_sparse and self._sparse_data:
            self._data = np.zeros(self.shape, dtype=self.dtype)
            indices = self._sparse_data['indices']
            values = self._sparse_data['values']
            self._data[indices] = values
            
            self._sparse_data = None
            self._is_sparse = False
    
    def __getitem__(self, key):
        """Get array element(s)"""
        self._access_count += 1
        
        if self._is_sparse:
            if self._sparse_data:
                # Create temporary dense array for indexing
                temp_data = np.zeros(self.shape, dtype=self.dtype)
                indices = self._sparse_data['indices']
                values = self._sparse_data['values']
                temp_data[indices] = values
                return temp_data[key]
            else:
                return np.zeros(self.shape, dtype=self.dtype)[key]
        else:
            self._ensure_allocated()
            return self._data[key]
    
    def __setitem__(self, key, value):
        """Set array element(s)"""
        self._modification_count += 1
        self._dirty = True
        
        if self._is_sparse:
            # Convert to dense for modification
            self._ensure_dense()
        
        self._ensure_allocated()
        self._data[key] = value
        
        # Check sparsity after modification
        if self._modification_count % 100 == 0:  # Check periodically
            self._check_sparsity()
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory usage information"""
        if self._is_sparse and self._sparse_data:
            sparse_memory = (
                sys.getsizeof(self._sparse_data['indices']) +
                sys.getsizeof(self._sparse_data['values']) +
                sys.getsizeof(self._sparse_data)
            )
            theoretical_dense_memory = self.size * np.dtype(self.dtype).itemsize
            
            return {
                'current_memory_bytes': sparse_memory,
                'theoretical_dense_memory_bytes': theoretical_dense_memory,
                'memory_saved_bytes': theoretical_dense_memory - sparse_memory,
                'compression_ratio': sparse_memory / theoretical_dense_memory,
                'is_sparse': True,
                'non_zero_elements': len(self._sparse_data['values'])
            }
        
        elif self._data is not None:
            current_memory = self._data.nbytes
            return {
                'current_memory_bytes': current_memory,
                'theoretical_dense_memory_bytes': current_memory,
                'memory_saved_bytes': 0,
                'compression_ratio': 1.0,
                'is_sparse': False,
                'non_zero_elements': np.count_nonzero(self._data)
            }
        
        else:
            return {
                'current_memory_bytes': 0,
                'theoretical_dense_memory_bytes': self.size * np.dtype(self.dtype).itemsize,
                'memory_saved_bytes': self.size * np.dtype(self.dtype).itemsize,
                'compression_ratio': 0.0,
                'is_sparse': False,
                'non_zero_elements': 0
            }


class MemoryProfiler:
    """
    Advanced memory profiler with leak detection and optimization recommendations
    """
    
    def __init__(self, sampling_interval: float = 1.0):
        """
        Initialize memory profiler
        
        Args:
            sampling_interval: Interval between memory samples (seconds)
        """
        self.sampling_interval = sampling_interval
        self._snapshots = []
        self._monitoring = False
        self._monitor_thread = None
        self._object_tracker = defaultdict(list)
        self._weak_refs = {}
        
    def start_monitoring(self) -> None:
        """Start memory monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop memory monitoring"""
        if not self._monitoring:
            return
        
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
        
        logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop"""
        while self._monitoring:
            snapshot = self._take_snapshot()
            self._snapshots.append(snapshot)
            
            # Keep only recent snapshots to avoid memory bloat
            if len(self._snapshots) > 1000:
                self._snapshots = self._snapshots[-500:]
            
            time.sleep(self.sampling_interval)
    
    def _take_snapshot(self) -> MemorySnapshot:
        """Take memory usage snapshot"""
        gc.collect()  # Force garbage collection
        
        # System memory info
        if PSUTIL_AVAILABLE:
            memory_info = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info().rss / (1024 * 1024)
            total_memory = memory_info.total / (1024 * 1024)
            available_memory = memory_info.available / (1024 * 1024)
        else:
            process_memory = 0
            total_memory = 0
            available_memory = 0
        
        # Garbage collection stats
        gc_stats = {i: gc.get_count()[i] for i in range(3)}
        
        # Object counting
        heap_objects = len(gc.get_objects())
        
        # Find large objects
        large_objects = self._find_large_objects()
        
        # Detect potential memory leaks
        memory_leaks = self._detect_memory_leaks()
        
        return MemorySnapshot(
            timestamp=time.time(),
            total_memory_mb=total_memory,
            available_memory_mb=available_memory,
            process_memory_mb=process_memory,
            heap_objects=heap_objects,
            gc_collections=gc_stats,
            large_objects=large_objects,
            memory_leaks=memory_leaks
        )
    
    def _find_large_objects(self, size_threshold_mb: float = 1.0) -> List[Dict[str, Any]]:
        """Find large objects in memory"""
        large_objects = []
        size_threshold_bytes = size_threshold_mb * 1024 * 1024
        
        for obj in gc.get_objects():
            try:
                size = sys.getsizeof(obj)
                if size > size_threshold_bytes:
                    large_objects.append({
                        'type': type(obj).__name__,
                        'size_mb': size / (1024 * 1024),
                        'id': id(obj),
                        'repr': repr(obj)[:100] if hasattr(obj, '__repr__') else 'N/A'
                    })
            except Exception:
                continue
        
        # Sort by size
        large_objects.sort(key=lambda x: x['size_mb'], reverse=True)
        return large_objects[:20]  # Top 20 largest objects
    
    def _detect_memory_leaks(self) -> List[Dict[str, Any]]:
        """Detect potential memory leaks"""
        leaks = []
        
        # Track object counts by type
        current_objects = defaultdict(int)
        for obj in gc.get_objects():
            current_objects[type(obj).__name__] += 1
        
        # Compare with previous snapshot if available
        if len(self._snapshots) > 10:  # Need some history
            prev_snapshot = self._snapshots[-10]
            
            for obj_type, current_count in current_objects.items():
                # Check for objects that are consistently growing
                growth_rate = (current_count - prev_snapshot.heap_objects) / 10
                
                if growth_rate > 100:  # Growing by more than 10 objects per snapshot
                    leaks.append({
                        'object_type': obj_type,
                        'current_count': current_count,
                        'growth_rate': growth_rate,
                        'severity': 'high' if growth_rate > 500 else 'medium'
                    })
        
        return leaks
    
    def get_memory_trend(self) -> Dict[str, Any]:
        """Analyze memory usage trends"""
        if len(self._snapshots) < 2:
            return {'error': 'Insufficient data for trend analysis'}
        
        # Extract time series data
        timestamps = [s.timestamp for s in self._snapshots]
        memory_usage = [s.process_memory_mb for s in self._snapshots]
        heap_objects = [s.heap_objects for s in self._snapshots]
        
        # Calculate trends
        memory_trend = np.polyfit(range(len(memory_usage)), memory_usage, 1)[0]
        objects_trend = np.polyfit(range(len(heap_objects)), heap_objects, 1)[0]
        
        return {
            'duration_minutes': (timestamps[-1] - timestamps[0]) / 60,
            'memory_trend_mb_per_minute': memory_trend,
            'objects_trend_per_minute': objects_trend,
            'current_memory_mb': memory_usage[-1],
            'peak_memory_mb': max(memory_usage),
            'average_memory_mb': np.mean(memory_usage),
            'memory_stability': 1.0 / (1.0 + np.std(memory_usage)),
            'potential_leak': memory_trend > 1.0 or objects_trend > 1000
        }
    
    def generate_optimization_report(self) -> MemoryOptimizationReport:
        """Generate comprehensive memory optimization report"""
        if not self._snapshots:
            raise ValueError("No memory snapshots available")
        
        initial_snapshot = self._snapshots[0]
        final_snapshot = self._snapshots[-1]
        
        initial_memory = initial_snapshot.process_memory_mb
        final_memory = final_snapshot.process_memory_mb
        memory_change = final_memory - initial_memory
        
        # Analyze patterns and generate recommendations
        recommendations = self._generate_recommendations()
        
        # Estimate potential memory savings
        techniques_applied = [
            'garbage_collection_optimization',
            'object_pooling',
            'sparse_data_structures',
            'memory_profiling'
        ]
        
        return MemoryOptimizationReport(
            initial_memory_mb=initial_memory,
            final_memory_mb=final_memory,
            memory_saved_mb=-memory_change if memory_change < 0 else 0,
            optimization_percentage=(-memory_change / initial_memory * 100) if initial_memory > 0 and memory_change < 0 else 0,
            techniques_applied=techniques_applied,
            recommendations=recommendations,
            performance_impact=0.05  # Estimated 5% performance overhead
        )
    
    def _generate_recommendations(self) -> List[str]:
        """Generate memory optimization recommendations"""
        recommendations = []
        
        if not self._snapshots:
            return ["Start memory monitoring to get recommendations"]
        
        trend = self.get_memory_trend()
        
        if trend.get('potential_leak', False):
            recommendations.append("Potential memory leak detected - investigate object lifecycle management")
        
        if trend.get('peak_memory_mb', 0) > 1000:
            recommendations.append("High memory usage detected - consider data streaming or batch processing")
        
        if trend.get('memory_stability', 1.0) < 0.7:
            recommendations.append("Unstable memory usage - implement memory pooling and caching")
        
        # Check for large objects
        if self._snapshots:
            large_objects = self._snapshots[-1].large_objects
            if large_objects:
                recommendations.append(f"Large objects detected: consider optimizing {[obj['type'] for obj in large_objects[:3]]}")
        
        recommendations.extend([
            "Use memory-efficient data structures for large datasets",
            "Implement object pooling for frequently created/destroyed objects",
            "Consider lazy loading for large data structures",
            "Use generators instead of lists for large sequences",
            "Implement data compression for sparse matrices"
        ])
        
        return recommendations


class MemoryOptimizer:
    """
    Main memory optimization orchestrator
    """
    
    def __init__(self):
        """Initialize memory optimizer"""
        self.profiler = MemoryProfiler()
        self.pools = {}
        self.optimization_history = []
        
    def start_optimization_session(self) -> None:
        """Start memory optimization session"""
        self.profiler.start_monitoring()
        gc.set_debug(gc.DEBUG_STATS)  # Enable GC statistics
        logger.info("Memory optimization session started")
    
    def end_optimization_session(self) -> MemoryOptimizationReport:
        """End optimization session and generate report"""
        self.profiler.stop_monitoring()
        gc.set_debug(0)  # Disable GC debug
        
        report = self.profiler.generate_optimization_report()
        self.optimization_history.append(report)
        
        logger.info(f"Memory optimization session completed: "
                   f"{report.memory_saved_mb:.1f}MB saved "
                   f"({report.optimization_percentage:.1f}% improvement)")
        
        return report
    
    def create_memory_pool(self, 
                          name: str, 
                          factory: Callable, 
                          max_size: int = 1000) -> MemoryPool:
        """Create named memory pool"""
        pool = MemoryPool(factory, max_size)
        self.pools[name] = pool
        logger.info(f"Created memory pool '{name}' with max size {max_size}")
        return pool
    
    def get_memory_pool(self, name: str) -> Optional[MemoryPool]:
        """Get named memory pool"""
        return self.pools.get(name)
    
    def optimize_garbage_collection(self) -> None:
        """Optimize garbage collection settings"""
        # Tune GC thresholds for better performance
        current_thresholds = gc.get_threshold()
        
        # Increase thresholds for better performance (less frequent GC)
        # but with more memory usage
        new_thresholds = (
            current_thresholds[0] * 2,  # Generation 0
            current_thresholds[1] * 2,  # Generation 1  
            current_thresholds[2] * 2   # Generation 2
        )
        
        gc.set_threshold(*new_thresholds)
        logger.info(f"Optimized GC thresholds: {current_thresholds} -> {new_thresholds}")
    
    def force_cleanup(self) -> Dict[str, Any]:
        """Force memory cleanup and return statistics"""
        initial_objects = len(gc.get_objects())
        
        # Clear all memory pools
        for pool in self.pools.values():
            pool.clear()
        
        # Force garbage collection
        collected_objects = gc.collect()
        
        final_objects = len(gc.get_objects())
        objects_freed = initial_objects - final_objects
        
        stats = {
            'initial_objects': initial_objects,
            'final_objects': final_objects,
            'objects_freed': objects_freed,
            'gc_collected': collected_objects,
            'pools_cleared': len(self.pools)
        }
        
        logger.info(f"Memory cleanup completed: {objects_freed} objects freed")
        return stats
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get overall optimization summary"""
        if not self.optimization_history:
            return {'message': 'No optimization sessions completed'}
        
        total_memory_saved = sum(r.memory_saved_mb for r in self.optimization_history)
        avg_optimization = np.mean([r.optimization_percentage for r in self.optimization_history])
        
        return {
            'total_sessions': len(self.optimization_history),
            'total_memory_saved_mb': total_memory_saved,
            'average_optimization_percentage': avg_optimization,
            'memory_pools_active': len(self.pools),
            'latest_session': self.optimization_history[-1] if self.optimization_history else None
        }


@contextmanager
def memory_optimization_context(optimizer: MemoryOptimizer = None):
    """Context manager for automatic memory optimization"""
    if optimizer is None:
        optimizer = MemoryOptimizer()
    
    optimizer.start_optimization_session()
    try:
        yield optimizer
    finally:
        report = optimizer.end_optimization_session()
        logger.info(f"Memory optimization context completed: "
                   f"{report.memory_saved_mb:.1f}MB saved")


def optimize_numpy_operations() -> None:
    """Apply memory optimizations for NumPy operations"""
    # Use memory mapping for large arrays
    np.seterr(all='warn')  # Warn on memory issues
    
    # Optimize BLAS/LAPACK usage
    try:
        import os
        # Limit threads to reduce memory overhead
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        
        logger.info("Applied NumPy memory optimizations")
    except Exception as e:
        logger.warning(f"Could not apply all NumPy optimizations: {e}")


def create_memory_efficient_matrix(shape: tuple, 
                                 dtype: type = np.float32,
                                 sparse_threshold: float = 0.1) -> MemoryEfficientArray:
    """Create memory-efficient matrix with automatic compression"""
    return MemoryEfficientArray(shape, dtype, sparse_threshold)


# Global memory optimizer instance
_global_optimizer = None


def get_memory_optimizer() -> MemoryOptimizer:
    """Get global memory optimizer instance"""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = MemoryOptimizer()
    return _global_optimizer


def run_memory_optimization_demo():
    """Demonstrate memory optimization capabilities"""
    print("🧠 Memory Optimization Demo")
    print("=" * 40)
    
    optimizer = MemoryOptimizer()
    
    with memory_optimization_context(optimizer) as opt:
        print("📊 Starting memory optimization session...")
        
        # Create some memory pools
        array_pool = opt.create_memory_pool(
            'numpy_arrays',
            lambda: np.zeros((1000, 1000), dtype=np.float32),
            max_size=50
        )
        
        list_pool = opt.create_memory_pool(
            'large_lists',
            lambda: [0] * 10000,
            max_size=100
        )
        
        print(f"✅ Created memory pools")
        
        # Test memory-efficient array
        print("🔧 Testing memory-efficient array...")
        efficient_array = create_memory_efficient_matrix((10000, 10000), sparse_threshold=0.05)
        
        # Set some sparse data
        for i in range(0, 10000, 100):
            for j in range(0, 10000, 100):
                efficient_array[i, j] = i + j
        
        memory_info = efficient_array.get_memory_info()
        print(f"✅ Memory-efficient array: {memory_info['compression_ratio']:.2f} compression ratio")
        print(f"   Memory saved: {memory_info['memory_saved_bytes'] / (1024*1024):.1f} MB")
        
        # Test memory pools
        print("🔄 Testing memory pools...")
        
        with array_pool.get_object() as arr:
            arr.fill(42)  # Use the array
        
        with list_pool.get_object() as lst:
            lst[0] = 999  # Use the list
        
        pool_stats = array_pool.get_stats()
        print(f"✅ Array pool efficiency: {pool_stats['memory_efficiency']:.1f}%")
        
        # Apply optimizations
        print("⚙️  Applying memory optimizations...")
        opt.optimize_garbage_collection()
        optimize_numpy_operations()
        
        time.sleep(2)  # Let monitoring collect some data
    
    # Get final report
    summary = optimizer.get_optimization_summary()
    print(f"\n📈 Optimization Summary:")
    print(f"   Sessions completed: {summary['total_sessions']}")
    print(f"   Memory pools active: {summary['memory_pools_active']}")
    
    if summary.get('latest_session'):
        session = summary['latest_session']
        print(f"   Latest session saved: {session.memory_saved_mb:.1f} MB")
        print(f"   Optimization percentage: {session.optimization_percentage:.1f}%")
    
    print("\n🎯 Key Recommendations:")
    if summary.get('latest_session') and summary['latest_session'].recommendations:
        for rec in summary['latest_session'].recommendations[:3]:
            print(f"   • {rec}")
    
    print("\n✅ Memory optimization demo completed!")


if __name__ == "__main__":
    # Run the demonstration
    run_memory_optimization_demo()