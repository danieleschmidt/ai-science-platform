"""Performance profiling and monitoring utilities"""

import time
import logging
import cProfile
import pstats
import io
import functools
import threading
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, asdict
from pathlib import Path
import psutil
import numpy as np
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics container"""
    execution_time: float
    cpu_percent: float
    memory_mb: float
    peak_memory_mb: float
    function_calls: int
    cache_hits: int = 0
    cache_misses: int = 0
    io_operations: int = 0


@dataclass
class ProfileResult:
    """Profiling result container"""
    function_name: str
    total_time: float
    call_count: int
    metrics: PerformanceMetrics
    hotspots: List[Dict[str, Any]] = None
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.hotspots is None:
            self.hotspots = []
        if self.recommendations is None:
            self.recommendations = []


class PerformanceMonitor:
    """Real-time performance monitoring"""
    
    def __init__(self):
        self._start_time = None
        self._start_memory = None
        self._peak_memory = 0
        self._process = psutil.Process()
        self._measurements = []
        self._lock = threading.Lock()
    
    def start(self) -> None:
        """Start performance monitoring"""
        with self._lock:
            self._start_time = time.time()
            self._start_memory = self._get_memory_mb()
            self._peak_memory = self._start_memory
    
    def stop(self) -> PerformanceMetrics:
        """Stop monitoring and return metrics"""
        with self._lock:
            if self._start_time is None:
                raise ValueError("Performance monitoring not started")
            
            end_time = time.time()
            execution_time = end_time - self._start_time
            
            current_memory = self._get_memory_mb()
            cpu_percent = self._process.cpu_percent()
            
            metrics = PerformanceMetrics(
                execution_time=execution_time,
                cpu_percent=cpu_percent,
                memory_mb=current_memory,
                peak_memory_mb=self._peak_memory,
                function_calls=0  # Will be updated if using with profiler
            )
            
            # Reset for next measurement
            self._start_time = None
            self._start_memory = None
            self._peak_memory = 0
            
            return metrics
    
    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB"""
        try:
            memory_info = self._process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            # Track peak memory
            if memory_mb > self._peak_memory:
                self._peak_memory = memory_mb
            
            return memory_mb
        except Exception:
            return 0.0
    
    @contextmanager
    def measure(self):
        """Context manager for measuring performance"""
        self.start()
        try:
            yield self
        finally:
            metrics = self.stop()
            self._measurements.append(metrics)
    
    def get_measurements(self) -> List[PerformanceMetrics]:
        """Get all measurements"""
        with self._lock:
            return self._measurements.copy()
    
    def clear_measurements(self) -> None:
        """Clear all measurements"""
        with self._lock:
            self._measurements.clear()


class ProfileManager:
    """Advanced profiling with hotspot detection and recommendations"""
    
    def __init__(self):
        self._profiles = {}
        self._monitor = PerformanceMonitor()
    
    def profile_function(self, 
                        func: Callable,
                        *args, 
                        detailed: bool = True,
                        sort_by: str = 'cumulative',
                        top_functions: int = 20,
                        **kwargs) -> ProfileResult:
        """Profile a single function call"""
        
        function_name = f"{func.__module__}.{func.__name__}"
        
        if detailed:
            # Use cProfile for detailed analysis
            profiler = cProfile.Profile()
            
            with self._monitor.measure():
                profiler.enable()
                result = func(*args, **kwargs)
                profiler.disable()
            
            # Get performance metrics
            metrics = self._monitor.get_measurements()[-1]
            
            # Analyze profiling data
            stats_stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stats_stream)
            stats.sort_stats(sort_by)
            
            # Get function call count
            total_calls = stats.total_calls
            metrics.function_calls = total_calls
            
            # Extract hotspots
            hotspots = self._extract_hotspots(stats, top_functions)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(metrics, hotspots)
            
            profile_result = ProfileResult(
                function_name=function_name,
                total_time=metrics.execution_time,
                call_count=total_calls,
                metrics=metrics,
                hotspots=hotspots,
                recommendations=recommendations
            )
        
        else:
            # Simple timing
            with self._monitor.measure():
                result = func(*args, **kwargs)
            
            metrics = self._monitor.get_measurements()[-1]
            
            profile_result = ProfileResult(
                function_name=function_name,
                total_time=metrics.execution_time,
                call_count=1,
                metrics=metrics
            )
        
        # Store profile
        self._profiles[function_name] = profile_result
        
        logger.info(f"Profiled {function_name}: {metrics.execution_time:.3f}s, "
                   f"Memory: {metrics.memory_mb:.1f}MB")
        
        return profile_result
    
    def _extract_hotspots(self, stats: pstats.Stats, top_count: int) -> List[Dict[str, Any]]:
        """Extract performance hotspots from profiling stats"""
        hotspots = []
        
        try:
            # Get top functions by cumulative time
            stats_list = []
            for func, (cc, nc, tt, ct, callers) in stats.stats.items():
                filename, line, func_name = func
                stats_list.append({
                    'function': f"{filename}:{line}({func_name})",
                    'calls': nc,
                    'total_time': tt,
                    'cumulative_time': ct,
                    'per_call_time': tt / nc if nc > 0 else 0
                })
            
            # Sort by cumulative time and take top N
            stats_list.sort(key=lambda x: x['cumulative_time'], reverse=True)
            hotspots = stats_list[:top_count]
            
        except Exception as e:
            logger.warning(f"Failed to extract hotspots: {e}")
        
        return hotspots
    
    def _generate_recommendations(self, 
                                metrics: PerformanceMetrics, 
                                hotspots: List[Dict[str, Any]]) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Memory recommendations
        if metrics.memory_mb > 500:  # > 500MB
            recommendations.append("Consider reducing memory usage - current usage is high")
        
        if metrics.peak_memory_mb > metrics.memory_mb * 1.5:
            recommendations.append("Memory usage spikes detected - investigate temporary allocations")
        
        # CPU recommendations
        if metrics.cpu_percent > 80:
            recommendations.append("High CPU usage detected - consider optimization or parallelization")
        
        # Execution time recommendations
        if metrics.execution_time > 10:
            recommendations.append("Long execution time - consider caching or algorithm optimization")
        
        # Hotspot-based recommendations
        if hotspots:
            # Find functions with high per-call time
            expensive_calls = [h for h in hotspots if h['per_call_time'] > 0.01]
            if expensive_calls:
                recommendations.append(
                    f"Optimize expensive function calls: {[h['function'] for h in expensive_calls[:3]]}"
                )
            
            # Find functions with many calls
            frequent_calls = [h for h in hotspots if h['calls'] > 1000]
            if frequent_calls:
                recommendations.append(
                    f"Consider caching for frequently called functions: {[h['function'] for h in frequent_calls[:3]]}"
                )
        
        return recommendations
    
    def compare_profiles(self, profile1_name: str, profile2_name: str) -> Dict[str, Any]:
        """Compare two performance profiles"""
        if profile1_name not in self._profiles or profile2_name not in self._profiles:
            raise ValueError("One or both profiles not found")
        
        p1 = self._profiles[profile1_name]
        p2 = self._profiles[profile2_name]
        
        comparison = {
            'profile1': profile1_name,
            'profile2': profile2_name,
            'time_difference': p2.total_time - p1.total_time,
            'time_ratio': p2.total_time / p1.total_time if p1.total_time > 0 else float('inf'),
            'memory_difference_mb': p2.metrics.memory_mb - p1.metrics.memory_mb,
            'call_count_difference': p2.call_count - p1.call_count,
            'performance_improvement': (p1.total_time - p2.total_time) / p1.total_time if p1.total_time > 0 else 0
        }
        
        # Add interpretation
        if comparison['performance_improvement'] > 0.1:
            comparison['interpretation'] = f"Profile 2 is {comparison['performance_improvement']:.1%} faster"
        elif comparison['performance_improvement'] < -0.1:
            comparison['interpretation'] = f"Profile 2 is {abs(comparison['performance_improvement']):.1%} slower"
        else:
            comparison['interpretation'] = "Performance is similar between profiles"
        
        return comparison
    
    def get_profile(self, function_name: str) -> Optional[ProfileResult]:
        """Get stored profile by function name"""
        return self._profiles.get(function_name)
    
    def get_all_profiles(self) -> Dict[str, ProfileResult]:
        """Get all stored profiles"""
        return self._profiles.copy()
    
    def clear_profiles(self) -> None:
        """Clear all stored profiles"""
        self._profiles.clear()
        self._monitor.clear_measurements()
    
    def export_profile_report(self, filepath: str) -> None:
        """Export comprehensive profiling report"""
        report_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'profiles': {name: asdict(profile) for name, profile in self._profiles.items()},
            'summary': {
                'total_profiles': len(self._profiles),
                'total_execution_time': sum(p.total_time for p in self._profiles.values()),
                'avg_execution_time': sum(p.total_time for p in self._profiles.values()) / len(self._profiles) if self._profiles else 0
            }
        }
        
        report_path = Path(filepath)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            import json
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            logger.info(f"Profile report exported to {filepath}")
        
        except Exception as e:
            logger.error(f"Failed to export profile report: {e}")
            raise


def profile_function(detailed: bool = True, 
                    store: bool = True,
                    sort_by: str = 'cumulative'):
    """Decorator for profiling functions"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not store:
                # Just monitor performance without storing
                monitor = PerformanceMonitor()
                with monitor.measure():
                    result = func(*args, **kwargs)
                
                metrics = monitor.get_measurements()[-1]
                logger.info(f"Function {func.__name__} executed in {metrics.execution_time:.3f}s")
                
            else:
                # Use profile manager
                profiler = get_profile_manager()
                profile_result = profiler.profile_function(
                    func, *args, detailed=detailed, sort_by=sort_by, **kwargs
                )
                result = func(*args, **kwargs)  # Function was already called during profiling
                
                # Log key metrics
                logger.info(f"Profiled {func.__name__}: "
                           f"Time: {profile_result.total_time:.3f}s, "
                           f"Calls: {profile_result.call_count}, "
                           f"Memory: {profile_result.metrics.memory_mb:.1f}MB")
            
            return result
        
        return wrapper
    return decorator


class AutoProfiler:
    """Automatic profiling with adaptive sampling"""
    
    def __init__(self, 
                 sample_rate: float = 0.1,
                 min_execution_time: float = 1.0,
                 max_profiles: int = 100):
        
        self.sample_rate = sample_rate
        self.min_execution_time = min_execution_time
        self.max_profiles = max_profiles
        self._profile_count = 0
        self._enabled = True
        self._profiler = ProfileManager()
    
    def should_profile(self, execution_time: float = None) -> bool:
        """Determine if profiling should be performed"""
        if not self._enabled:
            return False
        
        if self._profile_count >= self.max_profiles:
            return False
        
        if execution_time and execution_time < self.min_execution_time:
            return False
        
        import random
        return random.random() < self.sample_rate
    
    def auto_profile(self, func: Callable, *args, **kwargs) -> Any:
        """Automatically profile function if conditions are met"""
        # Quick execution to check timing
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        if self.should_profile(execution_time):
            # Re-run with profiling
            self._profiler.profile_function(func, *args, **kwargs)
            self._profile_count += 1
            
            logger.debug(f"Auto-profiled {func.__name__} "
                        f"({self._profile_count}/{self.max_profiles})")
        
        return result
    
    def enable(self) -> None:
        """Enable auto-profiling"""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable auto-profiling"""
        self._enabled = False
    
    def get_profiles(self) -> Dict[str, ProfileResult]:
        """Get collected profiles"""
        return self._profiler.get_all_profiles()


# Global instances
_profile_manager = None
_auto_profiler = None


def get_profile_manager() -> ProfileManager:
    """Get global profile manager"""
    global _profile_manager
    if _profile_manager is None:
        _profile_manager = ProfileManager()
    return _profile_manager


def get_auto_profiler() -> AutoProfiler:
    """Get global auto profiler"""
    global _auto_profiler
    if _auto_profiler is None:
        _auto_profiler = AutoProfiler()
    return _auto_profiler


@contextmanager
def performance_context(name: str = "operation"):
    """Context manager for measuring performance of code blocks"""
    monitor = PerformanceMonitor()
    start_time = time.time()
    
    with monitor.measure():
        yield monitor
    
    execution_time = time.time() - start_time
    logger.info(f"Performance context '{name}' completed in {execution_time:.3f}s")