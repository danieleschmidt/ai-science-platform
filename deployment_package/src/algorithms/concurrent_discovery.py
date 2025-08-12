"""Concurrent and scalable discovery engine for high-performance scientific automation"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Iterator
import logging
from dataclasses import dataclass
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import time
from itertools import islice

from .discovery import DiscoveryEngine, Discovery
from ..utils.performance import (
    cached, profiled, memory_optimized, ParallelProcessor, 
    get_profiler, get_memory_optimizer, PerformanceMetrics
)
from ..utils.error_handling import robust_execution, safe_array_operation, DiscoveryError
from ..utils.security import validate_input, sanitize_string

logger = logging.getLogger(__name__)


@dataclass
class BatchDiscoveryConfig:
    """Configuration for batch discovery operations"""
    batch_size: int = 100
    max_workers: int = None
    use_processes: bool = False
    enable_caching: bool = True
    memory_limit_mb: int = 1000
    timeout_seconds: Optional[float] = 300


class ConcurrentDiscoveryEngine(DiscoveryEngine):
    """High-performance concurrent discovery engine"""
    
    def __init__(self, discovery_threshold: float = 0.7, 
                 config: Optional[BatchDiscoveryConfig] = None):
        super().__init__(discovery_threshold)
        self.config = config or BatchDiscoveryConfig()
        
        # Initialize parallel processor
        self.parallel_processor = ParallelProcessor(
            max_workers=self.config.max_workers,
            use_processes=self.config.use_processes
        )
        
        # Performance tracking
        self.batch_metrics = []
        
        logger.info(f"ConcurrentDiscoveryEngine initialized with {self.config.max_workers} workers")
    
    @profiled("batch_discovery")
    @memory_optimized(preserve_precision=True)
    def discover_batch(self, 
                      datasets: List[Tuple[np.ndarray, Optional[np.ndarray]]], 
                      contexts: Optional[List[str]] = None) -> List[List[Discovery]]:
        """Discover patterns across multiple datasets concurrently
        
        Args:
            datasets: List of (data, targets) tuples
            contexts: Optional list of context strings for each dataset
            
        Returns:
            List of discovery lists, one per dataset
        """
        if not datasets:
            return []
        
        # Validate inputs
        for i, (data, targets) in enumerate(datasets):
            data = validate_input(data, f"dataset_{i}_data")
            if targets is not None:
                targets = validate_input(targets, f"dataset_{i}_targets")
        
        if contexts is None:
            contexts = [f"dataset_{i}" for i in range(len(datasets))]
        else:
            contexts = [sanitize_string(ctx, max_length=200) for ctx in contexts]
        
        if len(contexts) != len(datasets):
            raise ValueError(f"Number of contexts ({len(contexts)}) != datasets ({len(datasets)})")
        
        logger.info(f"Starting batch discovery on {len(datasets)} datasets")
        start_time = time.time()
        
        # Create discovery tasks
        discovery_tasks = []
        for i, ((data, targets), context) in enumerate(zip(datasets, contexts)):
            task = {
                'dataset_id': i,
                'data': data,
                'targets': targets,
                'context': context,
                'threshold': self.discovery_threshold
            }
            discovery_tasks.append(task)
        
        # Process in parallel
        try:
            results = self.parallel_processor.parallel_map(
                func=self._discover_single_dataset,
                items=discovery_tasks,
                show_progress=True
            )
            
            # Update global discovery count
            for discoveries in results:
                if discoveries:
                    self.discoveries.extend(discoveries)
            
            processing_time = time.time() - start_time
            total_discoveries = sum(len(discoveries) for discoveries in results if discoveries)
            
            logger.info(f"Batch discovery completed: {total_discoveries} discoveries in {processing_time:.3f}s")
            
            # Record performance metrics
            metrics = PerformanceMetrics(
                execution_time=processing_time,
                memory_usage_mb=0,  # Would need more complex tracking
                cpu_usage_percent=0,
                parallel_speedup=self._estimate_speedup(len(datasets), processing_time)
            )
            self.batch_metrics.append(metrics)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch discovery failed: {str(e)}")
            raise DiscoveryError(f"Batch discovery failed: {str(e)}")
    
    @cached(maxsize=64, ttl=3600)  # Cache for 1 hour
    def _discover_single_dataset(self, task: Dict[str, Any]) -> List[Discovery]:
        """Discover patterns in a single dataset (cached for performance)"""
        try:
            # Create temporary engine for this task
            temp_engine = DiscoveryEngine(discovery_threshold=task['threshold'])
            
            # Run discovery
            discoveries = temp_engine.discover(
                data=task['data'],
                targets=task['targets'],
                context=task['context']
            )
            
            # Update hypothesis count
            self.hypotheses_tested += temp_engine.hypotheses_tested
            
            return discoveries
            
        except Exception as e:
            logger.warning(f"Discovery failed for dataset {task['dataset_id']}: {str(e)}")
            return []
    
    @profiled("streaming_discovery")
    def discover_stream(self, 
                       data_stream: Iterator[Tuple[np.ndarray, Optional[np.ndarray]]],
                       window_size: int = 10,
                       overlap: int = 2) -> Iterator[List[Discovery]]:
        """Process streaming data for real-time discovery
        
        Args:
            data_stream: Iterator yielding (data, targets) tuples
            window_size: Number of samples to process together
            overlap: Number of samples to overlap between windows
            
        Yields:
            Lists of discoveries for each window
        """
        buffer = []
        window_count = 0
        
        logger.info(f"Starting streaming discovery with window_size={window_size}, overlap={overlap}")
        
        try:
            for data_point in data_stream:
                buffer.append(data_point)
                
                # Process when buffer is full
                if len(buffer) >= window_size:
                    window_count += 1
                    
                    # Create batch from buffer
                    batch = buffer.copy()
                    
                    # Discover patterns in current window
                    discoveries = self.discover_batch(
                        datasets=batch,
                        contexts=[f"stream_window_{window_count}_{i}" for i in range(len(batch))]
                    )
                    
                    # Flatten discoveries from all datasets in window
                    window_discoveries = []
                    for dataset_discoveries in discoveries:
                        window_discoveries.extend(dataset_discoveries)
                    
                    yield window_discoveries
                    
                    # Remove processed items but keep overlap
                    buffer = buffer[window_size - overlap:]
        
        except Exception as e:
            logger.error(f"Streaming discovery error: {str(e)}")
            raise DiscoveryError(f"Streaming discovery failed: {str(e)}")
    
    @profiled("adaptive_discovery")
    async def discover_adaptive(self, 
                               data: np.ndarray, 
                               targets: Optional[np.ndarray] = None,
                               context: str = "",
                               adaptation_rounds: int = 3) -> List[Discovery]:
        """Adaptive discovery that refines parameters based on initial results"""
        data = validate_input(data, "data")
        context = sanitize_string(context, max_length=200)
        
        if targets is not None:
            targets = validate_input(targets, "targets")
        
        logger.info(f"Starting adaptive discovery with {adaptation_rounds} rounds")
        
        best_discoveries = []
        best_threshold = self.discovery_threshold
        
        # Try multiple thresholds in parallel
        thresholds = np.linspace(0.3, 0.9, adaptation_rounds)
        
        tasks = []
        for threshold in thresholds:
            task = asyncio.create_task(
                self._async_discover_with_threshold(data, targets, context, threshold)
            )
            tasks.append((threshold, task))
        
        # Collect results
        threshold_results = []
        for threshold, task in tasks:
            try:
                discoveries = await task
                threshold_results.append((threshold, discoveries))
            except Exception as e:
                logger.warning(f"Adaptive discovery failed for threshold {threshold}: {str(e)}")
                threshold_results.append((threshold, []))
        
        # Select best threshold based on discovery quality
        best_score = 0
        for threshold, discoveries in threshold_results:
            if discoveries:
                # Score based on number of discoveries and average confidence
                avg_confidence = np.mean([d.confidence for d in discoveries])
                score = len(discoveries) * avg_confidence
                
                if score > best_score:
                    best_score = score
                    best_discoveries = discoveries
                    best_threshold = threshold
        
        # Update engine threshold if better one found
        if best_threshold != self.discovery_threshold:
            logger.info(f"Adapted discovery threshold from {self.discovery_threshold:.3f} to {best_threshold:.3f}")
            self.discovery_threshold = best_threshold
        
        return best_discoveries
    
    async def _async_discover_with_threshold(self, 
                                           data: np.ndarray, 
                                           targets: Optional[np.ndarray],
                                           context: str,
                                           threshold: float) -> List[Discovery]:
        """Asynchronous discovery with specific threshold"""
        loop = asyncio.get_event_loop()
        
        # Run discovery in thread pool to avoid blocking
        with ThreadPoolExecutor(max_workers=1) as executor:
            # Create temporary engine
            temp_engine = DiscoveryEngine(discovery_threshold=threshold)
            
            # Run discovery in thread
            future = executor.submit(temp_engine.discover, data, targets, context)
            discoveries = await loop.run_in_executor(None, lambda: future.result())
            
            return discoveries
    
    @profiled("hierarchical_discovery")
    def discover_hierarchical(self, 
                            data: np.ndarray, 
                            targets: Optional[np.ndarray] = None,
                            context: str = "",
                            max_depth: int = 3,
                            min_samples: int = 50) -> Dict[int, List[Discovery]]:
        """Hierarchical discovery at multiple resolution levels"""
        data = validate_input(data, "data")
        context = sanitize_string(context, max_length=200)
        
        if targets is not None:
            targets = validate_input(targets, "targets")
        
        if len(data) < min_samples:
            raise ValueError(f"Insufficient data for hierarchical discovery: {len(data)} < {min_samples}")
        
        logger.info(f"Starting hierarchical discovery with max_depth={max_depth}")
        
        hierarchical_discoveries = {}
        
        for depth in range(max_depth):
            # Calculate resolution for this level
            resolution_factor = 2 ** depth
            step_size = max(1, len(data) // (resolution_factor * 10))
            
            if step_size >= len(data):
                break
            
            # Subsample data at current resolution
            indices = np.arange(0, len(data), step_size)
            subsampled_data = data[indices]
            subsampled_targets = targets[indices] if targets is not None else None
            
            level_context = f"{context}_level_{depth}_resolution_{resolution_factor}"
            
            # Discover at this resolution
            discoveries = self.discover(subsampled_data, subsampled_targets, level_context)
            hierarchical_discoveries[depth] = discoveries
            
            logger.info(f"Level {depth}: {len(discoveries)} discoveries with {len(subsampled_data)} samples")
        
        return hierarchical_discoveries
    
    def _estimate_speedup(self, num_datasets: int, processing_time: float) -> Optional[float]:
        """Estimate parallel speedup compared to sequential processing"""
        if not hasattr(self, '_sequential_baseline'):
            return None
        
        expected_sequential_time = self._sequential_baseline * num_datasets
        if processing_time > 0:
            return expected_sequential_time / processing_time
        return None
    
    def benchmark_performance(self, 
                            test_datasets: List[Tuple[np.ndarray, Optional[np.ndarray]]],
                            run_sequential: bool = True) -> Dict[str, Any]:
        """Benchmark discovery performance"""
        logger.info(f"Benchmarking performance on {len(test_datasets)} datasets")
        
        results = {}
        
        # Benchmark parallel performance
        start_time = time.time()
        parallel_results = self.discover_batch(test_datasets)
        parallel_time = time.time() - start_time
        parallel_discoveries = sum(len(discoveries) for discoveries in parallel_results)
        
        results['parallel'] = {
            'total_time': parallel_time,
            'discoveries': parallel_discoveries,
            'datasets_per_second': len(test_datasets) / parallel_time if parallel_time > 0 else 0,
            'discoveries_per_second': parallel_discoveries / parallel_time if parallel_time > 0 else 0
        }
        
        # Benchmark sequential performance if requested
        if run_sequential and len(test_datasets) <= 10:  # Only for small datasets
            start_time = time.time()
            sequential_results = []
            for i, (data, targets) in enumerate(test_datasets):
                discoveries = self.discover(data, targets, f"sequential_test_{i}")
                sequential_results.append(discoveries)
            sequential_time = time.time() - start_time
            sequential_discoveries = sum(len(discoveries) for discoveries in sequential_results)
            
            results['sequential'] = {
                'total_time': sequential_time,
                'discoveries': sequential_discoveries,
                'datasets_per_second': len(test_datasets) / sequential_time if sequential_time > 0 else 0,
                'discoveries_per_second': sequential_discoveries / sequential_time if sequential_time > 0 else 0
            }
            
            # Calculate speedup
            if sequential_time > 0:
                results['speedup'] = sequential_time / parallel_time
                self._sequential_baseline = sequential_time / len(test_datasets)
            else:
                results['speedup'] = None
        
        logger.info(f"Benchmark complete: {results}")
        return results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        profiler = get_profiler()
        
        summary = {
            'total_discoveries': len(self.discoveries),
            'hypotheses_tested': self.hypotheses_tested,
            'batch_operations': len(self.batch_metrics),
            'average_batch_time': np.mean([m.execution_time for m in self.batch_metrics]) if self.batch_metrics else 0,
            'configuration': {
                'batch_size': self.config.batch_size,
                'max_workers': self.config.max_workers,
                'use_processes': self.config.use_processes,
                'discovery_threshold': self.discovery_threshold
            }
        }
        
        # Add profiling data if available
        avg_metrics = profiler.get_average_metrics('batch_discovery')
        if avg_metrics:
            summary['average_performance'] = {
                'execution_time': avg_metrics.execution_time,
                'memory_usage_mb': avg_metrics.memory_usage_mb,
                'cpu_usage_percent': avg_metrics.cpu_usage_percent
            }
        
        return summary
    
    def optimize_configuration(self, 
                              sample_datasets: List[Tuple[np.ndarray, Optional[np.ndarray]]]) -> BatchDiscoveryConfig:
        """Automatically optimize configuration based on sample datasets"""
        logger.info("Optimizing configuration based on sample data")
        
        # Test different configurations
        test_configs = [
            BatchDiscoveryConfig(batch_size=50, max_workers=2, use_processes=False),
            BatchDiscoveryConfig(batch_size=100, max_workers=4, use_processes=False),
            BatchDiscoveryConfig(batch_size=200, max_workers=8, use_processes=False),
            BatchDiscoveryConfig(batch_size=100, max_workers=2, use_processes=True),
        ]
        
        best_config = self.config
        best_performance = float('inf')
        
        # Limit sample size for optimization
        test_datasets = sample_datasets[:min(20, len(sample_datasets))]
        
        for config in test_configs:
            try:
                # Temporarily use test config
                original_config = self.config
                self.config = config
                self.parallel_processor = ParallelProcessor(
                    max_workers=config.max_workers,
                    use_processes=config.use_processes
                )
                
                # Benchmark
                start_time = time.time()
                self.discover_batch(test_datasets)
                performance = time.time() - start_time
                
                if performance < best_performance:
                    best_performance = performance
                    best_config = config
                
                logger.info(f"Config {config}: {performance:.3f}s")
                
            except Exception as e:
                logger.warning(f"Configuration test failed: {str(e)}")
            finally:
                # Restore original config
                self.config = original_config
        
        # Apply best configuration
        self.config = best_config
        self.parallel_processor = ParallelProcessor(
            max_workers=best_config.max_workers,
            use_processes=best_config.use_processes
        )
        
        logger.info(f"Optimized configuration: {best_config}")
        return best_config