"""Performance-optimized model implementations"""

import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache, wraps
import pickle
import hashlib
from dataclasses import dataclass
import logging

from ..models.simple import SimpleModel, ModelOutput
from ..utils.error_handling import robust_execution, ModelError
from ..utils.logging_config import setup_logging
from .caching import get_cache_manager

setup_logging()
logger = logging.getLogger(__name__)


@dataclass
class BatchOutput:
    """Output from batch processing"""
    predictions: List[np.ndarray]
    confidences: List[float]
    processing_time_ms: float
    batch_size: int
    parallel_workers: int


class CachedModel(SimpleModel):
    """Model with intelligent caching for repeated computations"""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, cache_size: int = 1000):
        super().__init__(input_dim, hidden_dim)
        self.cache_size = cache_size
        self.cache_manager = get_cache_manager()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info(f"CachedModel initialized with cache_size={cache_size}")
    
    def _get_input_hash(self, x: np.ndarray) -> str:
        """Generate hash for input array"""
        return hashlib.md5(x.tobytes()).hexdigest()
    
    @robust_execution(recovery_strategy='graceful_degradation')
    def forward(self, x: np.ndarray, use_cache: bool = True) -> ModelOutput:
        """Forward pass with caching"""
        if not use_cache:
            return super().forward(x)
        
        # Generate cache key
        input_hash = self._get_input_hash(x)
        weights_hash = hashlib.md5(pickle.dumps(self.weights)).hexdigest()
        cache_key = f"model_forward_{input_hash}_{weights_hash}"
        
        # Try to get from cache
        if use_cache:
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                self._cache_hits += 1
                logger.debug(f"Cache hit for key: {cache_key[:16]}...")
                return cached_result
            
            self._cache_misses += 1
            logger.debug(f"Cache miss for key: {cache_key[:16]}...")
        
        # Compute result
        result = super().forward(x)
        
        # Cache the result
        if use_cache:
            self.cache_manager.set(cache_key, result, ttl=3600)  # 1 hour TTL
        
        return result
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'cache_size': self.cache_size
        }


class BatchOptimizedModel(CachedModel):
    """Model optimized for batch processing with parallel execution"""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, 
                 max_workers: int = 4, optimal_batch_size: int = 32):
        super().__init__(input_dim, hidden_dim)
        self.max_workers = max_workers
        self.optimal_batch_size = optimal_batch_size
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._batch_stats = []
        logger.info(f"BatchOptimizedModel initialized: max_workers={max_workers}, batch_size={optimal_batch_size}")
    
    @robust_execution(recovery_strategy='graceful_degradation')
    def forward_batch(self, batch_inputs: List[np.ndarray], 
                     use_parallel: bool = True,
                     use_cache: bool = True) -> BatchOutput:
        """Process multiple inputs in parallel"""
        start_time = time.time()
        
        if not batch_inputs:
            raise ModelError("Empty batch provided to forward_batch")
        
        batch_size = len(batch_inputs)
        logger.info(f"Processing batch of size {batch_size}")
        
        if use_parallel and batch_size > 1:
            # Parallel processing
            predictions, confidences = self._process_parallel(batch_inputs, use_cache)
        else:
            # Sequential processing
            predictions, confidences = self._process_sequential(batch_inputs, use_cache)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Update batch statistics
        self._batch_stats.append({
            'batch_size': batch_size,
            'processing_time_ms': processing_time_ms,
            'parallel': use_parallel,
            'timestamp': time.time()
        })
        
        # Keep only recent stats
        if len(self._batch_stats) > 100:
            self._batch_stats = self._batch_stats[-50:]
        
        logger.info(f"Batch processed in {processing_time_ms:.1f}ms")
        
        return BatchOutput(
            predictions=predictions,
            confidences=confidences,
            processing_time_ms=processing_time_ms,
            batch_size=batch_size,
            parallel_workers=self.max_workers if use_parallel else 1
        )
    
    def _process_parallel(self, batch_inputs: List[np.ndarray], 
                         use_cache: bool) -> tuple[List[np.ndarray], List[float]]:
        """Process inputs in parallel"""
        futures = []
        
        # Submit all tasks
        for input_data in batch_inputs:
            future = self._executor.submit(self._safe_forward, input_data, use_cache)
            futures.append(future)
        
        # Collect results
        predictions = []
        confidences = []
        
        for future in as_completed(futures):
            try:
                output = future.result(timeout=30)  # 30 second timeout
                predictions.append(output.predictions)
                confidences.append(output.confidence)
            except Exception as e:
                logger.error(f"Parallel processing failed for input: {str(e)}")
                # Add fallback values
                predictions.append(np.array([0.0]))
                confidences.append(0.0)
        
        return predictions, confidences
    
    def _process_sequential(self, batch_inputs: List[np.ndarray], 
                           use_cache: bool) -> tuple[List[np.ndarray], List[float]]:
        """Process inputs sequentially"""
        predictions = []
        confidences = []
        
        for input_data in batch_inputs:
            try:
                output = self._safe_forward(input_data, use_cache)
                predictions.append(output.predictions)
                confidences.append(output.confidence)
            except Exception as e:
                logger.error(f"Sequential processing failed for input: {str(e)}")
                predictions.append(np.array([0.0]))
                confidences.append(0.0)
        
        return predictions, confidences
    
    def _safe_forward(self, x: np.ndarray, use_cache: bool) -> ModelOutput:
        """Thread-safe forward pass"""
        try:
            return self.forward(x, use_cache=use_cache)
        except Exception as e:
            logger.error(f"Safe forward failed: {str(e)}")
            # Return fallback output
            return ModelOutput(
                predictions=np.array([0.0]),
                confidence=0.0,
                metadata={'error': str(e), 'fallback': True}
            )
    
    def get_batch_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics"""
        if not self._batch_stats:
            return {}
        
        processing_times = [s['processing_time_ms'] for s in self._batch_stats]
        batch_sizes = [s['batch_size'] for s in self._batch_stats]
        
        return {
            'total_batches': len(self._batch_stats),
            'avg_processing_time_ms': np.mean(processing_times),
            'min_processing_time_ms': np.min(processing_times),
            'max_processing_time_ms': np.max(processing_times),
            'avg_batch_size': np.mean(batch_sizes),
            'total_inputs_processed': sum(batch_sizes),
            'throughput_inputs_per_second': sum(batch_sizes) / (sum(processing_times) / 1000) if sum(processing_times) > 0 else 0
        }
    
    def optimize_batch_size(self, test_inputs: List[np.ndarray], 
                           test_sizes: List[int] = None) -> int:
        """Automatically determine optimal batch size"""
        if test_sizes is None:
            test_sizes = [1, 2, 4, 8, 16, 32, 64]
        
        if not test_inputs:
            return self.optimal_batch_size
        
        logger.info("Optimizing batch size...")
        results = {}
        
        for batch_size in test_sizes:
            if batch_size > len(test_inputs):
                continue
            
            # Test with this batch size
            test_batch = test_inputs[:batch_size]
            start_time = time.time()
            
            try:
                self.forward_batch(test_batch, use_parallel=True, use_cache=False)
                processing_time = time.time() - start_time
                throughput = batch_size / processing_time
                results[batch_size] = throughput
                logger.debug(f"Batch size {batch_size}: {throughput:.1f} inputs/sec")
            except Exception as e:
                logger.warning(f"Batch size {batch_size} failed: {str(e)}")
                results[batch_size] = 0.0
        
        if not results:
            return self.optimal_batch_size
        
        # Find optimal batch size
        optimal_size = max(results.keys(), key=lambda k: results[k])
        self.optimal_batch_size = optimal_size
        
        logger.info(f"Optimal batch size determined: {optimal_size}")
        return optimal_size
    
    def __del__(self):
        """Cleanup thread pool executor"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)


class AdaptiveModel(BatchOptimizedModel):
    """Model that adapts its performance parameters based on usage patterns"""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64):
        super().__init__(input_dim, hidden_dim)
        self._performance_history = []
        self._adaptation_threshold = 10  # Adapt after 10 operations
        self._last_adaptation_time = time.time()
        logger.info("AdaptiveModel initialized with self-optimization")
    
    @robust_execution(recovery_strategy='graceful_degradation')
    def forward(self, x: np.ndarray, use_cache: bool = True) -> ModelOutput:
        """Adaptive forward pass with performance monitoring"""
        start_time = time.time()
        
        # Get result
        result = super().forward(x, use_cache=use_cache)
        
        # Track performance
        processing_time = (time.time() - start_time) * 1000
        self._performance_history.append({
            'processing_time_ms': processing_time,
            'input_shape': x.shape,
            'cache_used': use_cache,
            'timestamp': time.time()
        })
        
        # Trigger adaptation if needed
        if len(self._performance_history) >= self._adaptation_threshold:
            self._adapt_parameters()
        
        return result
    
    def _adapt_parameters(self):
        """Adapt model parameters based on performance history"""
        current_time = time.time()
        
        # Don't adapt too frequently
        if current_time - self._last_adaptation_time < 60:  # 1 minute cooldown
            return
        
        if not self._performance_history:
            return
        
        # Analyze recent performance
        recent_history = self._performance_history[-self._adaptation_threshold:]
        avg_processing_time = np.mean([h['processing_time_ms'] for h in recent_history])
        
        # Adapt cache usage
        cache_performance = {}
        for entry in recent_history:
            cache_key = entry['cache_used']
            if cache_key not in cache_performance:
                cache_performance[cache_key] = []
            cache_performance[cache_key].append(entry['processing_time_ms'])
        
        # Adapt batch size based on input patterns
        input_sizes = [np.prod(h['input_shape']) for h in recent_history]
        if len(set(input_sizes)) == 1:  # Consistent input sizes
            consistent_size = input_sizes[0]
            if consistent_size < 100:  # Small inputs
                self.optimal_batch_size = min(64, self.optimal_batch_size * 2)
            elif consistent_size > 1000:  # Large inputs
                self.optimal_batch_size = max(8, self.optimal_batch_size // 2)
        
        self._last_adaptation_time = current_time
        logger.info(f"Model adapted: batch_size={self.optimal_batch_size}, avg_time={avg_processing_time:.1f}ms")
        
        # Clear old history
        self._performance_history = self._performance_history[-50:]
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get adaptation performance statistics"""
        if not self._performance_history:
            return {}
        
        recent_history = self._performance_history[-20:]  # Last 20 operations
        
        processing_times = [h['processing_time_ms'] for h in recent_history]
        input_shapes = [h['input_shape'] for h in recent_history]
        
        return {
            'total_operations': len(self._performance_history),
            'recent_avg_processing_time_ms': np.mean(processing_times),
            'recent_std_processing_time_ms': np.std(processing_times),
            'input_shape_variety': len(set([str(shape) for shape in input_shapes])),
            'current_batch_size': self.optimal_batch_size,
            'last_adaptation_time': self._last_adaptation_time,
            'adaptation_frequency': len(self._performance_history) / max(1, (time.time() - self._last_adaptation_time))
        }