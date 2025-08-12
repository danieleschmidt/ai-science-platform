"""Scalable discovery engine with async processing and load balancing"""

import asyncio
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

from .discovery import DiscoveryEngine, Discovery
from ..performance.async_processing import AsyncTaskQueue, Task
from ..performance.auto_scaling import AutoScaler, LoadBalancer
from ..utils.error_handling import robust_execution, DiscoveryError
from ..utils.caching import LRUCache

logger = logging.getLogger(__name__)


@dataclass
class BatchDiscoveryResult:
    """Result from batch discovery processing"""
    batch_id: str
    discoveries: List[Discovery]
    processing_time: float
    worker_id: str
    success: bool
    error_message: Optional[str] = None


class ScalableDiscoveryEngine:
    """High-performance, scalable discovery engine with auto-scaling"""
    
    def __init__(self, 
                 discovery_threshold: float = 0.7,
                 max_workers: int = 8,
                 enable_caching: bool = True,
                 cache_size: int = 1000,
                 enable_auto_scaling: bool = True):
        
        self.discovery_threshold = discovery_threshold
        self.max_workers = max_workers
        self.enable_caching = enable_caching
        self.enable_auto_scaling = enable_auto_scaling
        
        # Core components
        self.base_engine = DiscoveryEngine(discovery_threshold)
        
        # Async processing
        self.task_queue = AsyncTaskQueue(
            max_concurrent_tasks=max_workers,
            max_queue_size=max_workers * 10
        )
        
        # Caching system
        if enable_caching:
            self.cache = LRUCache(max_size=cache_size, ttl=3600)
        else:
            self.cache = None
        
        # Load balancing
        self.load_balancer = LoadBalancer()
        
        # Auto-scaling
        if enable_auto_scaling:
            self.autoscaler = AutoScaler()
            self.autoscaler.set_scaling_callbacks(
                self._scale_up_workers,
                self._scale_down_workers  
            )
        else:
            self.autoscaler = None
        
        # Worker management
        self.discovery_workers = {}
        self.worker_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Statistics
        self.stats = {
            "total_discoveries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_processing_time": 0.0,
            "successful_batches": 0,
            "failed_batches": 0
        }
        
        logger.info(f"ScalableDiscoveryEngine initialized: workers={max_workers}, caching={enable_caching}")
    
    async def start(self) -> None:
        """Start the scalable discovery engine"""
        # Start async task queue
        await self.task_queue.start_workers()
        
        # Initialize discovery workers
        for i in range(self.max_workers):
            worker_id = f"discovery_worker_{i}"
            self.discovery_workers[worker_id] = DiscoveryEngine(self.discovery_threshold)
            self.load_balancer.register_worker(worker_id, self._process_discovery_task)
        
        # Start auto-scaling if enabled
        if self.autoscaler:
            self.autoscaler.start_monitoring()
        
        logger.info("ScalableDiscoveryEngine started successfully")
    
    async def discover_batch(self, 
                           data_batches: List[np.ndarray],
                           contexts: Optional[List[str]] = None,
                           batch_size: Optional[int] = None) -> List[BatchDiscoveryResult]:
        """Process multiple data batches concurrently"""
        
        if not data_batches:
            return []
        
        if contexts is None:
            contexts = [f"batch_{i}" for i in range(len(data_batches))]
        
        # Auto-batch if batch_size specified
        if batch_size and len(data_batches) > batch_size:
            data_batches = self._create_batches(data_batches, batch_size)
            contexts = [f"auto_batch_{i}" for i in range(len(data_batches))]
        
        # Submit discovery tasks
        task_ids = []
        for i, (data_batch, context) in enumerate(zip(data_batches, contexts)):
            task_id = f"discovery_batch_{i}_{time.time()}"
            
            await self.task_queue.submit_task(
                task_id=task_id,
                func=self._process_discovery_batch,
                data_batch,
                context,
                priority=1
            )
            task_ids.append(task_id)
        
        # Collect results
        results = []
        for task_id in task_ids:
            try:
                result = await self.task_queue.get_task_result(task_id, timeout=300.0)
                results.append(result)
                self.stats["successful_batches"] += 1
            except Exception as e:
                logger.error(f"Batch discovery failed: {task_id} - {e}")
                error_result = BatchDiscoveryResult(
                    batch_id=task_id,
                    discoveries=[],
                    processing_time=0.0,
                    worker_id="unknown",
                    success=False,
                    error_message=str(e)
                )
                results.append(error_result)
                self.stats["failed_batches"] += 1
        
        return results
    
    async def discover_streaming(self, 
                               data_stream: AsyncGenerator[np.ndarray, None],
                               context: str = "streaming") -> AsyncGenerator[BatchDiscoveryResult, None]:
        """Process streaming data for real-time discovery"""
        batch_counter = 0
        
        async for data_batch in data_stream:
            batch_id = f"stream_{context}_{batch_counter}"
            batch_counter += 1
            
            # Submit task
            task_id = await self.task_queue.submit_task(
                task_id=batch_id,
                func=self._process_discovery_batch,
                data_batch,
                f"{context}_stream_{batch_counter}",
                priority=2  # Higher priority for streaming
            )
            
            # Yield result as soon as it's ready
            try:
                result = await self.task_queue.get_task_result(task_id, timeout=60.0)
                yield result
            except Exception as e:
                logger.error(f"Streaming discovery failed: {batch_id} - {e}")
                yield BatchDiscoveryResult(
                    batch_id=batch_id,
                    discoveries=[],
                    processing_time=0.0,
                    worker_id="unknown",
                    success=False,
                    error_message=str(e)
                )
    
    def _process_discovery_batch(self, data: np.ndarray, context: str) -> BatchDiscoveryResult:
        """Process a single discovery batch"""
        batch_id = f"{context}_{hash(data.tobytes()) % 10000}"
        start_time = time.time()
        
        try:
            # Check cache first
            if self.cache:
                cache_key = self._generate_cache_key(data, context)
                cache_hit, cached_result = self.cache.get(cache_key)
                
                if cache_hit:
                    self.stats["cache_hits"] += 1
                    logger.debug(f"Cache hit for batch: {batch_id}")
                    return cached_result
                else:
                    self.stats["cache_misses"] += 1
            
            # Select worker using load balancer
            worker_id = self.load_balancer.select_worker(strategy="least_loaded")
            if not worker_id:
                raise DiscoveryError("No available workers")
            
            # Get worker engine
            worker_engine = self.discovery_workers[worker_id]
            
            # Process discovery
            discoveries = worker_engine.discover(data, context=context)
            
            # Update worker stats
            processing_time = time.time() - start_time
            self.load_balancer.update_worker_stats(
                worker_id,
                avg_response_time=processing_time,
                completed_tasks=self.load_balancer.worker_stats[worker_id]["completed_tasks"] + 1
            )
            
            # Create result
            result = BatchDiscoveryResult(
                batch_id=batch_id,
                discoveries=discoveries,
                processing_time=processing_time,
                worker_id=worker_id,
                success=True
            )
            
            # Cache result if enabled
            if self.cache:
                cache_key = self._generate_cache_key(data, context)
                self.cache.put(cache_key, result)
            
            # Update statistics
            self.stats["total_discoveries"] += len(discoveries)
            self._update_avg_processing_time(processing_time)
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Discovery batch processing failed: {batch_id} - {e}")
            
            return BatchDiscoveryResult(
                batch_id=batch_id,
                discoveries=[],
                processing_time=processing_time,
                worker_id=worker_id if 'worker_id' in locals() else "unknown",
                success=False,
                error_message=str(e)
            )
    
    def _generate_cache_key(self, data: np.ndarray, context: str) -> str:
        """Generate cache key for data and context"""
        data_hash = hash(data.tobytes())
        context_hash = hash(context)
        return f"discovery_{data_hash}_{context_hash}"
    
    def _update_avg_processing_time(self, processing_time: float) -> None:
        """Update average processing time statistics"""
        current_avg = self.stats["avg_processing_time"]
        total_batches = self.stats["successful_batches"] + self.stats["failed_batches"]
        
        if total_batches > 0:
            self.stats["avg_processing_time"] = (
                (current_avg * (total_batches - 1) + processing_time) / total_batches
            )
    
    def _create_batches(self, data_list: List[np.ndarray], batch_size: int) -> List[np.ndarray]:
        """Create batches from data list"""
        batches = []
        for i in range(0, len(data_list), batch_size):
            batch_data = data_list[i:i + batch_size]
            # Combine arrays in batch
            if len(batch_data) == 1:
                batches.append(batch_data[0])
            else:
                # Stack arrays if compatible
                try:
                    batches.append(np.vstack(batch_data))
                except ValueError:
                    # If stacking fails, process individually
                    batches.extend(batch_data)
        
        return batches
    
    def _scale_up_workers(self, additional_workers: int) -> None:
        """Scale up discovery workers"""
        logger.info(f"Scaling up: adding {additional_workers} workers")
        
        current_count = len(self.discovery_workers)
        
        for i in range(additional_workers):
            worker_id = f"discovery_worker_{current_count + i}"
            self.discovery_workers[worker_id] = DiscoveryEngine(self.discovery_threshold)
            self.load_balancer.register_worker(worker_id, self._process_discovery_task)
        
        # Expand thread pool if needed
        self.worker_pool._max_workers += additional_workers
    
    def _scale_down_workers(self, workers_to_remove: int) -> None:
        """Scale down discovery workers"""
        logger.info(f"Scaling down: removing {workers_to_remove} workers")
        
        # Remove workers (keep at least 1)
        worker_ids = list(self.discovery_workers.keys())
        workers_to_remove = min(workers_to_remove, len(worker_ids) - 1)
        
        for i in range(workers_to_remove):
            if len(self.discovery_workers) > 1:
                worker_id = worker_ids[-(i+1)]
                del self.discovery_workers[worker_id]
                if worker_id in self.load_balancer.workers:
                    del self.load_balancer.workers[worker_id]
                    del self.load_balancer.worker_stats[worker_id]
    
    def _process_discovery_task(self, *args, **kwargs) -> Any:
        """Process discovery task (callback for load balancer)"""
        # This is a placeholder for load balancer integration
        return self._process_discovery_batch(*args, **kwargs)
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        metrics = {
            "discovery_stats": self.stats.copy(),
            "task_queue_stats": self.task_queue.get_stats(),
            "worker_count": len(self.discovery_workers),
            "cache_stats": {
                "enabled": self.cache is not None,
                "hit_rate": (
                    self.stats["cache_hits"] / 
                    max(1, self.stats["cache_hits"] + self.stats["cache_misses"])
                ) if self.cache else 0.0
            }
        }
        
        if self.autoscaler:
            metrics["autoscaler_stats"] = self.autoscaler.get_scaling_stats()
        
        return metrics
    
    async def optimize_performance(self) -> Dict[str, str]:
        """Auto-optimize performance based on current metrics"""
        optimization_actions = []
        
        metrics = await self.get_performance_metrics()
        
        # Cache optimization
        if self.cache and metrics["cache_stats"]["hit_rate"] < 0.3:
            # Low cache hit rate, consider increasing cache size
            self.cache.max_size = min(self.cache.max_size * 2, 5000)
            optimization_actions.append("Increased cache size due to low hit rate")
        
        # Worker optimization
        avg_queue_time = metrics["task_queue_stats"].get("avg_execution_time", 0)
        if avg_queue_time > 5.0 and len(self.discovery_workers) < self.max_workers * 2:
            # High queue time, add workers if possible
            self._scale_up_workers(2)
            optimization_actions.append("Added workers due to high queue time")
        
        return {
            "optimizations_applied": optimization_actions,
            "timestamp": time.time()
        }
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the scalable discovery engine"""
        logger.info("Shutting down ScalableDiscoveryEngine")
        
        # Stop auto-scaling
        if self.autoscaler:
            self.autoscaler.stop_monitoring()
        
        # Shutdown task queue
        await self.task_queue.shutdown()
        
        # Shutdown worker pool
        self.worker_pool.shutdown(wait=True, timeout=30)
        
        logger.info("ScalableDiscoveryEngine shutdown complete")


# Convenience functions for easy usage

async def create_scalable_discovery_engine(
    discovery_threshold: float = 0.7,
    max_workers: int = 8,
    enable_auto_scaling: bool = True
) -> ScalableDiscoveryEngine:
    """Create and start a scalable discovery engine"""
    
    engine = ScalableDiscoveryEngine(
        discovery_threshold=discovery_threshold,
        max_workers=max_workers,
        enable_auto_scaling=enable_auto_scaling
    )
    
    await engine.start()
    return engine


async def batch_discover(
    data_batches: List[np.ndarray],
    discovery_threshold: float = 0.7,
    max_workers: int = 4
) -> List[BatchDiscoveryResult]:
    """Convenient function for batch discovery processing"""
    
    engine = await create_scalable_discovery_engine(
        discovery_threshold=discovery_threshold,
        max_workers=max_workers
    )
    
    try:
        results = await engine.discover_batch(data_batches)
        return results
    finally:
        await engine.shutdown()