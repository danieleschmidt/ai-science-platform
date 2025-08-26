"""Concurrent and parallel discovery processing"""

import logging
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Optional, Union, Tuple
import numpy as np
import time
from dataclasses import dataclass
import queue
from pathlib import Path

from ..algorithms.discovery import DiscoveryEngine, Discovery
from ..utils.data_utils import generate_sample_data
from .enhanced_caching import get_cache, cached

logger = logging.getLogger(__name__)


@dataclass
class ProcessingTask:
    """Task for concurrent processing"""
    task_id: str
    data: np.ndarray
    context: str
    parameters: Dict[str, Any]
    priority: int = 0


class ConcurrentDiscoveryEngine:
    """Concurrent discovery engine with parallel processing"""
    
    def __init__(self, max_workers: int = None, use_processes: bool = True,
                 cache_enabled: bool = True):
        self.max_workers = max_workers or min(8, mp.cpu_count())
        self.use_processes = use_processes
        self.cache_enabled = cache_enabled
        
        # Thread-safe components
        self.task_queue = queue.PriorityQueue()
        self.results = {}
        self.results_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'tasks_processed': 0,
            'total_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'parallel_speedup': 1.0
        }
        
        # Initialize cache
        if self.cache_enabled:
            self.cache = get_cache()
        
        logger.info(f"ConcurrentDiscoveryEngine initialized: "
                   f"workers={self.max_workers}, processes={use_processes}")
    
    @cached(ttl=1800, tags=['discovery'])
    def _cached_discovery(self, data_hash: str, threshold: float, 
                         context: str) -> List[Dict[str, Any]]:
        """Cached discovery computation"""
        engine = DiscoveryEngine(discovery_threshold=threshold)
        
        # Reconstruct data from hash (simplified - in practice would use better key)
        # For demo, generate consistent data based on hash
        np.random.seed(hash(data_hash) % (2**32))
        data = np.random.randn(100, 5)
        
        discoveries = engine.discover(data, context=context)
        
        # Convert to serializable format
        return [
            {
                'hypothesis': d.hypothesis,
                'confidence': float(d.confidence),
                'evidence_count': len(d.evidence),
                'metadata': d.metadata
            }
            for d in discoveries
        ]
    
    def discover_parallel(self, datasets: List[Tuple[np.ndarray, str]], 
                         threshold: float = 0.7, 
                         chunk_size: int = None) -> Dict[str, List[Discovery]]:
        """Discover patterns in multiple datasets concurrently"""
        
        start_time = time.time()
        results = {}
        
        # Determine optimal chunk size
        if chunk_size is None:
            chunk_size = max(1, len(datasets) // (self.max_workers * 2))
        
        # Create tasks
        tasks = []
        for i, (data, context) in enumerate(datasets):
            task = ProcessingTask(
                task_id=f"task_{i}",
                data=data,
                context=context,
                parameters={'threshold': threshold}
            )
            tasks.append(task)
        
        logger.info(f"Processing {len(tasks)} discovery tasks with {self.max_workers} workers")
        
        # Process tasks concurrently
        if self.use_processes and len(tasks) > 4:
            results = self._process_with_processes(tasks)
        else:
            results = self._process_with_threads(tasks)
        
        total_time = time.time() - start_time
        self.stats['total_processing_time'] += total_time
        self.stats['tasks_processed'] += len(tasks)
        
        # Calculate speedup (estimated)
        estimated_sequential_time = len(tasks) * 0.1  # Assume 0.1s per task
        speedup = estimated_sequential_time / total_time if total_time > 0 else 1.0
        self.stats['parallel_speedup'] = speedup
        
        logger.info(f"Parallel discovery completed: {len(results)} results, "
                   f"{total_time:.2f}s, speedup: {speedup:.1f}x")
        
        return results
    
    def _process_with_threads(self, tasks: List[ProcessingTask]) -> Dict[str, List[Discovery]]:
        """Process tasks using thread pool"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._process_single_task, task): task 
                for task in tasks
            }
            
            # Collect results
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    discoveries = future.result()
                    results[task.task_id] = discoveries
                except Exception as e:
                    logger.error(f"Task {task.task_id} failed: {e}")
                    results[task.task_id] = []
        
        return results
    
    def _process_with_processes(self, tasks: List[ProcessingTask]) -> Dict[str, List[Discovery]]:
        """Process tasks using process pool"""
        results = {}
        
        # Prepare tasks for multiprocessing (serialize data)
        serializable_tasks = []
        for task in tasks:
            serializable_tasks.append({
                'task_id': task.task_id,
                'data': task.data.tolist(),
                'context': task.context,
                'parameters': task.parameters
            })
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task_id = {
                executor.submit(_process_task_worker, task_data): task_data['task_id']
                for task_data in serializable_tasks
            }
            
            # Collect results
            for future in as_completed(future_to_task_id):
                task_id = future_to_task_id[future]
                try:
                    discoveries = future.result()
                    results[task_id] = discoveries
                except Exception as e:
                    logger.error(f"Process task {task_id} failed: {e}")
                    results[task_id] = []
        
        return results
    
    def _process_single_task(self, task: ProcessingTask) -> List[Discovery]:
        """Process a single discovery task"""
        try:
            threshold = task.parameters.get('threshold', 0.7)
            
            # Check cache first
            if self.cache_enabled:
                data_hash = str(hash(task.data.tobytes()))
                cached_result = self._cached_discovery(data_hash, threshold, task.context)
                if cached_result:
                    self.stats['cache_hits'] += 1
                    # Convert back to Discovery objects
                    return [
                        Discovery(
                            hypothesis=r['hypothesis'],
                            confidence=r['confidence'],
                            evidence=np.array([]),  # Simplified
                            metadata=r['metadata']
                        )
                        for r in cached_result
                    ]
            
            self.stats['cache_misses'] += 1
            
            # Perform discovery
            engine = DiscoveryEngine(discovery_threshold=threshold)
            discoveries = engine.discover(task.data, context=task.context)
            
            return discoveries
            
        except Exception as e:
            logger.error(f"Task processing error: {e}")
            return []
    
    def discover_streaming(self, data_stream: Callable[[], Optional[np.ndarray]],
                          context: str = "streaming", threshold: float = 0.7,
                          buffer_size: int = 10) -> None:
        """Process streaming data for real-time discovery"""
        
        logger.info(f"Starting streaming discovery with buffer size {buffer_size}")
        
        buffer = []
        
        while True:
            # Get next data chunk
            data_chunk = data_stream()
            if data_chunk is None:
                break
            
            buffer.append((data_chunk, f"{context}_chunk_{len(buffer)}"))
            
            # Process when buffer is full
            if len(buffer) >= buffer_size:
                results = self.discover_parallel(buffer, threshold=threshold)
                
                # Process results (e.g., trigger alerts, store, etc.)
                for task_id, discoveries in results.items():
                    if discoveries:
                        logger.info(f"Streaming discovery: {task_id} found {len(discoveries)} patterns")
                
                buffer.clear()
        
        # Process remaining buffer
        if buffer:
            results = self.discover_parallel(buffer, threshold=threshold)
            for task_id, discoveries in results.items():
                if discoveries:
                    logger.info(f"Final streaming discovery: {task_id} found {len(discoveries)} patterns")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        cache_stats = self.cache.get_stats() if self.cache_enabled else {}
        
        return {
            'tasks_processed': self.stats['tasks_processed'],
            'total_processing_time': self.stats['total_processing_time'],
            'avg_task_time': (self.stats['total_processing_time'] / 
                            max(1, self.stats['tasks_processed'])),
            'parallel_speedup': self.stats['parallel_speedup'],
            'cache_hit_rate': (self.stats['cache_hits'] / 
                             max(1, self.stats['cache_hits'] + self.stats['cache_misses'])),
            'max_workers': self.max_workers,
            'use_processes': self.use_processes,
            'cache_stats': cache_stats
        }


def _process_task_worker(task_data: Dict[str, Any]) -> List[Discovery]:
    """Worker function for multiprocessing"""
    try:
        # Reconstruct task
        data = np.array(task_data['data'])
        context = task_data['context']
        threshold = task_data['parameters'].get('threshold', 0.7)
        
        # Perform discovery
        engine = DiscoveryEngine(discovery_threshold=threshold)
        discoveries = engine.discover(data, context=context)
        
        return discoveries
        
    except Exception as e:
        logger.error(f"Worker process error: {e}")
        return []


class DiscoveryPipeline:
    """Pipeline for complex multi-stage discovery processing"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.stages = []
        self.metrics = {
            'total_processed': 0,
            'total_time': 0.0,
            'stage_times': {}
        }
    
    def add_stage(self, name: str, processor: Callable[[Any], Any]) -> 'DiscoveryPipeline':
        """Add processing stage to pipeline"""
        self.stages.append((name, processor))
        self.metrics['stage_times'][name] = 0.0
        return self
    
    def process_batch(self, input_data: List[Any]) -> List[Any]:
        """Process batch through all pipeline stages"""
        
        current_data = input_data
        start_time = time.time()
        
        for stage_name, processor in self.stages:
            stage_start = time.time()
            
            logger.debug(f"Pipeline stage: {stage_name} processing {len(current_data)} items")
            
            # Process stage with threading
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(processor, item) for item in current_data]
                current_data = [f.result() for f in as_completed(futures)]
            
            stage_time = time.time() - stage_start
            self.metrics['stage_times'][stage_name] += stage_time
            
            logger.debug(f"Stage {stage_name} completed in {stage_time:.2f}s")
        
        total_time = time.time() - start_time
        self.metrics['total_processed'] += len(input_data)
        self.metrics['total_time'] += total_time
        
        return current_data
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics"""
        return dict(self.metrics)


# Example usage and testing
if __name__ == "__main__":
    # Test concurrent discovery
    engine = ConcurrentDiscoveryEngine(max_workers=4)
    
    # Generate test datasets
    datasets = []
    for i in range(10):
        data, _ = generate_sample_data(size=50, data_type='normal')
        datasets.append((data, f"dataset_{i}"))
    
    # Run parallel discovery
    results = engine.discover_parallel(datasets, threshold=0.6)
    
    print(f"Processed {len(datasets)} datasets")
    print(f"Performance stats: {engine.get_performance_stats()}")
    
    # Test pipeline
    pipeline = DiscoveryPipeline(max_workers=2)
    pipeline.add_stage("normalize", lambda x: x / np.linalg.norm(x))
    pipeline.add_stage("detect", lambda x: x[x > 0.5])
    
    test_data = [np.random.randn(20) for _ in range(5)]
    results = pipeline.process_batch(test_data)
    
    print(f"Pipeline metrics: {pipeline.get_metrics()}")