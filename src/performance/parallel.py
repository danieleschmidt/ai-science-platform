"""Parallel processing and concurrent execution utilities"""

import time
import logging
from typing import List, Callable, Any, Dict, Optional, Tuple, Iterator
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import threading
import numpy as np
from functools import partial

logger = logging.getLogger(__name__)


class ParallelProcessor:
    """Advanced parallel processing with adaptive load balancing"""
    
    def __init__(self, 
                 max_workers: Optional[int] = None,
                 use_processes: bool = False,
                 chunk_size: Optional[int] = None):
        
        self.max_workers = max_workers or min(cpu_count(), 8)
        self.use_processes = use_processes
        self.chunk_size = chunk_size
        self._stats = {
            "tasks_completed": 0,
            "total_execution_time": 0.0,
            "avg_task_time": 0.0,
            "parallel_efficiency": 0.0
        }
        logger.info(f"ParallelProcessor initialized: workers={self.max_workers}, processes={use_processes}")
    
    def map(self, func: Callable, iterable: List[Any], 
           progress_callback: Optional[Callable[[int, int], None]] = None) -> List[Any]:
        """Execute function over iterable in parallel"""
        
        if not iterable:
            return []
        
        total_items = len(iterable)
        if total_items == 1:
            # No point in parallelizing single item
            return [func(iterable[0])]
        
        start_time = time.time()
        
        # Choose executor type
        ExecutorClass = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        try:
            with ExecutorClass(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_index = {
                    executor.submit(func, item): idx 
                    for idx, item in enumerate(iterable)
                }
                
                # Collect results in order
                results = [None] * total_items
                completed_count = 0
                
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    
                    try:
                        results[index] = future.result()
                        completed_count += 1
                        
                        # Progress callback
                        if progress_callback:
                            progress_callback(completed_count, total_items)
                        
                    except Exception as e:
                        logger.error(f"Task {index} failed: {e}")
                        results[index] = None
                
                execution_time = time.time() - start_time
                self._update_stats(total_items, execution_time)
                
                logger.info(f"Parallel execution completed: {total_items} tasks in {execution_time:.2f}s")
                return results
        
        except Exception as e:
            logger.error(f"Parallel execution failed: {e}")
            # Fallback to sequential execution
            logger.info("Falling back to sequential execution")
            return [func(item) for item in iterable]
    
    def map_chunked(self, func: Callable, iterable: List[Any], 
                    chunk_size: Optional[int] = None) -> List[Any]:
        """Execute function over chunked iterable for better memory efficiency"""
        
        chunk_size = chunk_size or self.chunk_size or max(1, len(iterable) // (self.max_workers * 2))
        
        # Create chunks
        chunks = [iterable[i:i + chunk_size] for i in range(0, len(iterable), chunk_size)]
        
        def process_chunk(chunk: List[Any]) -> List[Any]:
            return [func(item) for item in chunk]
        
        # Process chunks in parallel
        chunk_results = self.map(process_chunk, chunks)
        
        # Flatten results
        results = []
        for chunk_result in chunk_results:
            if chunk_result:
                results.extend(chunk_result)
        
        return results
    
    def starmap(self, func: Callable, iterable: List[Tuple], 
                progress_callback: Optional[Callable[[int, int], None]] = None) -> List[Any]:
        """Execute function with unpacked arguments in parallel"""
        
        def wrapper(args):
            return func(*args)
        
        return self.map(wrapper, iterable, progress_callback)
    
    def submit_batch(self, tasks: List[Tuple[Callable, Tuple, Dict]]) -> List[Any]:
        """Submit a batch of different tasks for parallel execution"""
        
        def execute_task(task_spec: Tuple[Callable, Tuple, Dict]) -> Any:
            func, args, kwargs = task_spec
            return func(*args, **kwargs)
        
        return self.map(execute_task, tasks)
    
    def _update_stats(self, num_tasks: int, execution_time: float) -> None:
        """Update execution statistics"""
        self._stats["tasks_completed"] += num_tasks
        self._stats["total_execution_time"] += execution_time
        self._stats["avg_task_time"] = execution_time / num_tasks if num_tasks > 0 else 0
        
        # Estimate parallel efficiency (simplified)
        sequential_estimate = self._stats["avg_task_time"] * num_tasks
        if sequential_estimate > 0:
            self._stats["parallel_efficiency"] = sequential_estimate / execution_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get parallel processing statistics"""
        return {
            **self._stats,
            "max_workers": self.max_workers,
            "use_processes": self.use_processes,
            "chunk_size": self.chunk_size
        }
    
    def reset_stats(self) -> None:
        """Reset statistics"""
        self._stats = {
            "tasks_completed": 0,
            "total_execution_time": 0.0,
            "avg_task_time": 0.0,
            "parallel_efficiency": 0.0
        }


class AdaptiveParallelProcessor(ParallelProcessor):
    """Parallel processor with adaptive worker management"""
    
    def __init__(self, 
                 initial_workers: int = 2,
                 max_workers: Optional[int] = None,
                 performance_window: int = 10):
        
        super().__init__(max_workers=max_workers or cpu_count())
        self.current_workers = initial_workers
        self.performance_window = performance_window
        self.performance_history = []
        self._lock = threading.Lock()
    
    def map(self, func: Callable, iterable: List[Any], 
           progress_callback: Optional[Callable[[int, int], None]] = None) -> List[Any]:
        """Execute with adaptive worker count"""
        
        # Temporarily set workers for this execution
        original_workers = self.max_workers
        self.max_workers = self.current_workers
        
        start_time = time.time()
        results = super().map(func, iterable, progress_callback)
        execution_time = time.time() - start_time
        
        # Restore original max workers
        self.max_workers = original_workers
        
        # Update performance tracking
        self._update_performance_tracking(len(iterable), execution_time)
        
        return results
    
    def _update_performance_tracking(self, num_tasks: int, execution_time: float) -> None:
        """Update performance tracking and adjust worker count"""
        with self._lock:
            throughput = num_tasks / execution_time if execution_time > 0 else 0
            
            self.performance_history.append({
                "workers": self.current_workers,
                "throughput": throughput,
                "execution_time": execution_time,
                "num_tasks": num_tasks
            })
            
            # Keep only recent history
            if len(self.performance_history) > self.performance_window:
                self.performance_history.pop(0)
            
            # Adjust worker count if we have enough data
            if len(self.performance_history) >= 3:
                self._adjust_worker_count()
    
    def _adjust_worker_count(self) -> None:
        """Adjust worker count based on performance history"""
        if len(self.performance_history) < 3:
            return
        
        recent_performance = self.performance_history[-3:]
        
        # Calculate performance trend
        throughputs = [p["throughput"] for p in recent_performance]
        avg_throughput = sum(throughputs) / len(throughputs)
        
        # Check if we should increase or decrease workers
        if avg_throughput > 0:
            # Try increasing workers if performance is good and not at max
            if (self.current_workers < self.max_workers and 
                all(t >= avg_throughput * 0.9 for t in throughputs[-2:])):
                
                self.current_workers = min(self.current_workers + 1, self.max_workers)
                logger.debug(f"Increased workers to {self.current_workers}")
            
            # Try decreasing workers if performance is declining
            elif (self.current_workers > 1 and 
                  throughputs[-1] < throughputs[0] * 0.8):
                
                self.current_workers = max(self.current_workers - 1, 1)
                logger.debug(f"Decreased workers to {self.current_workers}")


def parallel_discovery(discovery_engines: List[Any], 
                      data_batches: List[Tuple[np.ndarray, Optional[np.ndarray]]], 
                      contexts: List[str] = None,
                      max_workers: int = None) -> List[List[Any]]:
    """Execute multiple discovery processes in parallel"""
    
    if not discovery_engines or not data_batches:
        return []
    
    max_workers = max_workers or min(cpu_count(), len(data_batches))
    contexts = contexts or [""] * len(data_batches)
    
    def run_discovery(args: Tuple[Any, Tuple[np.ndarray, Optional[np.ndarray]], str]) -> List[Any]:
        """Run discovery on a single batch"""
        engine, (data, targets), context = args
        try:
            return engine.discover(data, targets, context)
        except Exception as e:
            logger.error(f"Discovery failed for batch: {e}")
            return []
    
    # Create task arguments
    if len(discovery_engines) == 1:
        # Use same engine for all batches
        tasks = [(discovery_engines[0], batch, context) 
                for batch, context in zip(data_batches, contexts)]
    else:
        # Use different engines (cycle if needed)
        tasks = []
        for i, (batch, context) in enumerate(zip(data_batches, contexts)):
            engine = discovery_engines[i % len(discovery_engines)]
            tasks.append((engine, batch, context))
    
    # Execute in parallel
    processor = ParallelProcessor(max_workers=max_workers)
    results = processor.map(run_discovery, tasks)
    
    logger.info(f"Parallel discovery completed: {len(tasks)} batches processed")
    return results


def parallel_experiment_runner(experiment_configs: List[Any],
                               experiment_func: Callable,
                               data_generator: Callable = None,
                               max_workers: int = None) -> Dict[str, List[Any]]:
    """Run multiple experiments in parallel"""
    
    if not experiment_configs:
        return {}
    
    max_workers = max_workers or min(cpu_count(), len(experiment_configs))
    
    def run_experiment(config: Any) -> Tuple[str, List[Any]]:
        """Run a single experiment configuration"""
        try:
            from ..experiments.runner import ExperimentRunner
            
            # Create temporary experiment runner
            runner = ExperimentRunner(f"/tmp/parallel_experiments/{config.name}")
            runner.register_experiment(config)
            
            # Generate data if needed
            if data_generator:
                data = data_generator()
            else:
                data = None
            
            # Run experiment
            results = runner.run_experiment(config.name, experiment_func, data)
            return config.name, results
            
        except Exception as e:
            logger.error(f"Experiment {config.name} failed: {e}")
            return config.name, []
    
    # Execute experiments in parallel
    processor = ParallelProcessor(max_workers=max_workers, use_processes=True)
    results = processor.map(run_experiment, experiment_configs)
    
    # Convert to dictionary
    experiment_results = {}
    for name, result in results:
        if name and result:
            experiment_results[name] = result
    
    logger.info(f"Parallel experiments completed: {len(experiment_results)} successful")
    return experiment_results


def batch_process_data(data_processor: Callable,
                       data_batches: List[np.ndarray],
                       batch_size: int = None,
                       max_workers: int = None) -> List[Any]:
    """Process data batches in parallel with automatic batching"""
    
    if not data_batches:
        return []
    
    # Auto-determine batch size if not provided
    if batch_size is None:
        total_elements = sum(batch.size for batch in data_batches)
        batch_size = max(1, total_elements // (max_workers or cpu_count()))
    
    # Group small batches together
    processed_batches = []
    current_batch = []
    current_size = 0
    
    for batch in data_batches:
        current_batch.append(batch)
        current_size += batch.size
        
        if current_size >= batch_size:
            # Concatenate batches if they're compatible
            try:
                if len(current_batch) > 1 and all(b.ndim == current_batch[0].ndim for b in current_batch):
                    combined_batch = np.concatenate(current_batch, axis=0)
                    processed_batches.append(combined_batch)
                else:
                    processed_batches.extend(current_batch)
            except Exception:
                processed_batches.extend(current_batch)
            
            current_batch = []
            current_size = 0
    
    # Add remaining batches
    if current_batch:
        processed_batches.extend(current_batch)
    
    # Process in parallel
    processor = ParallelProcessor(max_workers=max_workers)
    return processor.map(data_processor, processed_batches)


class ParallelBenchmark:
    """Benchmark parallel processing performance"""
    
    @staticmethod
    def benchmark_worker_counts(func: Callable,
                                data: List[Any],
                                worker_counts: List[int] = None) -> Dict[int, Dict[str, float]]:
        """Benchmark function with different worker counts"""
        
        if worker_counts is None:
            worker_counts = [1, 2, 4, cpu_count(), cpu_count() * 2]
        
        results = {}
        
        for workers in worker_counts:
            processor = ParallelProcessor(max_workers=workers)
            
            start_time = time.time()
            processor.map(func, data)
            execution_time = time.time() - start_time
            
            stats = processor.get_stats()
            
            results[workers] = {
                "execution_time": execution_time,
                "throughput": len(data) / execution_time if execution_time > 0 else 0,
                "efficiency": stats.get("parallel_efficiency", 0)
            }
            
            logger.info(f"Workers: {workers}, Time: {execution_time:.2f}s, "
                       f"Throughput: {results[workers]['throughput']:.2f} tasks/s")
        
        return results
    
    @staticmethod
    def find_optimal_workers(func: Callable,
                            data: List[Any],
                            max_workers: int = None) -> int:
        """Find optimal number of workers for given function and data"""
        
        max_workers = max_workers or cpu_count() * 2
        worker_counts = [1, 2, 4] + list(range(8, max_workers + 1, 4))
        
        benchmark_results = ParallelBenchmark.benchmark_worker_counts(
            func, data, worker_counts
        )
        
        # Find worker count with best throughput
        best_workers = max(
            benchmark_results.keys(),
            key=lambda w: benchmark_results[w]["throughput"]
        )
        
        logger.info(f"Optimal worker count: {best_workers} "
                   f"(throughput: {benchmark_results[best_workers]['throughput']:.2f})")
        
        return best_workers


# Global parallel processor instance
_parallel_processor = None


def get_parallel_processor() -> ParallelProcessor:
    """Get global parallel processor"""
    global _parallel_processor
    if _parallel_processor is None:
        _parallel_processor = ParallelProcessor()
    return _parallel_processor