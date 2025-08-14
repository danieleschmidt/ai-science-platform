"""Distributed processing for scientific workloads"""

import multiprocessing as mp
import threading
import time
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
import logging
import pickle
import queue
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


@dataclass
class ProcessingTask:
    """Individual processing task"""
    task_id: str
    function_name: str
    args: tuple
    kwargs: dict
    priority: int = 0
    retry_count: int = 0
    max_retries: int = 3
    timeout: float = 300.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingResult:
    """Result from processing task"""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    processing_time: float = 0.0
    worker_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class DistributedProcessor:
    """High-performance distributed processing system"""
    
    def __init__(self, max_workers: int = None, use_processes: bool = True):
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) + 4)
        self.use_processes = use_processes
        
        # Processing state
        self.task_queue = queue.PriorityQueue()
        self.result_queue = queue.Queue()
        self.is_running = False
        
        # Workers
        self.executor = None
        self.worker_threads = []
        
        # Statistics
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_processing_time": 0.0,
            "active_workers": 0,
            "queue_size": 0
        }
        
        # Function registry
        self.function_registry = {}
        
        # Performance monitoring
        self.performance_metrics = {
            "throughput": [],
            "latency": [],
            "error_rate": [],
            "resource_usage": []
        }
        
        logger.info(f"DistributedProcessor initialized: {self.max_workers} workers, "
                   f"mode={'processes' if use_processes else 'threads'}")
    
    def register_function(self, name: str, func: Callable) -> None:
        """Register a function for distributed execution"""
        self.function_registry[name] = func
        logger.info(f"Registered function: {name}")
    
    def start(self) -> None:
        """Start the distributed processing system"""
        if self.is_running:
            logger.warning("Processor already running")
            return
        
        self.is_running = True
        
        # Start executor
        if self.use_processes:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Start monitoring threads
        self.worker_threads = [
            threading.Thread(target=self._worker_loop, daemon=True),
            threading.Thread(target=self._stats_collector, daemon=True),
            threading.Thread(target=self._performance_monitor, daemon=True)
        ]
        
        for thread in self.worker_threads:
            thread.start()
        
        logger.info("Distributed processor started")
    
    def stop(self) -> None:
        """Stop the distributed processing system"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=True)
        
        # Wait for worker threads
        for thread in self.worker_threads:
            if thread.is_alive():
                thread.join(timeout=5.0)
        
        logger.info("Distributed processor stopped")
    
    def submit_task(self, function_name: str, *args, priority: int = 0, 
                   timeout: float = 300.0, **kwargs) -> str:
        """Submit a task for distributed processing"""
        task_id = f"task_{int(time.time() * 1000)}_{self.stats['tasks_submitted']}"
        
        task = ProcessingTask(
            task_id=task_id,
            function_name=function_name,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout
        )
        
        # Higher priority = lower number for queue ordering
        self.task_queue.put((-priority, time.time(), task))
        self.stats["tasks_submitted"] += 1
        
        logger.debug(f"Task submitted: {task_id}")
        return task_id
    
    def submit_batch(self, function_name: str, arg_list: List[tuple], 
                    priority: int = 0, timeout: float = 300.0) -> List[str]:
        """Submit multiple tasks as a batch"""
        task_ids = []
        
        for i, args in enumerate(arg_list):
            if isinstance(args, dict):
                task_id = self.submit_task(function_name, priority=priority, 
                                         timeout=timeout, **args)
            elif isinstance(args, (tuple, list)):
                task_id = self.submit_task(function_name, *args, priority=priority, 
                                         timeout=timeout)
            else:
                task_id = self.submit_task(function_name, args, priority=priority, 
                                         timeout=timeout)
            task_ids.append(task_id)
        
        logger.info(f"Batch submitted: {len(task_ids)} tasks for {function_name}")
        return task_ids
    
    def get_result(self, task_id: str, timeout: float = None) -> Optional[ProcessingResult]:
        """Get result for a specific task"""
        start_time = time.time()
        
        while timeout is None or (time.time() - start_time) < timeout:
            try:
                result = self.result_queue.get(timeout=1.0)
                if result.task_id == task_id:
                    return result
                else:
                    # Put back if not our result
                    self.result_queue.put(result)
            except queue.Empty:
                continue
        
        return None
    
    def get_results(self, task_ids: List[str], timeout: float = None) -> Dict[str, ProcessingResult]:
        """Get results for multiple tasks"""
        results = {}
        remaining_ids = set(task_ids)
        start_time = time.time()
        
        while remaining_ids and (timeout is None or (time.time() - start_time) < timeout):
            try:
                result = self.result_queue.get(timeout=1.0)
                if result.task_id in remaining_ids:
                    results[result.task_id] = result
                    remaining_ids.remove(result.task_id)
                else:
                    # Put back if not in our list
                    self.result_queue.put(result)
            except queue.Empty:
                continue
        
        return results
    
    def map(self, function_name: str, arg_list: List[Any], 
           timeout: float = 300.0, max_workers: int = None) -> List[ProcessingResult]:
        """Map function over list of arguments (similar to multiprocessing.map)"""
        
        # Submit all tasks
        task_ids = self.submit_batch(function_name, arg_list, timeout=timeout)
        
        # Collect results
        results = self.get_results(task_ids, timeout=timeout * len(arg_list))
        
        # Return in original order
        ordered_results = []
        for task_id in task_ids:
            if task_id in results:
                ordered_results.append(results[task_id])
            else:
                # Create failed result for missing tasks
                ordered_results.append(ProcessingResult(
                    task_id=task_id,
                    success=False,
                    error="Task timeout or failed to complete"
                ))
        
        return ordered_results
    
    def map_async(self, function_name: str, arg_list: List[Any], 
                 callback: Optional[Callable] = None, 
                 timeout: float = 300.0) -> List[str]:
        """Asynchronous map with optional callback for each result"""
        task_ids = self.submit_batch(function_name, arg_list, timeout=timeout)
        
        if callback:
            def result_handler():
                for task_id in task_ids:
                    result = self.get_result(task_id, timeout=timeout)
                    if result:
                        callback(result)
            
            threading.Thread(target=result_handler, daemon=True).start()
        
        return task_ids
    
    def _worker_loop(self) -> None:
        """Main worker loop for processing tasks"""
        while self.is_running:
            try:
                # Get next task
                try:
                    priority, timestamp, task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process task
                result = self._process_task(task)
                self.result_queue.put(result)
                
                # Update stats
                if result.success:
                    self.stats["tasks_completed"] += 1
                else:
                    self.stats["tasks_failed"] += 1
                
                self.stats["total_processing_time"] += result.processing_time
                
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
    
    def _process_task(self, task: ProcessingTask) -> ProcessingResult:
        """Process a single task"""
        start_time = time.time()
        worker_id = f"worker_{threading.current_thread().ident}"
        
        try:
            # Check if function is registered
            if task.function_name not in self.function_registry:
                raise ValueError(f"Function not registered: {task.function_name}")
            
            func = self.function_registry[task.function_name]
            
            # Execute with timeout
            if self.use_processes:
                future = self.executor.submit(func, *task.args, **task.kwargs)
                result = future.result(timeout=task.timeout)
            else:
                result = func(*task.args, **task.kwargs)
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                task_id=task.task_id,
                success=True,
                result=result,
                processing_time=processing_time,
                worker_id=worker_id
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                self.task_queue.put((-task.priority, time.time(), task))
                logger.warning(f"Task {task.task_id} failed, retrying ({task.retry_count}/{task.max_retries}): {error_msg}")
                
                return ProcessingResult(
                    task_id=task.task_id,
                    success=False,
                    error=f"Retry {task.retry_count}: {error_msg}",
                    processing_time=processing_time,
                    worker_id=worker_id
                )
            else:
                logger.error(f"Task {task.task_id} failed permanently: {error_msg}")
                
                return ProcessingResult(
                    task_id=task.task_id,
                    success=False,
                    error=error_msg,
                    processing_time=processing_time,
                    worker_id=worker_id
                )
    
    def _stats_collector(self) -> None:
        """Collect processing statistics"""
        while self.is_running:
            try:
                self.stats["queue_size"] = self.task_queue.qsize()
                self.stats["active_workers"] = threading.active_count() - 1  # Exclude main thread
                time.sleep(5.0)
            except Exception as e:
                logger.error(f"Stats collector error: {e}")
    
    def _performance_monitor(self) -> None:
        """Monitor performance metrics"""
        last_completed = 0
        last_time = time.time()
        
        while self.is_running:
            try:
                current_time = time.time()
                current_completed = self.stats["tasks_completed"]
                
                # Calculate throughput (tasks/second)
                time_delta = current_time - last_time
                if time_delta > 0:
                    throughput = (current_completed - last_completed) / time_delta
                    self.performance_metrics["throughput"].append((current_time, throughput))
                
                # Calculate error rate
                total_tasks = self.stats["tasks_completed"] + self.stats["tasks_failed"]
                error_rate = self.stats["tasks_failed"] / max(1, total_tasks)
                self.performance_metrics["error_rate"].append((current_time, error_rate))
                
                # Calculate average latency
                if current_completed > 0:
                    avg_latency = self.stats["total_processing_time"] / current_completed
                    self.performance_metrics["latency"].append((current_time, avg_latency))
                
                # Keep only recent metrics (last hour)
                cutoff_time = current_time - 3600
                for metric_name in self.performance_metrics:
                    self.performance_metrics[metric_name] = [
                        (t, v) for t, v in self.performance_metrics[metric_name] 
                        if t >= cutoff_time
                    ]
                
                last_completed = current_completed
                last_time = current_time
                
                time.sleep(10.0)
                
            except Exception as e:
                logger.error(f"Performance monitor error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        stats = self.stats.copy()
        
        # Add derived metrics
        total_tasks = stats["tasks_completed"] + stats["tasks_failed"]
        if total_tasks > 0:
            stats["success_rate"] = stats["tasks_completed"] / total_tasks
            stats["average_processing_time"] = stats["total_processing_time"] / stats["tasks_completed"] if stats["tasks_completed"] > 0 else 0
        else:
            stats["success_rate"] = 0.0
            stats["average_processing_time"] = 0.0
        
        return stats
    
    def get_performance_metrics(self) -> Dict[str, List[Tuple[float, float]]]:
        """Get performance metrics history"""
        return {name: metrics.copy() for name, metrics in self.performance_metrics.items()}
    
    def scale_workers(self, new_max_workers: int) -> None:
        """Dynamically scale the number of workers"""
        if not self.is_running:
            self.max_workers = new_max_workers
            return
        
        # For now, require restart to change worker count
        # In production, this would implement dynamic scaling
        logger.warning(f"Dynamic scaling not implemented. Current: {self.max_workers}, Requested: {new_max_workers}")
        self.max_workers = new_max_workers
    
    def clear_queues(self) -> None:
        """Clear all pending tasks and results"""
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("Queues cleared")
    
    def benchmark(self, function_name: str, test_data: List[Any], 
                 num_iterations: int = 3) -> Dict[str, Any]:
        """Benchmark processing performance"""
        logger.info(f"Starting benchmark: {function_name} with {len(test_data)} items, {num_iterations} iterations")
        
        benchmark_results = {
            "function_name": function_name,
            "test_data_size": len(test_data),
            "iterations": num_iterations,
            "results": []
        }
        
        for iteration in range(num_iterations):
            start_time = time.time()
            
            # Submit all tasks
            task_ids = self.submit_batch(function_name, test_data, priority=10)
            
            # Wait for completion
            results = self.get_results(task_ids, timeout=300.0)
            
            end_time = time.time()
            
            # Analyze results
            successful_results = [r for r in results.values() if r.success]
            failed_results = [r for r in results.values() if not r.success]
            
            iteration_result = {
                "iteration": iteration + 1,
                "total_time": end_time - start_time,
                "tasks_submitted": len(task_ids),
                "tasks_successful": len(successful_results),
                "tasks_failed": len(failed_results),
                "throughput": len(successful_results) / (end_time - start_time),
                "avg_task_time": np.mean([r.processing_time for r in successful_results]) if successful_results else 0,
                "success_rate": len(successful_results) / len(task_ids) if task_ids else 0
            }
            
            benchmark_results["results"].append(iteration_result)
            logger.info(f"Iteration {iteration + 1}: {iteration_result['throughput']:.2f} tasks/sec, "
                       f"{iteration_result['success_rate']:.2%} success rate")
        
        # Calculate overall statistics
        if benchmark_results["results"]:
            throughputs = [r["throughput"] for r in benchmark_results["results"]]
            success_rates = [r["success_rate"] for r in benchmark_results["results"]]
            
            benchmark_results["summary"] = {
                "avg_throughput": np.mean(throughputs),
                "max_throughput": np.max(throughputs),
                "min_throughput": np.min(throughputs),
                "std_throughput": np.std(throughputs),
                "avg_success_rate": np.mean(success_rates),
                "min_success_rate": np.min(success_rates)
            }
        
        logger.info(f"Benchmark complete: avg throughput = {benchmark_results['summary']['avg_throughput']:.2f} tasks/sec")
        return benchmark_results
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()