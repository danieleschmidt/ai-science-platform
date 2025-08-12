"""
Distributed Processing Framework
Advanced parallel processing and distributed computing for scalable bioneural fusion
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import logging
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Queue, Process, cpu_count
import threading
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

from ..algorithms.bioneural_pipeline import BioneuralOlfactoryPipeline, PipelineResult
from ..utils.validation import ValidationMixin
from ..utils.error_handling import robust_execution, safe_array_operation

logger = logging.getLogger(__name__)


@dataclass
class DistributedTask:
    """Represents a distributed processing task"""
    task_id: str
    signal: np.ndarray
    processing_config: Dict[str, Any]
    priority: int = 5
    timeout: Optional[float] = None


@dataclass
class DistributedResult:
    """Result from distributed processing"""
    task_id: str
    result: Optional[PipelineResult]
    processing_time: float
    worker_id: str
    success: bool
    error_message: Optional[str] = None


@dataclass
class WorkerStats:
    """Statistics for a worker process/thread"""
    worker_id: str
    tasks_processed: int
    total_processing_time: float
    average_processing_time: float
    errors: int
    last_activity: float


class DistributedWorker(ABC):
    """Abstract base class for distributed workers"""
    
    @abstractmethod
    def process_task(self, task: DistributedTask) -> DistributedResult:
        """Process a distributed task"""
        pass
    
    @abstractmethod
    def get_stats(self) -> WorkerStats:
        """Get worker statistics"""
        pass


class ThreadWorker(DistributedWorker):
    """Thread-based worker for I/O-bound tasks"""
    
    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.pipeline = BioneuralOlfactoryPipeline()
        self.tasks_processed = 0
        self.total_processing_time = 0.0
        self.errors = 0
        self.last_activity = time.time()
        self._lock = threading.Lock()
    
    @robust_execution(recovery_strategy='graceful_degradation')
    def process_task(self, task: DistributedTask) -> DistributedResult:
        """Process task using thread-based pipeline"""
        start_time = time.time()
        
        try:
            # Update activity timestamp
            with self._lock:
                self.last_activity = time.time()
            
            # Process signal
            result = self.pipeline.process(
                task.signal,
                signal_metadata=task.processing_config
            )
            
            processing_time = time.time() - start_time
            
            # Update statistics
            with self._lock:
                self.tasks_processed += 1
                self.total_processing_time += processing_time
            
            return DistributedResult(
                task_id=task.task_id,
                result=result,
                processing_time=processing_time,
                worker_id=self.worker_id,
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            with self._lock:
                self.errors += 1
                self.total_processing_time += processing_time
            
            logger.error(f"Worker {self.worker_id} failed to process task {task.task_id}: {e}")
            
            return DistributedResult(
                task_id=task.task_id,
                result=None,
                processing_time=processing_time,
                worker_id=self.worker_id,
                success=False,
                error_message=str(e)
            )
    
    def get_stats(self) -> WorkerStats:
        """Get worker statistics"""
        with self._lock:
            avg_time = (self.total_processing_time / self.tasks_processed 
                       if self.tasks_processed > 0 else 0.0)
            
            return WorkerStats(
                worker_id=self.worker_id,
                tasks_processed=self.tasks_processed,
                total_processing_time=self.total_processing_time,
                average_processing_time=avg_time,
                errors=self.errors,
                last_activity=self.last_activity
            )


class ProcessWorker:
    """Process-based worker for CPU-bound tasks"""
    
    def __init__(self, worker_id: str, task_queue: Queue, result_queue: Queue):
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.pipeline = None
        self.stats = {
            'tasks_processed': 0,
            'total_processing_time': 0.0,
            'errors': 0,
            'last_activity': time.time()
        }
    
    def run(self):
        """Main worker process loop"""
        # Initialize pipeline in worker process
        self.pipeline = BioneuralOlfactoryPipeline()
        
        logger.info(f"Process worker {self.worker_id} started")
        
        while True:
            try:
                # Get task with timeout
                task = self.task_queue.get(timeout=30.0)
                
                if task is None:  # Poison pill to stop worker
                    break
                
                # Process task
                result = self._process_task(task)
                
                # Send result
                self.result_queue.put(result)
                
                # Update stats
                self.stats['tasks_processed'] += 1
                self.stats['last_activity'] = time.time()
                
            except Exception as e:
                logger.error(f"Process worker {self.worker_id} error: {e}")
                self.stats['errors'] += 1
                
                # Send error result if we have task info
                if 'task' in locals():
                    error_result = DistributedResult(
                        task_id=task.task_id,
                        result=None,
                        processing_time=0.0,
                        worker_id=self.worker_id,
                        success=False,
                        error_message=str(e)
                    )
                    self.result_queue.put(error_result)
        
        logger.info(f"Process worker {self.worker_id} stopped")
    
    def _process_task(self, task: DistributedTask) -> DistributedResult:
        """Process individual task"""
        start_time = time.time()
        
        try:
            result = self.pipeline.process(
                task.signal,
                signal_metadata=task.processing_config
            )
            
            processing_time = time.time() - start_time
            self.stats['total_processing_time'] += processing_time
            
            return DistributedResult(
                task_id=task.task_id,
                result=result,
                processing_time=processing_time,
                worker_id=self.worker_id,
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.stats['total_processing_time'] += processing_time
            self.stats['errors'] += 1
            
            return DistributedResult(
                task_id=task.task_id,
                result=None,
                processing_time=processing_time,
                worker_id=self.worker_id,
                success=False,
                error_message=str(e)
            )


class DistributedProcessingManager(ValidationMixin):
    """
    Advanced Distributed Processing Manager
    
    Provides scalable parallel processing for bioneural olfactory fusion:
    1. Thread-based processing for I/O-bound workloads
    2. Process-based processing for CPU-bound workloads
    3. Adaptive load balancing and task distribution
    4. Comprehensive monitoring and statistics
    5. Fault tolerance and graceful degradation
    """
    
    def __init__(self, 
                 max_workers: Optional[int] = None,
                 worker_type: str = 'thread',
                 task_timeout: float = 300.0,
                 batch_size: int = 10):
        """
        Initialize distributed processing manager
        
        Args:
            max_workers: Maximum number of worker threads/processes
            worker_type: 'thread' for I/O-bound, 'process' for CPU-bound
            task_timeout: Maximum time per task in seconds
            batch_size: Default batch size for bulk processing
        """
        self.max_workers = max_workers or min(32, (cpu_count() or 1) + 4)
        self.worker_type = worker_type.lower()
        self.task_timeout = self.validate_positive_float(task_timeout, "task_timeout")
        self.batch_size = self.validate_positive_int(batch_size, "batch_size")
        
        if self.worker_type not in ['thread', 'process']:
            raise ValueError("worker_type must be 'thread' or 'process'")
        
        # Worker management
        self.workers: Dict[str, DistributedWorker] = {}
        self.active_tasks: Dict[str, DistributedTask] = {}
        
        # Statistics
        self.total_tasks_processed = 0
        self.total_processing_time = 0.0
        self.failed_tasks = 0
        
        # Thread/Process pools
        self.executor = None
        self.processes = []
        self.task_queue = None
        self.result_queue = None
        
        logger.info(f"DistributedProcessingManager initialized: {self.worker_type} workers, max={self.max_workers}")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
    
    def start(self):
        """Start the distributed processing system"""
        logger.info(f"Starting distributed processing with {self.worker_type} workers")
        
        if self.worker_type == 'thread':
            self._start_thread_workers()
        else:
            self._start_process_workers()
    
    def shutdown(self):
        """Shutdown the distributed processing system"""
        logger.info("Shutting down distributed processing")
        
        if self.worker_type == 'thread':
            self._shutdown_thread_workers()
        else:
            self._shutdown_process_workers()
    
    def _start_thread_workers(self):
        """Start thread-based workers"""
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Create thread workers
        for i in range(self.max_workers):
            worker_id = f"thread_worker_{i}"
            self.workers[worker_id] = ThreadWorker(worker_id)
    
    def _shutdown_thread_workers(self):
        """Shutdown thread-based workers"""
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
        self.workers.clear()
    
    def _start_process_workers(self):
        """Start process-based workers"""
        manager = Manager()
        self.task_queue = manager.Queue()
        self.result_queue = manager.Queue()
        
        # Start worker processes
        for i in range(self.max_workers):
            worker_id = f"process_worker_{i}"
            worker = ProcessWorker(worker_id, self.task_queue, self.result_queue)
            process = Process(target=worker.run)
            process.start()
            self.processes.append(process)
    
    def _shutdown_process_workers(self):
        """Shutdown process-based workers"""
        # Send poison pills to stop workers
        if self.task_queue:
            for _ in range(len(self.processes)):
                self.task_queue.put(None)
        
        # Wait for processes to finish
        for process in self.processes:
            process.join(timeout=10.0)
            if process.is_alive():
                logger.warning(f"Force terminating worker process {process.pid}")
                process.terminate()
                process.join()
        
        self.processes.clear()
    
    @robust_execution(recovery_strategy='graceful_degradation')
    def process_signals_batch(self, 
                             signals: List[np.ndarray],
                             processing_configs: Optional[List[Dict[str, Any]]] = None,
                             priority: int = 5) -> List[DistributedResult]:
        """
        Process a batch of signals in parallel
        
        Args:
            signals: List of chemical signals to process
            processing_configs: Optional processing configurations for each signal
            priority: Task priority (1-10, higher = more priority)
            
        Returns:
            List of processing results
        """
        if not signals:
            return []
        
        # Prepare tasks
        tasks = []
        configs = processing_configs or [{}] * len(signals)
        
        for i, (signal, config) in enumerate(zip(signals, configs)):
            task = DistributedTask(
                task_id=f"batch_task_{int(time.time())}_{i}",
                signal=signal,
                processing_config=config,
                priority=priority,
                timeout=self.task_timeout
            )
            tasks.append(task)
        
        # Process tasks
        if self.worker_type == 'thread':
            return self._process_tasks_threaded(tasks)
        else:
            return self._process_tasks_processes(tasks)
    
    def _process_tasks_threaded(self, tasks: List[DistributedTask]) -> List[DistributedResult]:
        """Process tasks using thread pool"""
        results = []
        
        # Submit tasks to thread pool
        future_to_task = {}
        for task in tasks:
            # Find available worker (simple round-robin)
            worker_id = list(self.workers.keys())[len(future_to_task) % len(self.workers)]
            worker = self.workers[worker_id]
            
            future = self.executor.submit(worker.process_task, task)
            future_to_task[future] = task
            self.active_tasks[task.task_id] = task
        
        # Collect results
        for future in as_completed(future_to_task.keys(), timeout=self.task_timeout + 30):
            task = future_to_task[future]
            
            try:
                result = future.result()
                results.append(result)
                
                # Update statistics
                self.total_tasks_processed += 1
                self.total_processing_time += result.processing_time
                
                if not result.success:
                    self.failed_tasks += 1
                
            except Exception as e:
                logger.error(f"Task {task.task_id} failed: {e}")
                
                error_result = DistributedResult(
                    task_id=task.task_id,
                    result=None,
                    processing_time=0.0,
                    worker_id="unknown",
                    success=False,
                    error_message=str(e)
                )
                results.append(error_result)
                self.failed_tasks += 1
            
            # Remove from active tasks
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
        
        return results
    
    def _process_tasks_processes(self, tasks: List[DistributedTask]) -> List[DistributedResult]:
        """Process tasks using process pool"""
        results = []
        
        # Submit tasks to process queue
        for task in tasks:
            self.task_queue.put(task)
            self.active_tasks[task.task_id] = task
        
        # Collect results
        tasks_remaining = len(tasks)
        timeout_time = time.time() + self.task_timeout + 30
        
        while tasks_remaining > 0 and time.time() < timeout_time:
            try:
                result = self.result_queue.get(timeout=1.0)
                results.append(result)
                
                # Update statistics
                self.total_tasks_processed += 1
                self.total_processing_time += result.processing_time
                
                if not result.success:
                    self.failed_tasks += 1
                
                # Remove from active tasks
                if result.task_id in self.active_tasks:
                    del self.active_tasks[result.task_id]
                
                tasks_remaining -= 1
                
            except Exception:
                continue  # Timeout, continue waiting
        
        # Handle any remaining tasks as timeouts
        for task_id, task in list(self.active_tasks.items()):
            if any(r.task_id == task_id for r in results):
                continue  # Already processed
            
            timeout_result = DistributedResult(
                task_id=task_id,
                result=None,
                processing_time=self.task_timeout,
                worker_id="timeout",
                success=False,
                error_message="Task timeout"
            )
            results.append(timeout_result)
            self.failed_tasks += 1
            del self.active_tasks[task_id]
        
        return results
    
    async def process_signals_async(self, 
                                  signals: List[np.ndarray],
                                  processing_configs: Optional[List[Dict[str, Any]]] = None) -> List[DistributedResult]:
        """
        Asynchronous signal processing with streaming results
        
        Args:
            signals: List of signals to process
            processing_configs: Optional configurations
            
        Returns:
            List of results as they become available
        """
        # Run synchronous processing in thread pool
        loop = asyncio.get_event_loop()
        
        result = await loop.run_in_executor(
            None,
            self.process_signals_batch,
            signals,
            processing_configs
        )
        
        return result
    
    def process_signal_stream(self, 
                            signal_generator: Callable[[], np.ndarray],
                            max_signals: Optional[int] = None,
                            callback: Optional[Callable[[DistributedResult], None]] = None) -> List[DistributedResult]:
        """
        Process continuous stream of signals
        
        Args:
            signal_generator: Function that generates signals
            max_signals: Maximum signals to process (None = unlimited)
            callback: Optional callback for each result
            
        Returns:
            List of all results
        """
        results = []
        processed = 0
        
        logger.info(f"Starting signal stream processing (max={max_signals})")
        
        try:
            while max_signals is None or processed < max_signals:
                try:
                    # Generate next signal
                    signal = signal_generator()
                    if signal is None:
                        break
                    
                    # Process single signal
                    batch_results = self.process_signals_batch([signal])
                    
                    if batch_results:
                        result = batch_results[0]
                        results.append(result)
                        
                        # Call callback if provided
                        if callback:
                            callback(result)
                        
                        processed += 1
                        
                        if processed % 100 == 0:
                            logger.info(f"Processed {processed} signals from stream")
                
                except StopIteration:
                    break
                except Exception as e:
                    logger.error(f"Error in signal stream processing: {e}")
                    break
        
        except KeyboardInterrupt:
            logger.info("Signal stream processing interrupted")
        
        logger.info(f"Signal stream processing complete: {len(results)} results")
        return results
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        # Worker statistics
        worker_stats = {}
        if self.worker_type == 'thread':
            for worker_id, worker in self.workers.items():
                worker_stats[worker_id] = asdict(worker.get_stats())
        
        # Overall statistics
        avg_processing_time = (self.total_processing_time / self.total_tasks_processed 
                             if self.total_tasks_processed > 0 else 0.0)
        
        success_rate = ((self.total_tasks_processed - self.failed_tasks) / self.total_tasks_processed 
                       if self.total_tasks_processed > 0 else 0.0)
        
        return {
            "system_config": {
                "max_workers": self.max_workers,
                "worker_type": self.worker_type,
                "task_timeout": self.task_timeout,
                "batch_size": self.batch_size
            },
            "overall_stats": {
                "total_tasks_processed": self.total_tasks_processed,
                "total_processing_time": self.total_processing_time,
                "average_processing_time": avg_processing_time,
                "failed_tasks": self.failed_tasks,
                "success_rate": success_rate,
                "active_tasks": len(self.active_tasks)
            },
            "worker_stats": worker_stats,
            "performance_metrics": {
                "throughput_per_second": self.total_tasks_processed / max(self.total_processing_time, 1e-10),
                "efficiency_score": success_rate * min(1.0, self.max_workers / max(1, len(self.active_tasks)))
            }
        }
    
    def monitor_performance(self, interval: float = 10.0, callback: Optional[Callable[[Dict], None]] = None):
        """
        Monitor system performance continuously
        
        Args:
            interval: Monitoring interval in seconds
            callback: Optional callback for performance data
        """
        logger.info(f"Starting performance monitoring (interval={interval}s)")
        
        def monitor_loop():
            while True:
                try:
                    stats = self.get_system_stats()
                    
                    if callback:
                        callback(stats)
                    else:
                        # Default logging
                        overall = stats["overall_stats"]
                        logger.info(
                            f"Performance: {overall['total_tasks_processed']} tasks, "
                            f"{overall['success_rate']:.3f} success rate, "
                            f"{overall['average_processing_time']:.4f}s avg time"
                        )
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    logger.error(f"Performance monitoring error: {e}")
                    time.sleep(interval)
        
        # Start monitoring in background thread
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        
        return monitor_thread
    
    def benchmark_throughput(self, 
                           num_signals: int = 100,
                           signal_dim: int = 128,
                           warmup_signals: int = 10) -> Dict[str, float]:
        """
        Benchmark system throughput
        
        Args:
            num_signals: Number of signals to process for benchmark
            signal_dim: Dimension of test signals
            warmup_signals: Number of warmup signals
            
        Returns:
            Benchmark metrics
        """
        logger.info(f"Starting throughput benchmark: {num_signals} signals, {signal_dim}D")
        
        # Generate test signals
        signals = [np.random.randn(signal_dim) for _ in range(num_signals + warmup_signals)]
        
        # Warmup
        if warmup_signals > 0:
            warmup_results = self.process_signals_batch(signals[:warmup_signals])
            logger.info(f"Warmup complete: {len(warmup_results)} signals processed")
        
        # Benchmark
        start_time = time.time()
        results = self.process_signals_batch(signals[warmup_signals:])
        total_time = time.time() - start_time
        
        # Calculate metrics
        successful_results = [r for r in results if r.success]
        
        benchmark_metrics = {
            "total_signals": num_signals,
            "successful_signals": len(successful_results),
            "failed_signals": len(results) - len(successful_results),
            "total_time": total_time,
            "throughput_signals_per_second": len(successful_results) / total_time,
            "average_processing_time": np.mean([r.processing_time for r in successful_results]),
            "median_processing_time": np.median([r.processing_time for r in successful_results]),
            "success_rate": len(successful_results) / len(results),
            "speedup_factor": len(successful_results) / (np.sum([r.processing_time for r in successful_results]) / max(total_time, 1e-10))
        }
        
        logger.info(f"Benchmark complete: {benchmark_metrics['throughput_signals_per_second']:.2f} signals/sec, {benchmark_metrics['success_rate']:.3f} success rate")
        
        return benchmark_metrics
    
    def summary(self) -> Dict[str, Any]:
        """Get comprehensive manager summary"""
        return {
            "manager_type": "DistributedProcessingManager",
            "configuration": {
                "max_workers": self.max_workers,
                "worker_type": self.worker_type,
                "task_timeout": self.task_timeout,
                "batch_size": self.batch_size
            },
            "statistics": self.get_system_stats(),
            "capabilities": [
                "Thread-based parallel processing",
                "Process-based parallel processing", 
                "Async signal processing",
                "Signal stream processing",
                "Performance monitoring",
                "Throughput benchmarking",
                "Fault tolerance and graceful degradation"
            ]
        }