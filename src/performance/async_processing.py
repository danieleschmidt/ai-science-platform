"""Asynchronous processing and task queue management for scalability"""

import asyncio
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import threading
import queue
import concurrent.futures
from datetime import datetime

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Represents a task in the async processing system"""
    id: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: int = 0  # Higher number = higher priority
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[Exception] = None
    retry_count: int = 0
    max_retries: int = 3


class AsyncTaskQueue:
    """High-performance async task queue with priority and load balancing"""
    
    def __init__(self, 
                 max_concurrent_tasks: int = 10,
                 max_queue_size: int = 1000,
                 enable_priorities: bool = True):
        
        self.max_concurrent_tasks = max_concurrent_tasks
        self.max_queue_size = max_queue_size
        self.enable_priorities = enable_priorities
        
        self._tasks: Dict[str, Task] = {}
        self._pending_queue = deque()
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._completed_tasks = deque(maxlen=100)  # Keep recent history
        
        self._shutdown = False
        self._stats = {
            "total_submitted": 0,
            "total_completed": 0,
            "total_failed": 0,
            "current_queue_size": 0,
            "peak_queue_size": 0,
            "avg_execution_time": 0.0
        }
        
        # Worker management
        self._workers: List[asyncio.Task] = []
        self._worker_semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        logger.info(f"AsyncTaskQueue initialized: max_concurrent={max_concurrent_tasks}")
    
    async def submit_task(self, 
                         task_id: str,
                         func: Union[Callable, Coroutine],
                         *args, 
                         priority: int = 0,
                         max_retries: int = 3,
                         **kwargs) -> str:
        """Submit a task for async execution"""
        
        if self._shutdown:
            raise RuntimeError("Task queue is shutdown")
        
        if len(self._pending_queue) >= self.max_queue_size:
            raise RuntimeError(f"Task queue full (max={self.max_queue_size})")
        
        # Create task
        task = Task(
            id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            max_retries=max_retries
        )
        
        self._tasks[task_id] = task
        
        # Add to queue based on priority
        if self.enable_priorities:
            self._insert_by_priority(task)
        else:
            self._pending_queue.append(task)
        
        self._stats["total_submitted"] += 1
        self._stats["current_queue_size"] = len(self._pending_queue)
        self._stats["peak_queue_size"] = max(self._stats["peak_queue_size"], 
                                           self._stats["current_queue_size"])
        
        logger.debug(f"Task submitted: {task_id} (priority={priority})")
        return task_id
    
    def _insert_by_priority(self, task: Task) -> None:
        """Insert task into queue maintaining priority order"""
        inserted = False
        for i, existing_task in enumerate(self._pending_queue):
            if task.priority > existing_task.priority:
                self._pending_queue.insert(i, task)
                inserted = True
                break
        
        if not inserted:
            self._pending_queue.append(task)
    
    async def start_workers(self, num_workers: Optional[int] = None) -> None:
        """Start worker tasks"""
        if num_workers is None:
            num_workers = self.max_concurrent_tasks
        
        self._workers = [
            asyncio.create_task(self._worker(f"worker_{i}"))
            for i in range(num_workers)
        ]
        
        logger.info(f"Started {num_workers} async workers")
    
    async def _worker(self, worker_id: str) -> None:
        """Worker task that processes queue"""
        logger.debug(f"Worker started: {worker_id}")
        
        while not self._shutdown:
            try:
                # Get next task
                if not self._pending_queue:
                    await asyncio.sleep(0.1)  # Brief wait
                    continue
                
                task = self._pending_queue.popleft()
                self._stats["current_queue_size"] = len(self._pending_queue)
                
                # Execute task with semaphore control
                async with self._worker_semaphore:
                    await self._execute_task(task, worker_id)
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1)  # Brief recovery pause
        
        logger.debug(f"Worker stopped: {worker_id}")
    
    async def _execute_task(self, task: Task, worker_id: str) -> None:
        """Execute a single task"""
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        
        try:
            logger.debug(f"Executing task {task.id} on {worker_id}")
            
            # Handle both sync and async functions
            if asyncio.iscoroutinefunction(task.func):
                result = await task.func(*task.args, **task.kwargs)
            else:
                # Run sync function in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, 
                    lambda: task.func(*task.args, **task.kwargs)
                )
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            
            self._stats["total_completed"] += 1
            self._update_avg_execution_time(task)
            
            # Move to completed tasks
            self._completed_tasks.append(task)
            
            logger.debug(f"Task completed: {task.id} in {task.completed_at - task.started_at:.3f}s")
            
        except Exception as e:
            task.error = e
            task.completed_at = time.time()
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                task.started_at = None
                
                # Re-queue for retry
                if self.enable_priorities:
                    self._insert_by_priority(task)
                else:
                    self._pending_queue.append(task)
                
                logger.warning(f"Task {task.id} failed, retrying ({task.retry_count}/{task.max_retries}): {e}")
            else:
                task.status = TaskStatus.FAILED
                self._stats["total_failed"] += 1
                logger.error(f"Task {task.id} failed permanently: {e}")
    
    def _update_avg_execution_time(self, task: Task) -> None:
        """Update average execution time statistics"""
        if task.started_at and task.completed_at:
            execution_time = task.completed_at - task.started_at
            current_avg = self._stats["avg_execution_time"]
            total_completed = self._stats["total_completed"]
            
            # Running average
            self._stats["avg_execution_time"] = (
                (current_avg * (total_completed - 1) + execution_time) / total_completed
            )
    
    async def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Wait for task completion and return result"""
        if task_id not in self._tasks:
            raise ValueError(f"Task not found: {task_id}")
        
        task = self._tasks[task_id]
        start_time = time.time()
        
        while task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
            if timeout and (time.time() - start_time) > timeout:
                raise asyncio.TimeoutError(f"Task {task_id} timeout after {timeout}s")
            
            await asyncio.sleep(0.1)
        
        if task.status == TaskStatus.COMPLETED:
            return task.result
        elif task.status == TaskStatus.FAILED:
            raise task.error
        else:
            raise RuntimeError(f"Task {task_id} in unexpected status: {task.status}")
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get current task status"""
        task = self._tasks.get(task_id)
        return task.status if task else None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            **self._stats,
            "current_running": len(self._running_tasks),
            "workers_active": len([w for w in self._workers if not w.done()]),
            "queue_utilization": self._stats["current_queue_size"] / self.max_queue_size,
            "completion_rate": (
                self._stats["total_completed"] / max(1, self._stats["total_submitted"])
            ),
            "failure_rate": (
                self._stats["total_failed"] / max(1, self._stats["total_submitted"])
            )
        }
    
    async def shutdown(self, timeout: float = 30.0) -> None:
        """Gracefully shutdown the task queue"""
        logger.info("Shutting down async task queue")
        self._shutdown = True
        
        # Wait for workers to finish
        if self._workers:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._workers, return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"Some workers didn't shutdown within {timeout}s")
                
                # Force cancel remaining workers
                for worker in self._workers:
                    if not worker.done():
                        worker.cancel()
        
        # Cancel any remaining running tasks
        for task in self._running_tasks.values():
            task.cancel()
        
        logger.info("Async task queue shutdown complete")


class DiscoveryTaskProcessor:
    """Specialized async processor for discovery operations"""
    
    def __init__(self, max_concurrent: int = 5):
        self.task_queue = AsyncTaskQueue(max_concurrent_tasks=max_concurrent)
        self._discovery_cache = {}
        self._cache_lock = asyncio.Lock()
    
    async def start(self) -> None:
        """Start the processor"""
        await self.task_queue.start_workers()
    
    async def process_discovery_batch(self, 
                                    data_batches: List[Any],
                                    discovery_params: Dict[str, Any]) -> List[Any]:
        """Process multiple discovery tasks concurrently"""
        from ..algorithms.discovery import DiscoveryEngine
        
        # Create discovery engine
        engine = DiscoveryEngine(**discovery_params)
        
        # Submit tasks for each batch
        task_ids = []
        for i, data_batch in enumerate(data_batches):
            task_id = f"discovery_batch_{i}_{time.time()}"
            await self.task_queue.submit_task(
                task_id,
                engine.discover,
                data_batch,
                context=f"batch_{i}",
                priority=1
            )
            task_ids.append(task_id)
        
        # Collect results
        results = []
        for task_id in task_ids:
            try:
                result = await self.task_queue.get_task_result(task_id, timeout=60.0)
                results.append(result)
            except Exception as e:
                logger.error(f"Discovery batch failed: {task_id} - {e}")
                results.append([])  # Empty result for failed batch
        
        return results
    
    async def shutdown(self) -> None:
        """Shutdown the processor"""
        await self.task_queue.shutdown()


# Global instance
_global_queue: Optional[AsyncTaskQueue] = None


async def get_async_queue(max_concurrent: int = 10) -> AsyncTaskQueue:
    """Get or create global async task queue"""
    global _global_queue
    if _global_queue is None:
        _global_queue = AsyncTaskQueue(max_concurrent_tasks=max_concurrent)
        await _global_queue.start_workers()
    return _global_queue


async def submit_async_task(task_id: str, 
                          func: Union[Callable, Coroutine],
                          *args,
                          priority: int = 0,
                          **kwargs) -> str:
    """Convenient function to submit task to global queue"""
    queue = await get_async_queue()
    return await queue.submit_task(task_id, func, *args, priority=priority, **kwargs)


async def get_async_result(task_id: str, timeout: Optional[float] = None) -> Any:
    """Get result from global queue"""
    queue = await get_async_queue()
    return await queue.get_task_result(task_id, timeout)


def async_task_decorator(priority: int = 0, 
                        max_retries: int = 3,
                        cache_results: bool = False):
    """Decorator to make functions async-task enabled"""
    
    def decorator(func: Callable) -> Callable:
        async def async_wrapper(*args, **kwargs):
            task_id = f"{func.__name__}_{hash((args, tuple(kwargs.items())))}"
            
            # Check cache if enabled
            if cache_results:
                # Simple in-memory cache (could be enhanced with Redis, etc.)
                if task_id in globals().get('_task_cache', {}):
                    return globals()['_task_cache'][task_id]
            
            queue = await get_async_queue()
            await queue.submit_task(
                task_id=task_id,
                func=func,
                *args,
                priority=priority,
                max_retries=max_retries,
                **kwargs
            )
            
            result = await queue.get_task_result(task_id)
            
            # Cache result if enabled
            if cache_results:
                if '_task_cache' not in globals():
                    globals()['_task_cache'] = {}
                globals()['_task_cache'][task_id] = result
            
            return result
        
        return async_wrapper
    return decorator