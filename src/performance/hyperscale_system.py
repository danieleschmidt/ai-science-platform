"""Hyperscale system for massive scientific computation"""

import numpy as np
import asyncio
import concurrent.futures
import multiprocessing as mp
import threading
import time
import logging
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from collections import deque, defaultdict
from abc import ABC, abstractmethod
import pickle
import queue
import psutil

from ..utils.error_handling import robust_execution, DiscoveryError
from ..utils.validation import ValidationMixin

logger = logging.getLogger(__name__)


@dataclass
class ComputationTask:
    """Single computation task for distributed processing"""
    task_id: str
    task_type: str
    data: Any
    parameters: Dict[str, Any]
    priority: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    estimated_duration: float = 1.0
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class ComputationResult:
    """Result from computation task"""
    task_id: str
    result: Any
    success: bool
    duration: float
    worker_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class WorkerNode:
    """Worker node in distributed system"""
    worker_id: str
    node_type: str  # 'cpu', 'gpu', 'memory_optimized'
    capabilities: List[str]
    max_concurrent_tasks: int
    current_load: float = 0.0
    total_completed: int = 0
    total_failed: int = 0
    last_heartbeat: datetime = field(default_factory=datetime.now)
    status: str = 'idle'  # 'idle', 'busy', 'error', 'offline'
    hardware_info: Dict[str, Any] = field(default_factory=dict)


class DistributedTaskQueue:
    """High-performance distributed task queue with priority scheduling"""
    
    def __init__(self, max_queue_size: int = 10000):
        """Initialize distributed task queue"""
        
        self.max_queue_size = max_queue_size
        
        # Multiple priority queues
        self.priority_queues = {
            'critical': asyncio.PriorityQueue(maxsize=1000),
            'high': asyncio.PriorityQueue(maxsize=2000),
            'normal': asyncio.PriorityQueue(maxsize=5000),
            'low': asyncio.PriorityQueue(maxsize=2000)
        }
        
        # Task tracking
        self.pending_tasks = {}
        self.running_tasks = {}
        self.completed_tasks = deque(maxlen=1000)
        self.failed_tasks = deque(maxlen=500)
        
        # Dependency management
        self.task_dependencies = {}
        self.dependency_graph = {}
        
        # Statistics
        self.stats = {
            'total_queued': 0,
            'total_completed': 0,
            'total_failed': 0,
            'avg_completion_time': 0.0,
            'throughput_per_second': 0.0
        }
        
        # Lock for thread safety
        self.lock = asyncio.Lock()
        
        logger.info("Distributed task queue initialized")
    
    async def submit_task(self, task: ComputationTask) -> bool:
        """Submit task to appropriate priority queue"""
        
        async with self.lock:
            # Check queue capacity
            total_queued = sum(q.qsize() for q in self.priority_queues.values())
            if total_queued >= self.max_queue_size:
                logger.warning("Task queue at capacity, rejecting new task")
                return False
            
            # Determine priority queue
            priority_level = self._determine_priority_level(task)
            
            # Handle dependencies
            if task.dependencies:
                await self._register_dependencies(task)
            
            # Add to appropriate queue
            try:
                priority_score = self._calculate_priority_score(task)
                await self.priority_queues[priority_level].put((priority_score, task))
                
                self.pending_tasks[task.task_id] = task
                self.stats['total_queued'] += 1
                
                logger.debug(f"Task {task.task_id} submitted to {priority_level} queue")
                return True
                
            except asyncio.QueueFull:
                logger.warning(f"Priority queue {priority_level} is full")
                return False
    
    async def get_next_task(self, worker_capabilities: List[str]) -> Optional[ComputationTask]:
        """Get next available task for worker with specific capabilities"""
        
        # Try queues in priority order
        for priority_level in ['critical', 'high', 'normal', 'low']:
            queue = self.priority_queues[priority_level]
            
            if not queue.empty():
                try:
                    priority_score, task = await asyncio.wait_for(queue.get(), timeout=0.1)
                    
                    # Check if worker can handle this task
                    if self._can_worker_handle_task(task, worker_capabilities):
                        # Check dependencies
                        if await self._are_dependencies_satisfied(task):
                            async with self.lock:
                                # Move to running tasks
                                if task.task_id in self.pending_tasks:
                                    del self.pending_tasks[task.task_id]
                                self.running_tasks[task.task_id] = task
                                
                            logger.debug(f"Assigned task {task.task_id} to worker with capabilities {worker_capabilities}")
                            return task
                        else:
                            # Dependencies not satisfied, put back in queue
                            await queue.put((priority_score, task))
                    else:
                        # Worker can't handle task, put back in queue
                        await queue.put((priority_score, task))
                        
                except asyncio.TimeoutError:
                    continue
        
        return None
    
    async def complete_task(self, result: ComputationResult):
        """Mark task as completed"""
        
        async with self.lock:
            task_id = result.task_id
            
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                del self.running_tasks[task_id]
                
                if result.success:
                    self.completed_tasks.append(result)
                    self.stats['total_completed'] += 1
                    
                    # Update average completion time
                    total_time = (self.stats['avg_completion_time'] * (self.stats['total_completed'] - 1) + 
                                result.duration) / self.stats['total_completed']
                    self.stats['avg_completion_time'] = total_time
                    
                    # Notify dependent tasks
                    await self._notify_dependent_tasks(task_id)
                    
                else:
                    self.failed_tasks.append(result)
                    self.stats['total_failed'] += 1
                    
                    # Retry logic for failed tasks
                    await self._handle_failed_task(task, result)
                
                logger.debug(f"Task {task_id} completed: success={result.success}")
    
    def _determine_priority_level(self, task: ComputationTask) -> str:
        """Determine which priority queue to use"""
        
        if task.priority >= 10:
            return 'critical'
        elif task.priority >= 7:
            return 'high'
        elif task.priority >= 3:
            return 'normal'
        else:
            return 'low'
    
    def _calculate_priority_score(self, task: ComputationTask) -> float:
        """Calculate priority score for ordering within queue"""
        
        base_score = task.priority * 1000
        
        # Age bonus (older tasks get higher priority)
        age_bonus = (datetime.now() - task.created_at).total_seconds() / 60  # Minutes
        
        # Duration penalty (shorter tasks get higher priority)
        duration_penalty = task.estimated_duration
        
        return base_score + age_bonus - duration_penalty
    
    def _can_worker_handle_task(self, task: ComputationTask, worker_capabilities: List[str]) -> bool:
        """Check if worker has required capabilities for task"""
        
        required_capabilities = task.metadata.get('required_capabilities', [])
        
        if not required_capabilities:
            return True  # Task has no special requirements
        
        return all(cap in worker_capabilities for cap in required_capabilities)
    
    async def _register_dependencies(self, task: ComputationTask):
        """Register task dependencies"""
        
        self.task_dependencies[task.task_id] = set(task.dependencies)
        
        # Build reverse dependency graph
        for dep_id in task.dependencies:
            if dep_id not in self.dependency_graph:
                self.dependency_graph[dep_id] = set()
            self.dependency_graph[dep_id].add(task.task_id)
    
    async def _are_dependencies_satisfied(self, task: ComputationTask) -> bool:
        """Check if all task dependencies are completed"""
        
        if task.task_id not in self.task_dependencies:
            return True
        
        remaining_deps = self.task_dependencies[task.task_id]
        return len(remaining_deps) == 0
    
    async def _notify_dependent_tasks(self, completed_task_id: str):
        """Notify tasks that depend on completed task"""
        
        if completed_task_id not in self.dependency_graph:
            return
        
        dependent_task_ids = self.dependency_graph[completed_task_id]
        
        for dep_task_id in dependent_task_ids:
            if dep_task_id in self.task_dependencies:
                self.task_dependencies[dep_task_id].discard(completed_task_id)
    
    async def _handle_failed_task(self, task: ComputationTask, result: ComputationResult):
        """Handle failed task with retry logic"""
        
        retry_count = task.metadata.get('retry_count', 0)
        max_retries = task.metadata.get('max_retries', 3)
        
        if retry_count < max_retries:
            # Create retry task
            retry_task = ComputationTask(
                task_id=f"{task.task_id}_retry_{retry_count + 1}",
                task_type=task.task_type,
                data=task.data,
                parameters=task.parameters,
                priority=max(1, task.priority - 1),  # Lower priority for retries
                estimated_duration=task.estimated_duration * 1.5,  # Expect longer duration
                dependencies=task.dependencies,
                metadata={**task.metadata, 'retry_count': retry_count + 1, 'original_task_id': task.task_id}
            )
            
            await self.submit_task(retry_task)
            logger.info(f"Requeued failed task {task.task_id} as {retry_task.task_id} (retry {retry_count + 1})")
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get comprehensive queue statistics"""
        
        queue_sizes = {level: queue.qsize() for level, queue in self.priority_queues.items()}
        
        return {
            **self.stats,
            'queue_sizes': queue_sizes,
            'pending_tasks': len(self.pending_tasks),
            'running_tasks': len(self.running_tasks),
            'total_queue_size': sum(queue_sizes.values()),
            'recent_failures': len(self.failed_tasks),
            'success_rate': (self.stats['total_completed'] / 
                           max(1, self.stats['total_completed'] + self.stats['total_failed'])) * 100
        }


class AdaptiveWorkerPool:
    """Adaptive worker pool that scales based on demand"""
    
    def __init__(self, 
                 min_workers: int = 2,
                 max_workers: int = None,
                 scale_up_threshold: float = 0.8,
                 scale_down_threshold: float = 0.3):
        """Initialize adaptive worker pool"""
        
        self.min_workers = min_workers
        self.max_workers = max_workers or min(mp.cpu_count() * 2, 32)
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        
        # Worker management
        self.workers = {}
        self.worker_stats = {}
        self.executor = None
        
        # Load balancing
        self.task_assignments = defaultdict(int)
        self.worker_loads = defaultdict(float)
        
        # Auto-scaling metrics
        self.scaling_metrics = {
            'cpu_usage': deque(maxlen=60),  # 1 minute of samples
            'queue_length': deque(maxlen=60),
            'task_completion_rate': deque(maxlen=60)
        }
        
        self.last_scale_decision = datetime.now()
        self.scaling_cooldown = 60  # Seconds between scaling decisions
        
        logger.info(f"Adaptive worker pool initialized: {min_workers}-{self.max_workers} workers")
    
    async def initialize(self, task_queue: DistributedTaskQueue):
        """Initialize worker pool"""
        
        self.task_queue = task_queue
        
        # Create initial worker pool
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers)
        
        # Start with minimum workers
        for i in range(self.min_workers):
            await self._create_worker(f"worker_{i}")
        
        # Start monitoring and scaling task
        asyncio.create_task(self._monitor_and_scale())
        
        logger.info(f"Worker pool initialized with {self.min_workers} workers")
    
    async def _create_worker(self, worker_id: str) -> WorkerNode:
        """Create new worker node"""
        
        # Detect worker capabilities based on system
        capabilities = self._detect_worker_capabilities()
        
        worker = WorkerNode(
            worker_id=worker_id,
            node_type=self._determine_node_type(),
            capabilities=capabilities,
            max_concurrent_tasks=self._calculate_max_concurrent_tasks(),
            hardware_info=self._get_hardware_info()
        )
        
        self.workers[worker_id] = worker
        self.worker_stats[worker_id] = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_duration': 0.0,
            'avg_task_time': 0.0,
            'cpu_time': 0.0,
            'memory_peak': 0.0
        }
        
        # Start worker task
        asyncio.create_task(self._worker_loop(worker))
        
        logger.info(f"Created worker {worker_id} with capabilities {capabilities}")
        return worker
    
    async def _worker_loop(self, worker: WorkerNode):
        """Main worker loop"""
        
        logger.info(f"Worker {worker.worker_id} started")
        
        while True:
            try:
                # Get next task
                task = await self.task_queue.get_next_task(worker.capabilities)
                
                if task is None:
                    # No tasks available, wait a bit
                    await asyncio.sleep(1.0)
                    worker.status = 'idle'
                    continue
                
                # Update worker status
                worker.status = 'busy'
                worker.current_load = min(1.0, worker.current_load + (1.0 / worker.max_concurrent_tasks))
                
                # Process task
                result = await self._process_task(worker, task)
                
                # Complete task
                await self.task_queue.complete_task(result)
                
                # Update worker statistics
                await self._update_worker_stats(worker, result)
                
                # Update load
                worker.current_load = max(0.0, worker.current_load - (1.0 / worker.max_concurrent_tasks))
                
            except Exception as e:
                logger.error(f"Worker {worker.worker_id} error: {e}")
                worker.status = 'error'
                await asyncio.sleep(5.0)  # Error recovery delay
                worker.status = 'idle'
    
    async def _process_task(self, worker: WorkerNode, task: ComputationTask) -> ComputationResult:
        """Process a single task"""
        
        start_time = time.time()
        
        try:
            # Execute task in process pool
            future = self.executor.submit(self._execute_task_function, task)
            result_data = await asyncio.get_event_loop().run_in_executor(None, future.result, 30.0)
            
            duration = time.time() - start_time
            
            return ComputationResult(
                task_id=task.task_id,
                result=result_data,
                success=True,
                duration=duration,
                worker_id=worker.worker_id,
                metrics={
                    'cpu_time': duration,
                    'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024  # MB
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            
            return ComputationResult(
                task_id=task.task_id,
                result=None,
                success=False,
                duration=duration,
                worker_id=worker.worker_id,
                error_message=str(e)
            )
    
    def _execute_task_function(self, task: ComputationTask) -> Any:
        """Execute task function (runs in separate process)"""
        
        # Task type dispatch
        if task.task_type == 'matrix_computation':
            return self._matrix_computation_task(task.data, task.parameters)
        elif task.task_type == 'data_analysis':
            return self._data_analysis_task(task.data, task.parameters)
        elif task.task_type == 'ml_training':
            return self._ml_training_task(task.data, task.parameters)
        elif task.task_type == 'scientific_simulation':
            return self._scientific_simulation_task(task.data, task.parameters)
        else:
            raise ValueError(f"Unknown task type: {task.task_type}")
    
    def _matrix_computation_task(self, data: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute matrix computation task"""
        
        operation = params.get('operation', 'multiply')
        
        if operation == 'multiply':
            # Matrix multiplication
            if data.ndim == 2:
                result = np.dot(data, data.T)
            else:
                result = data * 2
        elif operation == 'eigenvalue':
            # Eigenvalue decomposition
            if data.ndim == 2 and data.shape[0] == data.shape[1]:
                eigenvals, eigenvecs = np.linalg.eig(data)
                result = {'eigenvalues': eigenvals, 'eigenvectors': eigenvecs}
            else:
                result = {'error': 'Matrix must be square for eigenvalue decomposition'}
        elif operation == 'svd':
            # Singular Value Decomposition
            u, s, vt = np.linalg.svd(data)
            result = {'u': u, 's': s, 'vt': vt}
        else:
            result = {'error': f'Unknown matrix operation: {operation}'}
        
        return {
            'operation': operation,
            'input_shape': data.shape,
            'result': result,
            'computation_time': time.time()
        }
    
    def _data_analysis_task(self, data: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data analysis task"""
        
        analysis_type = params.get('analysis_type', 'descriptive')
        
        if analysis_type == 'descriptive':
            result = {
                'mean': np.mean(data),
                'std': np.std(data),
                'min': np.min(data),
                'max': np.max(data),
                'median': np.median(data),
                'percentiles': {
                    '25': np.percentile(data, 25),
                    '75': np.percentile(data, 75),
                    '90': np.percentile(data, 90),
                    '95': np.percentile(data, 95)
                }
            }
        elif analysis_type == 'correlation':
            if data.ndim == 2:
                result = {'correlation_matrix': np.corrcoef(data, rowvar=False)}
            else:
                result = {'error': 'Correlation requires 2D data'}
        elif analysis_type == 'clustering':
            # Simple k-means clustering
            from sklearn.cluster import KMeans
            n_clusters = params.get('n_clusters', 3)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data.reshape(-1, 1) if data.ndim == 1 else data)
            result = {
                'labels': labels,
                'centers': kmeans.cluster_centers_,
                'inertia': kmeans.inertia_
            }
        else:
            result = {'error': f'Unknown analysis type: {analysis_type}'}
        
        return {
            'analysis_type': analysis_type,
            'data_shape': data.shape,
            'result': result
        }
    
    def _ml_training_task(self, data: Dict[str, np.ndarray], params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute machine learning training task"""
        
        model_type = params.get('model_type', 'linear_regression')
        X = data.get('features')
        y = data.get('targets')
        
        if X is None or y is None:
            return {'error': 'Features and targets required for ML training'}
        
        if model_type == 'linear_regression':
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, r2_score
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            result = {
                'model_type': 'LinearRegression',
                'coefficients': model.coef_.tolist(),
                'intercept': float(model.intercept_),
                'r2_score': float(r2_score(y_test, y_pred)),
                'mse': float(mean_squared_error(y_test, y_pred)),
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
        elif model_type == 'neural_network':
            # Simple neural network with numpy
            result = self._simple_neural_network(X, y, params)
        else:
            result = {'error': f'Unknown model type: {model_type}'}
        
        return {
            'task_type': 'ml_training',
            'model_type': model_type,
            'data_shape': X.shape,
            'result': result
        }
    
    def _simple_neural_network(self, X: np.ndarray, y: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simple neural network implementation"""
        
        # Network parameters
        hidden_size = params.get('hidden_size', 10)
        learning_rate = params.get('learning_rate', 0.01)
        epochs = params.get('epochs', 100)
        
        # Initialize weights
        input_size = X.shape[1]
        output_size = 1 if y.ndim == 1 else y.shape[1]
        
        W1 = np.random.randn(input_size, hidden_size) * 0.01
        b1 = np.zeros((1, hidden_size))
        W2 = np.random.randn(hidden_size, output_size) * 0.01
        b2 = np.zeros((1, output_size))
        
        # Training loop
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            z1 = X @ W1 + b1
            a1 = np.maximum(0, z1)  # ReLU
            z2 = a1 @ W2 + b2
            a2 = z2  # Linear output
            
            # Loss (MSE)
            if y.ndim == 1:
                y_reshaped = y.reshape(-1, 1)
            else:
                y_reshaped = y
            
            loss = np.mean((a2 - y_reshaped) ** 2)
            losses.append(loss)
            
            # Backward pass
            dz2 = 2 * (a2 - y_reshaped) / len(X)
            dW2 = a1.T @ dz2
            db2 = np.sum(dz2, axis=0, keepdims=True)
            
            da1 = dz2 @ W2.T
            dz1 = da1 * (z1 > 0)  # ReLU derivative
            dW1 = X.T @ dz1
            db1 = np.sum(dz1, axis=0, keepdims=True)
            
            # Update weights
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2
        
        # Final predictions
        z1 = X @ W1 + b1
        a1 = np.maximum(0, z1)
        predictions = a1 @ W2 + b2
        
        final_loss = np.mean((predictions - y_reshaped) ** 2)
        
        return {
            'final_loss': float(final_loss),
            'training_losses': [float(l) for l in losses[-10:]],  # Last 10 losses
            'epochs': epochs,
            'hidden_size': hidden_size,
            'learning_rate': learning_rate
        }
    
    def _scientific_simulation_task(self, data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute scientific simulation task"""
        
        simulation_type = params.get('simulation_type', 'molecular_dynamics')
        
        if simulation_type == 'molecular_dynamics':
            return self._molecular_dynamics_simulation(data, params)
        elif simulation_type == 'monte_carlo':
            return self._monte_carlo_simulation(data, params)
        elif simulation_type == 'differential_equation':
            return self._differential_equation_simulation(data, params)
        else:
            return {'error': f'Unknown simulation type: {simulation_type}'}
    
    def _molecular_dynamics_simulation(self, data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Simplified molecular dynamics simulation"""
        
        n_particles = params.get('n_particles', 100)
        n_steps = params.get('n_steps', 1000)
        dt = params.get('dt', 0.001)
        temperature = params.get('temperature', 300.0)
        
        # Initialize random positions and velocities
        np.random.seed(42)
        positions = np.random.randn(n_particles, 3)
        velocities = np.random.randn(n_particles, 3) * np.sqrt(temperature / 100)
        
        # Simple integration loop
        kinetic_energies = []
        
        for step in range(n_steps):
            # Simple harmonic potential forces
            forces = -0.1 * positions
            
            # Velocity Verlet integration
            velocities += 0.5 * forces * dt
            positions += velocities * dt
            forces = -0.1 * positions  # Recalculate forces
            velocities += 0.5 * forces * dt
            
            # Calculate kinetic energy
            ke = 0.5 * np.sum(velocities**2)
            kinetic_energies.append(ke)
        
        return {
            'simulation_type': 'molecular_dynamics',
            'n_particles': n_particles,
            'n_steps': n_steps,
            'final_kinetic_energy': float(kinetic_energies[-1]),
            'avg_kinetic_energy': float(np.mean(kinetic_energies)),
            'energy_fluctuation': float(np.std(kinetic_energies))
        }
    
    def _monte_carlo_simulation(self, data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Monte Carlo simulation"""
        
        n_samples = params.get('n_samples', 10000)
        simulation_type = params.get('monte_carlo_type', 'pi_estimation')
        
        if simulation_type == 'pi_estimation':
            # Estimate π using Monte Carlo
            np.random.seed(42)
            x = np.random.uniform(-1, 1, n_samples)
            y = np.random.uniform(-1, 1, n_samples)
            
            inside_circle = np.sum(x**2 + y**2 <= 1)
            pi_estimate = 4 * inside_circle / n_samples
            
            return {
                'monte_carlo_type': 'pi_estimation',
                'n_samples': n_samples,
                'pi_estimate': float(pi_estimate),
                'error': float(abs(pi_estimate - np.pi)),
                'inside_circle': int(inside_circle)
            }
        
        elif simulation_type == 'integration':
            # Numerical integration using Monte Carlo
            def f(x):
                return x**2 * np.sin(x)
            
            np.random.seed(42)
            x_samples = np.random.uniform(0, np.pi, n_samples)
            y_samples = f(x_samples)
            
            integral_estimate = np.pi * np.mean(y_samples)
            
            return {
                'monte_carlo_type': 'integration',
                'n_samples': n_samples,
                'integral_estimate': float(integral_estimate),
                'function': 'x^2 * sin(x)',
                'integration_bounds': [0, 'π']
            }
        
        else:
            return {'error': f'Unknown Monte Carlo type: {simulation_type}'}
    
    def _differential_equation_simulation(self, data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Differential equation simulation"""
        
        eq_type = params.get('equation_type', 'lorenz')
        n_steps = params.get('n_steps', 1000)
        dt = params.get('dt', 0.01)
        
        if eq_type == 'lorenz':
            # Lorenz attractor
            sigma = params.get('sigma', 10.0)
            rho = params.get('rho', 28.0)
            beta = params.get('beta', 8.0/3.0)
            
            # Initial conditions
            x, y, z = 1.0, 1.0, 1.0
            trajectory = []
            
            for _ in range(n_steps):
                dx_dt = sigma * (y - x)
                dy_dt = x * (rho - z) - y
                dz_dt = x * y - beta * z
                
                x += dx_dt * dt
                y += dy_dt * dt
                z += dz_dt * dt
                
                trajectory.append([x, y, z])
            
            trajectory = np.array(trajectory)
            
            return {
                'equation_type': 'lorenz',
                'parameters': {'sigma': sigma, 'rho': rho, 'beta': beta},
                'n_steps': n_steps,
                'final_position': trajectory[-1].tolist(),
                'trajectory_std': np.std(trajectory, axis=0).tolist(),
                'trajectory_range': {
                    'x': [float(np.min(trajectory[:, 0])), float(np.max(trajectory[:, 0]))],
                    'y': [float(np.min(trajectory[:, 1])), float(np.max(trajectory[:, 1]))],
                    'z': [float(np.min(trajectory[:, 2])), float(np.max(trajectory[:, 2]))]
                }
            }
        
        else:
            return {'error': f'Unknown differential equation type: {eq_type}'}
    
    def _detect_worker_capabilities(self) -> List[str]:
        """Detect worker capabilities based on system"""
        
        capabilities = ['cpu_compute']
        
        # Check for GPU
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                capabilities.append('gpu_compute')
        except ImportError:
            pass
        
        # Check memory
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb > 16:
            capabilities.append('high_memory')
        
        # Check CPU cores
        if mp.cpu_count() > 8:
            capabilities.append('parallel_compute')
        
        capabilities.extend(['matrix_ops', 'data_analysis', 'ml_training', 'simulation'])
        
        return capabilities
    
    def _determine_node_type(self) -> str:
        """Determine node type based on hardware"""
        
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = mp.cpu_count()
        
        if memory_gb > 32:
            return 'memory_optimized'
        elif cpu_count > 16:
            return 'cpu_optimized'
        else:
            return 'general_purpose'
    
    def _calculate_max_concurrent_tasks(self) -> int:
        """Calculate maximum concurrent tasks based on hardware"""
        
        cpu_count = mp.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Conservative estimation
        cpu_based = cpu_count
        memory_based = int(memory_gb // 2)  # 2GB per task
        
        return min(cpu_based, memory_based, 8)  # Cap at 8 concurrent tasks
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information"""
        
        return {
            'cpu_count': mp.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'disk_gb': psutil.disk_usage('/').total / (1024**3),
            'platform': psutil.WINDOWS if hasattr(psutil, 'WINDOWS') else 'unix'
        }
    
    async def _update_worker_stats(self, worker: WorkerNode, result: ComputationResult):
        """Update worker statistics"""
        
        stats = self.worker_stats[worker.worker_id]
        
        if result.success:
            stats['tasks_completed'] += 1
            worker.total_completed += 1
        else:
            stats['tasks_failed'] += 1
            worker.total_failed += 1
        
        stats['total_duration'] += result.duration
        
        # Update averages
        total_tasks = stats['tasks_completed'] + stats['tasks_failed']
        if total_tasks > 0:
            stats['avg_task_time'] = stats['total_duration'] / total_tasks
        
        # Update metrics from result
        if result.metrics:
            stats['cpu_time'] += result.metrics.get('cpu_time', 0)
            stats['memory_peak'] = max(stats['memory_peak'], result.metrics.get('memory_usage', 0))
        
        worker.last_heartbeat = datetime.now()
    
    async def _monitor_and_scale(self):
        """Monitor system and make scaling decisions"""
        
        while True:
            try:
                # Collect metrics
                current_cpu = psutil.cpu_percent(interval=1)
                queue_stats = self.task_queue.get_queue_stats()
                current_queue_length = queue_stats['total_queue_size']
                
                # Update metrics
                self.scaling_metrics['cpu_usage'].append(current_cpu)
                self.scaling_metrics['queue_length'].append(current_queue_length)
                
                # Calculate completion rate
                total_completed = sum(stats['tasks_completed'] for stats in self.worker_stats.values())
                self.scaling_metrics['task_completion_rate'].append(total_completed)
                
                # Make scaling decision
                await self._make_scaling_decision()
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring and scaling: {e}")
                await asyncio.sleep(30)
    
    async def _make_scaling_decision(self):
        """Make scaling decision based on metrics"""
        
        now = datetime.now()
        
        # Check cooldown
        if (now - self.last_scale_decision).total_seconds() < self.scaling_cooldown:
            return
        
        current_workers = len([w for w in self.workers.values() if w.status != 'offline'])
        
        # Get recent metrics
        if len(self.scaling_metrics['cpu_usage']) < 5:
            return  # Not enough data
        
        avg_cpu = np.mean(list(self.scaling_metrics['cpu_usage'])[-5:])
        avg_queue_length = np.mean(list(self.scaling_metrics['queue_length'])[-5:])
        
        # Calculate load indicators
        cpu_load = avg_cpu / 100.0
        queue_load = min(1.0, avg_queue_length / 100.0)  # Normalize queue length
        
        # Combined load score
        combined_load = (cpu_load + queue_load) / 2.0
        
        # Scaling decisions
        if combined_load > self.scale_up_threshold and current_workers < self.max_workers:
            await self._scale_up()
        elif combined_load < self.scale_down_threshold and current_workers > self.min_workers:
            await self._scale_down()
    
    async def _scale_up(self):
        """Scale up worker pool"""
        
        new_worker_id = f"worker_{len(self.workers)}"
        await self._create_worker(new_worker_id)
        
        self.last_scale_decision = datetime.now()
        logger.info(f"Scaled up: created worker {new_worker_id}")
    
    async def _scale_down(self):
        """Scale down worker pool"""
        
        # Find least active worker
        least_active_worker = None
        min_load = float('inf')
        
        for worker in self.workers.values():
            if worker.status == 'idle' and worker.current_load < min_load:
                min_load = worker.current_load
                least_active_worker = worker
        
        if least_active_worker:
            least_active_worker.status = 'offline'
            self.last_scale_decision = datetime.now()
            logger.info(f"Scaled down: deactivated worker {least_active_worker.worker_id}")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get comprehensive worker pool statistics"""
        
        active_workers = [w for w in self.workers.values() if w.status != 'offline']
        
        total_completed = sum(w.total_completed for w in active_workers)
        total_failed = sum(w.total_failed for w in active_workers)
        avg_load = np.mean([w.current_load for w in active_workers]) if active_workers else 0.0
        
        return {
            'total_workers': len(self.workers),
            'active_workers': len(active_workers),
            'worker_statuses': {w.worker_id: w.status for w in self.workers.values()},
            'total_tasks_completed': total_completed,
            'total_tasks_failed': total_failed,
            'average_worker_load': float(avg_load),
            'success_rate': (total_completed / max(1, total_completed + total_failed)) * 100,
            'scaling_metrics': {
                'current_cpu_usage': list(self.scaling_metrics['cpu_usage'])[-1] if self.scaling_metrics['cpu_usage'] else 0,
                'current_queue_length': list(self.scaling_metrics['queue_length'])[-1] if self.scaling_metrics['queue_length'] else 0
            }
        }
    
    async def shutdown(self):
        """Shutdown worker pool gracefully"""
        
        logger.info("Shutting down worker pool...")
        
        # Mark all workers as offline
        for worker in self.workers.values():
            worker.status = 'offline'
        
        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=True)
        
        logger.info("Worker pool shutdown complete")


class HyperscaleComputeEngine(ValidationMixin):
    """Main hyperscale compute engine orchestrating distributed computation"""
    
    def __init__(self, 
                 min_workers: int = 4,
                 max_workers: int = 32,
                 queue_size: int = 10000):
        """Initialize hyperscale compute engine"""
        
        self.task_queue = DistributedTaskQueue(max_queue_size=queue_size)
        self.worker_pool = AdaptiveWorkerPool(
            min_workers=min_workers,
            max_workers=max_workers
        )
        
        # Performance monitoring
        self.performance_metrics = {
            'tasks_per_second': deque(maxlen=300),  # 5 minutes at 1 second intervals
            'average_task_duration': deque(maxlen=100),
            'system_utilization': deque(maxlen=300)
        }
        
        self.start_time = datetime.now()
        self.is_running = False
        
        logger.info("Hyperscale compute engine initialized")
    
    async def initialize(self):
        """Initialize the compute engine"""
        
        await self.worker_pool.initialize(self.task_queue)
        
        # Start performance monitoring
        asyncio.create_task(self._performance_monitor())
        
        self.is_running = True
        logger.info("Hyperscale compute engine started")
    
    async def submit_computation(self, 
                               task_type: str,
                               data: Any,
                               parameters: Dict[str, Any] = None,
                               priority: int = 5) -> str:
        """Submit computation task"""
        
        task_id = f"{task_type}_{int(time.time()*1000000)}"
        
        task = ComputationTask(
            task_id=task_id,
            task_type=task_type,
            data=data,
            parameters=parameters or {},
            priority=priority,
            estimated_duration=self._estimate_task_duration(task_type, data)
        )
        
        success = await self.task_queue.submit_task(task)
        
        if success:
            logger.info(f"Submitted computation task {task_id} (type: {task_type})")
            return task_id
        else:
            raise DiscoveryError(f"Failed to submit task {task_id}: queue full")
    
    async def submit_batch_computation(self, 
                                     tasks: List[Dict[str, Any]],
                                     batch_priority: int = 5) -> List[str]:
        """Submit batch of computation tasks"""
        
        task_ids = []
        
        for i, task_spec in enumerate(tasks):
            task_id = await self.submit_computation(
                task_type=task_spec['task_type'],
                data=task_spec['data'],
                parameters=task_spec.get('parameters', {}),
                priority=batch_priority - (i // 10)  # Slight priority decrease for later tasks
            )
            task_ids.append(task_id)
        
        logger.info(f"Submitted batch of {len(task_ids)} computation tasks")
        return task_ids
    
    async def submit_workflow(self, 
                            workflow_spec: Dict[str, Any]) -> Dict[str, List[str]]:
        """Submit complex workflow with dependencies"""
        
        workflow_id = workflow_spec.get('workflow_id', f"workflow_{int(time.time())}")
        stages = workflow_spec.get('stages', [])
        
        stage_task_ids = {}
        
        for stage in stages:
            stage_name = stage['name']
            stage_tasks = stage['tasks']
            stage_dependencies = stage.get('dependencies', [])
            
            # Convert stage dependencies to task dependencies
            task_dependencies = []
            for dep_stage in stage_dependencies:
                if dep_stage in stage_task_ids:
                    task_dependencies.extend(stage_task_ids[dep_stage])
            
            # Submit tasks for this stage
            stage_ids = []
            for task_spec in stage_tasks:
                task_id = f"{workflow_id}_{stage_name}_{len(stage_ids)}"
                
                task = ComputationTask(
                    task_id=task_id,
                    task_type=task_spec['task_type'],
                    data=task_spec['data'],
                    parameters=task_spec.get('parameters', {}),
                    priority=task_spec.get('priority', 5),
                    dependencies=task_dependencies,
                    metadata={'workflow_id': workflow_id, 'stage': stage_name}
                )
                
                success = await self.task_queue.submit_task(task)
                if success:
                    stage_ids.append(task_id)
            
            stage_task_ids[stage_name] = stage_ids
        
        logger.info(f"Submitted workflow {workflow_id} with {sum(len(ids) for ids in stage_task_ids.values())} tasks")
        return stage_task_ids
    
    def _estimate_task_duration(self, task_type: str, data: Any) -> float:
        """Estimate task duration based on type and data size"""
        
        base_durations = {
            'matrix_computation': 0.1,
            'data_analysis': 0.5,
            'ml_training': 5.0,
            'scientific_simulation': 2.0
        }
        
        base_duration = base_durations.get(task_type, 1.0)
        
        # Adjust based on data size
        if hasattr(data, 'size'):
            size_factor = np.log10(max(1, data.size))
            return base_duration * size_factor
        elif isinstance(data, dict):
            # For complex data structures
            return base_duration * 2.0
        else:
            return base_duration
    
    async def _performance_monitor(self):
        """Monitor system performance"""
        
        last_completed = 0
        
        while self.is_running:
            try:
                # Get current statistics
                queue_stats = self.task_queue.get_queue_stats()
                pool_stats = self.worker_pool.get_pool_stats()
                
                # Calculate tasks per second
                current_completed = queue_stats['total_completed']
                tasks_per_second = current_completed - last_completed
                self.performance_metrics['tasks_per_second'].append(tasks_per_second)
                last_completed = current_completed
                
                # Track average duration
                if queue_stats['avg_completion_time'] > 0:
                    self.performance_metrics['average_task_duration'].append(
                        queue_stats['avg_completion_time']
                    )
                
                # System utilization
                cpu_usage = pool_stats['scaling_metrics']['current_cpu_usage']
                self.performance_metrics['system_utilization'].append(cpu_usage)
                
                await asyncio.sleep(1)  # Monitor every second
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(5)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        queue_stats = self.task_queue.get_queue_stats()
        pool_stats = self.worker_pool.get_pool_stats()
        
        # Performance metrics
        recent_tps = list(self.performance_metrics['tasks_per_second'])[-10:]
        avg_tps = np.mean(recent_tps) if recent_tps else 0.0
        
        recent_duration = list(self.performance_metrics['average_task_duration'])[-10:]
        avg_duration = np.mean(recent_duration) if recent_duration else 0.0
        
        recent_utilization = list(self.performance_metrics['system_utilization'])[-10:]
        avg_utilization = np.mean(recent_utilization) if recent_utilization else 0.0
        
        return {
            'system_info': {
                'uptime_seconds': uptime,
                'status': 'running' if self.is_running else 'stopped',
                'start_time': self.start_time.isoformat()
            },
            'queue_status': queue_stats,
            'worker_status': pool_stats,
            'performance': {
                'tasks_per_second': float(avg_tps),
                'average_task_duration': float(avg_duration),
                'system_utilization': float(avg_utilization)
            },
            'capacity': {
                'max_workers': self.worker_pool.max_workers,
                'max_queue_size': self.task_queue.max_queue_size,
                'theoretical_max_tps': self.worker_pool.max_workers * (1.0 / max(0.1, avg_duration))
            }
        }
    
    async def run_benchmark(self, 
                          benchmark_type: str = 'mixed',
                          duration_seconds: int = 60) -> Dict[str, Any]:
        """Run system benchmark"""
        
        logger.info(f"Starting {duration_seconds}s {benchmark_type} benchmark")
        
        benchmark_start = time.time()
        submitted_tasks = []
        
        while (time.time() - benchmark_start) < duration_seconds:
            if benchmark_type == 'cpu_intensive':
                # CPU-intensive matrix operations
                data = np.random.randn(100, 100)
                task_id = await self.submit_computation(
                    'matrix_computation',
                    data,
                    {'operation': 'eigenvalue'},
                    priority=7
                )
                
            elif benchmark_type == 'memory_intensive':
                # Memory-intensive data analysis
                data = np.random.randn(10000, 50)
                task_id = await self.submit_computation(
                    'data_analysis',
                    data,
                    {'analysis_type': 'correlation'},
                    priority=6
                )
                
            elif benchmark_type == 'mixed':
                # Mix of different task types
                task_types = [
                    ('matrix_computation', np.random.randn(50, 50), {'operation': 'svd'}),
                    ('data_analysis', np.random.randn(1000, 10), {'analysis_type': 'descriptive'}),
                    ('ml_training', {
                        'features': np.random.randn(500, 5),
                        'targets': np.random.randn(500)
                    }, {'model_type': 'linear_regression'}),
                    ('scientific_simulation', {}, {'simulation_type': 'monte_carlo', 'n_samples': 1000})
                ]
                
                task_type, data, params = task_types[len(submitted_tasks) % len(task_types)]
                task_id = await self.submit_computation(task_type, data, params, priority=5)
            
            else:
                raise ValueError(f"Unknown benchmark type: {benchmark_type}")
            
            submitted_tasks.append(task_id)
            
            # Small delay to prevent overwhelming
            await asyncio.sleep(0.1)
        
        # Wait for all tasks to complete or timeout
        wait_start = time.time()
        max_wait = 120  # 2 minutes max wait
        
        while (time.time() - wait_start) < max_wait:
            queue_stats = self.task_queue.get_queue_stats()
            if queue_stats['running_tasks'] == 0 and queue_stats['total_queue_size'] == 0:
                break
            await asyncio.sleep(1)
        
        total_duration = time.time() - benchmark_start
        final_stats = self.get_system_status()
        
        return {
            'benchmark_type': benchmark_type,
            'duration_seconds': total_duration,
            'tasks_submitted': len(submitted_tasks),
            'final_system_status': final_stats,
            'benchmark_results': {
                'tasks_per_second': len(submitted_tasks) / duration_seconds,
                'average_completion_time': final_stats['performance']['average_task_duration'],
                'system_utilization': final_stats['performance']['system_utilization'],
                'success_rate': final_stats['queue_status']['success_rate']
            }
        }
    
    async def shutdown(self):
        """Shutdown compute engine gracefully"""
        
        logger.info("Shutting down hyperscale compute engine...")
        
        self.is_running = False
        
        # Wait for current tasks to complete (with timeout)
        shutdown_start = time.time()
        max_shutdown_time = 30  # 30 seconds max
        
        while (time.time() - shutdown_start) < max_shutdown_time:
            queue_stats = self.task_queue.get_queue_stats()
            if queue_stats['running_tasks'] == 0:
                break
            await asyncio.sleep(1)
        
        # Shutdown components
        await self.worker_pool.shutdown()
        
        logger.info("Hyperscale compute engine shutdown complete")


# Factory function for easy instantiation
def create_hyperscale_system(min_workers: int = 4, 
                           max_workers: int = 32,
                           queue_size: int = 10000) -> HyperscaleComputeEngine:
    """Create and return configured hyperscale compute system"""
    
    return HyperscaleComputeEngine(
        min_workers=min_workers,
        max_workers=max_workers,
        queue_size=queue_size
    )