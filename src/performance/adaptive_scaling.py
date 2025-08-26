"""Adaptive auto-scaling system for AI Science Platform"""

import time
import threading
import logging
import psutil
from typing import Dict, Any, List, Callable, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import queue
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class ScalingMetrics:
    """Metrics for scaling decisions"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    queue_length: int
    active_workers: int
    throughput_per_second: float
    response_time_ms: float
    error_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'queue_length': self.queue_length,
            'active_workers': self.active_workers,
            'throughput_per_second': self.throughput_per_second,
            'response_time_ms': self.response_time_ms,
            'error_rate': self.error_rate
        }


@dataclass
class ScalingPolicy:
    """Scaling policy configuration"""
    name: str
    min_workers: int = 1
    max_workers: int = 16
    scale_up_threshold: Dict[str, float] = field(default_factory=lambda: {
        'cpu_percent': 70.0,
        'memory_percent': 80.0,
        'queue_length': 10,
        'response_time_ms': 1000.0
    })
    scale_down_threshold: Dict[str, float] = field(default_factory=lambda: {
        'cpu_percent': 30.0,
        'memory_percent': 50.0,
        'queue_length': 2,
        'response_time_ms': 200.0
    })
    scale_up_cooldown: int = 60  # seconds
    scale_down_cooldown: int = 300  # seconds
    scale_step: int = 1


class ScalingDecisionEngine:
    """Makes intelligent scaling decisions based on metrics"""
    
    def __init__(self, policy: ScalingPolicy):
        self.policy = policy
        self.metrics_history: List[ScalingMetrics] = []
        self.last_scale_up = 0
        self.last_scale_down = 0
        
        logger.info(f"ScalingDecisionEngine initialized with policy: {policy.name}")
    
    def should_scale_up(self, metrics: ScalingMetrics) -> bool:
        """Determine if we should scale up"""
        now = time.time()
        
        # Check cooldown
        if now - self.last_scale_up < self.policy.scale_up_cooldown:
            return False
        
        # Check if we're at max capacity
        if metrics.active_workers >= self.policy.max_workers:
            return False
        
        # Check thresholds
        thresholds = self.policy.scale_up_threshold
        
        reasons = []
        if metrics.cpu_percent > thresholds.get('cpu_percent', 100):
            reasons.append(f"CPU: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > thresholds.get('memory_percent', 100):
            reasons.append(f"Memory: {metrics.memory_percent:.1f}%")
        
        if metrics.queue_length > thresholds.get('queue_length', float('inf')):
            reasons.append(f"Queue: {metrics.queue_length}")
        
        if metrics.response_time_ms > thresholds.get('response_time_ms', float('inf')):
            reasons.append(f"Response: {metrics.response_time_ms:.1f}ms")
        
        # Require multiple indicators for scaling up
        if len(reasons) >= 2:
            logger.info(f"Scale up triggered: {', '.join(reasons)}")
            return True
        
        return False
    
    def should_scale_down(self, metrics: ScalingMetrics) -> bool:
        """Determine if we should scale down"""
        now = time.time()
        
        # Check cooldown
        if now - self.last_scale_down < self.policy.scale_down_cooldown:
            return False
        
        # Check if we're at minimum capacity
        if metrics.active_workers <= self.policy.min_workers:
            return False
        
        # Check thresholds - all must be below threshold for scale down
        thresholds = self.policy.scale_down_threshold
        
        conditions = [
            metrics.cpu_percent < thresholds.get('cpu_percent', 0),
            metrics.memory_percent < thresholds.get('memory_percent', 0),
            metrics.queue_length < thresholds.get('queue_length', 0),
            metrics.response_time_ms < thresholds.get('response_time_ms', float('inf'))
        ]
        
        # Also check recent trend
        if len(self.metrics_history) >= 5:
            recent_metrics = self.metrics_history[-5:]
            avg_cpu = np.mean([m.cpu_percent for m in recent_metrics])
            avg_queue = np.mean([m.queue_length for m in recent_metrics])
            
            trend_stable = (
                avg_cpu < thresholds.get('cpu_percent', 0) and
                avg_queue < thresholds.get('queue_length', 0)
            )
            conditions.append(trend_stable)
        
        if all(conditions):
            logger.info("Scale down triggered: all metrics below threshold")
            return True
        
        return False
    
    def record_scaling_event(self, action: str):
        """Record scaling event"""
        now = time.time()
        if action == 'scale_up':
            self.last_scale_up = now
        elif action == 'scale_down':
            self.last_scale_down = now
    
    def add_metrics(self, metrics: ScalingMetrics):
        """Add metrics to history"""
        self.metrics_history.append(metrics)
        
        # Keep only last 100 metrics
        if len(self.metrics_history) > 100:
            self.metrics_history.pop(0)


class AdaptiveScaler:
    """Adaptive scaling system that manages worker pools"""
    
    def __init__(self, policy: ScalingPolicy, worker_factory: Callable[[], Any]):
        self.policy = policy
        self.worker_factory = worker_factory
        self.decision_engine = ScalingDecisionEngine(policy)
        
        # Worker management
        self.workers = []
        self.active_workers = 0
        self.worker_lock = threading.RLock()
        
        # Monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        self.metrics_queue = queue.Queue(maxsize=1000)
        
        # Statistics
        self.stats = {
            'scale_up_events': 0,
            'scale_down_events': 0,
            'total_tasks_processed': 0,
            'average_response_time': 0.0,
            'peak_workers': 0,
            'min_workers_used': float('inf')
        }
        
        # Initialize with minimum workers
        self._initialize_workers()
        
        logger.info(f"AdaptiveScaler initialized: {self.policy.name}")
    
    def _initialize_workers(self):
        """Initialize minimum number of workers"""
        with self.worker_lock:
            for _ in range(self.policy.min_workers):
                worker = self.worker_factory()
                self.workers.append(worker)
                self.active_workers += 1
    
    def start_monitoring(self, interval: int = 10):
        """Start adaptive scaling monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info(f"Adaptive scaling monitoring started (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop adaptive scaling monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Adaptive scaling monitoring stopped")
    
    def _monitoring_loop(self, interval: int):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                self.decision_engine.add_metrics(metrics)
                
                # Make scaling decisions
                if self.decision_engine.should_scale_up(metrics):
                    self._scale_up()
                elif self.decision_engine.should_scale_down(metrics):
                    self._scale_down()
                
                # Update statistics
                self._update_stats(metrics)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(interval)
    
    def _collect_metrics(self) -> ScalingMetrics:
        """Collect current system metrics"""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        # Queue and worker metrics
        queue_length = self.metrics_queue.qsize()
        
        # Throughput estimation (simplified)
        throughput = min(self.active_workers * 10, 100)  # Estimate
        
        # Response time estimation
        base_response = 100  # Base response time in ms
        load_factor = max(1, queue_length / max(1, self.active_workers))
        response_time = base_response * load_factor
        
        metrics = ScalingMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            queue_length=queue_length,
            active_workers=self.active_workers,
            throughput_per_second=throughput,
            response_time_ms=response_time
        )
        
        return metrics
    
    def _scale_up(self):
        """Add workers to the pool"""
        with self.worker_lock:
            if self.active_workers >= self.policy.max_workers:
                return
            
            # Add workers
            workers_to_add = min(
                self.policy.scale_step,
                self.policy.max_workers - self.active_workers
            )
            
            for _ in range(workers_to_add):
                try:
                    worker = self.worker_factory()
                    self.workers.append(worker)
                    self.active_workers += 1
                except Exception as e:
                    logger.error(f"Failed to create worker: {e}")
                    break
            
            self.stats['scale_up_events'] += 1
            self.stats['peak_workers'] = max(self.stats['peak_workers'], self.active_workers)
            
            self.decision_engine.record_scaling_event('scale_up')
            
            logger.info(f"Scaled up: added {workers_to_add} workers "
                       f"(total: {self.active_workers})")
    
    def _scale_down(self):
        """Remove workers from the pool"""
        with self.worker_lock:
            if self.active_workers <= self.policy.min_workers:
                return
            
            # Remove workers
            workers_to_remove = min(
                self.policy.scale_step,
                self.active_workers - self.policy.min_workers
            )
            
            for _ in range(workers_to_remove):
                if self.workers:
                    worker = self.workers.pop()
                    # Gracefully shutdown worker if possible
                    if hasattr(worker, 'shutdown'):
                        try:
                            worker.shutdown()
                        except:
                            pass
                    self.active_workers -= 1
            
            self.stats['scale_down_events'] += 1
            self.stats['min_workers_used'] = min(self.stats['min_workers_used'], 
                                                self.active_workers)
            
            self.decision_engine.record_scaling_event('scale_down')
            
            logger.info(f"Scaled down: removed {workers_to_remove} workers "
                       f"(total: {self.active_workers})")
    
    def _update_stats(self, metrics: ScalingMetrics):
        """Update internal statistics"""
        # Update running averages (simplified)
        alpha = 0.1  # Exponential moving average factor
        
        if self.stats['average_response_time'] == 0:
            self.stats['average_response_time'] = metrics.response_time_ms
        else:
            self.stats['average_response_time'] = (
                alpha * metrics.response_time_ms + 
                (1 - alpha) * self.stats['average_response_time']
            )
    
    def get_current_capacity(self) -> int:
        """Get current worker capacity"""
        return self.active_workers
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get scaling statistics"""
        return {
            'current_workers': self.active_workers,
            'min_workers': self.policy.min_workers,
            'max_workers': self.policy.max_workers,
            'scale_up_events': self.stats['scale_up_events'],
            'scale_down_events': self.stats['scale_down_events'],
            'peak_workers': self.stats['peak_workers'],
            'min_workers_used': self.stats['min_workers_used'] if self.stats['min_workers_used'] != float('inf') else 0,
            'average_response_time': self.stats['average_response_time'],
            'monitoring_active': self.monitoring_active,
            'policy_name': self.policy.name
        }
    
    def manually_scale(self, target_workers: int) -> bool:
        """Manually set number of workers"""
        target_workers = max(self.policy.min_workers, 
                           min(target_workers, self.policy.max_workers))
        
        with self.worker_lock:
            current = self.active_workers
            
            if target_workers > current:
                # Scale up
                for _ in range(target_workers - current):
                    try:
                        worker = self.worker_factory()
                        self.workers.append(worker)
                        self.active_workers += 1
                    except Exception as e:
                        logger.error(f"Manual scale up failed: {e}")
                        return False
                        
            elif target_workers < current:
                # Scale down
                for _ in range(current - target_workers):
                    if self.workers:
                        worker = self.workers.pop()
                        if hasattr(worker, 'shutdown'):
                            try:
                                worker.shutdown()
                            except:
                                pass
                        self.active_workers -= 1
            
            logger.info(f"Manual scaling: {current} → {self.active_workers} workers")
            return True


# Factory for creating mock workers
class MockWorker:
    """Mock worker for testing"""
    def __init__(self):
        self.active = True
    
    def shutdown(self):
        self.active = False


def create_mock_worker() -> MockWorker:
    """Factory function for mock workers"""
    return MockWorker()


# Example usage
if __name__ == "__main__":
    # Create scaling policy
    policy = ScalingPolicy(
        name="discovery_scaling",
        min_workers=2,
        max_workers=8,
        scale_up_threshold={
            'cpu_percent': 60.0,
            'queue_length': 5
        },
        scale_down_threshold={
            'cpu_percent': 20.0,
            'queue_length': 1
        }
    )
    
    # Create adaptive scaler
    scaler = AdaptiveScaler(policy, create_mock_worker)
    
    # Start monitoring
    scaler.start_monitoring(interval=5)
    
    try:
        # Simulate some load
        for i in range(10):
            # Add some fake load to queue
            try:
                scaler.metrics_queue.put(f"task_{i}", timeout=1)
            except queue.Full:
                pass
            
            time.sleep(2)
            print(f"Current capacity: {scaler.get_current_capacity()}")
        
        # Show final stats
        print("Scaling Statistics:")
        for key, value in scaler.get_scaling_stats().items():
            print(f"  {key}: {value}")
        
    finally:
        scaler.stop_monitoring()