"""Auto-scaling and dynamic load balancing for the AI Science Platform"""

import time
import logging
import threading
import psutil
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import deque
import asyncio

logger = logging.getLogger(__name__)


class ScalingDecision(Enum):
    """Scaling decisions"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"


@dataclass
class ResourceMetrics:
    """System resource metrics for scaling decisions"""
    cpu_percent: float
    memory_percent: float
    disk_io_percent: float
    network_io_mbps: float
    active_tasks: int
    queue_length: int
    avg_response_time: float
    timestamp: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "disk_io_percent": self.disk_io_percent,
            "network_io_mbps": self.network_io_mbps,
            "active_tasks": self.active_tasks,
            "queue_length": self.queue_length,
            "avg_response_time": self.avg_response_time,
            "timestamp": self.timestamp
        }


@dataclass
class ScalingThresholds:
    """Thresholds for auto-scaling decisions"""
    cpu_high: float = 80.0
    cpu_low: float = 30.0
    memory_high: float = 85.0
    memory_low: float = 40.0
    queue_high: int = 50
    queue_low: int = 5
    response_time_high: float = 5.0  # seconds
    min_workers: int = 1
    max_workers: int = 16
    scale_up_cooldown: float = 300.0  # 5 minutes
    scale_down_cooldown: float = 600.0  # 10 minutes


class AutoScaler:
    """Intelligent auto-scaling system with predictive capabilities"""
    
    def __init__(self, 
                 thresholds: Optional[ScalingThresholds] = None,
                 monitoring_interval: float = 60.0):
        
        self.thresholds = thresholds or ScalingThresholds()
        self.monitoring_interval = monitoring_interval
        
        # Historical data for trend analysis
        self.metrics_history = deque(maxlen=100)
        self.scaling_history = deque(maxlen=50)
        
        # Scaling state
        self.current_workers = self.thresholds.min_workers
        self.last_scale_up_time = 0.0
        self.last_scale_down_time = 0.0
        
        # Monitoring thread
        self._monitoring = False
        self._monitor_thread = None
        
        # Callbacks for scaling actions
        self._scale_up_callback: Optional[Callable[[int], None]] = None
        self._scale_down_callback: Optional[Callable[[int], None]] = None
        
        # Predictive modeling
        self._trend_weights = np.array([0.5, 0.3, 0.2])  # Recent, medium, old
        
        logger.info(f"AutoScaler initialized: workers={self.current_workers}, interval={monitoring_interval}s")
    
    def set_scaling_callbacks(self, 
                            scale_up_func: Callable[[int], None],
                            scale_down_func: Callable[[int], None]) -> None:
        """Set callback functions for scaling operations"""
        self._scale_up_callback = scale_up_func
        self._scale_down_callback = scale_down_func
        logger.info("Scaling callbacks registered")
    
    def start_monitoring(self) -> None:
        """Start auto-scaling monitoring"""
        if self._monitoring:
            logger.warning("Auto-scaling monitoring already running")
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitor_thread.start()
        
        logger.info("Auto-scaling monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop auto-scaling monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)
        
        logger.info("Auto-scaling monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self._monitoring:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Make scaling decision
                decision, target_workers = self._make_scaling_decision(metrics)
                
                # Execute scaling decision
                if decision != ScalingDecision.MAINTAIN:
                    self._execute_scaling_decision(decision, target_workers)
                
                # Sleep until next check
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in auto-scaling monitoring: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current system metrics"""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk I/O (simplified)
        disk_io = psutil.disk_io_counters()
        disk_io_percent = min(100, (disk_io.read_bytes + disk_io.write_bytes) / (1024**2))  # MB/s approximation
        
        # Network I/O (simplified)
        network_io = psutil.net_io_counters()
        network_io_mbps = (network_io.bytes_sent + network_io.bytes_recv) / (1024**2)
        
        # Task metrics (mock values - should be integrated with actual task system)
        active_tasks = self._get_active_tasks_count()
        queue_length = self._get_queue_length()
        avg_response_time = self._get_avg_response_time()
        
        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_io_percent=disk_io_percent,
            network_io_mbps=network_io_mbps,
            active_tasks=active_tasks,
            queue_length=queue_length,
            avg_response_time=avg_response_time,
            timestamp=time.time()
        )
    
    def _get_active_tasks_count(self) -> int:
        """Get current active tasks count"""
        # This should be integrated with the actual task system
        # For now, return a mock value based on CPU usage
        return max(1, int(psutil.cpu_percent() / 10))
    
    def _get_queue_length(self) -> int:
        """Get current queue length"""
        # This should be integrated with the actual task queue
        # For now, return a mock value
        return max(0, int((psutil.cpu_percent() - 50) / 5)) if psutil.cpu_percent() > 50 else 0
    
    def _get_avg_response_time(self) -> float:
        """Get average response time"""
        # This should be integrated with actual metrics
        # For now, correlate with system load
        load_factor = (psutil.cpu_percent() + psutil.virtual_memory().percent) / 200
        return 1.0 + load_factor * 3.0  # Base 1s + load-based increase
    
    def _make_scaling_decision(self, metrics: ResourceMetrics) -> Tuple[ScalingDecision, int]:
        """Make intelligent scaling decision based on metrics"""
        current_time = time.time()
        
        # Calculate scaling score
        scale_score = self._calculate_scale_score(metrics)
        
        # Check cooldown periods
        scale_up_ready = (current_time - self.last_scale_up_time) > self.thresholds.scale_up_cooldown
        scale_down_ready = (current_time - self.last_scale_down_time) > self.thresholds.scale_down_cooldown
        
        # Predictive trend analysis
        trend_factor = self._calculate_trend_factor()
        adjusted_score = scale_score + trend_factor
        
        # Make decision
        if adjusted_score > 0.7 and scale_up_ready and self.current_workers < self.thresholds.max_workers:
            # Scale up
            target_workers = min(
                self.thresholds.max_workers,
                self.current_workers + max(1, int(adjusted_score * 2))
            )
            return ScalingDecision.SCALE_UP, target_workers
            
        elif adjusted_score < -0.7 and scale_down_ready and self.current_workers > self.thresholds.min_workers:
            # Scale down
            target_workers = max(
                self.thresholds.min_workers,
                self.current_workers - max(1, int(abs(adjusted_score)))
            )
            return ScalingDecision.SCALE_DOWN, target_workers
        
        else:
            return ScalingDecision.MAINTAIN, self.current_workers
    
    def _calculate_scale_score(self, metrics: ResourceMetrics) -> float:
        """Calculate scaling score (-1 to 1, negative=scale down, positive=scale up)"""
        score = 0.0
        
        # CPU factor
        if metrics.cpu_percent > self.thresholds.cpu_high:
            score += (metrics.cpu_percent - self.thresholds.cpu_high) / 20
        elif metrics.cpu_percent < self.thresholds.cpu_low:
            score -= (self.thresholds.cpu_low - metrics.cpu_percent) / 30
        
        # Memory factor
        if metrics.memory_percent > self.thresholds.memory_high:
            score += (metrics.memory_percent - self.thresholds.memory_high) / 15
        elif metrics.memory_percent < self.thresholds.memory_low:
            score -= (self.thresholds.memory_low - metrics.memory_percent) / 40
        
        # Queue length factor
        if metrics.queue_length > self.thresholds.queue_high:
            score += (metrics.queue_length - self.thresholds.queue_high) / 25
        elif metrics.queue_length < self.thresholds.queue_low:
            score -= (self.thresholds.queue_low - metrics.queue_length) / 10
        
        # Response time factor
        if metrics.avg_response_time > self.thresholds.response_time_high:
            score += (metrics.avg_response_time - self.thresholds.response_time_high) / 2
        
        # Clamp score to [-1, 1]
        return max(-1.0, min(1.0, score))
    
    def _calculate_trend_factor(self) -> float:
        """Calculate predictive trend factor"""
        if len(self.metrics_history) < 3:
            return 0.0
        
        # Get recent metrics
        recent_metrics = list(self.metrics_history)[-3:]
        
        # Calculate trends for key metrics
        cpu_trend = self._calculate_metric_trend([m.cpu_percent for m in recent_metrics])
        memory_trend = self._calculate_metric_trend([m.memory_percent for m in recent_metrics])
        queue_trend = self._calculate_metric_trend([m.queue_length for m in recent_metrics])
        
        # Weighted average of trends
        trend_factor = (cpu_trend * 0.4 + memory_trend * 0.3 + queue_trend * 0.3) * 0.3
        
        return trend_factor
    
    def _calculate_metric_trend(self, values: List[float]) -> float:
        """Calculate trend for a metric (-1=decreasing, 1=increasing)"""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend
        x = np.arange(len(values))
        y = np.array(values)
        
        if np.std(y) == 0:
            return 0.0
        
        # Calculate slope normalized by range
        slope = np.polyfit(x, y, 1)[0]
        normalized_slope = slope / (np.max(y) - np.min(y) + 0.001)
        
        return max(-1.0, min(1.0, normalized_slope * 10))
    
    def _execute_scaling_decision(self, decision: ScalingDecision, target_workers: int) -> None:
        """Execute scaling decision"""
        old_workers = self.current_workers
        
        try:
            if decision == ScalingDecision.SCALE_UP:
                if self._scale_up_callback:
                    self._scale_up_callback(target_workers - old_workers)
                self.current_workers = target_workers
                self.last_scale_up_time = time.time()
                logger.info(f"Scaled UP: {old_workers} → {target_workers} workers")
                
            elif decision == ScalingDecision.SCALE_DOWN:
                if self._scale_down_callback:
                    self._scale_down_callback(old_workers - target_workers)
                self.current_workers = target_workers
                self.last_scale_down_time = time.time()
                logger.info(f"Scaled DOWN: {old_workers} → {target_workers} workers")
            
            # Record scaling event
            self.scaling_history.append({
                "timestamp": time.time(),
                "decision": decision.value,
                "old_workers": old_workers,
                "new_workers": target_workers,
                "metrics": self.metrics_history[-1] if self.metrics_history else None
            })
            
        except Exception as e:
            logger.error(f"Failed to execute scaling decision {decision}: {e}")
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics"""
        recent_metrics = list(self.metrics_history)[-10:] if self.metrics_history else []
        
        if recent_metrics:
            avg_cpu = np.mean([m.cpu_percent for m in recent_metrics])
            avg_memory = np.mean([m.memory_percent for m in recent_metrics])
            avg_queue = np.mean([m.queue_length for m in recent_metrics])
            avg_response_time = np.mean([m.avg_response_time for m in recent_metrics])
        else:
            avg_cpu = avg_memory = avg_queue = avg_response_time = 0.0
        
        scaling_events = len(self.scaling_history)
        scale_ups = sum(1 for event in self.scaling_history 
                       if event["decision"] == ScalingDecision.SCALE_UP.value)
        scale_downs = sum(1 for event in self.scaling_history 
                         if event["decision"] == ScalingDecision.SCALE_DOWN.value)
        
        return {
            "current_workers": self.current_workers,
            "monitoring_active": self._monitoring,
            "metrics_history_length": len(self.metrics_history),
            "scaling_events": scaling_events,
            "scale_ups": scale_ups,
            "scale_downs": scale_downs,
            "avg_cpu_10min": avg_cpu,
            "avg_memory_10min": avg_memory,
            "avg_queue_10min": avg_queue,
            "avg_response_time_10min": avg_response_time,
            "last_scale_up": self.last_scale_up_time,
            "last_scale_down": self.last_scale_down_time
        }
    
    def force_scale(self, target_workers: int) -> bool:
        """Manually force scaling to specific worker count"""
        if target_workers < self.thresholds.min_workers or target_workers > self.thresholds.max_workers:
            logger.error(f"Invalid worker count: {target_workers} (min={self.thresholds.min_workers}, max={self.thresholds.max_workers})")
            return False
        
        old_workers = self.current_workers
        
        if target_workers > old_workers:
            decision = ScalingDecision.SCALE_UP
        elif target_workers < old_workers:
            decision = ScalingDecision.SCALE_DOWN
        else:
            logger.info(f"Already at target worker count: {target_workers}")
            return True
        
        try:
            self._execute_scaling_decision(decision, target_workers)
            logger.info(f"Manual scaling: {old_workers} → {target_workers} workers")
            return True
        except Exception as e:
            logger.error(f"Manual scaling failed: {e}")
            return False


class LoadBalancer:
    """Intelligent load balancer for distributing tasks"""
    
    def __init__(self):
        self.workers = {}
        self.worker_stats = {}
        self._round_robin_index = 0
        
    def register_worker(self, worker_id: str, worker_callback: Callable) -> None:
        """Register a worker"""
        self.workers[worker_id] = worker_callback
        self.worker_stats[worker_id] = {
            "active_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "avg_response_time": 0.0,
            "last_used": 0.0,
            "health_score": 1.0
        }
        logger.info(f"Worker registered: {worker_id}")
    
    def select_worker(self, task_priority: int = 0, strategy: str = "least_loaded") -> Optional[str]:
        """Select best worker based on strategy"""
        if not self.workers:
            return None
        
        if strategy == "round_robin":
            return self._round_robin_selection()
        elif strategy == "least_loaded":
            return self._least_loaded_selection()
        elif strategy == "weighted_response_time":
            return self._weighted_response_time_selection()
        else:
            return self._least_loaded_selection()  # Default
    
    def _round_robin_selection(self) -> str:
        """Simple round-robin selection"""
        worker_ids = list(self.workers.keys())
        selected = worker_ids[self._round_robin_index % len(worker_ids)]
        self._round_robin_index += 1
        return selected
    
    def _least_loaded_selection(self) -> str:
        """Select worker with least active tasks"""
        return min(
            self.worker_stats.keys(),
            key=lambda w: self.worker_stats[w]["active_tasks"]
        )
    
    def _weighted_response_time_selection(self) -> str:
        """Select worker based on health score and response time"""
        scores = {}
        
        for worker_id, stats in self.worker_stats.items():
            # Calculate composite score
            load_factor = 1.0 / (stats["active_tasks"] + 1)
            response_factor = 1.0 / (stats["avg_response_time"] + 0.1)
            health_factor = stats["health_score"]
            
            scores[worker_id] = load_factor * response_factor * health_factor
        
        return max(scores.keys(), key=lambda w: scores[w])
    
    def update_worker_stats(self, worker_id: str, **stats) -> None:
        """Update worker statistics"""
        if worker_id in self.worker_stats:
            self.worker_stats[worker_id].update(stats)
            self.worker_stats[worker_id]["last_used"] = time.time()


# Global instances
_global_autoscaler: Optional[AutoScaler] = None
_global_load_balancer: Optional[LoadBalancer] = None


def get_autoscaler(thresholds: Optional[ScalingThresholds] = None) -> AutoScaler:
    """Get or create global auto-scaler"""
    global _global_autoscaler
    if _global_autoscaler is None:
        _global_autoscaler = AutoScaler(thresholds)
    return _global_autoscaler


def get_load_balancer() -> LoadBalancer:
    """Get or create global load balancer"""
    global _global_load_balancer
    if _global_load_balancer is None:
        _global_load_balancer = LoadBalancer()
    return _global_load_balancer