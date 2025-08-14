"""Comprehensive health monitoring system"""

import time
import psutil
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import threading
from collections import deque

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthMetrics:
    """System health metrics"""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    component_statuses: Dict[str, HealthStatus]
    performance_metrics: Dict[str, float]
    error_counts: Dict[str, int]
    availability_score: float


@dataclass
class ComponentHealth:
    """Individual component health status"""
    name: str
    status: HealthStatus
    last_check: float
    response_time: float
    error_rate: float
    uptime: float
    metadata: Dict[str, Any]


class HealthMonitor:
    """Comprehensive system health monitoring"""
    
    def __init__(self, check_interval: float = 30.0, history_size: int = 1000):
        self.check_interval = check_interval
        self.history_size = history_size
        
        # Health data storage
        self.health_history = deque(maxlen=history_size)
        self.component_registry = {}
        self.component_health = {}
        
        # Alert thresholds
        self.thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0,
            "error_rate": 0.05,
            "response_time": 5.0
        }
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        self.callbacks = []
        
        logger.info("HealthMonitor initialized")
    
    def register_component(self, name: str, health_check_func: Callable[[], Dict[str, Any]],
                          critical: bool = False) -> None:
        """Register a component for health monitoring"""
        self.component_registry[name] = {
            "health_check": health_check_func,
            "critical": critical,
            "last_check": 0,
            "error_count": 0,
            "total_checks": 0
        }
        
        self.component_health[name] = ComponentHealth(
            name=name,
            status=HealthStatus.UNKNOWN,
            last_check=0,
            response_time=0,
            error_rate=0,
            uptime=0,
            metadata={}
        )
        
        logger.info(f"Registered component for monitoring: {name}")
    
    def add_health_callback(self, callback: Callable[[HealthMetrics], None]) -> None:
        """Add callback for health status changes"""
        self.callbacks.append(callback)
    
    def start_monitoring(self) -> None:
        """Start continuous health monitoring"""
        if self.is_monitoring:
            logger.warning("Health monitoring already running")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                metrics = self.collect_health_metrics()
                self.health_history.append(metrics)
                
                # Trigger callbacks
                for callback in self.callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        logger.error(f"Health callback error: {e}")
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(self.check_interval)
    
    def collect_health_metrics(self) -> HealthMetrics:
        """Collect comprehensive health metrics"""
        timestamp = time.time()
        
        # System metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Component health checks
        component_statuses = {}
        performance_metrics = {}
        error_counts = {}
        
        for name, config in self.component_registry.items():
            try:
                start_time = time.time()
                health_data = config["health_check"]()
                response_time = time.time() - start_time
                
                # Update component health
                component = self.component_health[name]
                component.last_check = timestamp
                component.response_time = response_time
                component.metadata = health_data
                
                # Determine status
                status = self._determine_component_status(health_data, response_time)
                component.status = status
                component_statuses[name] = status
                
                # Update statistics
                config["total_checks"] += 1
                if status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                    config["error_count"] += 1
                
                component.error_rate = config["error_count"] / config["total_checks"]
                
                # Performance metrics
                performance_metrics[f"{name}_response_time"] = response_time
                performance_metrics[f"{name}_error_rate"] = component.error_rate
                
                error_counts[name] = config["error_count"]
                
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                component_statuses[name] = HealthStatus.CRITICAL
                config["error_count"] += 1
                error_counts[name] = config["error_count"]
        
        # Calculate overall availability
        availability_score = self._calculate_availability_score(component_statuses)
        
        metrics = HealthMetrics(
            timestamp=timestamp,
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            disk_usage=disk.percent,
            component_statuses=component_statuses,
            performance_metrics=performance_metrics,
            error_counts=error_counts,
            availability_score=availability_score
        )
        
        return metrics
    
    def _determine_component_status(self, health_data: Dict[str, Any], 
                                  response_time: float) -> HealthStatus:
        """Determine component health status"""
        
        # Check response time
        if response_time > self.thresholds["response_time"]:
            return HealthStatus.WARNING
        
        # Check health data indicators
        if "status" in health_data:
            status_str = health_data["status"].lower()
            if status_str in ["critical", "error", "failed"]:
                return HealthStatus.CRITICAL
            elif status_str in ["warning", "degraded"]:
                return HealthStatus.WARNING
            elif status_str in ["healthy", "ok", "running"]:
                return HealthStatus.HEALTHY
        
        # Check error rates
        if "error_rate" in health_data:
            if health_data["error_rate"] > self.thresholds["error_rate"]:
                return HealthStatus.WARNING
        
        # Default to healthy if no issues detected
        return HealthStatus.HEALTHY
    
    def _calculate_availability_score(self, component_statuses: Dict[str, HealthStatus]) -> float:
        """Calculate overall system availability score"""
        if not component_statuses:
            return 1.0
        
        status_weights = {
            HealthStatus.HEALTHY: 1.0,
            HealthStatus.WARNING: 0.7,
            HealthStatus.CRITICAL: 0.0,
            HealthStatus.UNKNOWN: 0.5
        }
        
        total_weight = 0
        weighted_score = 0
        
        for name, status in component_statuses.items():
            # Critical components have higher weight
            component_weight = 2.0 if self.component_registry[name]["critical"] else 1.0
            
            total_weight += component_weight
            weighted_score += status_weights[status] * component_weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def get_current_health(self) -> Optional[HealthMetrics]:
        """Get current health status"""
        if self.health_history:
            return self.health_history[-1]
        return self.collect_health_metrics()
    
    def get_health_history(self, hours: float = 24.0) -> List[HealthMetrics]:
        """Get health history for specified time period"""
        cutoff_time = time.time() - (hours * 3600)
        return [h for h in self.health_history if h.timestamp >= cutoff_time]
    
    def get_component_health(self, component_name: str) -> Optional[ComponentHealth]:
        """Get health status for specific component"""
        return self.component_health.get(component_name)
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for alert conditions"""
        alerts = []
        current_health = self.get_current_health()
        
        if not current_health:
            return alerts
        
        # System resource alerts
        if current_health.cpu_usage > self.thresholds["cpu_usage"]:
            alerts.append({
                "type": "system",
                "severity": "warning",
                "message": f"High CPU usage: {current_health.cpu_usage:.1f}%",
                "timestamp": current_health.timestamp
            })
        
        if current_health.memory_usage > self.thresholds["memory_usage"]:
            alerts.append({
                "type": "system",
                "severity": "warning",
                "message": f"High memory usage: {current_health.memory_usage:.1f}%",
                "timestamp": current_health.timestamp
            })
        
        if current_health.disk_usage > self.thresholds["disk_usage"]:
            alerts.append({
                "type": "system",
                "severity": "critical",
                "message": f"High disk usage: {current_health.disk_usage:.1f}%",
                "timestamp": current_health.timestamp
            })
        
        # Component alerts
        for name, status in current_health.component_statuses.items():
            if status == HealthStatus.CRITICAL:
                alerts.append({
                    "type": "component",
                    "severity": "critical",
                    "message": f"Component {name} is in critical state",
                    "component": name,
                    "timestamp": current_health.timestamp
                })
            elif status == HealthStatus.WARNING:
                alerts.append({
                    "type": "component",
                    "severity": "warning",
                    "message": f"Component {name} is in warning state",
                    "component": name,
                    "timestamp": current_health.timestamp
                })
        
        # Availability alerts
        if current_health.availability_score < 0.8:
            severity = "critical" if current_health.availability_score < 0.5 else "warning"
            alerts.append({
                "type": "availability",
                "severity": severity,
                "message": f"System availability low: {current_health.availability_score:.2%}",
                "timestamp": current_health.timestamp
            })
        
        return alerts
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        current_health = self.get_current_health()
        history = self.get_health_history(24.0)  # Last 24 hours
        
        if not current_health:
            return {"error": "No health data available"}
        
        # Calculate trends
        trends = {}
        if len(history) > 1:
            cpu_values = [h.cpu_usage for h in history]
            memory_values = [h.memory_usage for h in history]
            availability_values = [h.availability_score for h in history]
            
            trends = {
                "cpu_trend": "increasing" if cpu_values[-1] > np.mean(cpu_values[:-10]) else "stable",
                "memory_trend": "increasing" if memory_values[-1] > np.mean(memory_values[:-10]) else "stable",
                "availability_trend": "improving" if availability_values[-1] > np.mean(availability_values[:-10]) else "stable"
            }
        
        # Component summary
        component_summary = {}
        for name, health in self.component_health.items():
            component_summary[name] = {
                "status": health.status.value,
                "uptime_hours": (current_health.timestamp - health.last_check) / 3600,
                "error_rate": health.error_rate,
                "avg_response_time": health.response_time
            }
        
        return {
            "report_timestamp": current_health.timestamp,
            "overall_status": self._get_overall_status(current_health),
            "availability_score": current_health.availability_score,
            "system_metrics": {
                "cpu_usage": current_health.cpu_usage,
                "memory_usage": current_health.memory_usage,
                "disk_usage": current_health.disk_usage
            },
            "component_summary": component_summary,
            "trends": trends,
            "alerts": self.check_alerts(),
            "data_points": len(history),
            "monitoring_duration_hours": (history[-1].timestamp - history[0].timestamp) / 3600 if len(history) > 1 else 0
        }
    
    def _get_overall_status(self, health: HealthMetrics) -> str:
        """Determine overall system status"""
        if health.availability_score >= 0.95:
            return "healthy"
        elif health.availability_score >= 0.8:
            return "warning"
        else:
            return "critical"
    
    def export_health_data(self, filepath: str) -> None:
        """Export health data to file"""
        import json
        
        data = {
            "export_timestamp": time.time(),
            "health_history": [asdict(h) for h in self.health_history],
            "component_health": {name: asdict(health) for name, health in self.component_health.items()},
            "thresholds": self.thresholds
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Health data exported to {filepath}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        history = self.get_health_history(24.0)
        
        if not history:
            return {}
        
        cpu_values = [h.cpu_usage for h in history]
        memory_values = [h.memory_usage for h in history]
        availability_values = [h.availability_score for h in history]
        
        return {
            "cpu_stats": {
                "mean": np.mean(cpu_values),
                "max": np.max(cpu_values),
                "min": np.min(cpu_values),
                "std": np.std(cpu_values)
            },
            "memory_stats": {
                "mean": np.mean(memory_values),
                "max": np.max(memory_values),
                "min": np.min(memory_values),
                "std": np.std(memory_values)
            },
            "availability_stats": {
                "mean": np.mean(availability_values),
                "max": np.max(availability_values),
                "min": np.min(availability_values),
                "uptime_percentage": np.sum([1 for a in availability_values if a > 0.9]) / len(availability_values) * 100
            }
        }