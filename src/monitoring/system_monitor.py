"""Advanced system monitoring for AI Science Platform"""

import psutil
import time
import threading
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
import logging
from pathlib import Path

from ..utils.error_handling import robust_execution, PlatformError
from ..utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    network_sent_mb: float
    network_recv_mb: float
    active_processes: int
    load_average: List[float]
    gpu_available: bool = False
    gpu_memory_percent: float = 0.0


@dataclass
class ApplicationMetrics:
    """Application-specific metrics"""
    timestamp: str
    discoveries_made: int
    models_trained: int
    experiments_running: int
    errors_count: int
    average_response_time_ms: float
    memory_used_mb: float
    active_connections: int


class SystemMonitor:
    """Advanced system monitoring with alerting"""
    
    def __init__(self, 
                 monitoring_interval: float = 30.0,
                 alert_thresholds: Optional[Dict[str, float]] = None,
                 metrics_retention_hours: int = 24):
        self.monitoring_interval = monitoring_interval
        self.alert_thresholds = alert_thresholds or {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_usage_percent': 90.0,
            'error_rate': 0.1  # 10% error rate
        }
        self.metrics_retention_hours = metrics_retention_hours
        
        self._monitoring = False
        self._monitor_thread = None
        self._system_metrics: List[SystemMetrics] = []
        self._app_metrics: List[ApplicationMetrics] = []
        self._alert_callbacks: List[Callable] = []
        self._last_network_stats = None
        
        # Application state tracking
        self._app_state = {
            'discoveries_made': 0,
            'models_trained': 0,
            'experiments_running': 0,
            'errors_count': 0,
            'response_times': [],
            'start_time': time.time()
        }
        
        logger.info("SystemMonitor initialized")
    
    @robust_execution(recovery_strategy='graceful_degradation')
    def start_monitoring(self):
        """Start background monitoring"""
        if self._monitoring:
            logger.warning("Monitoring already started")
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("System monitoring stopped")
    
    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add callback for alert notifications"""
        self._alert_callbacks.append(callback)
        logger.debug("Alert callback added")
    
    @robust_execution(recovery_strategy='graceful_degradation')
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self._system_metrics.append(system_metrics)
                
                # Collect application metrics
                app_metrics = self._collect_application_metrics()
                self._app_metrics.append(app_metrics)
                
                # Check alert conditions
                self._check_alerts(system_metrics, app_metrics)
                
                # Clean old metrics
                self._cleanup_old_metrics()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {str(e)}")
                time.sleep(self.monitoring_interval)
    
    @robust_execution(recovery_strategy='graceful_degradation')
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system performance metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent
            
            # Network stats
            network = psutil.net_io_counters()
            network_sent_mb = network.bytes_sent / (1024**2)
            network_recv_mb = network.bytes_recv / (1024**2)
            
            # Process count
            active_processes = len(psutil.pids())
            
            # Load average (Unix systems)
            try:
                load_average = list(psutil.getloadavg())
            except AttributeError:
                # Windows doesn't have load average
                load_average = [cpu_percent / 100.0] * 3
            
            # GPU detection (basic)
            gpu_available = False
            gpu_memory_percent = 0.0
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_available = True
                    gpu_memory_percent = gpus[0].memoryUtil * 100
            except ImportError:
                pass
            
            return SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available_gb=memory_available_gb,
                disk_usage_percent=disk_usage_percent,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
                active_processes=active_processes,
                load_average=load_average,
                gpu_available=gpu_available,
                gpu_memory_percent=gpu_memory_percent
            )
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {str(e)}")
            raise PlatformError(f"System metrics collection failed: {str(e)}")
    
    def _collect_application_metrics(self) -> ApplicationMetrics:
        """Collect application-specific metrics"""
        current_time = time.time()
        uptime_hours = (current_time - self._app_state['start_time']) / 3600
        
        # Calculate average response time
        response_times = self._app_state['response_times']
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
        
        # Get current process memory usage
        process = psutil.Process()
        memory_used_mb = process.memory_info().rss / (1024**2)
        
        return ApplicationMetrics(
            timestamp=datetime.now().isoformat(),
            discoveries_made=self._app_state['discoveries_made'],
            models_trained=self._app_state['models_trained'],
            experiments_running=self._app_state['experiments_running'],
            errors_count=self._app_state['errors_count'],
            average_response_time_ms=avg_response_time,
            memory_used_mb=memory_used_mb,
            active_connections=0  # Would be populated from actual connection pool
        )
    
    def _check_alerts(self, system_metrics: SystemMetrics, app_metrics: ApplicationMetrics):
        """Check for alert conditions"""
        alerts = []
        
        # System alerts
        if system_metrics.cpu_percent > self.alert_thresholds['cpu_percent']:
            alerts.append({
                'type': 'high_cpu',
                'value': system_metrics.cpu_percent,
                'threshold': self.alert_thresholds['cpu_percent']
            })
        
        if system_metrics.memory_percent > self.alert_thresholds['memory_percent']:
            alerts.append({
                'type': 'high_memory',
                'value': system_metrics.memory_percent,
                'threshold': self.alert_thresholds['memory_percent']
            })
        
        if system_metrics.disk_usage_percent > self.alert_thresholds['disk_usage_percent']:
            alerts.append({
                'type': 'high_disk_usage',
                'value': system_metrics.disk_usage_percent,
                'threshold': self.alert_thresholds['disk_usage_percent']
            })
        
        # Application alerts
        total_operations = max(1, app_metrics.discoveries_made + app_metrics.models_trained)
        error_rate = app_metrics.errors_count / total_operations
        if error_rate > self.alert_thresholds['error_rate']:
            alerts.append({
                'type': 'high_error_rate',
                'value': error_rate,
                'threshold': self.alert_thresholds['error_rate']
            })
        
        # Trigger alert callbacks
        for alert in alerts:
            for callback in self._alert_callbacks:
                try:
                    callback(alert['type'], alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {str(e)}")
    
    def _cleanup_old_metrics(self):
        """Remove metrics older than retention period"""
        cutoff_time = datetime.now() - timedelta(hours=self.metrics_retention_hours)
        cutoff_str = cutoff_time.isoformat()
        
        self._system_metrics = [
            m for m in self._system_metrics 
            if m.timestamp >= cutoff_str
        ]
        
        self._app_metrics = [
            m for m in self._app_metrics 
            if m.timestamp >= cutoff_str
        ]
    
    def record_discovery(self):
        """Record a discovery event"""
        self._app_state['discoveries_made'] += 1
    
    def record_model_training(self):
        """Record a model training event"""
        self._app_state['models_trained'] += 1
    
    def record_error(self):
        """Record an error event"""
        self._app_state['errors_count'] += 1
    
    def record_response_time(self, response_time_ms: float):
        """Record a response time measurement"""
        self._app_state['response_times'].append(response_time_ms)
        # Keep only recent response times
        if len(self._app_state['response_times']) > 1000:
            self._app_state['response_times'] = self._app_state['response_times'][-500:]
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system and application metrics"""
        if not self._system_metrics or not self._app_metrics:
            return {}
        
        return {
            'system': asdict(self._system_metrics[-1]),
            'application': asdict(self._app_metrics[-1]),
            'monitoring_active': self._monitoring,
            'metrics_count': len(self._system_metrics)
        }
    
    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get summary statistics for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        cutoff_str = cutoff_time.isoformat()
        
        recent_system = [m for m in self._system_metrics if m.timestamp >= cutoff_str]
        recent_app = [m for m in self._app_metrics if m.timestamp >= cutoff_str]
        
        if not recent_system or not recent_app:
            return {}
        
        # Calculate system averages
        avg_cpu = sum(m.cpu_percent for m in recent_system) / len(recent_system)
        avg_memory = sum(m.memory_percent for m in recent_system) / len(recent_system)
        
        # Calculate application totals
        total_discoveries = recent_app[-1].discoveries_made - recent_app[0].discoveries_made
        total_errors = recent_app[-1].errors_count - recent_app[0].errors_count
        
        return {
            'period_hours': hours,
            'system_averages': {
                'cpu_percent': avg_cpu,
                'memory_percent': avg_memory
            },
            'application_totals': {
                'discoveries_made': total_discoveries,
                'errors_count': total_errors
            },
            'data_points': len(recent_system)
        }
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file"""
        try:
            data = {
                'export_time': datetime.now().isoformat(),
                'system_metrics': [asdict(m) for m in self._system_metrics],
                'application_metrics': [asdict(m) for m in self._app_metrics],
                'configuration': {
                    'monitoring_interval': self.monitoring_interval,
                    'alert_thresholds': self.alert_thresholds,
                    'retention_hours': self.metrics_retention_hours
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Metrics exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {str(e)}")
            raise PlatformError(f"Metrics export failed: {str(e)}")


# Global monitor instance
_global_monitor: Optional[SystemMonitor] = None


def get_system_monitor() -> SystemMonitor:
    """Get global system monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = SystemMonitor()
    return _global_monitor


def start_global_monitoring():
    """Start global system monitoring"""
    monitor = get_system_monitor()
    monitor.start_monitoring()


def stop_global_monitoring():
    """Stop global system monitoring"""
    monitor = get_system_monitor()
    monitor.stop_monitoring()