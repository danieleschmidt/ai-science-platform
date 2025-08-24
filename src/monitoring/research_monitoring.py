"""Advanced Research Monitoring and Health System"""

import numpy as np
import logging
import psutil
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import json

from ..utils.error_handling import robust_execution, MonitoringError
from ..utils.validation import ValidationMixin

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class MetricType(Enum):
    """Types of metrics to monitor"""
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    QUALITY = "quality"
    SECURITY = "security"
    BUSINESS = "business"


@dataclass
class MetricValue:
    """Individual metric value with metadata"""
    name: str
    value: float
    unit: str
    metric_type: MetricType
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    threshold_breached: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'type': self.metric_type.value,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags,
            'threshold_breached': self.threshold_breached
        }


@dataclass
class HealthCheck:
    """Health check definition and result"""
    name: str
    check_function: Callable[[], Tuple[HealthStatus, str, Dict[str, Any]]]
    interval_seconds: int
    timeout_seconds: int = 30
    last_run: Optional[datetime] = None
    last_status: HealthStatus = HealthStatus.UNKNOWN
    last_message: str = ""
    last_details: Dict[str, Any] = field(default_factory=dict)
    failure_count: int = 0
    max_failures: int = 3


@dataclass
class Alert:
    """Alert definition and state"""
    alert_id: str
    name: str
    description: str
    severity: str  # info, warning, error, critical
    metric_name: str
    condition: str  # >, <, ==, !=
    threshold: float
    duration_minutes: int = 5  # How long condition must persist
    triggered_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    notification_sent: bool = False
    
    def is_active(self) -> bool:
        """Check if alert is currently active"""
        return self.triggered_at is not None and self.resolved_at is None


class ResearchMonitoringSystem(ValidationMixin):
    """Comprehensive monitoring system for research operations"""
    
    def __init__(self, 
                 collection_interval: int = 60,
                 retention_hours: int = 168,  # 1 week
                 enable_auto_alerts: bool = True):
        """
        Initialize research monitoring system
        
        Args:
            collection_interval: Seconds between metric collections
            retention_hours: Hours to retain metrics
            enable_auto_alerts: Enable automatic alerting
        """
        self.collection_interval = collection_interval
        self.retention_hours = retention_hours
        self.enable_auto_alerts = enable_auto_alerts
        
        # Metric storage (in production, use time-series DB)
        self.metrics_buffer = deque(maxlen=10000)
        self.metric_history: Dict[str, deque] = {}
        
        # Health checks
        self.health_checks: Dict[str, HealthCheck] = {}
        self.overall_health = HealthStatus.UNKNOWN
        
        # Alerts
        self.alerts: Dict[str, Alert] = {}
        self.active_alerts: List[Alert] = []
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Initialize built-in checks and alerts
        self._initialize_default_health_checks()
        self._initialize_default_alerts()
        
        logger.info("ResearchMonitoringSystem initialized")
    
    def start_monitoring(self):
        """Start the monitoring system"""
        
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Research monitoring system started")
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Research monitoring system stopped")
    
    @robust_execution(recovery_strategy='continue_monitoring')
    def collect_system_metrics(self) -> List[MetricValue]:
        """Collect system-level metrics"""
        
        current_time = datetime.now()
        metrics = []
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics.append(MetricValue(
            name="cpu_usage_percent",
            value=cpu_percent,
            unit="percent",
            metric_type=MetricType.RESOURCE,
            timestamp=current_time,
            tags={"component": "system"}
        ))
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics.append(MetricValue(
            name="memory_usage_percent",
            value=memory.percent,
            unit="percent", 
            metric_type=MetricType.RESOURCE,
            timestamp=current_time,
            tags={"component": "system"}
        ))
        
        metrics.append(MetricValue(
            name="memory_available_gb",
            value=memory.available / (1024**3),
            unit="GB",
            metric_type=MetricType.RESOURCE,
            timestamp=current_time,
            tags={"component": "system"}
        ))
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        metrics.append(MetricValue(
            name="disk_usage_percent",
            value=(disk.used / disk.total) * 100,
            unit="percent",
            metric_type=MetricType.RESOURCE,
            timestamp=current_time,
            tags={"component": "system"}
        ))
        
        # Network metrics
        network = psutil.net_io_counters()
        metrics.append(MetricValue(
            name="network_bytes_sent",
            value=network.bytes_sent,
            unit="bytes",
            metric_type=MetricType.PERFORMANCE,
            timestamp=current_time,
            tags={"component": "network"}
        ))
        
        metrics.append(MetricValue(
            name="network_bytes_recv",
            value=network.bytes_recv,
            unit="bytes",
            metric_type=MetricType.PERFORMANCE,
            timestamp=current_time,
            tags={"component": "network"}
        ))
        
        return metrics
    
    def collect_research_metrics(self, 
                                research_data: Dict[str, Any],
                                operation_type: str = "general") -> List[MetricValue]:
        """Collect research-specific metrics"""
        
        current_time = datetime.now()
        metrics = []
        
        # Research quality metrics
        if 'hypothesis_count' in research_data:
            metrics.append(MetricValue(
                name="hypotheses_generated",
                value=research_data['hypothesis_count'],
                unit="count",
                metric_type=MetricType.QUALITY,
                timestamp=current_time,
                tags={"operation": operation_type}
            ))
        
        if 'validation_success_rate' in research_data:
            metrics.append(MetricValue(
                name="validation_success_rate",
                value=research_data['validation_success_rate'],
                unit="percent",
                metric_type=MetricType.QUALITY,
                timestamp=current_time,
                tags={"operation": operation_type}
            ))
        
        if 'breakthrough_score' in research_data:
            metrics.append(MetricValue(
                name="breakthrough_algorithm_score",
                value=research_data['breakthrough_score'],
                unit="score",
                metric_type=MetricType.QUALITY,
                timestamp=current_time,
                tags={"operation": operation_type}
            ))
        
        # Performance metrics
        if 'execution_time' in research_data:
            metrics.append(MetricValue(
                name="research_execution_time",
                value=research_data['execution_time'],
                unit="seconds",
                metric_type=MetricType.PERFORMANCE,
                timestamp=current_time,
                tags={"operation": operation_type}
            ))
        
        if 'data_processing_rate' in research_data:
            metrics.append(MetricValue(
                name="data_processing_rate",
                value=research_data['data_processing_rate'],
                unit="records_per_second",
                metric_type=MetricType.PERFORMANCE,
                timestamp=current_time,
                tags={"operation": operation_type}
            ))
        
        # Discovery metrics
        if 'causal_relationships_discovered' in research_data:
            metrics.append(MetricValue(
                name="causal_relationships_discovered",
                value=research_data['causal_relationships_discovered'],
                unit="count",
                metric_type=MetricType.QUALITY,
                timestamp=current_time,
                tags={"operation": operation_type}
            ))
        
        if 'cross_modal_insights' in research_data:
            metrics.append(MetricValue(
                name="cross_modal_insights",
                value=research_data['cross_modal_insights'],
                unit="count",
                metric_type=MetricType.QUALITY,
                timestamp=current_time,
                tags={"operation": operation_type}
            ))
        
        return metrics
    
    def record_metrics(self, metrics: List[MetricValue]):
        """Record metrics to storage"""
        
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=self.retention_hours)
        
        # Add to main buffer
        for metric in metrics:
            self.metrics_buffer.append(metric)
            
            # Add to metric-specific history
            if metric.name not in self.metric_history:
                self.metric_history[metric.name] = deque(maxlen=1000)
            
            self.metric_history[metric.name].append(metric)
        
        # Clean up old metrics
        self._cleanup_old_metrics(cutoff_time)
        
        # Check for threshold breaches
        if self.enable_auto_alerts:
            self._check_alert_conditions(metrics)
    
    def add_health_check(self, health_check: HealthCheck):
        """Add a custom health check"""
        self.health_checks[health_check.name] = health_check
        logger.info(f"Added health check: {health_check.name}")
    
    def run_health_checks(self) -> Dict[str, HealthCheck]:
        """Run all health checks and return results"""
        
        current_time = datetime.now()
        
        for name, check in self.health_checks.items():
            # Skip if too soon since last run
            if (check.last_run and 
                current_time - check.last_run < timedelta(seconds=check.interval_seconds)):
                continue
            
            try:
                # Run the check with timeout
                start_time = time.time()
                status, message, details = check.check_function()
                execution_time = time.time() - start_time
                
                # Update check results
                check.last_run = current_time
                check.last_status = status
                check.last_message = message
                check.last_details = details
                
                # Reset failure count on success
                if status == HealthStatus.HEALTHY:
                    check.failure_count = 0
                else:
                    check.failure_count += 1
                
                # Record health check metric
                self.record_metrics([MetricValue(
                    name=f"health_check_{name}",
                    value=1.0 if status == HealthStatus.HEALTHY else 0.0,
                    unit="boolean",
                    metric_type=MetricType.QUALITY,
                    timestamp=current_time,
                    tags={"check": name, "status": status.value}
                )])
                
            except Exception as e:
                logger.error(f"Health check {name} failed with exception: {e}")
                check.last_status = HealthStatus.CRITICAL
                check.last_message = f"Check failed with exception: {str(e)}"
                check.failure_count += 1
        
        # Update overall health
        self._update_overall_health()
        
        return self.health_checks
    
    def add_alert(self, alert: Alert):
        """Add a custom alert"""
        self.alerts[alert.alert_id] = alert
        logger.info(f"Added alert: {alert.name}")
    
    def get_metric_statistics(self, 
                             metric_name: str,
                             time_window_hours: int = 24) -> Dict[str, Any]:
        """Get statistics for a specific metric"""
        
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        if metric_name not in self.metric_history:
            return {"error": f"Metric {metric_name} not found"}
        
        # Filter metrics by time window
        recent_metrics = [m for m in self.metric_history[metric_name] 
                         if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return {"error": f"No recent data for metric {metric_name}"}
        
        values = [m.value for m in recent_metrics]
        
        return {
            "metric_name": metric_name,
            "time_window_hours": time_window_hours,
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": np.mean(values),
            "median": np.median(values),
            "std": np.std(values),
            "percentile_95": np.percentile(values, 95),
            "percentile_99": np.percentile(values, 99),
            "first_timestamp": min(m.timestamp for m in recent_metrics).isoformat(),
            "last_timestamp": max(m.timestamp for m in recent_metrics).isoformat()
        }
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        
        # Run health checks
        health_checks = self.run_health_checks()
        
        # System metrics summary
        system_metrics = {}
        for metric_name in ['cpu_usage_percent', 'memory_usage_percent', 'disk_usage_percent']:
            stats = self.get_metric_statistics(metric_name, time_window_hours=1)
            if 'error' not in stats:
                system_metrics[metric_name] = {
                    'current': stats.get('mean', 0),
                    'max_1h': stats.get('max', 0),
                    'trend': 'stable'  # Simplified
                }
        
        # Research metrics summary
        research_metrics = {}
        for metric_name in ['hypotheses_generated', 'validation_success_rate', 'breakthrough_algorithm_score']:
            stats = self.get_metric_statistics(metric_name, time_window_hours=24)
            if 'error' not in stats:
                research_metrics[metric_name] = {
                    'avg_24h': stats.get('mean', 0),
                    'max_24h': stats.get('max', 0),
                    'count_24h': stats.get('count', 0)
                }
        
        # Alert summary
        alert_summary = {
            'total_alerts': len(self.alerts),
            'active_alerts': len(self.active_alerts),
            'critical_alerts': len([a for a in self.active_alerts if a.severity == 'critical']),
            'warning_alerts': len([a for a in self.active_alerts if a.severity == 'warning'])
        }
        
        # Health check summary
        health_summary = {
            'overall_status': self.overall_health.value,
            'total_checks': len(health_checks),
            'healthy_checks': len([c for c in health_checks.values() if c.last_status == HealthStatus.HEALTHY]),
            'warning_checks': len([c for c in health_checks.values() if c.last_status == HealthStatus.WARNING]),
            'critical_checks': len([c for c in health_checks.values() if c.last_status == HealthStatus.CRITICAL])
        }
        
        return {
            'report_timestamp': datetime.now().isoformat(),
            'overall_health': self.overall_health.value,
            'system_metrics': system_metrics,
            'research_metrics': research_metrics,
            'health_checks': health_summary,
            'alerts': alert_summary,
            'recommendations': self._generate_health_recommendations()
        }
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        
        logger.info("Starting monitoring loop")
        
        while self.monitoring_active:
            try:
                # Collect system metrics
                system_metrics = self.collect_system_metrics()
                
                # Record metrics
                self.record_metrics(system_metrics)
                
                # Run health checks
                self.run_health_checks()
                
                # Sleep until next collection
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(min(self.collection_interval, 60))  # Fallback sleep
    
    def _initialize_default_health_checks(self):
        """Initialize default health checks"""
        
        # System resource health checks
        self.add_health_check(HealthCheck(
            name="cpu_usage",
            check_function=self._check_cpu_usage,
            interval_seconds=60
        ))
        
        self.add_health_check(HealthCheck(
            name="memory_usage",
            check_function=self._check_memory_usage,
            interval_seconds=60
        ))
        
        self.add_health_check(HealthCheck(
            name="disk_usage",
            check_function=self._check_disk_usage,
            interval_seconds=300  # 5 minutes
        ))
        
        # Research system health checks
        self.add_health_check(HealthCheck(
            name="research_pipeline",
            check_function=self._check_research_pipeline,
            interval_seconds=600  # 10 minutes
        ))
    
    def _initialize_default_alerts(self):
        """Initialize default alerts"""
        
        # CPU usage alert
        self.add_alert(Alert(
            alert_id="high_cpu_usage",
            name="High CPU Usage",
            description="CPU usage is above 90%",
            severity="warning",
            metric_name="cpu_usage_percent",
            condition=">",
            threshold=90.0,
            duration_minutes=5
        ))
        
        # Memory usage alert
        self.add_alert(Alert(
            alert_id="high_memory_usage",
            name="High Memory Usage", 
            description="Memory usage is above 85%",
            severity="warning",
            metric_name="memory_usage_percent",
            condition=">",
            threshold=85.0,
            duration_minutes=3
        ))
        
        # Disk usage alert
        self.add_alert(Alert(
            alert_id="high_disk_usage",
            name="High Disk Usage",
            description="Disk usage is above 80%",
            severity="critical",
            metric_name="disk_usage_percent",
            condition=">",
            threshold=80.0,
            duration_minutes=1
        ))
        
        # Research quality alert
        self.add_alert(Alert(
            alert_id="low_validation_success",
            name="Low Validation Success Rate",
            description="Hypothesis validation success rate is below 30%",
            severity="warning",
            metric_name="validation_success_rate",
            condition="<",
            threshold=30.0,
            duration_minutes=15
        ))
    
    def _check_cpu_usage(self) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        """Check CPU usage health"""
        
        cpu_percent = psutil.cpu_percent(interval=1)
        
        if cpu_percent > 95:
            return HealthStatus.CRITICAL, f"CPU usage critical: {cpu_percent:.1f}%", {"cpu_percent": cpu_percent}
        elif cpu_percent > 85:
            return HealthStatus.WARNING, f"CPU usage high: {cpu_percent:.1f}%", {"cpu_percent": cpu_percent}
        else:
            return HealthStatus.HEALTHY, f"CPU usage normal: {cpu_percent:.1f}%", {"cpu_percent": cpu_percent}
    
    def _check_memory_usage(self) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        """Check memory usage health"""
        
        memory = psutil.virtual_memory()
        
        if memory.percent > 95:
            return HealthStatus.CRITICAL, f"Memory usage critical: {memory.percent:.1f}%", {"memory_percent": memory.percent}
        elif memory.percent > 85:
            return HealthStatus.WARNING, f"Memory usage high: {memory.percent:.1f}%", {"memory_percent": memory.percent}
        else:
            return HealthStatus.HEALTHY, f"Memory usage normal: {memory.percent:.1f}%", {"memory_percent": memory.percent}
    
    def _check_disk_usage(self) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        """Check disk usage health"""
        
        disk = psutil.disk_usage('/')
        usage_percent = (disk.used / disk.total) * 100
        
        if usage_percent > 90:
            return HealthStatus.CRITICAL, f"Disk usage critical: {usage_percent:.1f}%", {"disk_percent": usage_percent}
        elif usage_percent > 80:
            return HealthStatus.WARNING, f"Disk usage high: {usage_percent:.1f}%", {"disk_percent": usage_percent}
        else:
            return HealthStatus.HEALTHY, f"Disk usage normal: {usage_percent:.1f}%", {"disk_percent": usage_percent}
    
    def _check_research_pipeline(self) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        """Check research pipeline health"""
        
        # Check if research metrics are being generated
        research_metrics = ['hypotheses_generated', 'validation_success_rate']
        recent_metrics = 0
        
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        for metric_name in research_metrics:
            if metric_name in self.metric_history:
                recent_count = len([m for m in self.metric_history[metric_name] 
                                  if m.timestamp > cutoff_time])
                recent_metrics += recent_count
        
        if recent_metrics == 0:
            return HealthStatus.WARNING, "No recent research activity", {"recent_metrics": recent_metrics}
        else:
            return HealthStatus.HEALTHY, f"Research pipeline active", {"recent_metrics": recent_metrics}
    
    def _cleanup_old_metrics(self, cutoff_time: datetime):
        """Clean up metrics older than retention period"""
        
        # Clean main buffer
        self.metrics_buffer = deque([m for m in self.metrics_buffer if m.timestamp > cutoff_time], 
                                   maxlen=self.metrics_buffer.maxlen)
        
        # Clean metric-specific history
        for metric_name in self.metric_history:
            self.metric_history[metric_name] = deque(
                [m for m in self.metric_history[metric_name] if m.timestamp > cutoff_time],
                maxlen=self.metric_history[metric_name].maxlen
            )
    
    def _check_alert_conditions(self, metrics: List[MetricValue]):
        """Check if any metrics breach alert conditions"""
        
        current_time = datetime.now()
        
        for metric in metrics:
            # Find relevant alerts for this metric
            relevant_alerts = [a for a in self.alerts.values() if a.metric_name == metric.name]
            
            for alert in relevant_alerts:
                # Check condition
                condition_met = self._evaluate_alert_condition(metric.value, alert.condition, alert.threshold)
                
                if condition_met:
                    # Check duration requirement
                    if alert.triggered_at is None:
                        alert.triggered_at = current_time
                    
                    duration = current_time - alert.triggered_at
                    if duration >= timedelta(minutes=alert.duration_minutes):
                        if not alert.is_active():
                            # Activate alert
                            self.active_alerts.append(alert)
                            logger.warning(f"Alert triggered: {alert.name} - {alert.description}")
                        
                        # Mark metric as breached
                        metric.threshold_breached = True
                else:
                    # Condition not met - reset or resolve
                    if alert.is_active():
                        alert.resolved_at = current_time
                        self.active_alerts = [a for a in self.active_alerts if a.alert_id != alert.alert_id]
                        logger.info(f"Alert resolved: {alert.name}")
                    
                    alert.triggered_at = None
    
    def _evaluate_alert_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate if alert condition is met"""
        
        if condition == ">":
            return value > threshold
        elif condition == "<":
            return value < threshold
        elif condition == "==":
            return abs(value - threshold) < 0.001  # Floating point comparison
        elif condition == "!=":
            return abs(value - threshold) >= 0.001
        else:
            return False
    
    def _update_overall_health(self):
        """Update overall system health status"""
        
        if not self.health_checks:
            self.overall_health = HealthStatus.UNKNOWN
            return
        
        health_states = [check.last_status for check in self.health_checks.values()]
        
        # Overall health is the worst individual health
        if HealthStatus.CRITICAL in health_states:
            self.overall_health = HealthStatus.CRITICAL
        elif HealthStatus.DEGRADED in health_states:
            self.overall_health = HealthStatus.DEGRADED
        elif HealthStatus.WARNING in health_states:
            self.overall_health = HealthStatus.WARNING
        elif all(state == HealthStatus.HEALTHY for state in health_states):
            self.overall_health = HealthStatus.HEALTHY
        else:
            self.overall_health = HealthStatus.WARNING  # Mixed states
    
    def _generate_health_recommendations(self) -> List[str]:
        """Generate health-based recommendations"""
        
        recommendations = []
        
        # Check active alerts
        if len(self.active_alerts) > 0:
            critical_alerts = [a for a in self.active_alerts if a.severity == 'critical']
            if critical_alerts:
                recommendations.append(f"Address {len(critical_alerts)} critical alerts immediately")
        
        # Check system resources
        cpu_stats = self.get_metric_statistics("cpu_usage_percent", 1)
        if 'error' not in cpu_stats and cpu_stats.get('mean', 0) > 80:
            recommendations.append("High CPU usage detected - consider scaling resources")
        
        memory_stats = self.get_metric_statistics("memory_usage_percent", 1)  
        if 'error' not in memory_stats and memory_stats.get('mean', 0) > 80:
            recommendations.append("High memory usage detected - optimize memory usage")
        
        # Check research metrics
        validation_stats = self.get_metric_statistics("validation_success_rate", 24)
        if 'error' not in validation_stats and validation_stats.get('mean', 100) < 50:
            recommendations.append("Low validation success rate - review hypothesis generation quality")
        
        return recommendations