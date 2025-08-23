"""Advanced monitoring system for autonomous research platform"""

import numpy as np
import logging
import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path
from collections import deque, defaultdict

from ..utils.error_handling import robust_execution, DiscoveryError
from ..utils.validation import ValidationMixin

logger = logging.getLogger(__name__)


@dataclass
class MetricSnapshot:
    """Single metric measurement snapshot"""
    timestamp: datetime
    metric_name: str
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    metric_pattern: str
    condition: str  # 'gt', 'lt', 'eq', 'anomaly'
    threshold: float
    window_minutes: int = 5
    min_occurrences: int = 1
    severity: str = 'warning'  # 'info', 'warning', 'error', 'critical'
    enabled: bool = True
    cooldown_minutes: int = 10


@dataclass
class Alert:
    """Alert instance"""
    rule_name: str
    message: str
    severity: str
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    metric_values: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedMetricsCollector:
    """Sophisticated metrics collection with anomaly detection"""
    
    def __init__(self, 
                 retention_hours: int = 24,
                 collection_interval_seconds: int = 10,
                 anomaly_detection: bool = True):
        
        self.retention_hours = retention_hours
        self.collection_interval = collection_interval_seconds
        self.anomaly_detection = anomaly_detection
        
        # Metric storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=int(3600 * retention_hours / collection_interval_seconds)))
        self.metric_stats: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Anomaly detection state
        self.baseline_windows = {}
        self.anomaly_thresholds = defaultdict(lambda: {'lower': None, 'upper': None})
        
        # Collection state
        self.collectors: Dict[str, Callable] = {}
        self.collection_thread = None
        self.running = False
        
        # Built-in collectors
        self._register_builtin_collectors()
        
        logger.info("Advanced metrics collector initialized")
    
    def _register_builtin_collectors(self):
        """Register built-in system and application metrics"""
        
        # System metrics
        self.register_collector("cpu_usage", self._collect_cpu_usage)
        self.register_collector("memory_usage", self._collect_memory_usage)
        self.register_collector("disk_usage", self._collect_disk_usage)
        self.register_collector("network_io", self._collect_network_io)
        
        # Application metrics
        self.register_collector("research_discoveries_per_minute", self._collect_discovery_rate)
        self.register_collector("algorithm_execution_time", self._collect_algorithm_performance)
        self.register_collector("model_accuracy", self._collect_model_performance)
        self.register_collector("data_processing_throughput", self._collect_throughput)
    
    def register_collector(self, metric_name: str, collector_func: Callable[[], float]):
        """Register a custom metric collector"""
        self.collectors[metric_name] = collector_func
        logger.info(f"Registered collector for metric: {metric_name}")
    
    def start_collection(self):
        """Start background metric collection"""
        if self.running:
            logger.warning("Metrics collection already running")
            return
        
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        logger.info("Started metrics collection thread")
    
    def stop_collection(self):
        """Stop background metric collection"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)
        logger.info("Stopped metrics collection")
    
    def _collection_loop(self):
        """Main collection loop running in background thread"""
        while self.running:
            try:
                self._collect_all_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_all_metrics(self):
        """Collect all registered metrics"""
        timestamp = datetime.now()
        
        for metric_name, collector in self.collectors.items():
            try:
                value = collector()
                if value is not None:
                    self._record_metric(metric_name, value, timestamp)
            except Exception as e:
                logger.warning(f"Failed to collect {metric_name}: {e}")
    
    def _record_metric(self, metric_name: str, value: float, timestamp: datetime):
        """Record a metric value with anomaly detection"""
        
        # Store metric
        self.metrics[metric_name].append(MetricSnapshot(
            timestamp=timestamp,
            metric_name=metric_name,
            value=value
        ))
        
        # Update statistics
        self._update_metric_stats(metric_name, value)
        
        # Anomaly detection
        if self.anomaly_detection:
            self._check_for_anomalies(metric_name, value, timestamp)
    
    def _update_metric_stats(self, metric_name: str, value: float):
        """Update running statistics for a metric"""
        
        if metric_name not in self.metric_stats:
            self.metric_stats[metric_name] = {
                'count': 0,
                'sum': 0.0,
                'sum_squares': 0.0,
                'min': value,
                'max': value,
                'recent_values': deque(maxlen=100)
            }
        
        stats = self.metric_stats[metric_name]
        stats['count'] += 1
        stats['sum'] += value
        stats['sum_squares'] += value * value
        stats['min'] = min(stats['min'], value)
        stats['max'] = max(stats['max'], value)
        stats['recent_values'].append(value)
        
        # Calculate derived statistics
        stats['mean'] = stats['sum'] / stats['count']
        if stats['count'] > 1:
            variance = (stats['sum_squares'] - stats['sum'] * stats['mean']) / (stats['count'] - 1)
            stats['std'] = np.sqrt(max(0, variance))
        else:
            stats['std'] = 0.0
    
    def _check_for_anomalies(self, metric_name: str, value: float, timestamp: datetime):
        """Check for anomalies in metric values"""
        
        stats = self.metric_stats.get(metric_name, {})
        
        # Need sufficient data for anomaly detection
        if stats.get('count', 0) < 20:
            return
        
        mean = stats['mean']
        std = stats['std']
        
        # Statistical anomaly detection (3-sigma rule)
        if std > 0:
            z_score = abs(value - mean) / std
            if z_score > 3.0:
                logger.warning(f"Anomaly detected in {metric_name}: value={value:.3f}, z_score={z_score:.2f}")
                
                # Could trigger alerts here
                self._handle_anomaly(metric_name, value, z_score, timestamp)
    
    def _handle_anomaly(self, metric_name: str, value: float, z_score: float, timestamp: datetime):
        """Handle detected anomaly"""
        
        anomaly_info = {
            'metric': metric_name,
            'value': value,
            'z_score': z_score,
            'timestamp': timestamp.isoformat(),
            'severity': 'high' if z_score > 5.0 else 'medium'
        }
        
        # Log anomaly
        logger.warning(f"ANOMALY DETECTED: {json.dumps(anomaly_info, default=str)}")
    
    # Built-in metric collectors
    
    def _collect_cpu_usage(self) -> float:
        """Collect CPU usage percentage"""
        try:
            return psutil.cpu_percent(interval=0.1)
        except:
            return 0.0
    
    def _collect_memory_usage(self) -> float:
        """Collect memory usage percentage"""
        try:
            return psutil.virtual_memory().percent
        except:
            return 0.0
    
    def _collect_disk_usage(self) -> float:
        """Collect disk usage percentage"""
        try:
            return psutil.disk_usage('/').percent
        except:
            return 0.0
    
    def _collect_network_io(self) -> float:
        """Collect network I/O rate"""
        try:
            net_io = psutil.net_io_counters()
            return float(net_io.bytes_sent + net_io.bytes_recv)
        except:
            return 0.0
    
    def _collect_discovery_rate(self) -> float:
        """Collect research discovery rate per minute"""
        # Placeholder - would integrate with actual discovery system
        return float(np.random.poisson(2))  # Simulate 2 discoveries per minute on average
    
    def _collect_algorithm_performance(self) -> float:
        """Collect average algorithm execution time"""
        # Placeholder - would integrate with actual algorithm timing
        return float(np.random.exponential(0.5))  # Simulate execution times
    
    def _collect_model_performance(self) -> float:
        """Collect current model accuracy"""
        # Placeholder - would integrate with actual model evaluation
        return float(np.random.beta(8, 2))  # Simulate accuracy between 0-1
    
    def _collect_throughput(self) -> float:
        """Collect data processing throughput"""
        # Placeholder - would integrate with actual data processing
        return float(np.random.gamma(2, 100))  # Simulate throughput
    
    def get_metric_history(self, metric_name: str, hours: int = 1) -> List[MetricSnapshot]:
        """Get historical data for a specific metric"""
        
        if metric_name not in self.metrics:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            snapshot for snapshot in self.metrics[metric_name]
            if snapshot.timestamp >= cutoff_time
        ]
    
    def get_metric_summary(self, metric_name: str) -> Dict[str, Any]:
        """Get statistical summary for a metric"""
        
        if metric_name not in self.metric_stats:
            return {"error": f"No data for metric {metric_name}"}
        
        stats = self.metric_stats[metric_name].copy()
        
        # Add percentiles from recent values
        if len(stats.get('recent_values', [])) > 0:
            recent = list(stats['recent_values'])
            stats['p50'] = float(np.percentile(recent, 50))
            stats['p90'] = float(np.percentile(recent, 90))
            stats['p95'] = float(np.percentile(recent, 95))
            stats['p99'] = float(np.percentile(recent, 99))
        
        # Remove deque object which isn't JSON serializable
        if 'recent_values' in stats:
            del stats['recent_values']
        
        return stats
    
    def get_all_metrics_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary for all collected metrics"""
        
        return {
            metric_name: self.get_metric_summary(metric_name)
            for metric_name in self.metrics.keys()
        }


class IntelligentAlertSystem:
    """Intelligent alerting system with adaptive thresholds"""
    
    def __init__(self, 
                 metrics_collector: AdvancedMetricsCollector,
                 alert_handlers: Optional[List[Callable]] = None):
        
        self.metrics_collector = metrics_collector
        self.alert_handlers = alert_handlers or []
        
        # Alert management
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.last_alert_times: Dict[str, datetime] = {}
        
        # Monitoring thread
        self.monitoring_thread = None
        self.running = False
        
        # Load default alert rules
        self._load_default_alert_rules()
        
        logger.info("Intelligent alert system initialized")
    
    def _load_default_alert_rules(self):
        """Load default alert rules for common scenarios"""
        
        default_rules = [
            AlertRule("high_cpu", "cpu_usage", "gt", 90.0, window_minutes=2, severity="warning"),
            AlertRule("high_memory", "memory_usage", "gt", 85.0, window_minutes=3, severity="warning"),
            AlertRule("disk_full", "disk_usage", "gt", 95.0, window_minutes=1, severity="critical"),
            AlertRule("low_discovery_rate", "research_discoveries_per_minute", "lt", 0.5, window_minutes=10, severity="info"),
            AlertRule("slow_algorithms", "algorithm_execution_time", "gt", 5.0, window_minutes=5, severity="warning"),
            AlertRule("poor_model_accuracy", "model_accuracy", "lt", 0.6, window_minutes=15, severity="warning"),
            AlertRule("low_throughput", "data_processing_throughput", "lt", 50.0, window_minutes=10, severity="info")
        ]
        
        for rule in default_rules:
            self.add_alert_rule(rule)
    
    def add_alert_rule(self, rule: AlertRule):
        """Add or update an alert rule"""
        self.alert_rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule"""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add a custom alert handler"""
        self.alert_handlers.append(handler)
    
    def start_monitoring(self):
        """Start alert monitoring"""
        if self.running:
            logger.warning("Alert monitoring already running")
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Started alert monitoring")
    
    def stop_monitoring(self):
        """Stop alert monitoring"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Stopped alert monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                self._check_alert_conditions()
                self._resolve_stale_alerts()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Error in alert monitoring: {e}")
                time.sleep(30)
    
    def _check_alert_conditions(self):
        """Check all alert rule conditions"""
        
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            try:
                self._evaluate_alert_rule(rule)
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule_name}: {e}")
    
    def _evaluate_alert_rule(self, rule: AlertRule):
        """Evaluate a specific alert rule"""
        
        # Get recent metric data
        window_start = datetime.now() - timedelta(minutes=rule.window_minutes)
        
        if rule.metric_pattern not in self.metrics_collector.metrics:
            return  # Metric not available
        
        # Get values within the window
        recent_snapshots = [
            snapshot for snapshot in self.metrics_collector.metrics[rule.metric_pattern]
            if snapshot.timestamp >= window_start
        ]
        
        if len(recent_snapshots) < rule.min_occurrences:
            return  # Not enough data points
        
        recent_values = [s.value for s in recent_snapshots]
        
        # Evaluate condition
        alert_triggered = False
        
        if rule.condition == "gt":
            alert_triggered = sum(1 for v in recent_values if v > rule.threshold) >= rule.min_occurrences
        elif rule.condition == "lt":
            alert_triggered = sum(1 for v in recent_values if v < rule.threshold) >= rule.min_occurrences
        elif rule.condition == "eq":
            alert_triggered = sum(1 for v in recent_values if abs(v - rule.threshold) < 0.001) >= rule.min_occurrences
        elif rule.condition == "anomaly":
            # Use z-score based anomaly detection
            if len(recent_values) > 5:
                mean_val = np.mean(recent_values)
                std_val = np.std(recent_values)
                if std_val > 0:
                    z_scores = [abs(v - mean_val) / std_val for v in recent_values]
                    alert_triggered = any(z > rule.threshold for z in z_scores)
        
        # Handle alert state
        if alert_triggered:
            self._trigger_alert(rule, recent_values)
        else:
            self._resolve_alert_if_active(rule.name)
    
    def _trigger_alert(self, rule: AlertRule, metric_values: List[float]):
        """Trigger an alert"""
        
        # Check cooldown period
        if rule.name in self.last_alert_times:
            time_since_last = datetime.now() - self.last_alert_times[rule.name]
            if time_since_last.total_seconds() < rule.cooldown_minutes * 60:
                return  # Still in cooldown
        
        # Create alert
        alert_message = self._generate_alert_message(rule, metric_values)
        
        alert = Alert(
            rule_name=rule.name,
            message=alert_message,
            severity=rule.severity,
            triggered_at=datetime.now(),
            metric_values=metric_values.copy(),
            metadata={
                "rule_condition": rule.condition,
                "threshold": rule.threshold,
                "window_minutes": rule.window_minutes
            }
        )
        
        # Store alert
        self.active_alerts[rule.name] = alert
        self.alert_history.append(alert)
        self.last_alert_times[rule.name] = alert.triggered_at
        
        # Notify handlers
        self._notify_alert_handlers(alert)
        
        logger.warning(f"ALERT TRIGGERED: {alert_message}")
    
    def _resolve_alert_if_active(self, rule_name: str):
        """Resolve an alert if it's currently active"""
        
        if rule_name in self.active_alerts:
            alert = self.active_alerts[rule_name]
            alert.resolved_at = datetime.now()
            del self.active_alerts[rule_name]
            
            logger.info(f"ALERT RESOLVED: {rule_name}")
    
    def _resolve_stale_alerts(self):
        """Resolve alerts that have been active too long"""
        
        max_active_time = timedelta(hours=2)  # Auto-resolve after 2 hours
        current_time = datetime.now()
        
        stale_alerts = [
            rule_name for rule_name, alert in self.active_alerts.items()
            if current_time - alert.triggered_at > max_active_time
        ]
        
        for rule_name in stale_alerts:
            self._resolve_alert_if_active(rule_name)
            logger.info(f"Auto-resolved stale alert: {rule_name}")
    
    def _generate_alert_message(self, rule: AlertRule, metric_values: List[float]) -> str:
        """Generate human-readable alert message"""
        
        avg_value = np.mean(metric_values)
        
        if rule.condition == "gt":
            return f"{rule.metric_pattern} is HIGH: {avg_value:.2f} > {rule.threshold} (last {rule.window_minutes}min)"
        elif rule.condition == "lt":
            return f"{rule.metric_pattern} is LOW: {avg_value:.2f} < {rule.threshold} (last {rule.window_minutes}min)"
        elif rule.condition == "eq":
            return f"{rule.metric_pattern} equals threshold: {avg_value:.2f} ≈ {rule.threshold}"
        elif rule.condition == "anomaly":
            return f"{rule.metric_pattern} shows ANOMALOUS behavior: {avg_value:.2f} (threshold: {rule.threshold})"
        else:
            return f"{rule.metric_pattern} triggered alert condition: {rule.condition}"
    
    def _notify_alert_handlers(self, alert: Alert):
        """Notify all registered alert handlers"""
        
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all currently active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for specified time period"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            alert for alert in self.alert_history
            if alert.triggered_at >= cutoff_time
        ]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alerting system statistics"""
        
        total_rules = len(self.alert_rules)
        enabled_rules = sum(1 for rule in self.alert_rules.values() if rule.enabled)
        active_alerts = len(self.active_alerts)
        
        # Alert frequency by severity
        severity_counts = defaultdict(int)
        for alert in self.alert_history[-100:]:  # Last 100 alerts
            severity_counts[alert.severity] += 1
        
        return {
            "total_rules": total_rules,
            "enabled_rules": enabled_rules,
            "active_alerts": active_alerts,
            "total_alerts_24h": len(self.get_alert_history(24)),
            "severity_distribution": dict(severity_counts),
            "most_frequent_alerts": self._get_most_frequent_alerts()
        }
    
    def _get_most_frequent_alerts(self) -> List[Dict[str, Any]]:
        """Get most frequently triggered alert rules"""
        
        rule_counts = defaultdict(int)
        for alert in self.alert_history[-100:]:
            rule_counts[alert.rule_name] += 1
        
        sorted_rules = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"rule_name": rule_name, "count": count}
            for rule_name, count in sorted_rules[:5]
        ]


class ResearchHealthChecker(ValidationMixin):
    """Comprehensive health checker for research platform"""
    
    def __init__(self, 
                 metrics_collector: AdvancedMetricsCollector,
                 alert_system: IntelligentAlertSystem):
        
        self.metrics_collector = metrics_collector
        self.alert_system = alert_system
        
        # Health check components
        self.health_checks = {
            'system_resources': self._check_system_resources,
            'research_pipeline': self._check_research_pipeline,
            'data_quality': self._check_data_quality,
            'model_performance': self._check_model_performance,
            'discovery_rate': self._check_discovery_rate,
            'storage_capacity': self._check_storage_capacity,
            'network_connectivity': self._check_network_connectivity
        }
        
        logger.info("Research health checker initialized")
    
    @robust_execution(recovery_strategy='graceful_degradation')
    def run_comprehensive_health_check(self) -> Dict[str, Any]:
        """Run all health checks and return comprehensive report"""
        
        logger.info("Starting comprehensive health check")
        
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'HEALTHY',
            'component_status': {},
            'recommendations': [],
            'metrics_summary': {},
            'alert_status': {}
        }
        
        # Run individual health checks
        failed_checks = 0
        warning_checks = 0
        
        for check_name, check_function in self.health_checks.items():
            try:
                check_result = check_function()
                health_report['component_status'][check_name] = check_result
                
                if check_result['status'] == 'FAILED':
                    failed_checks += 1
                elif check_result['status'] == 'WARNING':
                    warning_checks += 1
                
                # Add recommendations
                if 'recommendations' in check_result:
                    health_report['recommendations'].extend(check_result['recommendations'])
                
            except Exception as e:
                logger.error(f"Health check {check_name} failed: {e}")
                health_report['component_status'][check_name] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
                failed_checks += 1
        
        # Determine overall status
        if failed_checks > 0:
            health_report['overall_status'] = 'CRITICAL' if failed_checks > 2 else 'DEGRADED'
        elif warning_checks > 0:
            health_report['overall_status'] = 'WARNING'
        
        # Add metrics summary
        health_report['metrics_summary'] = self.metrics_collector.get_all_metrics_summary()
        
        # Add alert status
        health_report['alert_status'] = {
            'active_alerts': len(self.alert_system.get_active_alerts()),
            'alert_statistics': self.alert_system.get_alert_statistics()
        }
        
        logger.info(f"Health check completed: {health_report['overall_status']}")
        
        return health_report
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource availability"""
        
        result = {
            'status': 'HEALTHY',
            'details': {},
            'recommendations': []
        }
        
        try:
            # CPU check
            cpu_usage = psutil.cpu_percent(interval=1)
            result['details']['cpu_usage'] = cpu_usage
            
            if cpu_usage > 90:
                result['status'] = 'CRITICAL'
                result['recommendations'].append("CPU usage is critically high - consider scaling resources")
            elif cpu_usage > 75:
                result['status'] = 'WARNING'
                result['recommendations'].append("CPU usage is high - monitor for sustained load")
            
            # Memory check
            memory = psutil.virtual_memory()
            result['details']['memory_usage'] = memory.percent
            result['details']['memory_available_gb'] = memory.available / (1024**3)
            
            if memory.percent > 90:
                result['status'] = 'CRITICAL'
                result['recommendations'].append("Memory usage is critically high - restart processes or scale")
            elif memory.percent > 80:
                result['status'] = 'WARNING'
                result['recommendations'].append("Memory usage is high - monitor memory leaks")
            
            # Disk check
            disk = psutil.disk_usage('/')
            result['details']['disk_usage'] = disk.percent
            result['details']['disk_free_gb'] = disk.free / (1024**3)
            
            if disk.percent > 95:
                result['status'] = 'CRITICAL'
                result['recommendations'].append("Disk space critically low - clean up or expand storage")
            elif disk.percent > 85:
                result['status'] = 'WARNING'
                result['recommendations'].append("Disk space getting low - plan for cleanup or expansion")
            
        except Exception as e:
            result['status'] = 'ERROR'
            result['error'] = str(e)
        
        return result
    
    def _check_research_pipeline(self) -> Dict[str, Any]:
        """Check research pipeline health"""
        
        result = {
            'status': 'HEALTHY',
            'details': {},
            'recommendations': []
        }
        
        try:
            # Check discovery rate
            discovery_history = self.metrics_collector.get_metric_history('research_discoveries_per_minute', hours=1)
            
            if discovery_history:
                recent_discovery_rate = np.mean([s.value for s in discovery_history])
                result['details']['discovery_rate'] = recent_discovery_rate
                
                if recent_discovery_rate < 0.5:
                    result['status'] = 'WARNING'
                    result['recommendations'].append("Discovery rate is low - check algorithm performance")
                elif recent_discovery_rate < 0.1:
                    result['status'] = 'CRITICAL'
                    result['recommendations'].append("Discovery rate is critically low - investigate pipeline issues")
            
            # Check algorithm performance
            algo_history = self.metrics_collector.get_metric_history('algorithm_execution_time', hours=1)
            
            if algo_history:
                avg_execution_time = np.mean([s.value for s in algo_history])
                result['details']['avg_algorithm_time'] = avg_execution_time
                
                if avg_execution_time > 10.0:
                    result['status'] = 'WARNING'
                    result['recommendations'].append("Algorithm execution time is high - optimize algorithms")
            
            # Check data processing throughput
            throughput_history = self.metrics_collector.get_metric_history('data_processing_throughput', hours=1)
            
            if throughput_history:
                avg_throughput = np.mean([s.value for s in throughput_history])
                result['details']['data_throughput'] = avg_throughput
                
                if avg_throughput < 50:
                    result['status'] = 'WARNING'
                    result['recommendations'].append("Data processing throughput is low - check data pipeline")
            
        except Exception as e:
            result['status'] = 'ERROR'
            result['error'] = str(e)
        
        return result
    
    def _check_data_quality(self) -> Dict[str, Any]:
        """Check data quality metrics"""
        
        result = {
            'status': 'HEALTHY',
            'details': {},
            'recommendations': []
        }
        
        try:
            # Simulate data quality checks
            # In real implementation, this would check actual data
            
            data_completeness = np.random.uniform(0.95, 1.0)
            data_accuracy = np.random.uniform(0.90, 1.0)
            data_consistency = np.random.uniform(0.92, 1.0)
            
            result['details']['data_completeness'] = data_completeness
            result['details']['data_accuracy'] = data_accuracy
            result['details']['data_consistency'] = data_consistency
            
            if any(metric < 0.90 for metric in [data_completeness, data_accuracy, data_consistency]):
                result['status'] = 'WARNING'
                result['recommendations'].append("Data quality metrics below threshold - review data sources")
            
            if any(metric < 0.80 for metric in [data_completeness, data_accuracy, data_consistency]):
                result['status'] = 'CRITICAL'
                result['recommendations'].append("Critical data quality issues - immediate attention required")
            
        except Exception as e:
            result['status'] = 'ERROR'
            result['error'] = str(e)
        
        return result
    
    def _check_model_performance(self) -> Dict[str, Any]:
        """Check model performance metrics"""
        
        result = {
            'status': 'HEALTHY',
            'details': {},
            'recommendations': []
        }
        
        try:
            # Check model accuracy history
            accuracy_history = self.metrics_collector.get_metric_history('model_accuracy', hours=2)
            
            if accuracy_history:
                recent_accuracy = np.mean([s.value for s in accuracy_history[-10:]])  # Last 10 measurements
                accuracy_trend = self._calculate_trend([s.value for s in accuracy_history])
                
                result['details']['current_accuracy'] = recent_accuracy
                result['details']['accuracy_trend'] = accuracy_trend
                
                if recent_accuracy < 0.6:
                    result['status'] = 'WARNING'
                    result['recommendations'].append("Model accuracy is below acceptable threshold - retrain models")
                
                if recent_accuracy < 0.4:
                    result['status'] = 'CRITICAL'
                    result['recommendations'].append("Model accuracy is critically low - immediate model intervention needed")
                
                if accuracy_trend < -0.1:
                    result['status'] = 'WARNING'
                    result['recommendations'].append("Model accuracy is declining - investigate model drift")
            
        except Exception as e:
            result['status'] = 'ERROR'
            result['error'] = str(e)
        
        return result
    
    def _check_discovery_rate(self) -> Dict[str, Any]:
        """Check scientific discovery rate"""
        
        result = {
            'status': 'HEALTHY',
            'details': {},
            'recommendations': []
        }
        
        try:
            discovery_history = self.metrics_collector.get_metric_history('research_discoveries_per_minute', hours=4)
            
            if discovery_history:
                # Calculate discovery rate statistics
                discovery_rates = [s.value for s in discovery_history]
                current_rate = np.mean(discovery_rates[-10:]) if len(discovery_rates) >= 10 else np.mean(discovery_rates)
                historical_rate = np.mean(discovery_rates)
                rate_trend = self._calculate_trend(discovery_rates)
                
                result['details']['current_rate'] = current_rate
                result['details']['historical_average'] = historical_rate
                result['details']['trend'] = rate_trend
                
                # Thresholds based on expected discovery rates
                if current_rate < 0.5:
                    result['status'] = 'WARNING'
                    result['recommendations'].append("Discovery rate below target - review research algorithms")
                
                if current_rate < 0.2:
                    result['status'] = 'CRITICAL'
                    result['recommendations'].append("Discovery rate critically low - research pipeline may be stalled")
                
                if rate_trend < -0.5:
                    result['status'] = 'WARNING'
                    result['recommendations'].append("Discovery rate declining - investigate research efficiency")
            
        except Exception as e:
            result['status'] = 'ERROR'
            result['error'] = str(e)
        
        return result
    
    def _check_storage_capacity(self) -> Dict[str, Any]:
        """Check storage capacity and data retention"""
        
        result = {
            'status': 'HEALTHY',
            'details': {},
            'recommendations': []
        }
        
        try:
            # Check disk usage
            disk = psutil.disk_usage('/')
            
            result['details']['total_gb'] = disk.total / (1024**3)
            result['details']['used_gb'] = disk.used / (1024**3)
            result['details']['free_gb'] = disk.free / (1024**3)
            result['details']['usage_percent'] = (disk.used / disk.total) * 100
            
            # Storage thresholds
            if disk.percent > 90:
                result['status'] = 'CRITICAL'
                result['recommendations'].append("Storage critically full - immediate cleanup or expansion needed")
            elif disk.percent > 80:
                result['status'] = 'WARNING'
                result['recommendations'].append("Storage usage high - plan cleanup or expansion")
            
            # Check if we can write to storage
            try:
                test_file = Path('/tmp/health_check_test.txt')
                test_file.write_text('health check test')
                test_file.unlink()
                result['details']['write_test'] = 'PASS'
            except:
                result['status'] = 'CRITICAL'
                result['details']['write_test'] = 'FAIL'
                result['recommendations'].append("Cannot write to storage - check permissions and disk health")
            
        except Exception as e:
            result['status'] = 'ERROR'
            result['error'] = str(e)
        
        return result
    
    def _check_network_connectivity(self) -> Dict[str, Any]:
        """Check network connectivity for external services"""
        
        result = {
            'status': 'HEALTHY',
            'details': {},
            'recommendations': []
        }
        
        try:
            # Check network interface statistics
            net_io = psutil.net_io_counters()
            
            result['details']['bytes_sent'] = net_io.bytes_sent
            result['details']['bytes_recv'] = net_io.bytes_recv
            result['details']['packets_sent'] = net_io.packets_sent
            result['details']['packets_recv'] = net_io.packets_recv
            
            # Check for network errors
            if net_io.errin > 0 or net_io.errout > 0:
                result['status'] = 'WARNING'
                result['recommendations'].append(f"Network errors detected: in={net_io.errin}, out={net_io.errout}")
            
            # Check network activity (basic heuristic)
            if net_io.bytes_sent + net_io.bytes_recv < 1000:  # Very low network activity
                result['status'] = 'WARNING'
                result['recommendations'].append("Very low network activity - check connectivity")
            
        except Exception as e:
            result['status'] = 'ERROR'
            result['error'] = str(e)
        
        return result
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in values (positive = increasing, negative = decreasing)"""
        
        if len(values) < 2:
            return 0.0
        
        # Simple linear regression slope
        n = len(values)
        x = np.arange(n)
        y = np.array(values)
        
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
        
        return float(slope)
    
    def get_health_summary(self) -> Dict[str, str]:
        """Get quick health summary"""
        
        health_report = self.run_comprehensive_health_check()
        
        return {
            'overall_status': health_report['overall_status'],
            'timestamp': health_report['timestamp'],
            'active_alerts': str(health_report['alert_status']['active_alerts']),
            'failed_components': str(sum(1 for status in health_report['component_status'].values() 
                                       if status.get('status') in ['FAILED', 'CRITICAL', 'ERROR']))
        }


def create_monitoring_system() -> Tuple[AdvancedMetricsCollector, IntelligentAlertSystem, ResearchHealthChecker]:
    """Factory function to create complete monitoring system"""
    
    # Create metrics collector
    metrics_collector = AdvancedMetricsCollector(
        retention_hours=24,
        collection_interval_seconds=10,
        anomaly_detection=True
    )
    
    # Create alert system with default handlers
    def log_alert_handler(alert: Alert):
        logger.warning(f"ALERT: {alert.message} (severity: {alert.severity})")
    
    def console_alert_handler(alert: Alert):
        print(f"🚨 {alert.severity.upper()}: {alert.message}")
    
    alert_system = IntelligentAlertSystem(
        metrics_collector=metrics_collector,
        alert_handlers=[log_alert_handler, console_alert_handler]
    )
    
    # Create health checker
    health_checker = ResearchHealthChecker(
        metrics_collector=metrics_collector,
        alert_system=alert_system
    )
    
    logger.info("Complete monitoring system created")
    
    return metrics_collector, alert_system, health_checker