"""Intelligent alert system with adaptive thresholds"""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import threading
from collections import deque, defaultdict
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertState(Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    condition: str  # Expression to evaluate
    severity: AlertSeverity
    threshold: float
    evaluation_window: float  # seconds
    cooldown_period: float  # seconds
    enabled: bool = True
    tags: Dict[str, str] = field(default_factory=dict)
    description: str = ""


@dataclass
class Alert:
    """Individual alert instance"""
    id: str
    rule_name: str
    severity: AlertSeverity
    state: AlertState
    message: str
    timestamp: float
    resolved_timestamp: Optional[float] = None
    acknowledged_by: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertChannel:
    """Alert notification channel"""
    name: str
    channel_type: str  # email, webhook, slack, etc.
    config: Dict[str, Any]
    severity_filter: List[AlertSeverity] = field(default_factory=list)
    enabled: bool = True


class AlertSystem:
    """Intelligent alert system with adaptive thresholds and smart notifications"""
    
    def __init__(self, evaluation_interval: float = 30.0):
        self.evaluation_interval = evaluation_interval
        
        # Alert management
        self.alert_rules = {}
        self.active_alerts = {}
        self.alert_history = deque(maxlen=10000)
        self.alert_channels = {}
        
        # Adaptive thresholds
        self.adaptive_thresholds = {}
        self.baseline_data = defaultdict(lambda: deque(maxlen=1000))
        
        # State management
        self.is_running = False
        self.evaluation_thread = None
        self.lock = threading.Lock()
        
        # Suppression and escalation
        self.suppression_rules = []
        self.escalation_rules = {}
        self.alert_counters = defaultdict(int)
        
        # Callbacks
        self.alert_callbacks = []
        
        logger.info("AlertSystem initialized")
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add an alert rule"""
        with self.lock:
            self.alert_rules[rule.name] = rule
            logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str) -> None:
        """Remove an alert rule"""
        with self.lock:
            if rule_name in self.alert_rules:
                del self.alert_rules[rule_name]
                logger.info(f"Removed alert rule: {rule_name}")
    
    def add_alert_channel(self, channel: AlertChannel) -> None:
        """Add an alert notification channel"""
        self.alert_channels[channel.name] = channel
        logger.info(f"Added alert channel: {channel.name}")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add callback for alert events"""
        self.alert_callbacks.append(callback)
    
    def start(self) -> None:
        """Start alert evaluation"""
        if self.is_running:
            logger.warning("Alert system already running")
            return
        
        self.is_running = True
        self.evaluation_thread = threading.Thread(target=self._evaluation_loop, daemon=True)
        self.evaluation_thread.start()
        
        logger.info("Alert system started")
    
    def stop(self) -> None:
        """Stop alert evaluation"""
        self.is_running = False
        if self.evaluation_thread:
            self.evaluation_thread.join(timeout=5.0)
        
        logger.info("Alert system stopped")
    
    def _evaluation_loop(self) -> None:
        """Main alert evaluation loop"""
        while self.is_running:
            try:
                self._evaluate_rules()
                self._update_adaptive_thresholds()
                self._check_escalations()
                time.sleep(self.evaluation_interval)
            except Exception as e:
                logger.error(f"Alert evaluation error: {e}")
                time.sleep(self.evaluation_interval)
    
    def _evaluate_rules(self) -> None:
        """Evaluate all alert rules"""
        current_time = time.time()
        
        with self.lock:
            for rule_name, rule in self.alert_rules.items():
                if not rule.enabled:
                    continue
                
                try:
                    # Check if rule should fire
                    should_alert = self._evaluate_rule_condition(rule)
                    
                    alert_id = f"{rule_name}_{int(current_time)}"
                    
                    if should_alert:
                        # Check cooldown
                        if self._is_in_cooldown(rule_name, current_time):
                            continue
                        
                        # Create alert
                        alert = self._create_alert(alert_id, rule, current_time)
                        self._fire_alert(alert)
                    
                    else:
                        # Check if we should resolve existing alerts
                        self._check_alert_resolution(rule_name, current_time)
                
                except Exception as e:
                    logger.error(f"Error evaluating rule {rule_name}: {e}")
    
    def _evaluate_rule_condition(self, rule: AlertRule) -> bool:
        """Evaluate if alert rule condition is met"""
        # This is a simplified implementation
        # In a real system, you'd have a more sophisticated expression evaluator
        
        condition = rule.condition.lower()
        threshold = rule.threshold
        
        # Get recent metric data (would be injected from metrics collector)
        metric_name = self._extract_metric_name(condition)
        recent_values = self._get_recent_metric_values(metric_name, rule.evaluation_window)
        
        if not recent_values:
            return False
        
        current_value = recent_values[-1]
        
        # Simple condition evaluation
        if ">" in condition:
            return current_value > threshold
        elif "<" in condition:
            return current_value < threshold
        elif "avg" in condition:
            return np.mean(recent_values) > threshold
        elif "max" in condition:
            return np.max(recent_values) > threshold
        elif "spike" in condition:
            if len(recent_values) > 5:
                recent_avg = np.mean(recent_values[-5:])
                baseline_avg = np.mean(recent_values[:-5]) if len(recent_values) > 10 else recent_avg
                return (recent_avg - baseline_avg) / (baseline_avg + 1e-10) > threshold
        
        return False
    
    def _extract_metric_name(self, condition: str) -> str:
        """Extract metric name from condition string"""
        # Simple extraction - in reality would use proper parsing
        words = condition.split()
        for word in words:
            if "." in word or "_" in word:
                return word
        return "unknown_metric"
    
    def _get_recent_metric_values(self, metric_name: str, window_seconds: float) -> List[float]:
        """Get recent values for a metric (would integrate with MetricsCollector)"""
        # Placeholder implementation
        # In real system, would query MetricsCollector
        if metric_name in self.baseline_data:
            cutoff_time = time.time() - window_seconds
            return [v for v, t in self.baseline_data[metric_name] if t >= cutoff_time]
        return []
    
    def _is_in_cooldown(self, rule_name: str, current_time: float) -> bool:
        """Check if rule is in cooldown period"""
        for alert in reversed(self.alert_history):
            if (alert.rule_name == rule_name and 
                alert.state == AlertState.ACTIVE and
                current_time - alert.timestamp < self.alert_rules[rule_name].cooldown_period):
                return True
        return False
    
    def _create_alert(self, alert_id: str, rule: AlertRule, timestamp: float) -> Alert:
        """Create a new alert"""
        message = f"Alert: {rule.name} - {rule.description}"
        
        alert = Alert(
            id=alert_id,
            rule_name=rule.name,
            severity=rule.severity,
            state=AlertState.ACTIVE,
            message=message,
            timestamp=timestamp,
            tags=rule.tags.copy(),
            metadata={"threshold": rule.threshold, "condition": rule.condition}
        )
        
        return alert
    
    def _fire_alert(self, alert: Alert) -> None:
        """Fire an alert and send notifications"""
        with self.lock:
            self.active_alerts[alert.id] = alert
            self.alert_history.append(alert)
            self.alert_counters[alert.rule_name] += 1
        
        logger.warning(f"Alert fired: {alert.message}")
        
        # Send notifications
        self._send_notifications(alert)
        
        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def _send_notifications(self, alert: Alert) -> None:
        """Send alert notifications through configured channels"""
        for channel_name, channel in self.alert_channels.items():
            if not channel.enabled:
                continue
            
            # Check severity filter
            if channel.severity_filter and alert.severity not in channel.severity_filter:
                continue
            
            try:
                if channel.channel_type == "email":
                    self._send_email_notification(alert, channel)
                elif channel.channel_type == "webhook":
                    self._send_webhook_notification(alert, channel)
                elif channel.channel_type == "slack":
                    self._send_slack_notification(alert, channel)
                
            except Exception as e:
                logger.error(f"Failed to send notification via {channel_name}: {e}")
    
    def _send_email_notification(self, alert: Alert, channel: AlertChannel) -> None:
        """Send email notification"""
        config = channel.config
        
        msg = MimeMultipart()
        msg['From'] = config.get('from_email', 'alerts@ai-science-platform.com')
        msg['To'] = config.get('to_email', 'admin@ai-science-platform.com')
        msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.rule_name}"
        
        body = f"""
        Alert Details:
        - Rule: {alert.rule_name}
        - Severity: {alert.severity.value}
        - Message: {alert.message}
        - Timestamp: {time.ctime(alert.timestamp)}
        - Alert ID: {alert.id}
        
        Metadata: {alert.metadata}
        Tags: {alert.tags}
        """
        
        msg.attach(MimeText(body, 'plain'))
        
        # Would implement actual SMTP sending here
        logger.info(f"Email notification prepared for alert {alert.id}")
    
    def _send_webhook_notification(self, alert: Alert, channel: AlertChannel) -> None:
        """Send webhook notification"""
        # Would implement HTTP POST to webhook URL
        logger.info(f"Webhook notification prepared for alert {alert.id}")
    
    def _send_slack_notification(self, alert: Alert, channel: AlertChannel) -> None:
        """Send Slack notification"""
        # Would implement Slack API call
        logger.info(f"Slack notification prepared for alert {alert.id}")
    
    def _check_alert_resolution(self, rule_name: str, current_time: float) -> None:
        """Check if alerts should be automatically resolved"""
        alerts_to_resolve = []
        
        for alert_id, alert in self.active_alerts.items():
            if alert.rule_name == rule_name and alert.state == AlertState.ACTIVE:
                # Check if condition is no longer met
                rule = self.alert_rules[rule_name]
                if not self._evaluate_rule_condition(rule):
                    alerts_to_resolve.append(alert_id)
        
        for alert_id in alerts_to_resolve:
            self.resolve_alert(alert_id, "auto-resolved")
    
    def _update_adaptive_thresholds(self) -> None:
        """Update adaptive thresholds based on historical data"""
        current_time = time.time()
        
        for rule_name, rule in self.alert_rules.items():
            if rule.condition not in self.adaptive_thresholds:
                continue
            
            metric_name = self._extract_metric_name(rule.condition)
            recent_values = self._get_recent_metric_values(metric_name, 3600)  # Last hour
            
            if len(recent_values) < 30:
                continue  # Need sufficient data
            
            # Calculate adaptive threshold
            values_array = np.array(recent_values)
            mean = np.mean(values_array)
            std = np.std(values_array)
            
            # Set threshold at mean + 2 * std for anomaly detection
            adaptive_threshold = mean + 2 * std
            
            # Update rule threshold if significantly different
            if abs(adaptive_threshold - rule.threshold) / rule.threshold > 0.2:
                logger.info(f"Updating adaptive threshold for {rule_name}: {rule.threshold:.3f} -> {adaptive_threshold:.3f}")
                rule.threshold = adaptive_threshold
    
    def _check_escalations(self) -> None:
        """Check for alert escalations"""
        current_time = time.time()
        
        for rule_name, escalation_config in self.escalation_rules.items():
            alert_count = sum(1 for alert in self.active_alerts.values() 
                            if alert.rule_name == rule_name and alert.state == AlertState.ACTIVE)
            
            escalation_threshold = escalation_config.get("count_threshold", 5)
            escalation_window = escalation_config.get("time_window", 300)  # 5 minutes
            
            if alert_count >= escalation_threshold:
                # Create escalation alert
                escalation_alert = Alert(
                    id=f"escalation_{rule_name}_{int(current_time)}",
                    rule_name=f"escalation_{rule_name}",
                    severity=AlertSeverity.EMERGENCY,
                    state=AlertState.ACTIVE,
                    message=f"Alert escalation: {alert_count} {rule_name} alerts in {escalation_window}s",
                    timestamp=current_time,
                    metadata={"escalated_rule": rule_name, "alert_count": alert_count}
                )
                
                self._fire_alert(escalation_alert)
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        with self.lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.state = AlertState.ACKNOWLEDGED
                alert.acknowledged_by = acknowledged_by
                logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                return True
        return False
    
    def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Resolve an alert"""
        with self.lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.state = AlertState.RESOLVED
                alert.resolved_timestamp = time.time()
                
                # Move to history
                del self.active_alerts[alert_id]
                
                logger.info(f"Alert {alert_id} resolved by {resolved_by}")
                return True
        return False
    
    def suppress_alert(self, alert_id: str, duration_seconds: float) -> bool:
        """Temporarily suppress an alert"""
        with self.lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.state = AlertState.SUPPRESSED
                
                # Schedule unsuppression
                def unsuppress():
                    time.sleep(duration_seconds)
                    if alert_id in self.active_alerts:
                        self.active_alerts[alert_id].state = AlertState.ACTIVE
                
                threading.Thread(target=unsuppress, daemon=True).start()
                
                logger.info(f"Alert {alert_id} suppressed for {duration_seconds}s")
                return True
        return False
    
    def get_active_alerts(self, severity_filter: Optional[List[AlertSeverity]] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity"""
        alerts = list(self.active_alerts.values())
        
        if severity_filter:
            alerts = [a for a in alerts if a.severity in severity_filter]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_history(self, hours: float = 24.0) -> List[Alert]:
        """Get alert history for specified time period"""
        cutoff_time = time.time() - (hours * 3600)
        return [a for a in self.alert_history if a.timestamp >= cutoff_time]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert system statistics"""
        current_time = time.time()
        recent_alerts = self.get_alert_history(24.0)
        
        stats = {
            "active_alerts": len(self.active_alerts),
            "total_rules": len(self.alert_rules),
            "enabled_rules": sum(1 for rule in self.alert_rules.values() if rule.enabled),
            "alert_channels": len(self.alert_channels),
            "alerts_last_24h": len(recent_alerts),
            "severity_breakdown": defaultdict(int),
            "top_alerting_rules": {},
            "resolution_rate": 0.0
        }
        
        # Severity breakdown
        for alert in recent_alerts:
            stats["severity_breakdown"][alert.severity.value] += 1
        
        # Top alerting rules
        rule_counts = defaultdict(int)
        for alert in recent_alerts:
            rule_counts[alert.rule_name] += 1
        
        stats["top_alerting_rules"] = dict(sorted(rule_counts.items(), 
                                                key=lambda x: x[1], reverse=True)[:5])
        
        # Resolution rate
        resolved_count = sum(1 for alert in recent_alerts if alert.state == AlertState.RESOLVED)
        if recent_alerts:
            stats["resolution_rate"] = resolved_count / len(recent_alerts)
        
        return dict(stats)
    
    def export_alerts(self, filepath: str) -> None:
        """Export alert data to file"""
        import json
        
        export_data = {
            "export_timestamp": time.time(),
            "active_alerts": [asdict(alert) for alert in self.active_alerts.values()],
            "alert_history": [asdict(alert) for alert in self.alert_history],
            "alert_rules": {name: {
                "name": rule.name,
                "condition": rule.condition,
                "severity": rule.severity.value,
                "threshold": rule.threshold,
                "enabled": rule.enabled,
                "tags": rule.tags,
                "description": rule.description
            } for name, rule in self.alert_rules.items()},
            "statistics": self.get_alert_statistics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Alert data exported to {filepath}")
    
    def create_default_rules(self) -> None:
        """Create default alert rules for the AI Science Platform"""
        default_rules = [
            AlertRule(
                name="high_cpu_usage",
                condition="cpu_usage > threshold",
                severity=AlertSeverity.WARNING,
                threshold=80.0,
                evaluation_window=300,
                cooldown_period=600,
                description="CPU usage is high"
            ),
            AlertRule(
                name="high_memory_usage",
                condition="memory_usage > threshold",
                severity=AlertSeverity.WARNING,
                threshold=85.0,
                evaluation_window=300,
                cooldown_period=600,
                description="Memory usage is high"
            ),
            AlertRule(
                name="discovery_failures",
                condition="discovery_error_rate > threshold",
                severity=AlertSeverity.CRITICAL,
                threshold=0.1,
                evaluation_window=600,
                cooldown_period=300,
                description="High discovery failure rate"
            ),
            AlertRule(
                name="slow_processing",
                condition="avg_processing_time > threshold",
                severity=AlertSeverity.WARNING,
                threshold=5.0,
                evaluation_window=300,
                cooldown_period=600,
                description="Processing time is slow"
            )
        ]
        
        for rule in default_rules:
            self.add_alert_rule(rule)
        
        logger.info("Default alert rules created")