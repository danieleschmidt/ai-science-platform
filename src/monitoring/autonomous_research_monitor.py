"""
Autonomous Research Monitoring System
Real-time monitoring, alerting, and performance analytics for autonomous research operations

This module provides comprehensive monitoring capabilities for:
- Research process health and performance
- Discovery quality metrics
- Resource utilization tracking
- Anomaly detection and alerting
- Research velocity optimization
- Breakthrough prediction analytics
"""

import time
import logging
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import json
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ResearchMetrics:
    """Comprehensive research performance metrics"""
    timestamp: float
    
    # Core metrics
    hypotheses_generated_rate: float = 0.0
    breakthrough_discovery_rate: float = 0.0
    publication_readiness_rate: float = 0.0
    reproducibility_success_rate: float = 0.0
    
    # Quality metrics
    average_novelty_score: float = 0.0
    average_confidence_score: float = 0.0
    average_significance_score: float = 0.0
    cross_domain_connectivity: float = 0.0
    
    # Performance metrics
    computational_efficiency: float = 0.0
    memory_utilization: float = 0.0
    algorithm_convergence_rate: float = 0.0
    research_velocity: float = 0.0
    
    # Health metrics
    system_health_score: float = 1.0
    error_rate: float = 0.0
    success_rate: float = 1.0
    uptime_percentage: float = 100.0


@dataclass
class Alert:
    """Research monitoring alert"""
    alert_id: str
    severity: str  # 'info', 'warning', 'critical'
    component: str
    message: str
    timestamp: float
    resolved: bool = False
    resolution_time: Optional[float] = None


@dataclass
class AnomalyDetection:
    """Anomaly detection result"""
    metric_name: str
    current_value: float
    expected_value: float
    deviation_score: float
    is_anomaly: bool
    anomaly_type: str  # 'spike', 'drop', 'drift', 'oscillation'
    confidence: float


class AutonomousResearchMonitor:
    """
    Comprehensive monitoring system for autonomous research operations
    
    Features:
    - Real-time metrics collection and analysis
    - Anomaly detection using statistical methods
    - Performance optimization recommendations
    - Predictive analytics for breakthrough timing
    - Resource utilization tracking
    - Health monitoring and alerting
    """
    
    def __init__(self, 
                 metrics_retention_hours: int = 168,  # 1 week
                 alert_retention_hours: int = 720,    # 30 days
                 anomaly_sensitivity: float = 2.0,
                 monitoring_interval: float = 60.0):  # 1 minute
        
        self.metrics_retention_hours = metrics_retention_hours
        self.alert_retention_hours = alert_retention_hours
        self.anomaly_sensitivity = anomaly_sensitivity
        self.monitoring_interval = monitoring_interval
        
        # Data storage
        self.metrics_history = deque(maxlen=int(metrics_retention_hours * 60))
        self.alerts = deque(maxlen=int(alert_retention_hours * 6))  # 10-minute intervals
        self.anomalies_detected = deque(maxlen=1000)
        
        # Monitoring state
        self.is_monitoring = False
        self.start_time = time.time()
        self.last_metrics_update = 0.0
        self.monitoring_task = None
        
        # Performance tracking
        self.performance_baselines = {}
        self.trend_analyzers = {}
        self.predictive_models = {}
        
        # Alert handlers
        self.alert_handlers = []
        
        # Statistics for anomaly detection
        self.metric_statistics = defaultdict(lambda: {
            'history': deque(maxlen=1000),
            'mean': 0.0,
            'std': 0.1,
            'trend': 0.0
        })
        
        logger.info("AutonomousResearchMonitor initialized")
    
    async def start_monitoring(self):
        """Start continuous monitoring process"""
        if self.is_monitoring:
            logger.warning("Monitoring already active")
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("🔍 Autonomous research monitoring started")
        
        # Initial system health check
        await self._perform_system_health_check()
    
    async def stop_monitoring(self):
        """Stop monitoring process"""
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Autonomous research monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        try:
            while self.is_monitoring:
                cycle_start = time.time()
                
                # Collect metrics
                current_metrics = await self._collect_metrics()
                
                # Store metrics
                if current_metrics:
                    self.metrics_history.append(current_metrics)
                    self.last_metrics_update = time.time()
                    
                    # Analyze metrics for anomalies
                    await self._analyze_for_anomalies(current_metrics)
                    
                    # Update performance baselines
                    await self._update_baselines(current_metrics)
                    
                    # Generate alerts if needed
                    await self._check_alert_conditions(current_metrics)
                    
                    # Predictive analysis
                    await self._perform_predictive_analysis(current_metrics)
                
                # Cleanup old data
                await self._cleanup_old_data()
                
                # Sleep until next monitoring interval
                cycle_time = time.time() - cycle_start
                sleep_time = max(0, self.monitoring_interval - cycle_time)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    logger.warning(f"Monitoring cycle took {cycle_time:.2f}s, longer than interval {self.monitoring_interval}s")
                    
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            await self._generate_alert('critical', 'monitor', f"Monitoring loop error: {e}")
    
    async def _collect_metrics(self) -> Optional[ResearchMetrics]:
        """Collect current research metrics"""
        try:
            current_time = time.time()
            
            # Simulate metric collection (would integrate with actual research engine)
            metrics = ResearchMetrics(
                timestamp=current_time,
                hypotheses_generated_rate=self._simulate_metric('hypotheses_rate', 0.5, 1.5),
                breakthrough_discovery_rate=self._simulate_metric('breakthrough_rate', 0.1, 0.3),
                publication_readiness_rate=self._simulate_metric('publication_rate', 0.7, 0.95),
                reproducibility_success_rate=self._simulate_metric('reproducibility_rate', 0.8, 0.95),
                
                average_novelty_score=self._simulate_metric('novelty_score', 0.6, 0.9),
                average_confidence_score=self._simulate_metric('confidence_score', 0.7, 0.95),
                average_significance_score=self._simulate_metric('significance_score', 0.75, 0.98),
                cross_domain_connectivity=self._simulate_metric('cross_domain', 0.3, 0.8),
                
                computational_efficiency=self._simulate_metric('comp_efficiency', 0.6, 0.9),
                memory_utilization=self._simulate_metric('memory_util', 0.3, 0.8),
                algorithm_convergence_rate=self._simulate_metric('convergence_rate', 0.7, 0.95),
                research_velocity=self._simulate_metric('research_velocity', 0.5, 2.0),
                
                system_health_score=self._simulate_metric('health_score', 0.95, 1.0),
                error_rate=self._simulate_metric('error_rate', 0.0, 0.05),
                success_rate=self._simulate_metric('success_rate', 0.9, 1.0),
                uptime_percentage=self._simulate_metric('uptime', 95.0, 100.0)
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            await self._generate_alert('warning', 'metrics', f"Metrics collection failed: {e}")
            return None
    
    def _simulate_metric(self, metric_name: str, min_val: float, max_val: float) -> float:
        """Simulate realistic metric values with trends and noise"""
        
        stats = self.metric_statistics[metric_name]
        
        # Generate base value with trend
        if not stats['history']:
            # First time - random initialization
            base_value = np.random.uniform(min_val, max_val)
        else:
            # Continue trend with some noise
            last_value = stats['history'][-1]
            trend_component = stats['trend'] * 0.01  # 1% trend per cycle
            noise_component = np.random.normal(0, (max_val - min_val) * 0.02)  # 2% noise
            
            base_value = last_value + trend_component + noise_component
        
        # Keep within bounds
        base_value = np.clip(base_value, min_val, max_val)
        
        # Add occasional anomalies (5% chance)
        if np.random.random() < 0.05:
            anomaly_factor = np.random.choice([-1, 1]) * np.random.uniform(0.1, 0.3)
            base_value = np.clip(base_value * (1 + anomaly_factor), min_val, max_val)
        
        # Update statistics
        stats['history'].append(base_value)
        if len(stats['history']) >= 10:
            stats['mean'] = np.mean(list(stats['history'])[-50:])  # Recent mean
            stats['std'] = np.std(list(stats['history'])[-50:])   # Recent std
            
            # Simple trend calculation
            if len(stats['history']) >= 20:
                recent_values = list(stats['history'])[-20:]
                x = np.arange(len(recent_values))
                trend_slope = np.polyfit(x, recent_values, 1)[0]
                stats['trend'] = trend_slope
        
        return float(base_value)
    
    async def _analyze_for_anomalies(self, metrics: ResearchMetrics):
        """Analyze metrics for anomalous patterns"""
        
        current_time = time.time()
        metric_dict = {
            'hypotheses_generated_rate': metrics.hypotheses_generated_rate,
            'breakthrough_discovery_rate': metrics.breakthrough_discovery_rate,
            'publication_readiness_rate': metrics.publication_readiness_rate,
            'reproducibility_success_rate': metrics.reproducibility_success_rate,
            'average_novelty_score': metrics.average_novelty_score,
            'average_confidence_score': metrics.average_confidence_score,
            'average_significance_score': metrics.average_significance_score,
            'cross_domain_connectivity': metrics.cross_domain_connectivity,
            'computational_efficiency': metrics.computational_efficiency,
            'memory_utilization': metrics.memory_utilization,
            'algorithm_convergence_rate': metrics.algorithm_convergence_rate,
            'research_velocity': metrics.research_velocity,
            'system_health_score': metrics.system_health_score,
            'error_rate': metrics.error_rate,
            'success_rate': metrics.success_rate,
            'uptime_percentage': metrics.uptime_percentage
        }
        
        for metric_name, current_value in metric_dict.items():
            anomaly = await self._detect_metric_anomaly(metric_name, current_value)
            
            if anomaly and anomaly.is_anomaly:
                self.anomalies_detected.append(anomaly)
                
                # Generate alert for significant anomalies
                if anomaly.confidence > 0.8:
                    severity = 'critical' if anomaly.confidence > 0.95 else 'warning'
                    await self._generate_alert(
                        severity, 
                        'anomaly_detection',
                        f"Anomaly detected in {metric_name}: {anomaly.anomaly_type} "
                        f"(current: {current_value:.3f}, expected: {anomaly.expected_value:.3f})"
                    )
    
    async def _detect_metric_anomaly(self, metric_name: str, current_value: float) -> Optional[AnomalyDetection]:
        """Detect anomalies in a specific metric"""
        
        stats = self.metric_statistics[metric_name]
        
        if len(stats['history']) < 20:  # Need history for anomaly detection
            return None
        
        expected_value = stats['mean']
        deviation = abs(current_value - expected_value)
        std_dev = max(stats['std'], 0.01)  # Avoid division by zero
        
        # Z-score based anomaly detection
        z_score = deviation / std_dev
        is_anomaly = z_score > self.anomaly_sensitivity
        
        # Determine anomaly type
        anomaly_type = 'normal'
        if is_anomaly:
            if current_value > expected_value:
                if z_score > 3.0:
                    anomaly_type = 'spike'
                else:
                    anomaly_type = 'elevation'
            else:
                if z_score > 3.0:
                    anomaly_type = 'drop'
                else:
                    anomaly_type = 'depression'
            
            # Check for trend anomalies
            recent_trend = abs(stats['trend'])
            if recent_trend > std_dev * 0.1:  # Significant trend
                anomaly_type = 'drift'
        
        # Calculate confidence based on z-score
        confidence = min(1.0, z_score / 4.0)  # Normalize to 0-1
        
        return AnomalyDetection(
            metric_name=metric_name,
            current_value=current_value,
            expected_value=expected_value,
            deviation_score=z_score,
            is_anomaly=is_anomaly,
            anomaly_type=anomaly_type,
            confidence=confidence
        )
    
    async def _update_baselines(self, metrics: ResearchMetrics):
        """Update performance baselines based on recent metrics"""
        
        if len(self.metrics_history) < 100:  # Need sufficient history
            return
        
        # Calculate baselines from recent history
        recent_metrics = list(self.metrics_history)[-100:]  # Last 100 measurements
        
        baseline_metrics = {
            'hypotheses_generated_rate': np.mean([m.hypotheses_generated_rate for m in recent_metrics]),
            'breakthrough_discovery_rate': np.mean([m.breakthrough_discovery_rate for m in recent_metrics]),
            'publication_readiness_rate': np.mean([m.publication_readiness_rate for m in recent_metrics]),
            'average_novelty_score': np.mean([m.average_novelty_score for m in recent_metrics]),
            'research_velocity': np.mean([m.research_velocity for m in recent_metrics]),
            'system_health_score': np.mean([m.system_health_score for m in recent_metrics])
        }
        
        # Update baselines with exponential smoothing
        alpha = 0.1  # Smoothing factor
        for metric_name, new_baseline in baseline_metrics.items():
            if metric_name in self.performance_baselines:
                self.performance_baselines[metric_name] = (
                    alpha * new_baseline + 
                    (1 - alpha) * self.performance_baselines[metric_name]
                )
            else:
                self.performance_baselines[metric_name] = new_baseline
    
    async def _check_alert_conditions(self, metrics: ResearchMetrics):
        """Check for conditions that require alerts"""
        
        # Critical system health
        if metrics.system_health_score < 0.8:
            await self._generate_alert(
                'critical', 'system_health',
                f"System health critically low: {metrics.system_health_score:.3f}"
            )
        
        # High error rate
        if metrics.error_rate > 0.1:
            await self._generate_alert(
                'warning', 'error_rate',
                f"High error rate detected: {metrics.error_rate:.3f}"
            )
        
        # Low success rate
        if metrics.success_rate < 0.7:
            await self._generate_alert(
                'warning', 'success_rate',
                f"Success rate below threshold: {metrics.success_rate:.3f}"
            )
        
        # Memory utilization warning
        if metrics.memory_utilization > 0.9:
            await self._generate_alert(
                'warning', 'resource_usage',
                f"High memory utilization: {metrics.memory_utilization:.3f}"
            )
        
        # Research velocity drop
        if metrics.research_velocity < 0.3 and len(self.performance_baselines) > 0:
            baseline_velocity = self.performance_baselines.get('research_velocity', 1.0)
            if metrics.research_velocity < baseline_velocity * 0.5:
                await self._generate_alert(
                    'warning', 'performance',
                    f"Research velocity significantly below baseline: {metrics.research_velocity:.3f} vs {baseline_velocity:.3f}"
                )
        
        # Breakthrough discovery rate drop
        if metrics.breakthrough_discovery_rate < 0.05:
            await self._generate_alert(
                'info', 'discovery_rate',
                f"Low breakthrough discovery rate: {metrics.breakthrough_discovery_rate:.3f}"
            )
        
        # Publication readiness below expectations
        if metrics.publication_readiness_rate < 0.6:
            await self._generate_alert(
                'warning', 'publication_quality',
                f"Publication readiness rate low: {metrics.publication_readiness_rate:.3f}"
            )
    
    async def _perform_predictive_analysis(self, metrics: ResearchMetrics):
        """Perform predictive analysis for research optimization"""
        
        if len(self.metrics_history) < 50:  # Need sufficient history
            return
        
        # Predict breakthrough timing
        await self._predict_breakthrough_timing()
        
        # Analyze research velocity trends
        await self._analyze_velocity_trends()
        
        # Resource utilization forecasting
        await self._forecast_resource_needs(metrics)
    
    async def _predict_breakthrough_timing(self):
        """Predict when next breakthrough is likely to occur"""
        
        try:
            recent_metrics = list(self.metrics_history)[-50:]
            discovery_rates = [m.breakthrough_discovery_rate for m in recent_metrics]
            
            if not discovery_rates or all(rate == 0 for rate in discovery_rates):
                return
            
            # Simple trend analysis
            x = np.arange(len(discovery_rates))
            if len(set(discovery_rates)) > 1:  # Check for variance
                trend_slope = np.polyfit(x, discovery_rates, 1)[0]
                current_rate = discovery_rates[-1]
                
                if current_rate > 0:
                    # Estimate time to next breakthrough (hours)
                    time_to_breakthrough = 1.0 / current_rate if current_rate > 0.01 else 100.0
                    
                    # Adjust based on trend
                    if trend_slope > 0:
                        time_to_breakthrough *= 0.8  # Improving trend
                    elif trend_slope < -0.001:
                        time_to_breakthrough *= 1.3  # Declining trend
                    
                    # Update predictive model
                    self.predictive_models['breakthrough_timing'] = {
                        'predicted_hours': time_to_breakthrough,
                        'confidence': min(0.8, len(recent_metrics) / 50.0),
                        'trend_slope': trend_slope,
                        'current_rate': current_rate,
                        'updated': time.time()
                    }
                    
                    # Generate info alert for predictions
                    if time_to_breakthrough < 2.0:  # Within 2 hours
                        await self._generate_alert(
                            'info', 'prediction',
                            f"Breakthrough predicted within {time_to_breakthrough:.1f} hours"
                        )
        
        except Exception as e:
            logger.debug(f"Error in breakthrough prediction: {e}")
    
    async def _analyze_velocity_trends(self):
        """Analyze research velocity trends"""
        
        try:
            recent_metrics = list(self.metrics_history)[-30:]
            velocities = [m.research_velocity for m in recent_metrics]
            
            if len(velocities) < 10:
                return
            
            # Calculate trend and variability
            x = np.arange(len(velocities))
            trend_slope = np.polyfit(x, velocities, 1)[0]
            velocity_std = np.std(velocities)
            velocity_mean = np.mean(velocities)
            
            # Analyze patterns
            trend_analysis = {
                'trend_slope': trend_slope,
                'velocity_mean': velocity_mean,
                'velocity_std': velocity_std,
                'coefficient_of_variation': velocity_std / velocity_mean if velocity_mean > 0 else 0,
                'trend_strength': abs(trend_slope) / velocity_std if velocity_std > 0 else 0
            }
            
            # Store in trend analyzers
            self.trend_analyzers['research_velocity'] = trend_analysis
            
            # Generate insights
            if trend_analysis['trend_strength'] > 0.5:
                trend_direction = 'increasing' if trend_slope > 0 else 'decreasing'
                await self._generate_alert(
                    'info', 'trend_analysis',
                    f"Strong {trend_direction} trend in research velocity detected"
                )
        
        except Exception as e:
            logger.debug(f"Error in velocity trend analysis: {e}")
    
    async def _forecast_resource_needs(self, metrics: ResearchMetrics):
        """Forecast future resource requirements"""
        
        try:
            # Analyze memory utilization trend
            recent_metrics = list(self.metrics_history)[-20:]
            memory_utils = [m.memory_utilization for m in recent_metrics]
            
            if len(memory_utils) >= 5:
                x = np.arange(len(memory_utils))
                trend_slope = np.polyfit(x, memory_utils, 1)[0]
                
                current_memory = metrics.memory_utilization
                
                # Predict memory usage in next hour (assuming 1-minute intervals)
                predicted_memory = current_memory + (trend_slope * 60)
                
                # Alert if approaching capacity
                if predicted_memory > 0.85:
                    await self._generate_alert(
                        'warning', 'resource_forecast',
                        f"Memory utilization may reach {predicted_memory:.3f} within 1 hour"
                    )
                
                # Store forecast
                self.predictive_models['memory_forecast'] = {
                    'predicted_utilization': predicted_memory,
                    'trend_slope': trend_slope,
                    'current_utilization': current_memory,
                    'forecast_horizon_hours': 1.0,
                    'updated': time.time()
                }
        
        except Exception as e:
            logger.debug(f"Error in resource forecasting: {e}")
    
    async def _perform_system_health_check(self):
        """Perform comprehensive system health check"""
        
        health_checks = {
            'monitoring_system': await self._check_monitoring_health(),
            'data_integrity': await self._check_data_integrity(),
            'performance_baseline': await self._check_performance_baseline(),
            'alert_system': await self._check_alert_system_health(),
            'anomaly_detection': await self._check_anomaly_detection_health()
        }
        
        overall_health = np.mean(list(health_checks.values()))
        
        logger.info(f"System health check completed. Overall health: {overall_health:.3f}")
        
        for component, health in health_checks.items():
            if health < 0.8:
                await self._generate_alert(
                    'warning', 'health_check',
                    f"{component} health below threshold: {health:.3f}"
                )
        
        return health_checks
    
    async def _check_monitoring_health(self) -> float:
        """Check monitoring system health"""
        health_score = 1.0
        
        # Check if metrics are being updated
        if self.last_metrics_update > 0:
            time_since_update = time.time() - self.last_metrics_update
            if time_since_update > self.monitoring_interval * 3:  # 3x expected interval
                health_score *= 0.5
        
        # Check metrics history size
        if len(self.metrics_history) == 0:
            health_score *= 0.3
        elif len(self.metrics_history) < 10:
            health_score *= 0.7
        
        return health_score
    
    async def _check_data_integrity(self) -> float:
        """Check data integrity"""
        if not self.metrics_history:
            return 0.5
        
        health_score = 1.0
        
        # Check for data consistency
        recent_metrics = list(self.metrics_history)[-10:]
        
        # Check for null or invalid values
        invalid_count = 0
        total_checks = 0
        
        for metrics in recent_metrics:
            metric_values = [
                metrics.hypotheses_generated_rate,
                metrics.breakthrough_discovery_rate,
                metrics.publication_readiness_rate,
                metrics.system_health_score
            ]
            
            for value in metric_values:
                total_checks += 1
                if np.isnan(value) or np.isinf(value) or value < 0:
                    invalid_count += 1
        
        if total_checks > 0:
            validity_rate = 1.0 - (invalid_count / total_checks)
            health_score *= validity_rate
        
        return health_score
    
    async def _check_performance_baseline(self) -> float:
        """Check performance baseline health"""
        if not self.performance_baselines:
            return 0.6  # No baselines established yet
        
        health_score = 1.0
        
        # Check baseline completeness
        expected_baselines = [
            'hypotheses_generated_rate',
            'breakthrough_discovery_rate',
            'research_velocity'
        ]
        
        present_baselines = sum(1 for baseline in expected_baselines 
                              if baseline in self.performance_baselines)
        completeness = present_baselines / len(expected_baselines)
        health_score *= completeness
        
        return health_score
    
    async def _check_alert_system_health(self) -> float:
        """Check alert system health"""
        health_score = 1.0
        
        # Check recent alert generation
        recent_time = time.time() - 3600  # Last hour
        recent_alerts = [alert for alert in self.alerts if alert.timestamp > recent_time]
        
        # Alert system should generate some alerts (but not too many)
        alert_count = len(recent_alerts)
        if alert_count == 0:
            health_score *= 0.8  # Might indicate alert system not working
        elif alert_count > 50:
            health_score *= 0.6  # Too many alerts might indicate issues
        
        return health_score
    
    async def _check_anomaly_detection_health(self) -> float:
        """Check anomaly detection health"""
        health_score = 1.0
        
        # Check if anomaly detection is finding anomalies occasionally
        recent_time = time.time() - 7200  # Last 2 hours
        recent_anomalies = [anomaly for anomaly in self.anomalies_detected 
                           if hasattr(anomaly, 'timestamp') and anomaly.timestamp > recent_time]
        
        # Should detect some anomalies but not too many
        anomaly_count = len(recent_anomalies) if hasattr(self, 'anomalies_detected') else 0
        if anomaly_count == 0:
            health_score *= 0.9  # Might be too conservative
        elif anomaly_count > 20:
            health_score *= 0.7  # Might be too sensitive
        
        return health_score
    
    async def _generate_alert(self, severity: str, component: str, message: str):
        """Generate monitoring alert"""
        
        current_time = time.time()
        alert_id = f"{component}_{int(current_time)}_{np.random.randint(1000)}"
        
        alert = Alert(
            alert_id=alert_id,
            severity=severity,
            component=component,
            message=message,
            timestamp=current_time
        )
        
        self.alerts.append(alert)
        
        # Log alert
        log_level = {
            'info': logging.INFO,
            'warning': logging.WARNING,
            'critical': logging.ERROR
        }.get(severity, logging.INFO)
        
        logger.log(log_level, f"[{severity.upper()}] {component}: {message}")
        
        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    async def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        
        current_time = time.time()
        
        # Clean old alerts
        cutoff_time = current_time - (self.alert_retention_hours * 3600)
        original_alert_count = len(self.alerts)
        
        # Filter alerts to keep only recent ones
        self.alerts = deque(
            [alert for alert in self.alerts if alert.timestamp > cutoff_time],
            maxlen=self.alerts.maxlen
        )
        
        cleaned_alerts = original_alert_count - len(self.alerts)
        if cleaned_alerts > 0:
            logger.debug(f"Cleaned up {cleaned_alerts} old alerts")
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add custom alert handler"""
        self.alert_handlers.append(handler)
        logger.info("Added custom alert handler")
    
    def get_current_metrics(self) -> Optional[ResearchMetrics]:
        """Get most recent metrics"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of metrics over specified time period"""
        
        cutoff_time = time.time() - (hours * 3600)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return {'error': 'No metrics available for specified time period'}
        
        # Calculate summary statistics
        summary = {
            'time_period_hours': hours,
            'metric_count': len(recent_metrics),
            'averages': {
                'hypotheses_generated_rate': np.mean([m.hypotheses_generated_rate for m in recent_metrics]),
                'breakthrough_discovery_rate': np.mean([m.breakthrough_discovery_rate for m in recent_metrics]),
                'publication_readiness_rate': np.mean([m.publication_readiness_rate for m in recent_metrics]),
                'average_novelty_score': np.mean([m.average_novelty_score for m in recent_metrics]),
                'research_velocity': np.mean([m.research_velocity for m in recent_metrics]),
                'system_health_score': np.mean([m.system_health_score for m in recent_metrics])
            },
            'trends': {},
            'anomalies_detected': len([a for a in self.anomalies_detected 
                                     if hasattr(a, 'timestamp') and getattr(a, 'timestamp', 0) > cutoff_time]),
            'alerts_generated': len([a for a in self.alerts if a.timestamp > cutoff_time])
        }
        
        # Calculate trends
        for metric_name in summary['averages'].keys():
            metric_values = [getattr(m, metric_name) for m in recent_metrics]
            if len(metric_values) > 1:
                x = np.arange(len(metric_values))
                trend_slope = np.polyfit(x, metric_values, 1)[0]
                summary['trends'][metric_name] = trend_slope
        
        return summary
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of alerts over specified time period"""
        
        cutoff_time = time.time() - (hours * 3600)
        recent_alerts = [a for a in self.alerts if a.timestamp > cutoff_time]
        
        if not recent_alerts:
            return {'message': 'No alerts in specified time period'}
        
        # Categorize alerts
        alert_summary = {
            'total_alerts': len(recent_alerts),
            'by_severity': defaultdict(int),
            'by_component': defaultdict(int),
            'resolved_count': sum(1 for a in recent_alerts if a.resolved),
            'unresolved_count': sum(1 for a in recent_alerts if not a.resolved),
            'recent_critical': []
        }
        
        for alert in recent_alerts:
            alert_summary['by_severity'][alert.severity] += 1
            alert_summary['by_component'][alert.component] += 1
            
            # Include recent critical alerts
            if alert.severity == 'critical' and not alert.resolved:
                alert_summary['recent_critical'].append({
                    'alert_id': alert.alert_id,
                    'component': alert.component,
                    'message': alert.message,
                    'timestamp': alert.timestamp
                })
        
        # Convert defaultdicts to regular dicts
        alert_summary['by_severity'] = dict(alert_summary['by_severity'])
        alert_summary['by_component'] = dict(alert_summary['by_component'])
        
        return alert_summary
    
    def get_anomaly_report(self, hours: int = 24) -> Dict[str, Any]:
        """Get anomaly detection report"""
        
        cutoff_time = time.time() - (hours * 3600)
        recent_anomalies = [a for a in self.anomalies_detected 
                           if hasattr(a, 'timestamp') and getattr(a, 'timestamp', 0) > cutoff_time]
        
        if not recent_anomalies:
            return {'message': 'No anomalies detected in specified time period'}
        
        # Analyze anomalies
        anomaly_report = {
            'total_anomalies': len(recent_anomalies),
            'by_metric': defaultdict(int),
            'by_type': defaultdict(int),
            'high_confidence_anomalies': [],
            'metrics_with_anomalies': set()
        }
        
        for anomaly in recent_anomalies:
            anomaly_report['by_metric'][anomaly.metric_name] += 1
            anomaly_report['by_type'][anomaly.anomaly_type] += 1
            anomaly_report['metrics_with_anomalies'].add(anomaly.metric_name)
            
            # Include high-confidence anomalies
            if anomaly.confidence > 0.8:
                anomaly_report['high_confidence_anomalies'].append({
                    'metric_name': anomaly.metric_name,
                    'current_value': anomaly.current_value,
                    'expected_value': anomaly.expected_value,
                    'anomaly_type': anomaly.anomaly_type,
                    'confidence': anomaly.confidence
                })
        
        # Convert sets and defaultdicts
        anomaly_report['by_metric'] = dict(anomaly_report['by_metric'])
        anomaly_report['by_type'] = dict(anomaly_report['by_type'])
        anomaly_report['metrics_with_anomalies'] = list(anomaly_report['metrics_with_anomalies'])
        
        return anomaly_report
    
    def get_predictive_insights(self) -> Dict[str, Any]:
        """Get predictive insights and forecasts"""
        
        insights = {
            'breakthrough_prediction': self.predictive_models.get('breakthrough_timing', {}),
            'resource_forecast': self.predictive_models.get('memory_forecast', {}),
            'velocity_trends': self.trend_analyzers.get('research_velocity', {}),
            'performance_baselines': dict(self.performance_baselines),
            'recommendations': self._generate_optimization_recommendations()
        }
        
        return insights
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on monitoring data"""
        
        recommendations = []
        
        # Check recent performance
        if self.metrics_history:
            recent_metrics = self.metrics_history[-1]
            
            # Memory optimization
            if recent_metrics.memory_utilization > 0.8:
                recommendations.append("Consider implementing memory optimization or garbage collection")
            
            # Computational efficiency
            if recent_metrics.computational_efficiency < 0.7:
                recommendations.append("Review and optimize computational algorithms for better efficiency")
            
            # Research velocity
            if recent_metrics.research_velocity < 0.5:
                recommendations.append("Investigate bottlenecks in research pipeline to improve velocity")
            
            # Discovery rate
            if recent_metrics.breakthrough_discovery_rate < 0.1:
                recommendations.append("Consider adjusting discovery thresholds or algorithm parameters")
            
            # Publication readiness
            if recent_metrics.publication_readiness_rate < 0.7:
                recommendations.append("Review quality criteria for publication-ready research")
        
        # Check for frequent anomalies
        recent_anomalies = list(self.anomalies_detected)[-20:] if self.anomalies_detected else []
        if len(recent_anomalies) > 10:
            recommendations.append("High anomaly detection rate - consider adjusting sensitivity parameters")
        
        # Check alert frequency
        recent_alerts = [a for a in self.alerts if a.timestamp > time.time() - 3600]
        if len(recent_alerts) > 10:
            recommendations.append("High alert frequency - review alert thresholds and conditions")
        
        if not recommendations:
            recommendations.append("System operating within normal parameters")
        
        return recommendations
    
    async def generate_monitoring_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        
        current_time = time.time()
        
        report = {
            'report_timestamp': current_time,
            'monitoring_period_hours': hours,
            'system_overview': {
                'monitoring_uptime_hours': (current_time - self.start_time) / 3600,
                'is_monitoring_active': self.is_monitoring,
                'total_metrics_collected': len(self.metrics_history),
                'last_metrics_update': self.last_metrics_update
            },
            'current_metrics': self.get_current_metrics().__dict__ if self.get_current_metrics() else {},
            'metrics_summary': self.get_metrics_summary(hours),
            'alert_summary': self.get_alert_summary(hours),
            'anomaly_report': self.get_anomaly_report(hours),
            'predictive_insights': self.get_predictive_insights(),
            'system_health': await self._perform_system_health_check(),
            'performance_recommendations': self._generate_optimization_recommendations()
        }
        
        return report


# Example usage and testing
async def demo_monitoring_system():
    """Demonstrate the autonomous research monitoring system"""
    
    logger.info("🔍 Starting Autonomous Research Monitoring Demo")
    
    # Initialize monitor
    monitor = AutonomousResearchMonitor(
        metrics_retention_hours=1,  # Short retention for demo
        monitoring_interval=5.0     # 5-second intervals for demo
    )
    
    # Add custom alert handler
    async def custom_alert_handler(alert: Alert):
        print(f"🚨 ALERT [{alert.severity.upper()}] {alert.component}: {alert.message}")
    
    monitor.add_alert_handler(custom_alert_handler)
    
    # Start monitoring
    await monitor.start_monitoring()
    
    # Let it run for a demo period
    await asyncio.sleep(60)  # 1 minute demo
    
    # Generate report
    report = await monitor.generate_monitoring_report(hours=1)
    
    print("\n📊 Monitoring Report Summary:")
    print(f"  Metrics Collected: {report['system_overview']['total_metrics_collected']}")
    print(f"  Alerts Generated: {report['alert_summary'].get('total_alerts', 0)}")
    print(f"  Anomalies Detected: {report['anomaly_report'].get('total_anomalies', 0)}")
    
    # Stop monitoring
    await monitor.stop_monitoring()
    
    print("✅ Monitoring demo completed")
    
    return report


if __name__ == "__main__":
    # Run monitoring demo
    asyncio.run(demo_monitoring_system())