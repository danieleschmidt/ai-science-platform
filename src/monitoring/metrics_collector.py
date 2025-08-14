"""Advanced metrics collection and analysis"""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Metric:
    """Individual metric data point"""
    name: str
    value: Union[float, int]
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class MetricSummary:
    """Summary statistics for a metric"""
    name: str
    count: int
    sum: float
    min: float
    max: float
    mean: float
    std: float
    percentiles: Dict[str, float]
    latest_value: float
    latest_timestamp: float


class MetricsCollector:
    """Advanced metrics collection and aggregation system"""
    
    def __init__(self, buffer_size: int = 10000, aggregation_interval: float = 60.0):
        self.buffer_size = buffer_size
        self.aggregation_interval = aggregation_interval
        
        # Storage
        self.metrics_buffer = deque(maxlen=buffer_size)
        self.metric_stores = defaultdict(lambda: deque(maxlen=1000))
        self.aggregated_metrics = {}
        
        # Configuration
        self.custom_aggregators = {}
        self.metric_configs = {}
        
        # State
        self.is_collecting = False
        self.aggregation_thread = None
        self.lock = threading.Lock()
        
        # Built-in metrics
        self.system_metrics = {
            "discovery_operations": 0,
            "model_predictions": 0,
            "api_requests": 0,
            "processing_time": deque(maxlen=1000),
            "error_counts": defaultdict(int)
        }
        
        logger.info("MetricsCollector initialized")
    
    def record_metric(self, name: str, value: Union[float, int], 
                     tags: Optional[Dict[str, str]] = None,
                     metric_type: MetricType = MetricType.GAUGE) -> None:
        """Record a metric value"""
        with self.lock:
            metric = Metric(
                name=name,
                value=value,
                timestamp=time.time(),
                tags=tags or {},
                metric_type=metric_type
            )
            
            self.metrics_buffer.append(metric)
            self.metric_stores[name].append(metric)
            
            # Update built-in counters
            if metric_type == MetricType.COUNTER:
                self.system_metrics[name] = self.system_metrics.get(name, 0) + value
    
    def record_timer(self, name: str, duration: float, 
                    tags: Optional[Dict[str, str]] = None) -> None:
        """Record a timing metric"""
        self.record_metric(name, duration, tags, MetricType.TIMER)
        self.system_metrics["processing_time"].append(duration)
    
    def increment_counter(self, name: str, value: int = 1, 
                         tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric"""
        self.record_metric(name, value, tags, MetricType.COUNTER)
    
    def record_histogram(self, name: str, value: float, 
                        tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric"""
        self.record_metric(name, value, tags, MetricType.HISTOGRAM)
    
    def time_function(self, metric_name: str, tags: Optional[Dict[str, str]] = None):
        """Decorator to time function execution"""
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    self.record_timer(f"{metric_name}_success", time.time() - start_time, tags)
                    return result
                except Exception as e:
                    self.record_timer(f"{metric_name}_error", time.time() - start_time, tags)
                    self.increment_counter(f"{metric_name}_errors", tags=tags)
                    raise
            return wrapper
        return decorator
    
    def start_collection(self) -> None:
        """Start metrics collection and aggregation"""
        if self.is_collecting:
            logger.warning("Metrics collection already running")
            return
        
        self.is_collecting = True
        self.aggregation_thread = threading.Thread(target=self._aggregation_loop, daemon=True)
        self.aggregation_thread.start()
        
        logger.info("Metrics collection started")
    
    def stop_collection(self) -> None:
        """Stop metrics collection"""
        self.is_collecting = False
        if self.aggregation_thread:
            self.aggregation_thread.join(timeout=5.0)
        
        logger.info("Metrics collection stopped")
    
    def _aggregation_loop(self) -> None:
        """Main aggregation loop"""
        while self.is_collecting:
            try:
                self._aggregate_metrics()
                time.sleep(self.aggregation_interval)
            except Exception as e:
                logger.error(f"Metrics aggregation error: {e}")
                time.sleep(self.aggregation_interval)
    
    def _aggregate_metrics(self) -> None:
        """Aggregate metrics and compute summaries"""
        with self.lock:
            current_time = time.time()
            
            for metric_name, metric_data in self.metric_stores.items():
                if not metric_data:
                    continue
                
                # Get recent metrics (last aggregation interval)
                recent_metrics = [
                    m for m in metric_data 
                    if current_time - m.timestamp <= self.aggregation_interval
                ]
                
                if not recent_metrics:
                    continue
                
                # Compute summary
                values = [m.value for m in recent_metrics]
                summary = self._compute_metric_summary(metric_name, values, recent_metrics[-1])
                
                self.aggregated_metrics[metric_name] = summary
    
    def _compute_metric_summary(self, name: str, values: List[float], 
                               latest_metric: Metric) -> MetricSummary:
        """Compute summary statistics for metric values"""
        if not values:
            return MetricSummary(
                name=name,
                count=0,
                sum=0.0,
                min=0.0,
                max=0.0,
                mean=0.0,
                std=0.0,
                percentiles={},
                latest_value=0.0,
                latest_timestamp=time.time()
            )
        
        values_array = np.array(values)
        
        # Compute percentiles
        percentiles = {}
        for p in [50, 90, 95, 99]:
            percentiles[f"p{p}"] = float(np.percentile(values_array, p))
        
        return MetricSummary(
            name=name,
            count=len(values),
            sum=float(np.sum(values_array)),
            min=float(np.min(values_array)),
            max=float(np.max(values_array)),
            mean=float(np.mean(values_array)),
            std=float(np.std(values_array)),
            percentiles=percentiles,
            latest_value=float(latest_metric.value),
            latest_timestamp=latest_metric.timestamp
        )
    
    def get_metric_summary(self, metric_name: str) -> Optional[MetricSummary]:
        """Get summary for specific metric"""
        return self.aggregated_metrics.get(metric_name)
    
    def get_all_metrics(self) -> Dict[str, MetricSummary]:
        """Get all aggregated metrics"""
        return self.aggregated_metrics.copy()
    
    def get_recent_metrics(self, metric_name: str, seconds: float = 300) -> List[Metric]:
        """Get recent metrics for a specific metric name"""
        cutoff_time = time.time() - seconds
        
        with self.lock:
            if metric_name in self.metric_stores:
                return [m for m in self.metric_stores[metric_name] if m.timestamp >= cutoff_time]
        
        return []
    
    def query_metrics(self, name_pattern: str = None, 
                     tags: Optional[Dict[str, str]] = None,
                     start_time: Optional[float] = None,
                     end_time: Optional[float] = None) -> List[Metric]:
        """Query metrics with filters"""
        results = []
        
        with self.lock:
            for metric_name, metric_data in self.metric_stores.items():
                # Filter by name pattern
                if name_pattern and name_pattern not in metric_name:
                    continue
                
                for metric in metric_data:
                    # Filter by time range
                    if start_time and metric.timestamp < start_time:
                        continue
                    if end_time and metric.timestamp > end_time:
                        continue
                    
                    # Filter by tags
                    if tags:
                        if not all(metric.tags.get(k) == v for k, v in tags.items()):
                            continue
                    
                    results.append(metric)
        
        return sorted(results, key=lambda m: m.timestamp)
    
    def register_custom_aggregator(self, metric_name: str, 
                                  aggregator_func: Callable[[List[Metric]], Dict[str, Any]]) -> None:
        """Register custom aggregation function for specific metric"""
        self.custom_aggregators[metric_name] = aggregator_func
        logger.info(f"Registered custom aggregator for metric: {metric_name}")
    
    def configure_metric(self, metric_name: str, config: Dict[str, Any]) -> None:
        """Configure metric-specific settings"""
        self.metric_configs[metric_name] = config
        logger.info(f"Configured metric: {metric_name}")
    
    def get_top_metrics(self, metric_type: str = "value", top_k: int = 10) -> List[Dict[str, Any]]:
        """Get top metrics by various criteria"""
        metric_rankings = []
        
        for name, summary in self.aggregated_metrics.items():
            if metric_type == "value":
                score = summary.latest_value
            elif metric_type == "volume":
                score = summary.count
            elif metric_type == "variability":
                score = summary.std
            elif metric_type == "max":
                score = summary.max
            else:
                score = summary.mean
            
            metric_rankings.append({
                "name": name,
                "score": score,
                "summary": summary
            })
        
        return sorted(metric_rankings, key=lambda x: x["score"], reverse=True)[:top_k]
    
    def compute_metric_correlations(self, metric_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Compute correlations between metrics"""
        correlations = {}
        
        # Get metric values for correlation analysis
        metric_values = {}
        for name in metric_names:
            recent_metrics = self.get_recent_metrics(name, 3600)  # Last hour
            if len(recent_metrics) > 10:  # Need sufficient data
                metric_values[name] = [m.value for m in recent_metrics]
        
        # Compute pairwise correlations
        for name1 in metric_values:
            correlations[name1] = {}
            for name2 in metric_values:
                if name1 != name2:
                    try:
                        # Align time series (simple approach)
                        min_len = min(len(metric_values[name1]), len(metric_values[name2]))
                        values1 = metric_values[name1][-min_len:]
                        values2 = metric_values[name2][-min_len:]
                        
                        correlation = np.corrcoef(values1, values2)[0, 1]
                        correlations[name1][name2] = float(correlation) if not np.isnan(correlation) else 0.0
                    except Exception as e:
                        logger.error(f"Correlation computation error for {name1}-{name2}: {e}")
                        correlations[name1][name2] = 0.0
        
        return correlations
    
    def detect_anomalies(self, metric_name: str, threshold: float = 2.0) -> List[Dict[str, Any]]:
        """Detect anomalies in metric values using statistical methods"""
        recent_metrics = self.get_recent_metrics(metric_name, 3600)
        
        if len(recent_metrics) < 30:
            return []  # Need sufficient data for anomaly detection
        
        values = np.array([m.value for m in recent_metrics])
        mean = np.mean(values)
        std = np.std(values)
        
        anomalies = []
        for metric in recent_metrics:
            z_score = abs(metric.value - mean) / (std + 1e-10)
            if z_score > threshold:
                anomalies.append({
                    "timestamp": metric.timestamp,
                    "value": metric.value,
                    "z_score": z_score,
                    "expected_range": [mean - threshold * std, mean + threshold * std]
                })
        
        return sorted(anomalies, key=lambda x: x["z_score"], reverse=True)
    
    def export_metrics(self, filepath: str, format: str = "json") -> None:
        """Export metrics data to file"""
        import json
        
        export_data = {
            "export_timestamp": time.time(),
            "aggregated_metrics": {name: {
                "name": summary.name,
                "count": summary.count,
                "sum": summary.sum,
                "min": summary.min,
                "max": summary.max,
                "mean": summary.mean,
                "std": summary.std,
                "percentiles": summary.percentiles,
                "latest_value": summary.latest_value,
                "latest_timestamp": summary.latest_timestamp
            } for name, summary in self.aggregated_metrics.items()},
            "system_metrics": {
                "discovery_operations": self.system_metrics["discovery_operations"],
                "model_predictions": self.system_metrics["model_predictions"],
                "api_requests": self.system_metrics["api_requests"],
                "avg_processing_time": np.mean(self.system_metrics["processing_time"]) if self.system_metrics["processing_time"] else 0,
                "error_counts": dict(self.system_metrics["error_counts"])
            }
        }
        
        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
        
        logger.info(f"Metrics exported to {filepath}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get metrics data formatted for dashboard display"""
        current_time = time.time()
        
        # Key performance indicators
        kpis = {}
        if "processing_time" in self.aggregated_metrics:
            kpis["avg_processing_time"] = self.aggregated_metrics["processing_time"].mean
        
        if "discovery_operations" in self.aggregated_metrics:
            kpis["total_discoveries"] = self.aggregated_metrics["discovery_operations"].sum
        
        # System health indicators
        health_indicators = {
            "total_operations": self.system_metrics["discovery_operations"],
            "error_rate": sum(self.system_metrics["error_counts"].values()) / max(1, self.system_metrics["api_requests"]),
            "avg_response_time": np.mean(self.system_metrics["processing_time"]) if self.system_metrics["processing_time"] else 0
        }
        
        # Recent activity
        recent_activity = {}
        for name, summary in self.aggregated_metrics.items():
            if current_time - summary.latest_timestamp <= 300:  # Last 5 minutes
                recent_activity[name] = {
                    "latest_value": summary.latest_value,
                    "count": summary.count,
                    "trend": "up" if summary.latest_value > summary.mean else "down"
                }
        
        return {
            "timestamp": current_time,
            "kpis": kpis,
            "health_indicators": health_indicators,
            "recent_activity": recent_activity,
            "metric_count": len(self.aggregated_metrics),
            "data_points": sum(summary.count for summary in self.aggregated_metrics.values())
        }