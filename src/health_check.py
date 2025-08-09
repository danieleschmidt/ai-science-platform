"""Health check and monitoring utilities for AI Science Platform"""

import time
import psutil
import logging
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class HealthMetrics:
    """Health metrics for system monitoring"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    process_count: int
    uptime_seconds: float
    status: str = "healthy"
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


@dataclass
class ComponentHealth:
    """Health status for individual components"""
    component_name: str
    status: str
    last_check: str
    response_time_ms: float
    error_count: int = 0
    warning_count: int = 0
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class HealthChecker:
    """Comprehensive system health monitoring"""
    
    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self.start_time = time.time()
        self.health_history: List[HealthMetrics] = []
        self.component_health: Dict[str, ComponentHealth] = {}
        self.alert_thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_usage_percent": 90.0,
            "response_time_ms": 1000.0
        }
        self._monitoring = False
        self._monitor_thread = None
        self.health_checks: Dict[str, Callable[[], ComponentHealth]] = {}
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self) -> None:
        """Register default component health checks"""
        self.register_health_check("discovery_engine", self._check_discovery_engine)
        self.register_health_check("experiment_runner", self._check_experiment_runner)
        self.register_health_check("file_system", self._check_file_system)
        self.register_health_check("logging_system", self._check_logging_system)
    
    def register_health_check(self, component_name: str, check_func: Callable[[], ComponentHealth]) -> None:
        """Register a custom health check function"""
        self.health_checks[component_name] = check_func
        logger.info(f"Registered health check for {component_name}")
    
    def get_system_metrics(self) -> HealthMetrics:
        """Get current system health metrics"""
        try:
            # CPU and memory metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent
            disk_free_gb = disk.free / (1024**3)
            
            # Process metrics
            process_count = len(psutil.pids())
            
            # Uptime
            uptime_seconds = time.time() - self.start_time
            
            # Determine status
            errors = []
            warnings = []
            
            if cpu_percent > self.alert_thresholds["cpu_percent"]:
                errors.append(f"High CPU usage: {cpu_percent:.1f}%")
            elif cpu_percent > 70:
                warnings.append(f"Elevated CPU usage: {cpu_percent:.1f}%")
            
            if memory_percent > self.alert_thresholds["memory_percent"]:
                errors.append(f"High memory usage: {memory_percent:.1f}%")
            elif memory_percent > 75:
                warnings.append(f"Elevated memory usage: {memory_percent:.1f}%")
            
            if disk_usage_percent > self.alert_thresholds["disk_usage_percent"]:
                errors.append(f"Low disk space: {disk_usage_percent:.1f}% used")
            elif disk_usage_percent > 80:
                warnings.append(f"Disk space warning: {disk_usage_percent:.1f}% used")
            
            status = "unhealthy" if errors else ("warning" if warnings else "healthy")
            
            metrics = HealthMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available_gb=memory_available_gb,
                disk_usage_percent=disk_usage_percent,
                disk_free_gb=disk_free_gb,
                process_count=process_count,
                uptime_seconds=uptime_seconds,
                status=status,
                errors=errors,
                warnings=warnings
            )
            
            return metrics
        
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return HealthMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_available_gb=0.0,
                disk_usage_percent=0.0,
                disk_free_gb=0.0,
                process_count=0,
                uptime_seconds=time.time() - self.start_time,
                status="error",
                errors=[f"Failed to collect system metrics: {e}"]
            )
    
    def check_component_health(self) -> Dict[str, ComponentHealth]:
        """Check health of all registered components"""
        health_results = {}
        
        for component_name, check_func in self.health_checks.items():
            try:
                start_time = time.time()
                health_result = check_func()
                response_time = (time.time() - start_time) * 1000
                
                # Update response time
                health_result.response_time_ms = response_time
                
                # Check response time threshold
                if response_time > self.alert_thresholds["response_time_ms"]:
                    health_result.warning_count += 1
                    if "response_time" not in health_result.details:
                        health_result.details["response_time"] = []
                    health_result.details["response_time"].append(
                        f"Slow response: {response_time:.2f}ms"
                    )
                
                health_results[component_name] = health_result
                self.component_health[component_name] = health_result
                
            except Exception as e:
                logger.error(f"Health check failed for {component_name}: {e}")
                health_results[component_name] = ComponentHealth(
                    component_name=component_name,
                    status="error",
                    last_check=datetime.now().isoformat(),
                    response_time_ms=0.0,
                    error_count=1,
                    details={"error": str(e)}
                )
        
        return health_results
    
    def _check_discovery_engine(self) -> ComponentHealth:
        """Check discovery engine health"""
        try:
            from .algorithms.discovery import DiscoveryEngine
            
            # Test basic functionality
            engine = DiscoveryEngine(discovery_threshold=0.7)
            test_data = np.random.randn(100, 1)
            test_targets = test_data.flatten() + np.random.randn(100) * 0.1
            
            # Try to generate hypothesis
            hypothesis = engine.generate_hypothesis(test_data, "health_check")
            
            # Try to test hypothesis
            is_valid, metrics = engine.test_hypothesis(hypothesis, test_data, test_targets)
            
            return ComponentHealth(
                component_name="discovery_engine",
                status="healthy",
                last_check=datetime.now().isoformat(),
                response_time_ms=0.0,  # Will be set by caller
                details={
                    "hypothesis_generated": True,
                    "hypothesis_tested": True,
                    "test_data_shape": test_data.shape,
                    "metrics": metrics
                }
            )
        
        except Exception as e:
            return ComponentHealth(
                component_name="discovery_engine",
                status="error",
                last_check=datetime.now().isoformat(),
                response_time_ms=0.0,
                error_count=1,
                details={"error": str(e)}
            )
    
    def _check_experiment_runner(self) -> ComponentHealth:
        """Check experiment runner health"""
        try:
            from .experiments.runner import ExperimentRunner, ExperimentConfig
            
            # Test basic functionality
            runner = ExperimentRunner("/tmp/health_check_experiments")
            
            # Create a test config
            config = ExperimentConfig(
                name="health_check",
                description="Health check test",
                parameters={"test": True},
                metrics_to_track=["test_metric"],
                num_runs=1
            )
            
            runner.register_experiment(config)
            
            return ComponentHealth(
                component_name="experiment_runner",
                status="healthy",
                last_check=datetime.now().isoformat(),
                response_time_ms=0.0,
                details={
                    "config_registered": True,
                    "experiments_count": len(runner.experiments)
                }
            )
        
        except Exception as e:
            return ComponentHealth(
                component_name="experiment_runner",
                status="error",
                last_check=datetime.now().isoformat(),
                response_time_ms=0.0,
                error_count=1,
                details={"error": str(e)}
            )
    
    def _check_file_system(self) -> ComponentHealth:
        """Check file system health"""
        try:
            # Check critical directories
            critical_dirs = ["logs", "experiment_results"]
            dir_status = {}
            
            for dir_name in critical_dirs:
                dir_path = Path(dir_name)
                if not dir_path.exists():
                    dir_path.mkdir(parents=True, exist_ok=True)
                
                # Test write permissions
                test_file = dir_path / "health_check.tmp"
                try:
                    test_file.write_text("health check")
                    test_file.unlink()
                    dir_status[dir_name] = "writable"
                except Exception:
                    dir_status[dir_name] = "read_only"
            
            # Check if any directories are not writable
            non_writable = [d for d, s in dir_status.items() if s != "writable"]
            status = "error" if non_writable else "healthy"
            
            return ComponentHealth(
                component_name="file_system",
                status=status,
                last_check=datetime.now().isoformat(),
                response_time_ms=0.0,
                error_count=len(non_writable),
                details={
                    "directory_status": dir_status,
                    "non_writable_dirs": non_writable
                }
            )
        
        except Exception as e:
            return ComponentHealth(
                component_name="file_system",
                status="error",
                last_check=datetime.now().isoformat(),
                response_time_ms=0.0,
                error_count=1,
                details={"error": str(e)}
            )
    
    def _check_logging_system(self) -> ComponentHealth:
        """Check logging system health"""
        try:
            # Test logging functionality
            test_logger = logging.getLogger("health_check")
            test_message = f"Health check at {datetime.now().isoformat()}"
            
            # Test different log levels
            test_logger.debug(test_message)
            test_logger.info(test_message)
            test_logger.warning(test_message)
            
            # Check if log directory exists
            logs_dir = Path("logs")
            log_files_exist = logs_dir.exists() and any(logs_dir.iterdir())
            
            return ComponentHealth(
                component_name="logging_system",
                status="healthy",
                last_check=datetime.now().isoformat(),
                response_time_ms=0.0,
                details={
                    "test_message_logged": True,
                    "log_directory_exists": logs_dir.exists(),
                    "log_files_exist": log_files_exist
                }
            )
        
        except Exception as e:
            return ComponentHealth(
                component_name="logging_system",
                status="error",
                last_check=datetime.now().isoformat(),
                response_time_ms=0.0,
                error_count=1,
                details={"error": str(e)}
            )
    
    def start_monitoring(self) -> None:
        """Start continuous health monitoring"""
        if self._monitoring:
            logger.warning("Health monitoring is already running")
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info(f"Health monitoring started with {self.check_interval}s interval")
    
    def stop_monitoring(self) -> None:
        """Stop continuous health monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Health monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Continuous monitoring loop"""
        while self._monitoring:
            try:
                # Collect system metrics
                metrics = self.get_system_metrics()
                self.health_history.append(metrics)
                
                # Limit history size
                if len(self.health_history) > 1000:
                    self.health_history = self.health_history[-500:]
                
                # Check component health
                self.check_component_health()
                
                # Log alerts
                if metrics.errors:
                    logger.error(f"System health alerts: {metrics.errors}")
                if metrics.warnings:
                    logger.warning(f"System health warnings: {metrics.warnings}")
                
                # Wait for next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary"""
        current_metrics = self.get_system_metrics()
        component_health = self.check_component_health()
        
        # Calculate health trends
        recent_history = self.health_history[-10:] if len(self.health_history) >= 10 else self.health_history
        
        if recent_history:
            avg_cpu = sum(m.cpu_percent for m in recent_history) / len(recent_history)
            avg_memory = sum(m.memory_percent for m in recent_history) / len(recent_history)
        else:
            avg_cpu = current_metrics.cpu_percent
            avg_memory = current_metrics.memory_percent
        
        # Overall health status
        system_status = current_metrics.status
        component_statuses = [c.status for c in component_health.values()]
        
        if "error" in component_statuses or system_status == "error":
            overall_status = "error"
        elif "warning" in component_statuses or system_status == "warning":
            overall_status = "warning"
        else:
            overall_status = "healthy"
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "system_metrics": asdict(current_metrics),
            "component_health": {name: asdict(health) for name, health in component_health.items()},
            "trends": {
                "avg_cpu_10_checks": avg_cpu,
                "avg_memory_10_checks": avg_memory,
                "history_length": len(self.health_history)
            },
            "uptime_hours": current_metrics.uptime_seconds / 3600
        }
    
    def export_health_report(self, filepath: str) -> None:
        """Export comprehensive health report to file"""
        health_summary = self.get_health_summary()
        
        report_path = Path(filepath)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            import json
            with open(report_path, 'w') as f:
                json.dump(health_summary, f, indent=2, default=str)
            
            logger.info(f"Health report exported to {filepath}")
        
        except Exception as e:
            logger.error(f"Failed to export health report: {e}")
            raise


# Global health checker instance
_health_checker = None


def get_health_checker(check_interval: int = 60) -> HealthChecker:
    """Get or create global health checker"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker(check_interval)
    return _health_checker


def start_health_monitoring(check_interval: int = 60) -> None:
    """Start global health monitoring"""
    health_checker = get_health_checker(check_interval)
    health_checker.start_monitoring()


def stop_health_monitoring() -> None:
    """Stop global health monitoring"""
    if _health_checker:
        _health_checker.stop_monitoring()


def get_health_status() -> Dict[str, Any]:
    """Get current health status"""
    if _health_checker:
        return _health_checker.get_health_summary()
    else:
        # Return basic status if monitoring not started
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "unknown",
            "message": "Health monitoring not started"
        }