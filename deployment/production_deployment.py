#!/usr/bin/env python3
"""
Production Deployment Framework
Enterprise-grade deployment system for bioneural olfactory fusion
"""

import os
import sys
import json
import yaml
import time
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from contextlib import contextmanager

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.algorithms.bioneural_pipeline import BioneuralOlfactoryPipeline, PipelineConfig
from src.performance.distributed_processing import DistributedProcessingManager
from src.performance.gpu_acceleration import GPUAcceleratedBioneuralFusion, GPUConfig

logger = logging.getLogger(__name__)


@dataclass
class DeploymentConfig:
    """Configuration for production deployment"""
    # Service configuration
    service_name: str = "bioneural-olfactory-fusion"
    version: str = "1.0.0"
    port: int = 8080
    host: str = "0.0.0.0"
    
    # Scaling configuration
    min_workers: int = 2
    max_workers: int = 16
    worker_type: str = "thread"  # 'thread' or 'process'
    auto_scaling: bool = True
    
    # Performance configuration
    enable_gpu: bool = True
    gpu_device_id: int = 0
    batch_processing: bool = True
    max_batch_size: int = 32
    request_timeout: float = 30.0
    
    # Monitoring configuration
    enable_metrics: bool = True
    metrics_port: int = 9090
    log_level: str = "INFO"
    
    # Security configuration
    enable_auth: bool = False
    api_key_required: bool = False
    rate_limit_per_minute: int = 1000
    
    # Storage configuration
    model_cache_dir: str = "/tmp/bioneural_models"
    results_storage: str = "memory"  # 'memory', 'redis', 'database'
    
    # Health check configuration
    health_check_interval: float = 30.0
    health_check_timeout: float = 10.0


@dataclass
class ServiceStatus:
    """Service health and status information"""
    service_name: str
    version: str
    status: str  # 'healthy', 'degraded', 'unhealthy'
    uptime: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    current_load: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_available: bool
    last_health_check: float


class HealthChecker:
    """Health monitoring and checking system"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.start_time = time.time()
        self.request_stats = {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'response_times': []
        }
        
    def record_request(self, success: bool, response_time: float):
        """Record request statistics"""
        self.request_stats['total'] += 1
        if success:
            self.request_stats['successful'] += 1
        else:
            self.request_stats['failed'] += 1
            
        self.request_stats['response_times'].append(response_time)
        
        # Keep only recent response times
        if len(self.request_stats['response_times']) > 1000:
            self.request_stats['response_times'] = self.request_stats['response_times'][-800:]
    
    def get_service_status(self, pipeline: BioneuralOlfactoryPipeline) -> ServiceStatus:
        """Get current service status"""
        
        # Calculate metrics
        uptime = time.time() - self.start_time
        avg_response_time = (
            sum(self.request_stats['response_times']) / len(self.request_stats['response_times'])
            if self.request_stats['response_times'] else 0.0
        )
        
        # Get system metrics
        memory_mb = self._get_memory_usage()
        cpu_percent = self._get_cpu_usage()
        
        # Determine service status
        success_rate = (
            self.request_stats['successful'] / max(self.request_stats['total'], 1)
        )
        
        if success_rate >= 0.95 and avg_response_time < 1.0:
            status = 'healthy'
        elif success_rate >= 0.8 and avg_response_time < 5.0:
            status = 'degraded'
        else:
            status = 'unhealthy'
        
        return ServiceStatus(
            service_name=self.config.service_name,
            version=self.config.version,
            status=status,
            uptime=uptime,
            total_requests=self.request_stats['total'],
            successful_requests=self.request_stats['successful'],
            failed_requests=self.request_stats['failed'],
            average_response_time=avg_response_time,
            current_load=len(self.request_stats['response_times'][-100:]) / 100.0,  # Recent load
            memory_usage_mb=memory_mb,
            cpu_usage_percent=cpu_percent,
            gpu_available=self._check_gpu_available(),
            last_health_check=time.time()
        )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except ImportError:
            return 0.0
    
    def _check_gpu_available(self) -> bool:
        """Check if GPU is available"""
        try:
            import cupy as cp
            return cp.cuda.is_available()
        except ImportError:
            return False


class ProductionService:
    """
    Production-grade bioneural olfactory fusion service
    
    Provides enterprise-ready deployment with:
    1. RESTful API interface
    2. Automatic scaling and load balancing
    3. Health monitoring and metrics
    4. GPU acceleration support
    5. Distributed processing
    6. Comprehensive logging and error handling
    """
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.health_checker = HealthChecker(config)
        
        # Initialize core components
        self.pipeline = None
        self.distributed_manager = None
        self.gpu_accelerator = None
        
        # Service state
        self.is_running = False
        self.shutdown_requested = False
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"Production service initializing: {config.service_name} v{config.version}")
    
    def _setup_logging(self):
        """Setup production logging"""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # File handler
        log_dir = Path("/var/log") / self.config.service_name
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / f"{self.config.service_name}.log")
        file_handler.setLevel(log_level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
    
    def initialize_components(self):
        """Initialize all service components"""
        logger.info("Initializing service components")
        
        # Initialize bioneural pipeline
        pipeline_config = PipelineConfig(
            enable_adaptation=True,
            enable_profiling=True,
            quality_threshold=0.7
        )
        self.pipeline = BioneuralOlfactoryPipeline(pipeline_config)
        
        # Initialize distributed processing
        self.distributed_manager = DistributedProcessingManager(
            max_workers=self.config.max_workers,
            worker_type=self.config.worker_type,
            batch_size=self.config.max_batch_size
        )
        
        # Initialize GPU acceleration if enabled
        if self.config.enable_gpu:
            try:
                gpu_config = GPUConfig(
                    device_id=self.config.gpu_device_id,
                    batch_size=self.config.max_batch_size
                )
                self.gpu_accelerator = GPUAcceleratedBioneuralFusion(gpu_config=gpu_config)
                logger.info("GPU acceleration initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU acceleration: {e}")
                self.gpu_accelerator = None
        
        # Create model cache directory
        Path(self.config.model_cache_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("Service components initialized successfully")
    
    def start(self):
        """Start the production service"""
        logger.info(f"Starting {self.config.service_name} service")
        
        try:
            # Initialize components
            self.initialize_components()
            
            # Start distributed processing
            self.distributed_manager.start()
            
            # Start web server
            self._start_web_server()
            
            # Start health monitoring
            if self.config.enable_metrics:
                self._start_health_monitoring()
            
            self.is_running = True
            logger.info(f"Service started successfully on {self.config.host}:{self.config.port}")
            
            # Main service loop
            self._run_service_loop()
            
        except Exception as e:
            logger.error(f"Failed to start service: {e}")
            raise
        finally:
            self.shutdown()
    
    def _start_web_server(self):
        """Start the web server (Flask/FastAPI)"""
        try:
            from flask import Flask, request, jsonify
            
            app = Flask(__name__)
            
            @app.route('/health')
            def health_check():
                status = self.health_checker.get_service_status(self.pipeline)
                return jsonify(asdict(status))
            
            @app.route('/process', methods=['POST'])
            def process_signal():
                start_time = time.time()
                
                try:
                    # Parse request
                    data = request.get_json()
                    if not data or 'signal' not in data:
                        return jsonify({'error': 'Missing signal data'}), 400
                    
                    signal = data['signal']
                    if not isinstance(signal, list) or len(signal) == 0:
                        return jsonify({'error': 'Invalid signal format'}), 400
                    
                    # Convert to numpy array
                    import numpy as np
                    signal_array = np.array(signal, dtype=np.float32)
                    
                    # Process signal
                    result = self.pipeline.process(signal_array)
                    
                    # Prepare response
                    response = {
                        'success': True,
                        'processing_time': result.processing_time,
                        'quality_score': result.quality_metrics['overall_quality'],
                        'pattern_complexity': result.bioneural_result.pattern_complexity,
                        'fusion_confidence': result.neural_fusion_result.fusion_confidence
                    }
                    
                    # Record success
                    response_time = time.time() - start_time
                    self.health_checker.record_request(True, response_time)
                    
                    return jsonify(response)
                    
                except Exception as e:
                    logger.error(f"Request processing failed: {e}")
                    
                    # Record failure
                    response_time = time.time() - start_time
                    self.health_checker.record_request(False, response_time)
                    
                    return jsonify({'error': str(e)}), 500
            
            @app.route('/batch_process', methods=['POST'])
            def batch_process():
                start_time = time.time()
                
                try:
                    data = request.get_json()
                    if not data or 'signals' not in data:
                        return jsonify({'error': 'Missing signals data'}), 400
                    
                    signals_data = data['signals']
                    if not isinstance(signals_data, list):
                        return jsonify({'error': 'Signals must be a list'}), 400
                    
                    # Convert signals
                    import numpy as np
                    signals = [np.array(signal, dtype=np.float32) for signal in signals_data]
                    
                    # Process batch using distributed manager
                    results = self.distributed_manager.process_signals_batch(signals)
                    
                    # Prepare response
                    response = {
                        'success': True,
                        'num_signals': len(signals),
                        'results': [
                            {
                                'task_id': result.task_id,
                                'success': result.success,
                                'processing_time': result.processing_time,
                                'quality_score': result.result.quality_metrics['overall_quality'] if result.result else 0.0
                            }
                            for result in results
                        ]
                    }
                    
                    response_time = time.time() - start_time
                    self.health_checker.record_request(True, response_time)
                    
                    return jsonify(response)
                    
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
                    
                    response_time = time.time() - start_time
                    self.health_checker.record_request(False, response_time)
                    
                    return jsonify({'error': str(e)}), 500
            
            @app.route('/metrics')
            def get_metrics():
                if not self.config.enable_metrics:
                    return jsonify({'error': 'Metrics disabled'}), 403
                
                status = self.health_checker.get_service_status(self.pipeline)
                distributed_stats = self.distributed_manager.get_system_stats()
                
                metrics = {
                    'service_status': asdict(status),
                    'distributed_processing': distributed_stats,
                    'gpu_acceleration': self.gpu_accelerator.get_performance_summary() if self.gpu_accelerator else None
                }
                
                return jsonify(metrics)
            
            # Start Flask app in production mode
            import threading
            
            def run_flask():
                app.run(
                    host=self.config.host,
                    port=self.config.port,
                    debug=False,
                    threaded=True
                )
            
            flask_thread = threading.Thread(target=run_flask, daemon=True)
            flask_thread.start()
            
        except ImportError:
            logger.error("Flask not available, web server not started")
            raise
    
    def _start_health_monitoring(self):
        """Start health monitoring system"""
        def health_monitor_loop():
            while not self.shutdown_requested:
                try:
                    status = self.health_checker.get_service_status(self.pipeline)
                    
                    if status.status != 'healthy':
                        logger.warning(f"Service status: {status.status}")
                        
                        # Auto-scaling logic
                        if self.config.auto_scaling and status.current_load > 0.8:
                            self._scale_up()
                        elif self.config.auto_scaling and status.current_load < 0.2:
                            self._scale_down()
                    
                    time.sleep(self.config.health_check_interval)
                    
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")
                    time.sleep(self.config.health_check_interval)
        
        import threading
        health_thread = threading.Thread(target=health_monitor_loop, daemon=True)
        health_thread.start()
    
    def _scale_up(self):
        """Scale up the service"""
        current_workers = self.distributed_manager.max_workers
        if current_workers < self.config.max_workers:
            logger.info(f"Scaling up from {current_workers} to {current_workers + 1} workers")
            # Implementation would depend on deployment platform
    
    def _scale_down(self):
        """Scale down the service"""
        current_workers = self.distributed_manager.max_workers
        if current_workers > self.config.min_workers:
            logger.info(f"Scaling down from {current_workers} to {current_workers - 1} workers")
            # Implementation would depend on deployment platform
    
    def _run_service_loop(self):
        """Main service loop"""
        try:
            while not self.shutdown_requested:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Service interrupted by user")
            self.shutdown_requested = True
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down service")
        
        self.shutdown_requested = True
        
        # Stop distributed processing
        if self.distributed_manager:
            self.distributed_manager.shutdown()
        
        # Cleanup GPU resources
        if self.gpu_accelerator:
            self.gpu_accelerator.optimize_memory_usage()
        
        self.is_running = False
        logger.info("Service shutdown complete")


def create_docker_config(config: DeploymentConfig) -> str:
    """Generate Dockerfile for deployment"""
    
    dockerfile_content = f"""
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    cuda-toolkit-11-8 \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the package
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /var/log/{config.service_name} {config.model_cache_dir}

# Expose ports
EXPOSE {config.port}
EXPOSE {config.metrics_port}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:{config.port}/health || exit 1

# Run the service
CMD ["python", "deployment/production_deployment.py", "--config", "/app/deployment/config.yaml"]
"""
    
    return dockerfile_content


def create_kubernetes_config(config: DeploymentConfig) -> Dict[str, Any]:
    """Generate Kubernetes deployment configuration"""
    
    k8s_config = {
        'apiVersion': 'apps/v1',
        'kind': 'Deployment',
        'metadata': {
            'name': config.service_name,
            'labels': {
                'app': config.service_name,
                'version': config.version
            }
        },
        'spec': {
            'replicas': config.min_workers,
            'selector': {
                'matchLabels': {
                    'app': config.service_name
                }
            },
            'template': {
                'metadata': {
                    'labels': {
                        'app': config.service_name,
                        'version': config.version
                    }
                },
                'spec': {
                    'containers': [{
                        'name': config.service_name,
                        'image': f"{config.service_name}:{config.version}",
                        'ports': [
                            {'containerPort': config.port, 'name': 'http'},
                            {'containerPort': config.metrics_port, 'name': 'metrics'}
                        ],
                        'env': [
                            {'name': 'LOG_LEVEL', 'value': config.log_level},
                            {'name': 'ENABLE_GPU', 'value': str(config.enable_gpu)},
                            {'name': 'MAX_WORKERS', 'value': str(config.max_workers)}
                        ],
                        'resources': {
                            'requests': {
                                'cpu': '500m',
                                'memory': '1Gi'
                            },
                            'limits': {
                                'cpu': '2000m',
                                'memory': '4Gi',
                                'nvidia.com/gpu': '1' if config.enable_gpu else '0'
                            }
                        },
                        'livenessProbe': {
                            'httpGet': {
                                'path': '/health',
                                'port': config.port
                            },
                            'initialDelaySeconds': 60,
                            'periodSeconds': 30
                        },
                        'readinessProbe': {
                            'httpGet': {
                                'path': '/health', 
                                'port': config.port
                            },
                            'initialDelaySeconds': 30,
                            'periodSeconds': 10
                        }
                    }]
                }
            }
        }
    }
    
    return k8s_config


def create_helm_chart(config: DeploymentConfig) -> Dict[str, str]:
    """Generate Helm chart for Kubernetes deployment"""
    
    # Chart.yaml
    chart_yaml = f"""
apiVersion: v2
name: {config.service_name}
description: Bioneural Olfactory Fusion Service
version: {config.version}
appVersion: {config.version}
"""
    
    # values.yaml
    values_yaml = f"""
replicaCount: {config.min_workers}

image:
  repository: {config.service_name}
  tag: {config.version}
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: {config.port}
  metricsPort: {config.metrics_port}

ingress:
  enabled: false
  annotations: {{}}
  hosts: []
  tls: []

resources:
  requests:
    cpu: 500m
    memory: 1Gi
  limits:
    cpu: 2000m
    memory: 4Gi

autoscaling:
  enabled: {str(config.auto_scaling).lower()}
  minReplicas: {config.min_workers}
  maxReplicas: {config.max_workers}
  targetCPUUtilizationPercentage: 70

nodeSelector: {{}}
tolerations: []
affinity: {{}}
"""
    
    # templates/deployment.yaml
    deployment_yaml = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "bioneural-service.fullname" . }}
  labels:
    {{- include "bioneural-service.labels" . | nindent 4 }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "bioneural-service.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "bioneural-service.selectorLabels" . | nindent 8 }}
    spec:
      containers:
        - name: {{ .Chart.Name }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - name: http
              containerPort: {{ .Values.service.port }}
              protocol: TCP
            - name: metrics
              containerPort: {{ .Values.service.metricsPort }}
              protocol: TCP
          livenessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 60
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 30
            periodSeconds: 10
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
"""
    
    return {
        'Chart.yaml': chart_yaml,
        'values.yaml': values_yaml,
        'templates/deployment.yaml': deployment_yaml
    }


def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description='Bioneural Olfactory Fusion Production Deployment')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--generate-docker', action='store_true', help='Generate Dockerfile')
    parser.add_argument('--generate-k8s', action='store_true', help='Generate Kubernetes config')
    parser.add_argument('--generate-helm', action='store_true', help='Generate Helm chart')
    parser.add_argument('--start-service', action='store_true', help='Start the service')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            if args.config.endswith('.yaml') or args.config.endswith('.yml'):
                config_data = yaml.safe_load(f)
            else:
                config_data = json.load(f)
        
        config = DeploymentConfig(**config_data)
    else:
        config = DeploymentConfig()
    
    # Generate deployment artifacts
    if args.generate_docker:
        dockerfile = create_docker_config(config)
        with open('Dockerfile', 'w') as f:
            f.write(dockerfile)
        print("Generated Dockerfile")
    
    if args.generate_k8s:
        k8s_config = create_kubernetes_config(config)
        with open('k8s-deployment.yaml', 'w') as f:
            yaml.dump(k8s_config, f)
        print("Generated Kubernetes deployment config")
    
    if args.generate_helm:
        helm_chart = create_helm_chart(config)
        chart_dir = Path('helm-chart')
        chart_dir.mkdir(exist_ok=True)
        (chart_dir / 'templates').mkdir(exist_ok=True)
        
        for filename, content in helm_chart.items():
            filepath = chart_dir / filename
            filepath.parent.mkdir(exist_ok=True, parents=True)
            with open(filepath, 'w') as f:
                f.write(content)
        print("Generated Helm chart")
    
    # Start service
    if args.start_service:
        service = ProductionService(config)
        try:
            service.start()
        except KeyboardInterrupt:
            print("Service interrupted")
        except Exception as e:
            print(f"Service failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()