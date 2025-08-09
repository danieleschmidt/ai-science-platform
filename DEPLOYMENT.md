# AI Science Platform - Deployment Guide

## 🚀 Production Deployment

### Prerequisites

- Python 3.8+ 
- Virtual environment support
- Git
- 8GB+ RAM recommended for large-scale experiments
- Multi-core CPU for optimal parallel processing

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ai-science-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the platform in development mode
pip install -e .
```

### Configuration

#### Environment Variables

```bash
export AI_SCIENCE_PLATFORM_LOG_LEVEL=INFO
export AI_SCIENCE_PLATFORM_MAX_WORKERS=8
export AI_SCIENCE_PLATFORM_CACHE_SIZE=256
export AI_SCIENCE_PLATFORM_DATA_DIR=/path/to/data
export AI_SCIENCE_PLATFORM_RESULTS_DIR=/path/to/results
```

#### Configuration File (Optional)

Create `config.json` in the project root:

```json
{
    "discovery": {
        "default_threshold": 0.7,
        "max_hypotheses": 10,
        "enable_caching": true
    },
    "performance": {
        "max_workers": 8,
        "use_processes": false,
        "batch_size": 100,
        "memory_limit_mb": 2000
    },
    "security": {
        "max_file_size_mb": 100,
        "enable_input_validation": true,
        "audit_logging": true
    },
    "experiments": {
        "default_runs": 5,
        "results_retention_days": 90,
        "auto_cleanup": true
    }
}
```

### Docker Deployment

#### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY examples/ ./examples/
COPY setup.py .
COPY README.md .

# Install the application
RUN pip install -e .

# Create directories for data and results
RUN mkdir -p /app/data /app/results

# Set environment variables
ENV AI_SCIENCE_PLATFORM_DATA_DIR=/app/data
ENV AI_SCIENCE_PLATFORM_RESULTS_DIR=/app/results
ENV AI_SCIENCE_PLATFORM_LOG_LEVEL=INFO

# Expose port for potential web interface
EXPOSE 8080

# Default command
CMD ["python", "-m", "examples.complete_platform_demo"]
```

#### docker-compose.yml

```yaml
version: '3.8'

services:
  ai-science-platform:
    build: .
    container_name: ai-science-platform
    environment:
      - AI_SCIENCE_PLATFORM_LOG_LEVEL=INFO
      - AI_SCIENCE_PLATFORM_MAX_WORKERS=4
    volumes:
      - ./data:/app/data
      - ./results:/app/results
    ports:
      - "8080:8080"
    restart: unless-stopped
    
  redis-cache:
    image: redis:alpine
    container_name: ai-science-cache
    ports:
      - "6379:6379"
    restart: unless-stopped
```

### Kubernetes Deployment

#### deployment.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-science-platform
  labels:
    app: ai-science-platform
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-science-platform
  template:
    metadata:
      labels:
        app: ai-science-platform
    spec:
      containers:
      - name: ai-science-platform
        image: ai-science-platform:latest
        ports:
        - containerPort: 8080
        env:
        - name: AI_SCIENCE_PLATFORM_LOG_LEVEL
          value: "INFO"
        - name: AI_SCIENCE_PLATFORM_MAX_WORKERS
          value: "4"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: results-volume
          mountPath: /app/results
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: data-pvc
      - name: results-volume
        persistentVolumeClaim:
          claimName: results-pvc
```

### Cloud Deployment

#### AWS ECS

```json
{
  "family": "ai-science-platform",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "ai-science-platform",
      "image": "your-ecr-repo/ai-science-platform:latest",
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "AI_SCIENCE_PLATFORM_LOG_LEVEL",
          "value": "INFO"
        },
        {
          "name": "AI_SCIENCE_PLATFORM_MAX_WORKERS",
          "value": "8"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/ai-science-platform",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### Google Cloud Run

```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: ai-science-platform
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        run.googleapis.com/memory: "4Gi"
        run.googleapis.com/cpu: "2"
        run.googleapis.com/execution-environment: gen2
    spec:
      containers:
      - image: gcr.io/your-project/ai-science-platform:latest
        ports:
        - containerPort: 8080
        env:
        - name: AI_SCIENCE_PLATFORM_LOG_LEVEL
          value: INFO
        - name: AI_SCIENCE_PLATFORM_MAX_WORKERS
          value: "8"
        resources:
          limits:
            memory: 4Gi
            cpu: "2"
```

## ⚡ Performance Tuning

### Memory Optimization

```python
# Configure memory limits
from src.utils.performance import get_memory_optimizer

optimizer = get_memory_optimizer()

# Process large datasets in batches
def process_large_dataset(data):
    batch_size = 1000  # Adjust based on available memory
    results = []
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batch_result = process_batch(batch)
        results.extend(batch_result)
        
        # Force garbage collection
        import gc
        gc.collect()
    
    return results
```

### Parallel Processing Configuration

```python
from src.algorithms.concurrent_discovery import ConcurrentDiscoveryEngine, BatchDiscoveryConfig
from src.utils.performance import ParallelProcessor

# Optimize for CPU-bound tasks
cpu_config = BatchDiscoveryConfig(
    batch_size=50,
    max_workers=8,
    use_processes=True,  # Use processes for CPU-bound work
    memory_limit_mb=2000
)

# Optimize for I/O-bound tasks
io_config = BatchDiscoveryConfig(
    batch_size=200,
    max_workers=16,
    use_processes=False,  # Use threads for I/O-bound work
    memory_limit_mb=1000
)
```

### Caching Configuration

```python
from src.utils.performance import get_cache

# Configure global cache
cache = get_cache(maxsize=512, ttl=3600)  # 1 hour TTL

# Use function-level caching
from src.utils.performance import cached

@cached(maxsize=128, ttl=1800)  # 30 minute TTL
def expensive_computation(data):
    # Your expensive computation here
    return result
```

## 🔒 Security Configuration

### Production Security Settings

```python
from src.utils.security import SecurityConfig, SecurityValidator

# Production security configuration
security_config = SecurityConfig(
    max_file_size_mb=50,  # Limit file sizes
    allowed_file_extensions=['.npy', '.csv', '.json'],
    enable_data_validation=True,
    enable_input_sanitization=True,
    enable_audit_logging=True,
    max_array_size=1_000_000,  # 1M elements max
    max_memory_usage_mb=500
)

validator = SecurityValidator(security_config)
```

### Input Validation

```python
from src.utils.security import validate_input, sanitize_string

def secure_discovery_endpoint(data, context):
    # Validate inputs
    validated_data = validate_input(data, "discovery_data")
    sanitized_context = sanitize_string(context, max_length=100)
    
    # Proceed with discovery
    engine = DiscoveryEngine()
    return engine.discover(validated_data, context=sanitized_context)
```

## 📊 Monitoring and Logging

### Logging Configuration

```python
import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
        }
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'DEBUG',
            'formatter': 'detailed',
            'class': 'logging.FileHandler',
            'filename': 'ai_science_platform.log',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}

logging.config.dictConfig(LOGGING_CONFIG)
```

### Performance Monitoring

```python
from src.utils.performance import get_profiler

profiler = get_profiler()

# Monitor discovery performance
profiler.start_profiling("batch_discovery")
# ... perform discovery operations
metrics = profiler.end_profiling("batch_discovery")

print(f"Discovery took {metrics.execution_time:.3f}s")
print(f"Memory usage: {metrics.memory_usage_mb:.1f}MB")
```

### Health Checks

```python
def health_check():
    """System health check endpoint"""
    health_status = {
        'status': 'healthy',
        'timestamp': time.time(),
        'version': '1.0.0',
        'components': {}
    }
    
    try:
        # Test discovery engine
        from src.algorithms.discovery import DiscoveryEngine
        engine = DiscoveryEngine()
        health_status['components']['discovery'] = 'healthy'
        
        # Test experiment runner
        from src.experiments.runner import ExperimentRunner
        runner = ExperimentRunner()
        health_status['components']['experiments'] = 'healthy'
        
        # Test memory usage
        import psutil
        memory_percent = psutil.virtual_memory().percent
        health_status['memory_usage_percent'] = memory_percent
        
        if memory_percent > 90:
            health_status['status'] = 'warning'
            health_status['warnings'] = ['High memory usage']
            
    except Exception as e:
        health_status['status'] = 'unhealthy'
        health_status['error'] = str(e)
    
    return health_status
```

## 🧪 Testing in Production

### Smoke Tests

```bash
#!/bin/bash
# smoke_test.sh

echo "Running AI Science Platform smoke tests..."

# Test basic functionality
python3 -c "
import sys
sys.path.insert(0, 'src')

from algorithms.discovery import DiscoveryEngine
from utils.data_utils import generate_sample_data

# Generate test data
data, targets = generate_sample_data(size=100, seed=42)

# Test discovery
engine = DiscoveryEngine(discovery_threshold=0.5)
discoveries = engine.discover(data, targets)

print(f'Smoke test passed: {len(discoveries)} discoveries found')
" || exit 1

echo "All smoke tests passed!"
```

### Load Testing

```python
import concurrent.futures
import time
from src.algorithms.concurrent_discovery import ConcurrentDiscoveryEngine
from src.utils.data_utils import generate_sample_data

def load_test(num_concurrent=10, datasets_per_thread=5):
    """Load test the platform"""
    
    def worker_task(worker_id):
        # Create datasets
        datasets = []
        for i in range(datasets_per_thread):
            data, targets = generate_sample_data(size=200, seed=worker_id*100+i)
            datasets.append((data, targets))
        
        # Run discovery
        engine = ConcurrentDiscoveryEngine(discovery_threshold=0.6)
        results = engine.discover_batch(datasets)
        
        return sum(len(discoveries) for discoveries in results)
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        futures = [executor.submit(worker_task, i) for i in range(num_concurrent)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    duration = time.time() - start_time
    total_discoveries = sum(results)
    
    print(f"Load test results:")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Total discoveries: {total_discoveries}")
    print(f"  Discoveries/second: {total_discoveries/duration:.1f}")
    print(f"  Datasets processed: {num_concurrent * datasets_per_thread}")

if __name__ == "__main__":
    load_test()
```

## 🚨 Troubleshooting

### Common Issues

#### Memory Issues
```python
# If you encounter memory issues:
# 1. Reduce batch sizes
config.batch_size = 50

# 2. Enable memory optimization
from src.utils.performance import memory_optimized

@memory_optimized(preserve_precision=False)
def process_data(data):
    return your_processing_function(data)

# 3. Use process-based parallelism for isolation
config.use_processes = True
```

#### Performance Issues
```python
# 1. Profile your code
from src.utils.performance import profiled

@profiled("slow_operation")
def slow_operation():
    # Your code here
    pass

# 2. Optimize array operations
from src.utils.performance import get_memory_optimizer

optimizer = get_memory_optimizer()
optimized_data = optimizer.optimize_array_dtype(data)

# 3. Use caching
from src.utils.performance import cached

@cached(maxsize=128)
def expensive_function(param):
    # Expensive computation
    return result
```

#### Discovery Issues
```python
# If discovery results are poor:
# 1. Adjust threshold
engine = DiscoveryEngine(discovery_threshold=0.5)  # Lower = more discoveries

# 2. Check data quality
from src.utils.data_utils import validate_data
validation = validate_data(data, targets)
if not validation['valid']:
    print(f"Data issues: {validation['issues']}")

# 3. Use adaptive discovery
from src.algorithms.concurrent_discovery import ConcurrentDiscoveryEngine
engine = ConcurrentDiscoveryEngine()
discoveries = await engine.discover_adaptive(data, targets)
```

### Support and Maintenance

- **Logs Location**: `/app/logs/ai_science_platform.log`
- **Configuration**: `/app/config.json`
- **Data Directory**: `/app/data`
- **Results Directory**: `/app/results`

### Backup Strategy

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/ai-science-platform"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p "$BACKUP_DIR/$DATE"

# Backup configuration
cp config.json "$BACKUP_DIR/$DATE/"

# Backup results
tar -czf "$BACKUP_DIR/$DATE/results.tar.gz" results/

# Backup logs
tar -czf "$BACKUP_DIR/$DATE/logs.tar.gz" logs/

echo "Backup completed: $BACKUP_DIR/$DATE"
```

## 🎯 Next Steps

1. **Scale Up**: Increase worker counts and batch sizes based on your hardware
2. **Integrate**: Connect with your existing data pipelines and ML workflows
3. **Extend**: Add custom discovery algorithms and model architectures
4. **Monitor**: Set up comprehensive monitoring and alerting
5. **Optimize**: Profile and optimize for your specific use cases

For additional support, consult the main README.md and example files.