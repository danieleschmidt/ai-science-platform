# AI Science Platform - API Documentation

## 🚀 Core API Reference

### Discovery Engine API

#### `DiscoveryEngine`

**Location**: `src/algorithms/discovery.py`

```python
class DiscoveryEngine:
    def __init__(self, discovery_threshold: float = 0.7, 
                 model_config: Dict[str, Any] = None)
```

**Methods**:

##### `discover(data, context="default") -> List[Discovery]`
Execute discovery on dataset with hypothesis testing.

**Parameters**:
- `data` (np.ndarray): Input dataset for analysis
- `context` (str): Context identifier for discovery

**Returns**: List of Discovery objects with hypotheses and confidence scores

**Example**:
```python
engine = DiscoveryEngine(discovery_threshold=0.7)
data = np.random.randn(100, 5)
discoveries = engine.discover(data, context="experiment_1")

for discovery in discoveries:
    print(f"Hypothesis: {discovery.hypothesis}")
    print(f"Confidence: {discovery.confidence:.3f}")
    print(f"P-value: {discovery.p_value:.6f}")
```

##### `summary() -> Dict[str, Any]`
Get comprehensive discovery statistics.

**Returns**: Dictionary with discovery metrics

**Example**:
```python
stats = engine.summary()
print(f"Hypotheses tested: {stats['hypotheses_tested']}")
print(f"Average confidence: {stats['avg_confidence']:.3f}")
```

### Scalable Discovery Engine API

#### `ScalableDiscoveryEngine`

**Location**: `src/algorithms/scalable_discovery.py`

```python
class ScalableDiscoveryEngine:
    def __init__(self, discovery_threshold: float = 0.7,
                 max_workers: int = 8,
                 enable_caching: bool = True,
                 cache_size: int = 1000,
                 enable_auto_scaling: bool = True)
```

##### `async start() -> None`
Initialize and start the scalable discovery engine.

##### `async discover_batch(data_batches, contexts=None, batch_size=None) -> List[BatchDiscoveryResult]`
Process multiple data batches concurrently.

**Parameters**:
- `data_batches` (List[np.ndarray]): List of data arrays to process
- `contexts` (List[str], optional): Context identifiers for each batch
- `batch_size` (int, optional): Auto-batch size for large datasets

**Returns**: List of BatchDiscoveryResult objects

**Example**:
```python
engine = await create_scalable_discovery_engine(max_workers=4)

data_batches = [np.random.randn(50, 4) for _ in range(10)]
results = await engine.discover_batch(data_batches)

for result in results:
    if result.success:
        print(f"Batch {result.batch_id}: {len(result.discoveries)} discoveries")
        print(f"Processing time: {result.processing_time:.2f}s")
```

##### `async discover_streaming(data_stream, context="streaming") -> AsyncGenerator[BatchDiscoveryResult, None]`
Process streaming data for real-time discovery.

**Parameters**:
- `data_stream` (AsyncGenerator[np.ndarray, None]): Async data stream
- `context` (str): Base context identifier

**Yields**: BatchDiscoveryResult objects as they complete

**Example**:
```python
async def data_generator():
    for i in range(20):
        yield np.random.randn(25, 3)
        await asyncio.sleep(0.1)

async for result in engine.discover_streaming(data_generator()):
    print(f"Stream result: {len(result.discoveries)} discoveries")
```

### Performance APIs

#### Async Task Queue

**Location**: `src/performance/async_processing.py`

```python
class AsyncTaskQueue:
    def __init__(self, max_concurrent_tasks: int = 10, 
                 max_queue_size: int = 100)
```

##### `async submit_task(task_id, func, *args, priority=0, **kwargs) -> str`
Submit task to async processing queue.

**Parameters**:
- `task_id` (str): Unique task identifier
- `func` (Callable): Function to execute
- `*args`: Function arguments
- `priority` (int): Task priority (higher = more urgent)
- `**kwargs`: Function keyword arguments

**Returns**: Task ID for result retrieval

##### `async get_task_result(task_id, timeout=None) -> Any`
Retrieve task result by ID.

**Example**:
```python
queue = AsyncTaskQueue(max_concurrent_tasks=4)
await queue.start_workers()

def compute_task(x, y):
    return x * y + np.random.randn()

task_id = await queue.submit_task("calc_1", compute_task, 5, 3, priority=1)
result = await queue.get_task_result(task_id, timeout=10.0)
```

#### Auto-Scaling

**Location**: `src/performance/auto_scaling.py`

```python
class AutoScaler:
    def __init__(self, thresholds: ScalingThresholds = None,
                 monitoring_interval: float = 10.0)
```

##### `set_scaling_callbacks(scale_up_func, scale_down_func)`
Configure scaling callbacks for resource management.

##### `start_monitoring() -> None`
Begin monitoring system metrics for scaling decisions.

**Example**:
```python
from src.performance.auto_scaling import AutoScaler, ScalingThresholds

thresholds = ScalingThresholds(min_workers=2, max_workers=16)
scaler = AutoScaler(thresholds, monitoring_interval=5.0)

def scale_up(count):
    print(f"Scaling up by {count} workers")

def scale_down(count):
    print(f"Scaling down by {count} workers")

scaler.set_scaling_callbacks(scale_up, scale_down)
scaler.start_monitoring()
```

### Reliability APIs

#### Circuit Breaker

**Location**: `src/utils/circuit_breaker.py`

```python
class CircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig)
```

##### `call(func, *args, **kwargs) -> Any`
Execute function with circuit breaker protection.

**Example**:
```python
from src.utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

config = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=30.0,
    expected_errors=[ConnectionError, TimeoutError]
)

breaker = CircuitBreaker(config)

def risky_operation():
    # Potentially failing operation
    if np.random.random() < 0.3:
        raise ConnectionError("Network failure")
    return "Success"

try:
    result = breaker.call(risky_operation)
    print(f"Result: {result}")
except Exception as e:
    print(f"Circuit breaker prevented execution: {e}")
```

##### Decorator Usage
```python
from src.utils.circuit_breaker import circuit_breaker

@circuit_breaker("my_service", failure_threshold=3, recovery_timeout=15.0)
def call_external_service():
    # Service call implementation
    pass
```

#### Retry Mechanism

**Location**: `src/utils/retry.py`

```python
class RetryHandler:
    def __init__(self, config: RetryConfig)
```

##### `execute(func, *args, **kwargs) -> Any`
Execute function with retry logic.

**Example**:
```python
from src.utils.retry import RetryHandler, RetryConfig

config = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=30.0,
    backoff_multiplier=2.0,
    jitter=True
)

handler = RetryHandler(config)

def unstable_function():
    if np.random.random() < 0.6:
        raise TemporaryError("Temporary failure")
    return "Success"

try:
    result = handler.execute(unstable_function)
    print(f"Result: {result}")
except Exception as e:
    print(f"All retry attempts failed: {e}")
```

##### Decorator Usage
```python
from src.utils.retry import retry_on_temporary_error

@retry_on_temporary_error(max_attempts=3, base_delay=0.5)
def fetch_data_from_api():
    # API call implementation
    pass
```

### Caching APIs

#### Cache Manager

**Location**: `src/performance/caching.py`

##### `@cached_function(cache_disabled=False, ttl=None, memory_only=False)`
Decorator for automatic function result caching.

**Example**:
```python
from src.performance.caching import cached_function

@cached_function(ttl=3600, memory_only=False)
def expensive_computation(data: np.ndarray) -> np.ndarray:
    # Expensive processing
    return np.fft.fft2(data)

# First call - computed and cached
result1 = expensive_computation(np.random.randn(100, 100))

# Second call - retrieved from cache
result2 = expensive_computation(np.random.randn(100, 100))
```

##### Manual Cache Control
```python
from src.performance.caching import get_cache_manager

cache = get_cache_manager()

# Get cache statistics
stats = cache.stats()
print(f"Memory cache hit rate: {stats['memory_cache']['hit_rate']:.2%}")

# Clear all caches
cache.clear()

# Enable/disable caching
cache.disable()
cache.enable()
```

### Health Monitoring API

#### Health Checker

**Location**: `src/health_check.py`

```python
def get_health_checker() -> HealthChecker
```

##### `get_health_summary() -> Dict[str, Any]`
Get comprehensive system health status.

**Example**:
```python
from src.health_check import get_health_checker

checker = get_health_checker()
health = checker.get_health_summary()

print(f"Overall Status: {health['overall_status']}")
print(f"System Uptime: {health.get('uptime', 'Unknown')}")
print(f"Memory Usage: {health['system_metrics']['memory_percent']:.1f}%")

for component, status in health['component_health'].items():
    print(f"{component}: {status['status']}")
```

### CLI API

#### Command Line Interface

**Location**: `src/cli.py`

```bash
# System status check
python -m src.cli status

# Run discovery analysis
python -m src.cli discover --threshold 0.7 --context "research_1"

# System health monitoring
python -m src.cli health

# Performance benchmarking
python -m src.cli benchmark

# Cache management
python -m src.cli cache --clear
python -m src.cli cache --stats
```

**Programmatic CLI Access**:
```python
import sys
from src.cli import main

# Simulate CLI call
old_argv = sys.argv
sys.argv = ['ai-science', 'status']
main()
sys.argv = old_argv
```

### Utility APIs

#### Data Processing

**Example Helper Functions**:
```python
# Generate sample scientific data
def generate_sample_data(samples: int = 100, 
                        features: int = 5, 
                        noise_level: float = 0.1) -> np.ndarray:
    """Generate sample data for testing discovery algorithms"""
    base_signal = np.random.randn(samples, features)
    noise = np.random.randn(samples, features) * noise_level
    return base_signal + noise

# Data preprocessing
def preprocess_scientific_data(data: np.ndarray, 
                             normalize: bool = True,
                             remove_outliers: bool = True) -> np.ndarray:
    """Preprocess scientific data for discovery analysis"""
    processed = data.copy()
    
    if remove_outliers:
        # Remove outliers using IQR method
        Q1 = np.percentile(processed, 25, axis=0)
        Q3 = np.percentile(processed, 75, axis=0)
        IQR = Q3 - Q1
        mask = ((processed >= Q1 - 1.5 * IQR) & 
                (processed <= Q3 + 1.5 * IQR)).all(axis=1)
        processed = processed[mask]
    
    if normalize:
        # Z-score normalization
        processed = (processed - np.mean(processed, axis=0)) / np.std(processed, axis=0)
    
    return processed
```

## 🔧 Configuration APIs

### Environment Configuration

**Example configuration loading**:
```python
import os
from pathlib import Path

def load_configuration(environment: str = "development") -> Dict[str, Any]:
    """Load environment-specific configuration"""
    config_file = Path(f".env.{environment}")
    
    config = {
        "discovery_threshold": float(os.getenv("DISCOVERY_THRESHOLD", "0.7")),
        "max_workers": int(os.getenv("MAX_WORKERS", "4")),
        "cache_enabled": os.getenv("CACHE_ENABLED", "true").lower() == "true",
        "auto_scaling": os.getenv("AUTO_SCALING", "false").lower() == "true",
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "backup_enabled": os.getenv("BACKUP_ENABLED", "false").lower() == "true",
        "health_check_enabled": os.getenv("HEALTH_CHECK_ENABLED", "false").lower() == "true"
    }
    
    return config
```

## 📊 Performance Monitoring APIs

### Metrics Collection

```python
from src.algorithms.scalable_discovery import create_scalable_discovery_engine

async def get_system_performance_metrics():
    """Collect comprehensive system performance metrics"""
    engine = await create_scalable_discovery_engine()
    
    metrics = await engine.get_performance_metrics()
    
    performance_report = {
        "discovery_performance": {
            "total_discoveries": metrics["discovery_stats"]["total_discoveries"],
            "avg_processing_time": metrics["discovery_stats"]["avg_processing_time"],
            "successful_batches": metrics["discovery_stats"]["successful_batches"],
            "failed_batches": metrics["discovery_stats"]["failed_batches"]
        },
        "caching_performance": {
            "hit_rate": metrics["cache_stats"]["hit_rate"],
            "cache_enabled": metrics["cache_stats"]["enabled"]
        },
        "scaling_performance": {
            "current_workers": metrics["worker_count"],
            "auto_scaling_active": "autoscaler_stats" in metrics
        }
    }
    
    await engine.shutdown()
    return performance_report

# Usage
import asyncio
metrics = asyncio.run(get_system_performance_metrics())
print(f"Discovery success rate: {metrics['discovery_performance']['successful_batches']/(metrics['discovery_performance']['successful_batches'] + metrics['discovery_performance']['failed_batches'])*100:.1f}%")
```

## 🚀 Quick Start Examples

### Basic Discovery Workflow
```python
import numpy as np
from src.algorithms.discovery import DiscoveryEngine

# 1. Create discovery engine
engine = DiscoveryEngine(discovery_threshold=0.75)

# 2. Prepare scientific data
experimental_data = np.random.randn(200, 6)

# 3. Execute discovery
discoveries = engine.discover(experimental_data, context="experiment_alpha")

# 4. Analyze results
print(f"Found {len(discoveries)} significant discoveries")
for i, discovery in enumerate(discoveries[:3]):
    print(f"\nDiscovery {i+1}:")
    print(f"  Hypothesis: {discovery.hypothesis}")
    print(f"  Confidence: {discovery.confidence:.3f}")
    print(f"  Statistical significance: p={discovery.p_value:.6f}")

# 5. Get summary statistics
summary = engine.summary()
print(f"\nSummary: {summary['hypotheses_tested']} hypotheses tested")
print(f"Average confidence: {summary['avg_confidence']:.3f}")
```

### High-Performance Batch Processing
```python
import asyncio
import numpy as np
from src.algorithms.scalable_discovery import create_scalable_discovery_engine

async def large_scale_discovery():
    # 1. Initialize scalable engine
    engine = await create_scalable_discovery_engine(
        discovery_threshold=0.7,
        max_workers=8,
        enable_auto_scaling=True
    )
    
    # 2. Prepare multiple datasets
    datasets = [np.random.randn(100, 4) for _ in range(20)]
    contexts = [f"experiment_batch_{i}" for i in range(20)]
    
    # 3. Process all batches concurrently
    results = await engine.discover_batch(datasets, contexts)
    
    # 4. Analyze batch results
    successful_batches = [r for r in results if r.success]
    total_discoveries = sum(len(r.discoveries) for r in successful_batches)
    avg_processing_time = np.mean([r.processing_time for r in successful_batches])
    
    print(f"Processed {len(successful_batches)}/{len(results)} batches successfully")
    print(f"Total discoveries: {total_discoveries}")
    print(f"Average processing time: {avg_processing_time:.2f}s")
    
    # 5. Get performance metrics
    metrics = await engine.get_performance_metrics()
    print(f"Cache hit rate: {metrics['cache_stats']['hit_rate']:.1%}")
    
    await engine.shutdown()

# Run the example
asyncio.run(large_scale_discovery())
```

### Production Deployment Example
```python
# deploy_to_production.py
from deploy import Deployer
import logging

logging.basicConfig(level=logging.INFO)

def deploy_ai_science_platform():
    """Deploy AI Science Platform to production"""
    
    deployer = Deployer()
    
    try:
        # Deploy to production environment
        deployer.deploy(environment="production")
        
        # Generate deployment report
        report = deployer.generate_deployment_report()
        
        print("\n🎉 Deployment Summary:")
        print(f"Status: {report['deployment_status']}")
        print(f"Platform: {report['platform']}")
        print(f"Python Version: {report['python_version']}")
        print(f"Deployment Time: {report['deployment_time']}")
        
        print("\n✅ Production deployment completed successfully!")
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = deploy_ai_science_platform()
    exit(0 if success else 1)
```

---

This comprehensive API documentation covers all major components and provides practical examples for each functionality. The APIs are designed for both simple use cases and complex, production-scale scientific research applications.

**🤖 Generated with Claude Code**
**Co-Authored-By: Claude <noreply@anthropic.com>**