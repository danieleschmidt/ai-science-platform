"""Enhanced AI Science Platform Demo - Generation 3 Features"""

import sys
import time
import logging
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.logging_config import setup_logging
from src.algorithms.discovery import DiscoveryEngine
from src.models.simple import SimpleDiscoveryModel
from src.utils.data_utils import generate_sample_data

# Import Generation 3 enhancements
from src.performance.enhanced_caching import get_cache, cached
from src.performance.concurrent_discovery import ConcurrentDiscoveryEngine
from src.performance.adaptive_scaling import AdaptiveScaler, ScalingPolicy, create_mock_worker

logger = logging.getLogger(__name__)


def demo_enhanced_caching():
    """Demonstrate intelligent caching system"""
    print("\n🚀 GENERATION 3: Enhanced Caching Demo")
    print("=" * 50)
    
    cache = get_cache()
    
    @cached(ttl=300, tags=['demo'])
    def expensive_discovery_computation(data_size: int, complexity: int):
        """Simulate expensive computation"""
        time.sleep(0.2)  # Simulate processing time
        data, _ = generate_sample_data(size=data_size, data_type='normal')
        engine = DiscoveryEngine(discovery_threshold=0.6)
        discoveries = engine.discover(data, context=f"complexity_{complexity}")
        return len(discoveries)
    
    # Test caching performance
    print("Testing caching performance...")
    
    # First call (cache miss)
    start = time.time()
    result1 = expensive_discovery_computation(100, 1)
    first_call_time = time.time() - start
    
    # Second call (cache hit)
    start = time.time()
    result2 = expensive_discovery_computation(100, 1)
    cached_call_time = time.time() - start
    
    print(f"First call (cache miss): {first_call_time:.3f}s → {result1} discoveries")
    print(f"Second call (cache hit): {cached_call_time:.6f}s → {result2} discoveries")
    print(f"Speedup: {first_call_time / cached_call_time:.1f}x")
    
    # Show cache statistics
    stats = cache.get_stats()
    print(f"\nCache Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def demo_concurrent_discovery():
    """Demonstrate concurrent discovery processing"""
    print("\n🚀 GENERATION 3: Concurrent Discovery Demo")
    print("=" * 50)
    
    # Create concurrent discovery engine
    engine = ConcurrentDiscoveryEngine(max_workers=4, use_processes=False)
    
    # Generate multiple datasets
    print("Generating test datasets...")
    datasets = []
    for i in range(12):
        data, _ = generate_sample_data(size=75, data_type='normal')
        datasets.append((data, f"dataset_{i}"))
    
    print(f"Created {len(datasets)} datasets for concurrent processing")
    
    # Process datasets concurrently
    print("Processing datasets concurrently...")
    start = time.time()
    results = engine.discover_parallel(datasets, threshold=0.6)
    concurrent_time = time.time() - start
    
    # Count total discoveries
    total_discoveries = sum(len(discoveries) for discoveries in results.values())
    
    print(f"Concurrent processing completed:")
    print(f"  Time: {concurrent_time:.2f}s")
    print(f"  Total discoveries: {total_discoveries}")
    print(f"  Datasets processed: {len(results)}")
    
    # Show performance stats
    perf_stats = engine.get_performance_stats()
    print(f"\nPerformance Statistics:")
    for key, value in perf_stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")


def demo_adaptive_scaling():
    """Demonstrate adaptive auto-scaling"""
    print("\n🚀 GENERATION 3: Adaptive Scaling Demo")
    print("=" * 50)
    
    # Create scaling policy
    policy = ScalingPolicy(
        name="discovery_demo_scaling",
        min_workers=2,
        max_workers=6,
        scale_up_threshold={
            'cpu_percent': 50.0,
            'queue_length': 3,
            'response_time_ms': 500.0
        },
        scale_down_threshold={
            'cpu_percent': 20.0,
            'queue_length': 1,
            'response_time_ms': 200.0
        },
        scale_up_cooldown=5,  # Faster for demo
        scale_down_cooldown=10
    )
    
    # Create adaptive scaler
    scaler = AdaptiveScaler(policy, create_mock_worker)
    
    print(f"Initial workers: {scaler.get_current_capacity()}")
    
    # Start monitoring
    scaler.start_monitoring(interval=2)
    
    try:
        print("Simulating load patterns...")
        
        # Simulate increasing load
        for i in range(8):
            # Add load to queue
            for j in range(i):
                try:
                    scaler.metrics_queue.put(f"task_{i}_{j}", timeout=0.1)
                except:
                    pass
            
            current_workers = scaler.get_current_capacity()
            queue_size = scaler.metrics_queue.qsize()
            
            print(f"Step {i+1}: Workers={current_workers}, Queue={queue_size}")
            time.sleep(3)
        
        print("\nReducing load...")
        
        # Clear queue to simulate reduced load
        while not scaler.metrics_queue.empty():
            try:
                scaler.metrics_queue.get_nowait()
            except:
                break
        
        # Wait for scale down
        for i in range(5):
            current_workers = scaler.get_current_capacity()
            queue_size = scaler.metrics_queue.qsize()
            print(f"Cooldown {i+1}: Workers={current_workers}, Queue={queue_size}")
            time.sleep(3)
        
        # Show final statistics
        stats = scaler.get_scaling_stats()
        print(f"\nFinal Scaling Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
    finally:
        scaler.stop_monitoring()


def demo_integrated_platform():
    """Demonstrate integrated platform with all Generation 3 features"""
    print("\n🚀 GENERATION 3: Integrated Platform Demo")
    print("=" * 50)
    
    # Initialize components
    print("Initializing enhanced platform components...")
    
    # Enhanced caching
    cache = get_cache()
    
    # Concurrent discovery
    concurrent_engine = ConcurrentDiscoveryEngine(max_workers=4, cache_enabled=True)
    
    # Adaptive scaling
    scaling_policy = ScalingPolicy(
        name="integrated_platform",
        min_workers=1,
        max_workers=4
    )
    scaler = AdaptiveScaler(scaling_policy, create_mock_worker)
    
    print("✅ All components initialized")
    
    # Demonstrate integrated workflow
    print("\nExecuting integrated scientific discovery workflow...")
    
    # Step 1: Generate research datasets
    datasets = []
    for i in range(6):
        data, _ = generate_sample_data(size=100, data_type='normal')
        datasets.append((data, f"research_dataset_{i}"))
    
    print(f"Generated {len(datasets)} research datasets")
    
    # Step 2: Process with caching and concurrency
    start = time.time()
    discovery_results = concurrent_engine.discover_parallel(
        datasets, 
        threshold=0.65
    )
    processing_time = time.time() - start
    
    # Step 3: Analyze results
    total_discoveries = sum(len(results) for results in discovery_results.values())
    successful_datasets = len([r for r in discovery_results.values() if r])
    
    print(f"\n📊 Integrated Platform Results:")
    print(f"  Processing time: {processing_time:.2f}s")
    print(f"  Datasets processed: {len(datasets)}")
    print(f"  Successful analyses: {successful_datasets}")
    print(f"  Total discoveries: {total_discoveries}")
    
    # Show component performance
    cache_stats = cache.get_stats()
    concurrent_stats = concurrent_engine.get_performance_stats()
    
    print(f"\n🎯 Performance Metrics:")
    print(f"  Cache hit rate: {cache_stats.get('hit_rate', 0):.2%}")
    print(f"  Parallel speedup: {concurrent_stats.get('parallel_speedup', 1):.1f}x")
    print(f"  Memory utilization: {cache_stats.get('memory_usage_mb', 0):.1f}MB")
    
    print(f"\n✅ Generation 3 platform demonstration complete!")


def main():
    """Main demonstration function"""
    print("🧬 AI SCIENCE PLATFORM - GENERATION 3 DEMONSTRATION")
    print("=" * 60)
    print("Showcasing advanced optimization and scaling features")
    print()
    
    # Setup logging
    setup_logging()
    logger.info("Starting Generation 3 platform demonstration")
    
    try:
        # Run individual demos
        demo_enhanced_caching()
        demo_concurrent_discovery()
        demo_adaptive_scaling()
        demo_integrated_platform()
        
        print("\n🎉 ALL GENERATION 3 DEMOS COMPLETED SUCCESSFULLY!")
        print("\n📈 Platform Features Demonstrated:")
        print("  ✅ Intelligent caching with adaptive eviction")
        print("  ✅ Concurrent discovery processing")
        print("  ✅ Adaptive auto-scaling based on load")
        print("  ✅ Integrated performance optimization")
        
        return True
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)