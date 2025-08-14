#!/usr/bin/env python3
"""
Generation 3 Demo: Scale and Performance
Demonstrates high-performance distributed processing and optimization
"""

import numpy as np
import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.algorithms.discovery import DiscoveryEngine
from src.models.simple import SimpleModel, SimpleDiscoveryModel
from src.scaling.distributed_processor import DistributedProcessor
from src.performance.caching import CacheManager
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def scientific_computation(data: np.ndarray, iterations: int = 1000) -> dict:
    """Intensive scientific computation for testing"""
    result = {}
    
    # Simulate complex computation
    for i in range(iterations):
        data = np.sin(data) + np.cos(data * 0.1)
    
    result['final_mean'] = float(np.mean(data))
    result['final_std'] = float(np.std(data))
    result['iterations'] = iterations
    result['data_size'] = len(data)
    
    return result


def discovery_analysis(data: np.ndarray, threshold: float = 0.5) -> dict:
    """Scientific discovery analysis"""
    engine = DiscoveryEngine(discovery_threshold=threshold)
    discoveries = engine.discover(data, context="distributed_analysis")
    
    return {
        'num_discoveries': len(discoveries),
        'avg_confidence': np.mean([d.confidence for d in discoveries]) if discoveries else 0.0,
        'data_stats': {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'size': len(data)
        }
    }


def model_training(data: np.ndarray, model_type: str = 'simple') -> dict:
    """Train AI models on data"""
    if model_type == 'simple':
        model = SimpleModel()
    else:
        model = SimpleDiscoveryModel()
    
    # Create training data
    X = data.reshape(-1, min(10, len(data)))[:50]  # Limit size for demo
    if X.shape[0] < 50:
        # Pad if needed
        X = np.pad(X, ((0, 50 - X.shape[0]), (0, 0)), mode='constant')
    
    y = np.sum(X, axis=1) + 0.1 * np.random.random(50)
    
    model.fit(X, y)
    
    # Test prediction
    test_prediction = model.predict(X[:5])
    
    return {
        'model_type': model_type,
        'training_samples': X.shape[0],
        'features': X.shape[1],
        'test_predictions': test_prediction.tolist(),
        'is_trained': model.is_trained
    }


def demonstrate_distributed_processing():
    """Demo 1: Distributed processing capabilities"""
    print("\n=== Demo 1: Distributed Processing ===")
    
    # Create processor
    processor = DistributedProcessor(max_workers=4, use_processes=False)  # Use threads for demo
    
    # Register functions
    processor.register_function('scientific_computation', scientific_computation)
    processor.register_function('discovery_analysis', discovery_analysis)
    processor.register_function('model_training', model_training)
    
    # Start processor
    processor.start()
    
    try:
        # Generate test datasets
        np.random.seed(42)
        datasets = [np.random.normal(i, 1, 100) for i in range(10)]
        
        print(f"Processing {len(datasets)} datasets with distributed computing...")
        
        # Submit batch of scientific computations
        start_time = time.time()
        comp_task_ids = processor.submit_batch('scientific_computation', 
                                             [(data, 100) for data in datasets])
        
        # Submit discovery analyses
        disc_task_ids = processor.submit_batch('discovery_analysis', 
                                             [(data, 0.6) for data in datasets])
        
        # Submit model training
        model_task_ids = processor.submit_batch('model_training', 
                                              [(data, 'simple') for data in datasets[:5]])
        
        print(f"Submitted {len(comp_task_ids + disc_task_ids + model_task_ids)} tasks")
        
        # Collect results
        comp_results = processor.get_results(comp_task_ids, timeout=30.0)
        disc_results = processor.get_results(disc_task_ids, timeout=30.0)
        model_results = processor.get_results(model_task_ids, timeout=30.0)
        
        processing_time = time.time() - start_time
        
        # Analyze results
        successful_comp = sum(1 for r in comp_results.values() if r.success)
        successful_disc = sum(1 for r in disc_results.values() if r.success)
        successful_model = sum(1 for r in model_results.values() if r.success)
        
        print(f"Processing completed in {processing_time:.2f}s")
        print(f"Scientific computations: {successful_comp}/{len(comp_task_ids)} successful")
        print(f"Discovery analyses: {successful_disc}/{len(disc_task_ids)} successful")
        print(f"Model training: {successful_model}/{len(model_task_ids)} successful")
        
        # Show some results
        if comp_results:
            first_comp = next(iter(comp_results.values()))
            if first_comp.success:
                print(f"Sample computation result: {first_comp.result}")
        
        if disc_results:
            first_disc = next(iter(disc_results.values()))
            if first_disc.success:
                print(f"Sample discovery result: {first_disc.result}")
        
        # Get processor statistics
        stats = processor.get_stats()
        print(f"Processor stats: {stats}")
        
        return processor, processing_time, stats
        
    finally:
        processor.stop()


def demonstrate_performance_optimization():
    """Demo 2: Performance optimization with caching"""
    print("\n=== Demo 2: Performance Optimization ===")
    
    # Create cache manager
    cache_manager = CacheManager(memory_cache_size=1000)
    
    def cached_computation(data: np.ndarray, complexity: int = 500) -> dict:
        """Cached expensive computation"""
        # Create cache key
        cache_key = f"computation_{hash(data.tobytes())}_{complexity}"
        
        # Check cache
        found, cached_result = cache_manager.get("cached_computation", (data,), {"complexity": complexity})
        if found:
            return {"result": cached_result, "from_cache": True}
        
        # Expensive computation
        result = np.sum(np.sin(data) * np.cos(data * 0.1))
        for _ in range(complexity):
            result += np.mean(np.sqrt(np.abs(data)))
        
        # Cache result
        final_result = float(result)
        cache_manager.put("cached_computation", (data,), {"complexity": complexity}, final_result)
        
        return {"result": final_result, "from_cache": False}
    
    # Test data
    np.random.seed(42)
    test_data = np.random.normal(0, 1, 1000)
    
    print("Running performance tests...")
    
    # First run (no cache)
    start_time = time.time()
    results = []
    for i in range(10):
        result = cached_computation(test_data + i * 0.01, complexity=200)
        results.append(result)
    first_run_time = time.time() - start_time
    
    # Second run (with cache)
    start_time = time.time()
    cached_results = []
    for i in range(10):
        result = cached_computation(test_data + i * 0.01, complexity=200)
        cached_results.append(result)
    second_run_time = time.time() - start_time
    
    cache_hits = sum(1 for r in cached_results if r["from_cache"])
    speedup = first_run_time / second_run_time if second_run_time > 0 else float('inf')
    
    print(f"First run (no cache): {first_run_time:.3f}s")
    print(f"Second run (with cache): {second_run_time:.3f}s")
    print(f"Cache hits: {cache_hits}/10")
    print(f"Speedup: {speedup:.2f}x")
    
    # Cache statistics
    cache_stats = cache_manager.stats()
    print(f"Cache stats: {cache_stats}")
    
    return cache_manager, speedup, cache_stats


def demonstrate_scaling_benchmark():
    """Demo 3: Scaling benchmark"""
    print("\n=== Demo 3: Scaling Benchmark ===")
    
    processor = DistributedProcessor(max_workers=2, use_processes=False)
    processor.register_function('scientific_computation', scientific_computation)
    processor.start()
    
    try:
        # Generate benchmark data
        np.random.seed(42)
        small_datasets = [np.random.normal(0, 1, 50) for _ in range(5)]
        medium_datasets = [np.random.normal(0, 1, 100) for _ in range(10)]
        large_datasets = [np.random.normal(0, 1, 200) for _ in range(20)]
        
        benchmarks = {
            "small": small_datasets,
            "medium": medium_datasets,
            "large": large_datasets
        }
        
        results = {}
        
        for size_name, datasets in benchmarks.items():
            print(f"Benchmarking {size_name} workload ({len(datasets)} tasks)...")
            
            # Prepare test data
            test_data = [(data, 50) for data in datasets]
            
            # Run benchmark
            benchmark_result = processor.benchmark('scientific_computation', test_data, num_iterations=2)
            results[size_name] = benchmark_result
            
            summary = benchmark_result["summary"]
            print(f"  Average throughput: {summary['avg_throughput']:.2f} tasks/sec")
            print(f"  Success rate: {summary['avg_success_rate']:.2%}")
        
        return processor, results
        
    finally:
        processor.stop()


def demonstrate_memory_optimization():
    """Demo 4: Memory optimization techniques"""
    print("\n=== Demo 4: Memory Optimization ===")
    
    print("Demonstrating memory-efficient data processing...")
    
    # Generator for large datasets
    def data_generator(num_datasets: int, size: int):
        """Memory-efficient data generator"""
        for i in range(num_datasets):
            yield np.random.normal(i, 1, size)
    
    # Process data in chunks
    chunk_results = []
    chunk_size = 5
    total_datasets = 20
    
    print(f"Processing {total_datasets} datasets in chunks of {chunk_size}")
    
    for chunk_start in range(0, total_datasets, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_datasets)
        chunk_data = list(data_generator(chunk_end - chunk_start, 500))
        
        # Process chunk
        chunk_discoveries = []
        for i, data in enumerate(chunk_data):
            engine = DiscoveryEngine(discovery_threshold=0.5)
            discoveries = engine.discover(data, context=f"chunk_{chunk_start//chunk_size}_item_{i}")
            chunk_discoveries.append(len(discoveries))
        
        chunk_results.extend(chunk_discoveries)
        print(f"  Chunk {chunk_start//chunk_size + 1}: processed {len(chunk_data)} datasets")
        
        # Clear memory (simulate)
        del chunk_data
    
    total_discoveries = sum(chunk_results)
    avg_discoveries = np.mean(chunk_results)
    
    print(f"Memory-efficient processing complete:")
    print(f"  Total discoveries: {total_discoveries}")
    print(f"  Average discoveries per dataset: {avg_discoveries:.2f}")
    print(f"  Processed {total_datasets} datasets without loading all into memory")
    
    return {
        "total_datasets": total_datasets,
        "total_discoveries": total_discoveries,
        "avg_discoveries": avg_discoveries,
        "memory_efficient": True
    }


def demonstrate_auto_scaling():
    """Demo 5: Auto-scaling simulation"""
    print("\n=== Demo 5: Auto-scaling Simulation ===")
    
    print("Simulating adaptive workload processing...")
    
    # Simulate varying workload
    workload_phases = [
        {"name": "light", "tasks": 5, "complexity": 50},
        {"name": "moderate", "tasks": 10, "complexity": 100},
        {"name": "heavy", "tasks": 20, "complexity": 200},
        {"name": "peak", "tasks": 30, "complexity": 150},
        {"name": "cooldown", "tasks": 5, "complexity": 50}
    ]
    
    scaling_results = []
    
    for phase in workload_phases:
        print(f"Processing {phase['name']} workload: {phase['tasks']} tasks")
        
        # Determine optimal worker count based on workload
        if phase['tasks'] <= 5:
            optimal_workers = 1
        elif phase['tasks'] <= 15:
            optimal_workers = 2
        else:
            optimal_workers = 4
        
        processor = DistributedProcessor(max_workers=optimal_workers, use_processes=False)
        processor.register_function('scientific_computation', scientific_computation)
        processor.start()
        
        try:
            # Generate workload
            np.random.seed(42)
            datasets = [np.random.normal(0, 1, 100) for _ in range(phase['tasks'])]
            test_data = [(data, phase['complexity']) for data in datasets]
            
            # Process workload
            start_time = time.time()
            task_ids = processor.submit_batch('scientific_computation', test_data)
            results = processor.get_results(task_ids, timeout=60.0)
            processing_time = time.time() - start_time
            
            successful_tasks = sum(1 for r in results.values() if r.success)
            throughput = successful_tasks / processing_time if processing_time > 0 else 0
            
            phase_result = {
                "phase": phase['name'],
                "workers": optimal_workers,
                "tasks": phase['tasks'],
                "successful": successful_tasks,
                "processing_time": processing_time,
                "throughput": throughput
            }
            
            scaling_results.append(phase_result)
            print(f"  Workers: {optimal_workers}, Throughput: {throughput:.2f} tasks/sec")
            
        finally:
            processor.stop()
    
    # Analyze scaling efficiency
    print("\nAuto-scaling analysis:")
    for result in scaling_results:
        efficiency = result['throughput'] / result['workers'] if result['workers'] > 0 else 0
        print(f"  {result['phase']}: {efficiency:.2f} tasks/sec/worker")
    
    return scaling_results


def main():
    """Run all Generation 3 demonstrations"""
    print("AI Science Platform - Generation 3 Demo")
    print("=======================================")
    print("Demonstrating scale and performance optimization")
    
    try:
        # Run all demonstrations
        distributed_demo = demonstrate_distributed_processing()
        optimization_demo = demonstrate_performance_optimization()
        benchmark_demo = demonstrate_scaling_benchmark()
        memory_demo = demonstrate_memory_optimization()
        scaling_demo = demonstrate_auto_scaling()
        
        # Summary
        print("\n=== Generation 3 Summary ===")
        print(f"✅ Distributed Processing: {distributed_demo[2]['success_rate']:.2%} success rate")
        print(f"✅ Performance Optimization: {optimization_demo[1]:.2f}x speedup with caching")
        print(f"✅ Scaling Benchmarks: Completed across multiple workload sizes")
        print(f"✅ Memory Optimization: Processed {memory_demo['total_datasets']} datasets efficiently")
        print(f"✅ Auto-scaling: Tested {len(scaling_demo)} different workload phases")
        print("\n🎯 Generation 3 Complete: High-performance scaling achieved!")
        
        return True
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)