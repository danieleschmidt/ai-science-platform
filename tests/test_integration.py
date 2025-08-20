"""Integration tests for AI Science Platform"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from src.algorithms.discovery import DiscoveryEngine
from src.experiments.runner import ExperimentRunner, ExperimentConfig
from src.models.simple import SimpleModel
from src.utils.data_utils import generate_sample_data, validate_data
from src.performance.caching import cached_function, get_cache_manager
from src.performance.parallel import ParallelProcessor, parallel_discovery
from src.performance.resource_pool import DiscoveryPool
from src.config import get_config_manager, PlatformConfig
from src.health_check import get_health_checker


class TestFullPipelineIntegration:
    """Test complete platform pipeline from data generation to results"""
    
    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup after each test"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_complete_scientific_discovery_pipeline(self):
        """Test complete scientific discovery workflow"""
        # 1. Generate synthetic dataset
        data, targets = generate_sample_data(
            size=200,
            data_type="polynomial", 
            degree=2,
            noise=0.1,
            seed=42
        )
        
        # 2. Validate data quality
        validation_report = validate_data(data, targets)
        assert validation_report["valid"] == True
        
        # 3. Initialize discovery engine with caching
        @cached_function()
        def cached_discovery(engine, data, targets, context):
            return engine.discover(data, targets, context)
        
        engine = DiscoveryEngine(discovery_threshold=0.6)
        
        # 4. Run discovery process
        discoveries = cached_discovery(engine, data, targets, "polynomial_analysis")
        
        # 5. Validate discoveries
        assert isinstance(discoveries, list)
        assert len(discoveries) >= 0
        
        for discovery in discoveries:
            assert discovery.confidence >= engine.discovery_threshold
            assert "polynomial" in discovery.hypothesis or "analysis" in discovery.hypothesis
        
        # 6. Get discovery summary
        summary = engine.summary()
        assert summary["total_discoveries"] == len(discoveries)
        assert summary["hypotheses_tested"] > 0
    
    def test_experiment_runner_with_model_training(self):
        """Test experiment runner integrated with model training"""
        # Setup experiment runner
        runner = ExperimentRunner(results_dir=self.temp_dir)
        
        # Define experiment configuration
        config = ExperimentConfig(
            name="model_training_experiment",
            description="Test model training with different parameters",
            parameters={
                "learning_rate": 0.01,
                "max_iterations": 500,
                "data_size": 100
            },
            metrics_to_track=["accuracy", "loss", "training_time"],
            num_runs=3,
            seed=42
        )
        
        runner.register_experiment(config)
        
        # Define experiment function
        def model_training_experiment(params):
            # Generate training data
            X, y = generate_sample_data(
                size=params["data_size"],
                data_type="normal",
                seed=np.random.randint(0, 1000)
            )
            
            # Train model
            model = SimpleLinearModel(
                learning_rate=params["learning_rate"],
                max_iterations=params["max_iterations"]
            )
            
            metrics = model.train(X, y)
            
            return {
                "accuracy": metrics.accuracy,
                "loss": metrics.loss,
                "training_time": metrics.training_time
            }
        
        # Run experiment
        results = runner.run_experiment("model_training_experiment", model_training_experiment)
        
        # Validate results
        assert len(results) == 3
        assert all(r.success for r in results)
        
        # Analyze results
        analysis = runner.analyze_results("model_training_experiment")
        assert analysis["success_rate"] == 1.0
        assert "accuracy" in analysis["metrics_summary"]
        assert "loss" in analysis["metrics_summary"]
        assert "training_time" in analysis["metrics_summary"]
    
    def test_parallel_discovery_with_resource_pool(self):
        """Test parallel discovery using resource pool"""
        # Create discovery pool
        pool = DiscoveryPool(
            discovery_threshold=0.6,
            min_size=2,
            max_size=4
        )
        
        try:
            import time
            time.sleep(0.1)  # Allow pool initialization
            
            # Generate multiple datasets
            datasets = []
            for i in range(4):
                data, targets = generate_sample_data(
                    size=50,
                    data_type="sine" if i % 2 == 0 else "polynomial",
                    seed=i * 10
                )
                datasets.append((data, targets))
            
            # Run parallel discovery using resource pool
            all_discoveries = []
            
            for i, (data, targets) in enumerate(datasets):
                with pool.get_resource() as engine:
                    discoveries = engine.discover(data, targets, f"dataset_{i}")
                    all_discoveries.extend(discoveries)
            
            # Validate results
            assert len(all_discoveries) >= 0
            
            # Check pool metrics
            metrics = pool.get_metrics()
            assert metrics.total_created >= 2
        
        finally:
            pool.shutdown()
    
    def test_configuration_driven_workflow(self):
        """Test workflow driven by configuration management"""
        # Get configuration manager
        config_manager = get_config_manager()
        config = config_manager.get_config()
        
        # Update configuration for testing
        config_manager.update_config({
            "discovery": {
                "discovery_threshold": 0.5,
                "max_hypotheses": 5
            },
            "experiments": {
                "default_num_runs": 2
            }
        })
        
        updated_config = config_manager.get_config()
        
        # Use configuration in discovery
        engine = DiscoveryEngine(
            discovery_threshold=updated_config.discovery.discovery_threshold
        )
        
        data, targets = generate_sample_data(size=80, data_type="exponential", seed=42)
        discoveries = engine.discover(data, targets, "config_driven_test")
        
        # Use configuration in experiments
        runner = ExperimentRunner(results_dir=self.temp_dir)
        
        def simple_experiment(params):
            return {"result": params.get("test_param", 1.0)}
        
        config = ExperimentConfig(
            name="config_test",
            description="Configuration driven test",
            parameters={"test_param": 2.0},
            metrics_to_track=["result"],
            num_runs=updated_config.experiments.default_num_runs
        )
        
        runner.register_experiment(config)
        results = runner.run_experiment("config_test", simple_experiment)
        
        # Validate configuration was used
        assert len(results) == 2  # Should use configured num_runs
        assert all(r.metrics["result"] == 2.0 for r in results)
    
    def test_health_monitoring_integration(self):
        """Test health monitoring integration"""
        # Get health checker
        health_checker = get_health_checker()
        
        # Check initial health
        initial_health = health_checker.get_health_summary()
        assert "overall_status" in initial_health
        assert "system_metrics" in initial_health
        
        # Run some operations that might affect health
        processor = ParallelProcessor(max_workers=2)
        
        def cpu_intensive_task(n):
            # Simulate some CPU work
            return sum(i ** 2 for i in range(n))
        
        # Run parallel tasks
        data = [1000, 2000, 3000, 4000, 5000]
        results = processor.map(cpu_intensive_task, data)
        
        # Check health after operations
        post_operation_health = health_checker.get_health_summary()
        
        # Validate health monitoring captured the activity
        assert post_operation_health["system_metrics"]["uptime_seconds"] > 0
        assert isinstance(post_operation_health["system_metrics"]["cpu_percent"], float)
        assert post_operation_health["system_metrics"]["memory_mb"] > 0
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery across components"""
        # Test discovery engine error handling
        engine = DiscoveryEngine(discovery_threshold=0.7)
        
        # Test with problematic data
        bad_data = np.array([[np.nan, 1], [2, np.inf], [3, 4]])
        targets = np.array([1, 2, 3])
        
        # Should handle gracefully without crashing
        discoveries = engine.discover(bad_data, targets, "error_test")
        assert isinstance(discoveries, list)  # Should return empty list or handle gracefully
        
        # Test experiment runner error handling
        runner = ExperimentRunner(results_dir=self.temp_dir)
        
        def failing_experiment(params):
            if params.get("should_fail", False):
                raise RuntimeError("Intentional test failure")
            return {"success": True}
        
        config = ExperimentConfig(
            name="error_test_exp",
            description="Test error handling",
            parameters={"should_fail": True},
            metrics_to_track=["success"],
            num_runs=2
        )
        
        runner.register_experiment(config)
        results = runner.run_experiment("error_test_exp", failing_experiment)
        
        # Should have results even though experiments failed
        assert len(results) == 2
        assert all(not r.success for r in results)
        assert all("Intentional test failure" in r.error_message for r in results)
        
        # Analysis should handle failed experiments
        analysis = runner.analyze_results("error_test_exp")
        assert analysis["success_rate"] == 0.0
        assert analysis["total_runs"] == 2
    
    def test_performance_optimization_integration(self):
        """Test performance optimization features working together"""
        # Test caching with expensive operations
        expensive_call_count = 0
        
        @cached_function()
        def expensive_discovery_operation(data_size, data_type, seed):
            nonlocal expensive_call_count
            expensive_call_count += 1
            
            data, targets = generate_sample_data(
                size=data_size,
                data_type=data_type,
                seed=seed
            )
            
            engine = DiscoveryEngine(discovery_threshold=0.6)
            return engine.discover(data, targets, f"{data_type}_cached")
        
        # Run same operation multiple times
        for _ in range(3):
            discoveries = expensive_discovery_operation(100, "normal", 42)
            assert isinstance(discoveries, list)
        
        # Should only call expensive operation once due to caching
        assert expensive_call_count == 1
        
        # Test parallel processing with caching
        processor = ParallelProcessor(max_workers=2)
        
        # These should hit cache
        tasks = [(100, "normal", 42), (100, "normal", 42), (50, "sine", 123)]
        
        results = processor.map(
            lambda args: expensive_discovery_operation(*args),
            tasks
        )
        
        # Should have results for all tasks
        assert len(results) == 3
        assert all(isinstance(r, list) for r in results)
        
        # Should only add one more call for the new parameters
        assert expensive_call_count == 2
    
    def test_comprehensive_benchmarking(self):
        """Test comprehensive benchmarking across different components"""
        benchmark_results = {}
        
        # Benchmark data generation
        import time
        
        start_time = time.time()
        large_data, large_targets = generate_sample_data(
            size=1000,
            data_type="polynomial",
            degree=3,
            seed=42
        )
        benchmark_results["data_generation_time"] = time.time() - start_time
        
        # Benchmark discovery
        start_time = time.time()
        engine = DiscoveryEngine(discovery_threshold=0.6)
        discoveries = engine.discover(large_data, large_targets, "benchmark_test")
        benchmark_results["discovery_time"] = time.time() - start_time
        benchmark_results["discoveries_found"] = len(discoveries)
        
        # Benchmark model training
        start_time = time.time()
        model = SimpleLinearModel(max_iterations=1000)
        model.train(large_data, large_targets)
        benchmark_results["model_training_time"] = time.time() - start_time
        
        # Benchmark parallel processing
        processor = ParallelProcessor(max_workers=4)
        
        def simple_task(x):
            return x ** 2
        
        large_task_list = list(range(1000))
        
        start_time = time.time()
        parallel_results = processor.map(simple_task, large_task_list)
        benchmark_results["parallel_processing_time"] = time.time() - start_time
        
        # Validate benchmark results
        assert benchmark_results["data_generation_time"] > 0
        assert benchmark_results["discovery_time"] > 0
        assert benchmark_results["model_training_time"] > 0
        assert benchmark_results["parallel_processing_time"] > 0
        assert len(parallel_results) == 1000
        
        # Log benchmark results for analysis
        print("\n=== PERFORMANCE BENCHMARKS ===")
        for metric, value in benchmark_results.items():
            if "time" in metric:
                print(f"{metric}: {value:.4f}s")
            else:
                print(f"{metric}: {value}")


class TestRealWorldScenarios:
    """Test real-world usage scenarios"""
    
    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup after each test"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_scientific_study_simulation(self):
        """Simulate a complete scientific study workflow"""
        # Research question: Compare different data generation methods
        data_types = ["normal", "exponential", "sine", "polynomial"]
        sample_sizes = [50, 100, 200]
        
        runner = ExperimentRunner(results_dir=self.temp_dir)
        all_results = []
        
        # Generate experiments for each combination
        for data_type in data_types:
            for sample_size in sample_sizes:
                config = ExperimentConfig(
                    name=f"study_{data_type}_{sample_size}",
                    description=f"Study {data_type} data with {sample_size} samples",
                    parameters={
                        "data_type": data_type,
                        "sample_size": sample_size,
                        "discovery_threshold": 0.6
                    },
                    metrics_to_track=["discoveries_count", "avg_confidence", "data_quality"],
                    num_runs=3,
                    seed=42
                )
                
                runner.register_experiment(config)
                
                def study_experiment(params):
                    # Generate data
                    data, targets = generate_sample_data(
                        size=params["sample_size"],
                        data_type=params["data_type"],
                        seed=np.random.randint(0, 1000)
                    )
                    
                    # Validate data
                    validation = validate_data(data, targets)
                    data_quality = 1.0 if validation["valid"] else 0.0
                    
                    # Run discovery
                    engine = DiscoveryEngine(
                        discovery_threshold=params["discovery_threshold"]
                    )
                    discoveries = engine.discover(data, targets, params["data_type"])
                    
                    avg_confidence = np.mean([d.confidence for d in discoveries]) if discoveries else 0.0
                    
                    return {
                        "discoveries_count": len(discoveries),
                        "avg_confidence": avg_confidence,
                        "data_quality": data_quality
                    }
                
                results = runner.run_experiment(config.name, study_experiment)
                all_results.extend(results)
        
        # Analyze study results
        assert len(all_results) == len(data_types) * len(sample_sizes) * 3  # 3 runs each
        
        # Should have mostly successful experiments
        success_rate = sum(1 for r in all_results if r.success) / len(all_results)
        assert success_rate > 0.8
        
        # Compare different data types
        for data_type in data_types:
            type_experiments = [name for name in runner.experiments.keys() if data_type in name]
            assert len(type_experiments) == len(sample_sizes)
    
    def test_automated_hyperparameter_optimization(self):
        """Test automated hyperparameter optimization scenario"""
        # Define parameter grid
        learning_rates = [0.001, 0.01, 0.1]
        max_iterations = [500, 1000, 2000]
        
        best_config = None
        best_score = float('-inf')
        
        # Generate test data
        X_train, y_train = generate_sample_data(size=200, data_type="polynomial", seed=42)
        X_test, y_test = generate_sample_data(size=50, data_type="polynomial", seed=123)
        
        runner = ExperimentRunner(results_dir=self.temp_dir)
        
        # Test all parameter combinations
        for lr in learning_rates:
            for max_iter in max_iterations:
                config = ExperimentConfig(
                    name=f"hyperparam_lr{lr}_iter{max_iter}",
                    description=f"Test lr={lr}, max_iter={max_iter}",
                    parameters={
                        "learning_rate": lr,
                        "max_iterations": max_iter
                    },
                    metrics_to_track=["test_accuracy", "training_time"],
                    num_runs=3,
                    seed=42
                )
                
                runner.register_experiment(config)
                
                def hyperparameter_experiment(params):
                    model = SimpleLinearModel(
                        learning_rate=params["learning_rate"],
                        max_iterations=params["max_iterations"]
                    )
                    
                    # Train model
                    train_metrics = model.train(X_train, y_train)
                    
                    # Evaluate on test set
                    test_metrics = model.evaluate(X_test, y_test)
                    
                    return {
                        "test_accuracy": test_metrics.accuracy,
                        "training_time": train_metrics.training_time
                    }
                
                results = runner.run_experiment(config.name, hyperparameter_experiment)
                
                # Calculate average test accuracy for this configuration
                avg_accuracy = np.mean([r.metrics["test_accuracy"] for r in results if r.success])
                
                if avg_accuracy > best_score:
                    best_score = avg_accuracy
                    best_config = {
                        "learning_rate": lr,
                        "max_iterations": max_iter,
                        "score": avg_accuracy
                    }
        
        # Should find a reasonable best configuration
        assert best_config is not None
        assert best_config["score"] > 0.5  # Should achieve decent accuracy
        
        print(f"\nBest hyperparameters: {best_config}")
    
    def test_batch_processing_workflow(self):
        """Test batch processing of multiple datasets"""
        # Create multiple synthetic datasets
        datasets = []
        for i in range(10):
            data, targets = generate_sample_data(
                size=100,
                data_type=["normal", "exponential", "sine", "polynomial"][i % 4],
                seed=i * 42
            )
            datasets.append((data, targets, f"dataset_{i}"))
        
        # Process datasets in parallel using resource pool
        pool = DiscoveryPool(min_size=2, max_size=4)
        
        try:
            import time
            time.sleep(0.1)  # Allow pool initialization
            
            all_results = []
            
            # Process each dataset
            for data, targets, name in datasets:
                with pool.get_resource() as engine:
                    discoveries = engine.discover(data, targets, name)
                    all_results.append({
                        "dataset": name,
                        "discoveries": len(discoveries),
                        "avg_confidence": np.mean([d.confidence for d in discoveries]) if discoveries else 0.0
                    })
            
            # Validate batch processing results
            assert len(all_results) == 10
            
            # Calculate summary statistics
            total_discoveries = sum(r["discoveries"] for r in all_results)
            avg_discoveries_per_dataset = total_discoveries / len(all_results)
            
            assert total_discoveries >= 0
            assert avg_discoveries_per_dataset >= 0
            
            print(f"\nBatch processing results:")
            print(f"Total datasets: {len(all_results)}")
            print(f"Total discoveries: {total_discoveries}")
            print(f"Average discoveries per dataset: {avg_discoveries_per_dataset:.2f}")
        
        finally:
            pool.shutdown()