#!/usr/bin/env python3
"""Basic test runner for AI Science Platform"""

import sys
import os
import traceback
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from algorithms.discovery import DiscoveryEngine, Discovery
        print("  ✅ Discovery engine import successful")
        
        from experiments.runner import ExperimentRunner, ExperimentConfig
        print("  ✅ Experiment runner import successful")
        
        from models.base import BaseModel, LinearModel, PolynomialModel
        print("  ✅ Base models import successful")
        
        from utils.data_utils import generate_sample_data, validate_data
        print("  ✅ Data utilities import successful")
        
        from utils.error_handling import robust_execution, ErrorHandler
        print("  ✅ Error handling import successful")
        
        from utils.security import SecurityValidator, validate_input
        print("  ✅ Security utilities import successful")
        
        from utils.performance import cached, profiled, ParallelProcessor
        print("  ✅ Performance utilities import successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import failed: {str(e)}")
        traceback.print_exc()
        return False


def test_basic_functionality():
    """Test basic functionality of core components"""
    print("\nTesting basic functionality...")
    
    try:
        # Test data generation
        from utils.data_utils import generate_sample_data, validate_data
        
        data, targets = generate_sample_data(size=100, data_type="sine", seed=42)
        validation = validate_data(data, targets)
        
        assert validation['valid'], "Data validation failed"
        assert data.shape[0] == 100, "Data size incorrect"
        print("  ✅ Data generation and validation working")
        
        # Test discovery engine
        from algorithms.discovery import DiscoveryEngine
        
        engine = DiscoveryEngine(discovery_threshold=0.6)
        discoveries = engine.discover(data, targets, "test_context")
        
        assert isinstance(discoveries, list), "Discoveries should be a list"
        print(f"  ✅ Discovery engine working ({len(discoveries)} discoveries found)")
        
        # Test model training
        from models.base import LinearModel, PolynomialModel
        
        linear_model = LinearModel(random_seed=42)
        linear_model.fit(data, targets)
        predictions = linear_model.predict(data)
        score = linear_model.score(data, targets)
        
        assert predictions.shape[0] == data.shape[0], "Prediction shape incorrect"
        assert isinstance(score, (int, float)), "Score should be numeric"
        print(f"  ✅ Linear model training working (score: {score:.3f})")
        
        poly_model = PolynomialModel(degree=2, random_seed=42)
        poly_model.fit(data, targets)
        poly_score = poly_model.score(data, targets)
        
        print(f"  ✅ Polynomial model training working (score: {poly_score:.3f})")
        
        # Test experiment runner
        from experiments.runner import ExperimentRunner, ExperimentConfig
        
        runner = ExperimentRunner("./test_results")
        config = ExperimentConfig(
            name="test_experiment",
            description="Test experiment",
            parameters={"threshold": 0.6},
            metrics_to_track=["accuracy"],
            num_runs=2,
            seed=42
        )
        runner.register_experiment(config)
        
        def test_exp_func(params):
            return {"accuracy": np.random.random()}
        
        results = runner.run_experiment("test_experiment", test_exp_func)
        assert len(results) == 2, "Should have 2 experiment results"
        print("  ✅ Experiment runner working")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Functionality test failed: {str(e)}")
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling and security"""
    print("\nTesting error handling and security...")
    
    try:
        from utils.security import SecurityValidator, validate_input
        from utils.error_handling import ErrorHandler
        
        # Test security validation
        validator = SecurityValidator()
        
        # Test valid input
        valid_data = np.array([1, 2, 3, 4, 5])
        is_valid, error = validator.validate_array_input(valid_data, "test_data")
        assert is_valid, f"Valid data should pass validation: {error}"
        print("  ✅ Security validation working")
        
        # Test error handler
        handler = ErrorHandler(log_errors=False, raise_on_error=False)
        
        def failing_function():
            raise ValueError("Test error")
        
        recovered, result = handler.handle_error(
            "test_func", "test_module", ValueError("Test error")
        )
        assert not recovered, "Should not recover without strategy"
        print("  ✅ Error handling working")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error handling test failed: {str(e)}")
        traceback.print_exc()
        return False


def test_performance_features():
    """Test performance optimization features"""
    print("\nTesting performance features...")
    
    try:
        from utils.performance import cached, profiled, MemoryOptimizer, LRUCache
        
        # Test caching
        cache = LRUCache(maxsize=5)
        cache.put("key1", "value1")
        value, hit = cache.get("key1")
        assert hit and value == "value1", "Cache should return stored value"
        print("  ✅ LRU cache working")
        
        # Test memory optimizer
        optimizer = MemoryOptimizer()
        large_array = np.random.random((100, 100)).astype(np.float64)
        optimized = optimizer.optimize_array_dtype(large_array, preserve_precision=False)
        
        memory_saved = large_array.nbytes - optimized.nbytes
        print(f"  ✅ Memory optimizer working (saved {memory_saved} bytes)")
        
        # Test cached decorator
        call_count = 0
        
        @cached(maxsize=5)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        result1 = expensive_function(5)
        result2 = expensive_function(5)
        
        assert result1 == result2 == 10, "Cached function should return correct result"
        assert call_count == 1, "Function should only be called once due to caching"
        print("  ✅ Function caching working")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Performance features test failed: {str(e)}")
        traceback.print_exc()
        return False


def test_advanced_features():
    """Test advanced platform features"""
    print("\nTesting advanced features...")
    
    try:
        # Test concurrent discovery (simplified)
        from algorithms.concurrent_discovery import ConcurrentDiscoveryEngine, BatchDiscoveryConfig
        from utils.data_utils import generate_sample_data
        
        # Create test datasets
        datasets = []
        for i in range(3):
            data, targets = generate_sample_data(size=50, data_type="polynomial", seed=42+i)
            datasets.append((data, targets))
        
        # Test batch discovery
        config = BatchDiscoveryConfig(batch_size=2, max_workers=2, use_processes=False)
        concurrent_engine = ConcurrentDiscoveryEngine(discovery_threshold=0.5, config=config)
        
        batch_results = concurrent_engine.discover_batch(datasets)
        assert len(batch_results) == 3, "Should have results for all datasets"
        print(f"  ✅ Concurrent discovery working ({sum(len(r) for r in batch_results)} total discoveries)")
        
        # Test hierarchical discovery
        data, targets = generate_sample_data(size=200, data_type="sine", seed=42)
        hierarchical_results = concurrent_engine.discover_hierarchical(
            data, targets, "test_hierarchical", max_depth=2, min_samples=50
        )
        assert len(hierarchical_results) > 0, "Should have hierarchical results"
        print(f"  ✅ Hierarchical discovery working ({len(hierarchical_results)} levels)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Advanced features test failed: {str(e)}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("🧪 AI Science Platform - Comprehensive Testing")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Error Handling", test_error_handling),
        ("Performance Features", test_performance_features),
        ("Advanced Features", test_advanced_features)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {str(e)}")
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Platform is ready for deployment!")
    else:
        print(f"⚠️  {total - passed} tests failed - Review and fix issues")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)