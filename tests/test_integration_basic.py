"""Basic integration tests for AI Science Platform"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from src.models.simple import SimpleModel, SimpleDiscoveryModel
from src.algorithms.discovery import DiscoveryEngine
from src.utils.data_utils import generate_sample_data
from src.performance.caching import LRUCache
from src.utils.security import SecurityValidator


class TestBasicIntegration:
    """Basic integration tests that should work reliably"""
    
    def test_model_discovery_integration(self):
        """Test integration between models and discovery algorithms"""
        # Create a simple model
        model = SimpleDiscoveryModel(input_dim=5, hidden_dim=16)
        
        # Create discovery engine
        engine = DiscoveryEngine(discovery_threshold=0.6)
        
        # Generate test data
        features, targets = generate_sample_data(size=50, data_type='normal')
        data = np.hstack([features, np.random.randn(50, 4)])  # Make it 5D
        
        # Use discovery engine
        discoveries = engine.discover(data, context="integration_test")
        
        # Verify integration works
        if isinstance(discoveries, list) and len(discoveries) > 0:
            # Test discovery with model
            for discovery in discoveries[:3]:  # Test first 3
                # Use model to analyze discovery
                hypothesis = discovery.hypothesis
                assert isinstance(hypothesis, str)
                assert len(hypothesis) > 0
                
        # Test model directly
        test_sample = data[0]
        model_output = model.forward(test_sample)
        
        assert hasattr(model_output, 'predictions')
        assert hasattr(model_output, 'confidence')
        assert model_output.confidence >= 0.0
        assert model_output.confidence <= 1.0
    
    def test_caching_security_integration(self):
        """Test integration between caching and security systems"""
        # Create cache
        cache = LRUCache(max_size=10, ttl=60)
        
        # Create security validator
        validator = SecurityValidator()
        
        # Test secure data handling with cache
        test_data = np.random.randn(20, 3)
        
        # Validate data first
        is_valid, error_msg = validator.validate_array_input(test_data, "test_data")
        assert is_valid, f"Data validation failed: {error_msg}"
        
        # Cache the validated data
        data_key = "test_data_validated"
        cache.put(data_key, test_data.tolist())  # Convert to list for JSON serialization
        
        # Retrieve from cache
        found, cached_data = cache.get(data_key)
        assert found
        assert cached_data is not None
        
        # Verify integrity
        cached_array = np.array(cached_data)
        assert cached_array.shape == test_data.shape
        assert np.allclose(cached_array, test_data)
    
    def test_data_pipeline_integration(self):
        """Test complete data processing pipeline"""
        # Generate source data
        features, targets = generate_sample_data(size=100, data_type='sine')
        raw_data = np.hstack([features.reshape(-1, 1), np.random.randn(100, 7)])  # Make it 8D
        
        # Validate with security
        validator = SecurityValidator()
        is_valid, _ = validator.validate_array_input(raw_data, "raw_data")
        assert is_valid
        
        # Process with model
        model = SimpleModel(input_dim=8, hidden_dim=32, output_dim=3)
        
        # Test batch processing
        batch_size = 10
        results = []
        
        for i in range(0, len(raw_data), batch_size):
            batch = raw_data[i:i + batch_size]
            
            for sample in batch:
                output = model.forward(sample)
                results.append(output)
        
        # Verify pipeline results
        assert len(results) == len(raw_data)
        
        for result in results[:5]:  # Check first 5
            assert hasattr(result, 'predictions')
            assert hasattr(result, 'confidence')
            assert len(result.predictions) == 3  # Output dimension
    
    def test_model_training_integration(self):
        """Test model training with data validation"""
        # Create model
        model = SimpleModel(input_dim=4, hidden_dim=16, output_dim=2)
        
        # Generate training data
        features, targets = generate_sample_data(size=80, data_type='normal')
        X_train = np.hstack([features.reshape(-1, 1), np.random.randn(80, 3)])  # Make it 4D
        y_train = np.random.randn(80, 2)  # Match output dimensions
        
        # Validate data
        validator = SecurityValidator()
        is_valid_x, _ = validator.validate_array_input(X_train, "X_train")
        is_valid_y, _ = validator.validate_array_input(y_train, "y_train")
        
        assert is_valid_x and is_valid_y
        
        # Train model
        model.fit(X_train, y_train, epochs=20, learning_rate=0.01)
        
        # Verify training worked
        assert model.trained
        
        # Test prediction
        test_sample = X_train[0]
        prediction = model.predict(test_sample.reshape(1, -1))
        
        assert len(prediction) == 2  # Output dimension
        assert all(np.isfinite(prediction))
    
    def test_error_handling_integration(self):
        """Test error handling across components"""
        # Test with invalid data
        model = SimpleModel(input_dim=3)
        
        # Wrong input dimension should be handled gracefully
        wrong_input = np.random.randn(2)  # 2D input for 3D model
        
        with pytest.raises(ValueError):
            model.forward(wrong_input)
        
        # Test security validation with bad data
        validator = SecurityValidator()
        
        # Test with invalid array
        bad_data = np.array([np.inf, np.nan, 1.0])
        is_valid, error_msg = validator.validate_array_input(bad_data, "bad_data")
        
        assert not is_valid
        assert "infinite values" in error_msg or "NaN values" in error_msg
        
        # Test discovery engine with empty data
        engine = DiscoveryEngine()
        result = engine.discover(np.array([]).reshape(0, 5))
        
        # Should handle gracefully (return error dict or empty list)
        assert isinstance(result, (list, dict))
    
    def test_performance_monitoring_integration(self):
        """Test performance monitoring across components"""
        import time
        
        # Create cache for performance testing
        cache = LRUCache(max_size=100)
        
        # Test cache performance
        test_data = {"key1": "value1", "key2": [1, 2, 3], "key3": {"nested": True}}
        
        # Measure cache operations
        start_time = time.time()
        
        for i in range(50):
            key = f"test_key_{i}"
            cache.put(key, test_data)
        
        put_time = time.time() - start_time
        
        start_time = time.time()
        
        for i in range(50):
            key = f"test_key_{i}"
            found, data = cache.get(key)
            assert found
        
        get_time = time.time() - start_time
        
        # Verify reasonable performance
        assert put_time < 1.0  # Should complete in under 1 second
        assert get_time < 1.0  # Should complete in under 1 second
        
        # Check cache statistics
        stats = cache.stats()
        assert stats['hits'] >= 50
        assert stats['size'] <= 100  # Within size limit
    
    def test_file_operations_integration(self):
        """Test file operations with security validation"""
        validator = SecurityValidator()
        
        # Test with temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test data
            test_data = {"experiment": "integration_test", "results": [1, 2, 3]}
            
            # Generate secure filename
            filename = validator.generate_secure_filename("test_data", ".json")
            file_path = temp_path / filename
            
            # Validate file path
            is_valid, _ = validator.validate_file_path(file_path)
            assert is_valid
            
            # Write and read file
            import json
            with open(file_path, 'w') as f:
                json.dump(test_data, f)
            
            # Verify file exists and has content
            assert file_path.exists()
            assert file_path.stat().st_size > 0
            
            # Load and validate
            result, success, error = validator.safe_json_load(file_path)
            
            assert success, f"Failed to load JSON: {error}"
            assert result == test_data
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # 1. Generate and validate data
        features, targets = generate_sample_data(size=30, data_type='polynomial')
        raw_data = np.hstack([features.reshape(-1, 1), np.random.randn(30, 4)])  # Make it 5D
        
        validator = SecurityValidator()
        is_valid, _ = validator.validate_array_input(raw_data, "workflow_data")
        assert is_valid
        
        # 2. Create and train model
        model = SimpleDiscoveryModel(input_dim=5, hidden_dim=20)
        
        # Generate some training targets
        targets = np.random.randn(30, 1)  # Single target for simplicity
        model.fit(raw_data, targets, epochs=15)
        
        # 3. Run discovery
        engine = DiscoveryEngine(discovery_threshold=0.5)  # Lower threshold for testing
        discoveries = engine.discover(raw_data, context="end_to_end_test")
        
        # 4. Process discoveries with model
        discovery_results = []
        
        if isinstance(discoveries, list):
            for discovery in discoveries[:3]:  # Process first 3
                # Use model for discovery analysis
                test_sample = raw_data[0]  # Use first sample as example
                model_result = model.predict_discovery(test_sample)
                
                discovery_results.append({
                    'discovery': discovery,
                    'model_analysis': model_result
                })
        
        # 5. Cache results
        cache = LRUCache(max_size=50)
        cache.put("end_to_end_results", {
            'num_discoveries': len(discoveries) if isinstance(discoveries, list) else 0,
            'model_trained': model.trained,
            'data_shape': raw_data.shape,
            'workflow_status': 'completed'
        })
        
        # 6. Verify workflow completion
        found, cached_results = cache.get("end_to_end_results")
        assert found
        assert cached_results['workflow_status'] == 'completed'
        assert cached_results['model_trained'] is True


if __name__ == "__main__":
    pytest.main([__file__])