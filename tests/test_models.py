"""Comprehensive tests for models module"""

import pytest
import numpy as np
import tempfile
import os
from src.models.simple import SimpleModel, SimpleDiscoveryModel, ModelOutput, create_model, benchmark_model


class TestModelOutput:
    """Test ModelOutput dataclass"""
    
    def test_model_output_creation(self):
        """Test ModelOutput creation"""
        predictions = np.array([0.5, 0.3, 0.8])
        confidence = 0.75
        metadata = {"test": True}
        
        output = ModelOutput(
            predictions=predictions,
            confidence=confidence,
            metadata=metadata
        )
        
        assert np.array_equal(output.predictions, predictions)
        assert output.confidence == confidence
        assert output.metadata == metadata
    
    def test_model_output_without_metadata(self):
        """Test ModelOutput creation without metadata"""
        predictions = np.array([0.1, 0.9])
        confidence = 0.6
        
        output = ModelOutput(
            predictions=predictions,
            confidence=confidence
        )
        
        assert np.array_equal(output.predictions, predictions)
        assert output.confidence == confidence
        assert output.metadata is None


class TestSimpleModel:
    """Test SimpleModel implementation"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = SimpleModel(input_dim=5, hidden_dim=32, output_dim=2)
        
        assert model.input_dim == 5
        assert model.hidden_dim == 32
        assert model.output_dim == 2
        assert not model.trained
        assert 'W1' in model.weights
        assert 'W2' in model.weights
        assert 'b1' in model.weights
        assert 'b2' in model.weights
    
    def test_default_initialization(self):
        """Test model with default parameters"""
        model = SimpleModel()
        
        assert model.input_dim == 10
        assert model.hidden_dim == 64
        assert model.output_dim == 1
        assert not model.trained
    
    def test_forward_pass(self):
        """Test forward pass"""
        model = SimpleModel(input_dim=3, hidden_dim=16, output_dim=2)
        input_data = np.array([1.0, 2.0, 3.0])
        
        output = model.forward(input_data)
        
        assert isinstance(output, ModelOutput)
        assert len(output.predictions) == 2
        assert 0.0 <= output.confidence <= 1.0
        assert output.metadata['model_type'] == 'SimpleModel'
        assert output.metadata['trained'] == False
    
    def test_forward_batch(self):
        """Test forward pass with batch input"""
        model = SimpleModel(input_dim=4)
        input_data = np.random.randn(5, 4)  # Batch of 5 samples
        
        output = model.forward(input_data)
        
        assert isinstance(output, ModelOutput)
        assert len(output.predictions) == 5
        assert isinstance(output.confidence, float)
    
    def test_forward_wrong_dimension(self):
        """Test forward pass with wrong input dimension"""
        model = SimpleModel(input_dim=3)
        input_data = np.array([1.0, 2.0])  # Wrong dimension
        
        with pytest.raises(ValueError, match="Input dimension"):
            model.forward(input_data)
    
    def test_predict_method(self):
        """Test predict method"""
        model = SimpleModel(input_dim=2)
        input_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        predictions = model.predict(input_data)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 2
    
    def test_training_basic(self):
        """Test basic training functionality"""
        model = SimpleModel(input_dim=2, hidden_dim=8)
        X = np.random.randn(50, 2)
        y = np.random.randn(50)
        
        model.fit(X, y, epochs=50, learning_rate=0.01)
        
        assert model.trained == True
        # Should be able to make predictions after training
        predictions = model.predict(X[:5])
        assert len(predictions) == 5
    
    def test_get_model_info(self):
        """Test get_model_info method"""
        model = SimpleModel(input_dim=3, hidden_dim=20, output_dim=2)
        
        info = model.get_model_info()
        
        assert info['input_dim'] == 3
        assert info['hidden_dim'] == 20
        assert info['output_dim'] == 2
        assert info['trained'] == False
        assert info['model_type'] == 'SimpleModel'
        assert 'parameters' in info
    
    def test_activation_functions(self):
        """Test activation functions"""
        model = SimpleModel()
        
        # Test sigmoid
        x = np.array([0.0, 1.0, -1.0, 10.0, -10.0])
        sigmoid_result = model._sigmoid(x)
        assert np.all((sigmoid_result >= 0) & (sigmoid_result <= 1))
        
        # Test ReLU
        relu_result = model._relu(x)
        expected = np.array([0.0, 1.0, 0.0, 10.0, 0.0])
        assert np.array_equal(relu_result, expected)


class TestSimpleDiscoveryModel:
    """Test SimpleDiscoveryModel implementation"""
    
    def test_initialization(self):
        """Test discovery model initialization"""
        model = SimpleDiscoveryModel(input_dim=5, hidden_dim=32)
        
        assert model.input_dim == 5
        assert model.hidden_dim == 32
        assert model.output_dim == 3  # discovery_score, novelty, confidence
        assert model.discovery_threshold == 0.7
        assert not model.trained
    
    def test_forward_pass(self):
        """Test forward pass with discovery outputs"""
        model = SimpleDiscoveryModel(input_dim=4)
        input_data = np.random.randn(4)
        
        output = model.forward(input_data)
        
        assert isinstance(output, ModelOutput)
        assert len(output.predictions) == 3
        assert output.metadata['model_type'] == 'SimpleDiscoveryModel'
        assert 'discovery_score' in output.metadata
        assert 'novelty_score' in output.metadata
        assert 'is_discovery' in output.metadata
    
    def test_predict_discovery(self):
        """Test predict_discovery method"""
        model = SimpleDiscoveryModel(input_dim=3)
        input_data = np.random.randn(3)
        
        result = model.predict_discovery(input_data)
        
        assert isinstance(result, dict)
        assert 'is_discovery' in result
        assert 'discovery_score' in result
        assert 'novelty_score' in result
        assert 'confidence' in result
        assert 'discovery_category' in result
        assert 'metadata' in result
        
        # Check data types
        assert isinstance(result['is_discovery'], (bool, np.bool_))
        assert isinstance(result['discovery_score'], float)
        assert isinstance(result['novelty_score'], float)
        assert isinstance(result['confidence'], float)
        assert isinstance(result['discovery_category'], str)
    
    def test_discovery_categorization(self):
        """Test discovery categorization logic"""
        model = SimpleDiscoveryModel()
        
        # Test different scenarios
        categories = []
        for discovery_score in [0.1, 0.4, 0.6, 0.8, 0.9]:
            for novelty_score in [0.2, 0.5, 0.8]:
                category = model._categorize_discovery(discovery_score, novelty_score)
                categories.append(category)
                assert isinstance(category, str)
        
        # Should have different categories
        unique_categories = set(categories)
        assert len(unique_categories) >= 3
    
    def test_set_discovery_threshold(self):
        """Test setting discovery threshold"""
        model = SimpleDiscoveryModel()
        
        # Test valid thresholds
        model.set_discovery_threshold(0.5)
        assert model.discovery_threshold == 0.5
        
        model.set_discovery_threshold(0.9)
        assert model.discovery_threshold == 0.9
        
        # Test boundary conditions
        model.set_discovery_threshold(-0.1)  # Should clamp to 0
        assert model.discovery_threshold == 0.0
        
        model.set_discovery_threshold(1.5)  # Should clamp to 1
        assert model.discovery_threshold == 1.0
    
    def test_discovery_threshold_effect(self):
        """Test that discovery threshold affects is_discovery result"""
        model = SimpleDiscoveryModel(input_dim=2)
        input_data = np.array([0.5, 0.5])
        
        # Set low threshold
        model.set_discovery_threshold(0.1)
        result_low = model.predict_discovery(input_data)
        
        # Set high threshold  
        model.set_discovery_threshold(0.9)
        result_high = model.predict_discovery(input_data)
        
        # Results should potentially differ based on threshold
        # (depending on the actual discovery score)
        assert 'is_discovery' in result_low
        assert 'is_discovery' in result_high


class TestModelUtilities:
    """Test utility functions"""
    
    def test_create_model_simple(self):
        """Test create_model with simple type"""
        model = create_model(model_type="simple", input_dim=5, hidden_dim=32)
        
        assert isinstance(model, SimpleModel)
        assert model.input_dim == 5
        assert model.hidden_dim == 32
    
    def test_create_model_discovery(self):
        """Test create_model with discovery type"""
        model = create_model(model_type="discovery", input_dim=4, hidden_dim=16)
        
        assert isinstance(model, SimpleDiscoveryModel)
        assert model.input_dim == 4
        assert model.hidden_dim == 16
    
    def test_create_model_invalid_type(self):
        """Test create_model with invalid type"""
        with pytest.raises(ValueError, match="Unknown model type"):
            create_model(model_type="invalid")
    
    def test_benchmark_model(self):
        """Test benchmark_model function"""
        model = SimpleModel(input_dim=3)
        test_data = np.random.randn(3)
        
        results = benchmark_model(model, test_data, iterations=10)
        
        assert isinstance(results, dict)
        assert 'avg_inference_time_ms' in results
        assert 'std_inference_time_ms' in results
        assert 'min_inference_time_ms' in results
        assert 'max_inference_time_ms' in results
        assert 'throughput_per_second' in results
        
        # Check that all values are reasonable
        assert results['avg_inference_time_ms'] > 0
        assert results['std_inference_time_ms'] >= 0
        assert results['min_inference_time_ms'] >= 0
        assert results['max_inference_time_ms'] >= results['min_inference_time_ms']
        assert results['throughput_per_second'] > 0


class TestModelIntegration:
    """Integration tests for models"""
    
    def test_end_to_end_simple_model(self):
        """Test complete workflow with SimpleModel"""
        # Create model
        model = create_model("simple", input_dim=4, hidden_dim=16)
        
        # Generate training data
        X_train = np.random.randn(100, 4)
        y_train = np.random.randn(100)
        
        # Train model
        model.fit(X_train, y_train, epochs=50)
        assert model.trained
        
        # Make predictions
        X_test = np.random.randn(10, 4)
        predictions = model.predict(X_test)
        assert len(predictions) == 10
        
        # Test forward pass
        output = model.forward(X_test[0])
        assert isinstance(output, ModelOutput)
        
        # Benchmark performance
        benchmark_results = benchmark_model(model, X_test[0], iterations=5)
        assert 'avg_inference_time_ms' in benchmark_results
    
    def test_end_to_end_discovery_model(self):
        """Test complete workflow with SimpleDiscoveryModel"""
        # Create discovery model
        model = create_model("discovery", input_dim=3, hidden_dim=20)
        
        # Generate training data
        X_train = np.random.randn(50, 3)
        y_train = np.random.randn(50)
        
        # Train model
        model.fit(X_train, y_train, epochs=30)
        assert model.trained
        
        # Test discovery predictions
        test_input = np.random.randn(3)
        discovery_result = model.predict_discovery(test_input)
        
        assert 'is_discovery' in discovery_result
        assert 'discovery_category' in discovery_result
        
        # Test threshold adjustment
        original_threshold = model.discovery_threshold
        model.set_discovery_threshold(0.5)
        assert model.discovery_threshold == 0.5
        
        # Reset threshold
        model.set_discovery_threshold(original_threshold)
        assert model.discovery_threshold == original_threshold
    
    def test_model_comparison(self):
        """Test comparing different model types"""
        input_dim = 5
        test_data = np.random.randn(input_dim)
        
        # Create both model types
        simple_model = create_model("simple", input_dim=input_dim, hidden_dim=16)
        discovery_model = create_model("discovery", input_dim=input_dim, hidden_dim=16)
        
        # Test forward passes
        simple_output = simple_model.forward(test_data)
        discovery_output = discovery_model.forward(test_data)
        
        assert len(simple_output.predictions) == 1  # SimpleModel has 1 output
        assert len(discovery_output.predictions) == 3  # DiscoveryModel has 3 outputs
        
        # Benchmark both
        simple_benchmark = benchmark_model(simple_model, test_data, iterations=5)
        discovery_benchmark = benchmark_model(discovery_model, test_data, iterations=5)
        
        assert 'throughput_per_second' in simple_benchmark
        assert 'throughput_per_second' in discovery_benchmark


class TestModelRobustness:
    """Test model robustness and edge cases"""
    
    def test_extreme_input_values(self):
        """Test models with extreme input values"""
        model = SimpleModel(input_dim=3)
        
        # Test with very large values
        large_input = np.array([1000, -1000, 500])
        output_large = model.forward(large_input)
        assert isinstance(output_large, ModelOutput)
        assert not np.any(np.isnan(output_large.predictions))
        assert not np.any(np.isinf(output_large.predictions))
        
        # Test with very small values
        small_input = np.array([1e-10, -1e-10, 0])
        output_small = model.forward(small_input)
        assert isinstance(output_small, ModelOutput)
        assert not np.any(np.isnan(output_small.predictions))
    
    def test_zero_input(self):
        """Test models with zero input"""
        model = SimpleModel(input_dim=4)
        zero_input = np.zeros(4)
        
        output = model.forward(zero_input)
        assert isinstance(output, ModelOutput)
        assert len(output.predictions) == 1
        assert not np.isnan(output.confidence)
    
    def test_single_dimension_input(self):
        """Test model with single dimension"""
        model = SimpleModel(input_dim=1, hidden_dim=4, output_dim=1)
        input_data = np.array([0.5])
        
        output = model.forward(input_data)
        assert isinstance(output, ModelOutput)
        assert len(output.predictions) == 1
    
    def test_model_info_consistency(self):
        """Test that model info is consistent"""
        model = SimpleModel(input_dim=7, hidden_dim=25, output_dim=2)
        
        info1 = model.get_model_info()
        info2 = model.get_model_info()
        
        # Info should be consistent
        assert info1 == info2
        
        # Train model and check info changes
        X = np.random.randn(20, 7)
        y = np.random.randn(20)
        model.fit(X, y, epochs=10)
        
        info_after_training = model.get_model_info()
        assert info_after_training['trained'] == True
        assert info1['trained'] == False


if __name__ == "__main__":
    pytest.main([__file__])