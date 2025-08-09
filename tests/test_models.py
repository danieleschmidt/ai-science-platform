"""Tests for models module"""

import pytest
import numpy as np
import tempfile
import os
from src.models.base import BaseModel, SimpleLinearModel, ModelMetrics


class TestModelMetrics:
    
    def test_initialization(self):
        """Test ModelMetrics initialization"""
        metrics = ModelMetrics()
        assert metrics.accuracy == 0.0
        assert metrics.loss == float('inf')
        assert metrics.training_time == 0.0
        assert metrics.inference_time == 0.0
        assert metrics.memory_usage == 0.0
        assert isinstance(metrics.additional_metrics, dict)
    
    def test_custom_initialization(self):
        """Test ModelMetrics with custom values"""
        metrics = ModelMetrics(
            accuracy=0.95,
            loss=0.05,
            training_time=10.5,
            additional_metrics={"f1_score": 0.93}
        )
        assert metrics.accuracy == 0.95
        assert metrics.loss == 0.05
        assert metrics.training_time == 10.5
        assert metrics.additional_metrics["f1_score"] == 0.93


class MockModel(BaseModel):
    """Mock model for testing BaseModel abstract methods"""
    
    def __init__(self, fail_predict=False, fail_train=False):
        super().__init__("MockModel", {"fail_predict": fail_predict, "fail_train": fail_train})
        self.fail_predict = fail_predict
        self.fail_train = fail_train
        self.training_data = None
    
    def train(self, X, y, **kwargs):
        if self.fail_train:
            raise RuntimeError("Training failed")
        
        self.training_data = (X.copy(), y.copy())
        self.is_trained = True
        
        # Mock metrics
        self.metrics = ModelMetrics(
            accuracy=0.85,
            loss=0.15,
            training_time=1.0,
            additional_metrics={"r2_score": 0.80}
        )
        
        self._model_parameters = {"weights": np.random.randn(X.shape[1]), "bias": 0.1}
        return self.metrics
    
    def predict(self, X):
        if self.fail_predict:
            raise RuntimeError("Prediction failed")
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Mock predictions
        return X.mean(axis=1) + 0.1
    
    def evaluate(self, X, y):
        predictions = self.predict(X)
        mse = np.mean((predictions - y) ** 2)
        
        return ModelMetrics(
            accuracy=0.82,
            loss=mse,
            additional_metrics={"mae": np.mean(np.abs(predictions - y))}
        )


class TestBaseModel:
    
    def test_initialization(self):
        """Test BaseModel initialization"""
        model = MockModel()
        
        assert model.model_name == "MockModel"
        assert not model.is_trained
        assert isinstance(model.metrics, ModelMetrics)
        assert isinstance(model._model_parameters, dict)
    
    def test_custom_initialization(self):
        """Test BaseModel with custom config"""
        config = {"param1": 10, "param2": "test"}
        model = MockModel()
        model.config = config
        
        assert model.config == config
    
    def test_train_success(self):
        """Test successful training"""
        model = MockModel()
        X = np.random.randn(100, 3)
        y = np.random.randn(100)
        
        metrics = model.train(X, y)
        
        assert model.is_trained
        assert isinstance(metrics, ModelMetrics)
        assert metrics.accuracy == 0.85
        assert "r2_score" in metrics.additional_metrics
        assert model.training_data[0].shape == X.shape
    
    def test_train_failure(self):
        """Test training failure"""
        model = MockModel(fail_train=True)
        X = np.random.randn(50, 2)
        y = np.random.randn(50)
        
        with pytest.raises(RuntimeError, match="Training failed"):
            model.train(X, y)
        
        assert not model.is_trained
    
    def test_predict_success(self):
        """Test successful prediction"""
        model = MockModel()
        X_train = np.random.randn(100, 3)
        y_train = np.random.randn(100)
        X_test = np.random.randn(20, 3)
        
        model.train(X_train, y_train)
        predictions = model.predict(X_test)
        
        assert predictions.shape == (20,)
        assert isinstance(predictions, np.ndarray)
    
    def test_predict_not_trained(self):
        """Test prediction without training"""
        model = MockModel()
        X = np.random.randn(10, 3)
        
        with pytest.raises(ValueError, match="Model must be trained before making predictions"):
            model.predict(X)
    
    def test_predict_failure(self):
        """Test prediction failure"""
        model = MockModel(fail_predict=True)
        X_train = np.random.randn(50, 2)
        y_train = np.random.randn(50)
        X_test = np.random.randn(10, 2)
        
        model.train(X_train, y_train)
        
        with pytest.raises(RuntimeError, match="Prediction failed"):
            model.predict(X_test)
    
    def test_evaluate(self):
        """Test model evaluation"""
        model = MockModel()
        X_train = np.random.randn(100, 3)
        y_train = np.random.randn(100)
        X_test = np.random.randn(20, 3)
        y_test = np.random.randn(20)
        
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        
        assert isinstance(metrics, ModelMetrics)
        assert metrics.accuracy == 0.82
        assert "mae" in metrics.additional_metrics
    
    def test_fit_transform(self):
        """Test fit_transform convenience method"""
        model = MockModel()
        X = np.random.randn(50, 2)
        y = np.random.randn(50)
        
        predictions, metrics = model.fit_transform(X, y)
        
        assert model.is_trained
        assert predictions.shape == (50,)
        assert isinstance(metrics, ModelMetrics)
    
    def test_save_load_pickle(self):
        """Test save/load with pickle format"""
        model = MockModel()
        X = np.random.randn(30, 2)
        y = np.random.randn(30)
        model.train(X, y)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            filepath = f.name
        
        try:
            model.save_model(filepath)
            
            # Create new model and load
            new_model = MockModel()
            new_model.load_model(filepath)
            
            assert new_model.is_trained
            assert new_model.model_name == "MockModel"
            assert new_model.metrics.accuracy == 0.85
            assert new_model._model_parameters["bias"] == 0.1
        
        finally:
            os.unlink(filepath)
    
    def test_save_load_json(self):
        """Test save/load with JSON format"""
        model = MockModel()
        X = np.random.randn(30, 2)  
        y = np.random.randn(30)
        model.train(X, y)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            model.save_model(filepath)
            
            # Create new model and load
            new_model = MockModel()
            new_model.load_model(filepath)
            
            assert new_model.is_trained
            assert new_model.model_name == "MockModel"
        
        finally:
            os.unlink(filepath)
    
    def test_get_model_info(self):
        """Test get_model_info method"""
        model = MockModel()
        X = np.random.randn(40, 3)
        y = np.random.randn(40)
        model.train(X, y)
        
        info = model.get_model_info()
        
        assert info["model_name"] == "MockModel"
        assert info["is_trained"] == True
        assert "metrics" in info
        assert info["metrics"]["accuracy"] == 0.85
        assert "parameter_count" in info
    
    def test_reset_model(self):
        """Test model reset"""
        model = MockModel()
        X = np.random.randn(30, 2)
        y = np.random.randn(30)
        model.train(X, y)
        
        assert model.is_trained
        
        model.reset_model()
        
        assert not model.is_trained
        assert model.metrics.accuracy == 0.0
        assert len(model._model_parameters) == 0
    
    def test_string_representation(self):
        """Test string representation"""
        model = MockModel()
        str_repr = str(model)
        
        assert "MockModel" in str_repr
        assert "trained=False" in str_repr
        
        # Train and check again
        X = np.random.randn(20, 2)
        y = np.random.randn(20)
        model.train(X, y)
        
        str_repr = str(model)
        assert "trained=True" in str_repr
        assert "accuracy=0.850" in str_repr


class TestSimpleLinearModel:
    
    def test_initialization(self):
        """Test SimpleLinearModel initialization"""
        model = SimpleLinearModel(learning_rate=0.05, max_iterations=500)
        
        assert model.model_name == "SimpleLinearModel"
        assert model.learning_rate == 0.05
        assert model.max_iterations == 500
        assert model.weights is None
        assert model.bias is None
    
    def test_train_linear_data(self):
        """Test training on linear data"""
        # Generate linear data: y = 2x + 1 + noise
        np.random.seed(42)
        X = np.random.randn(200, 1)
        y = 2 * X.flatten() + 1 + np.random.randn(200) * 0.1
        
        model = SimpleLinearModel(learning_rate=0.01, max_iterations=1000)
        metrics = model.train(X, y)
        
        assert model.is_trained
        assert isinstance(metrics, ModelMetrics)
        assert metrics.accuracy > 0.8  # Should achieve good R²
        assert metrics.loss < 1.0  # Should have low MSE
        assert model.weights is not None
        assert model.bias is not None
        
        # Check if learned parameters are reasonable
        assert abs(model.weights[0] - 2.0) < 0.2  # Should be close to 2
        assert abs(model.bias - 1.0) < 0.2  # Should be close to 1
    
    def test_train_multidimensional(self):
        """Test training on multi-dimensional data"""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        true_weights = np.array([1.5, -2.0, 0.5])
        y = X.dot(true_weights) + np.random.randn(100) * 0.1
        
        model = SimpleLinearModel(learning_rate=0.01, max_iterations=2000)
        metrics = model.train(X, y)
        
        assert model.is_trained
        assert model.weights.shape == (3,)
        assert metrics.accuracy > 0.85
        
        # Check learned weights are reasonable
        for i, true_weight in enumerate(true_weights):
            assert abs(model.weights[i] - true_weight) < 0.3
    
    def test_predict(self):
        """Test prediction"""
        np.random.seed(42)
        X_train = np.random.randn(100, 2)
        y_train = X_train[:, 0] - X_train[:, 1] + np.random.randn(100) * 0.1
        
        model = SimpleLinearModel(max_iterations=1000)
        model.train(X_train, y_train)
        
        X_test = np.random.randn(20, 2)
        predictions = model.predict(X_test)
        
        assert predictions.shape == (20,)
        assert isinstance(predictions, np.ndarray)
    
    def test_evaluate(self):
        """Test evaluation"""
        np.random.seed(42)
        X_train = np.random.randn(150, 1)
        y_train = 3 * X_train.flatten() + np.random.randn(150) * 0.2
        
        X_test = np.random.randn(50, 1)  
        y_test = 3 * X_test.flatten() + np.random.randn(50) * 0.2
        
        model = SimpleLinearModel(max_iterations=1500)
        model.train(X_train, y_train)
        
        eval_metrics = model.evaluate(X_test, y_test)
        
        assert isinstance(eval_metrics, ModelMetrics)
        assert eval_metrics.accuracy > 0.8  # Should have good R²
        assert "r2_score" in eval_metrics.additional_metrics
        assert "mae" in eval_metrics.additional_metrics
        assert "rmse" in eval_metrics.additional_metrics
    
    def test_single_feature(self):
        """Test with single feature data"""
        np.random.seed(42)
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])  # Perfect linear relationship
        
        model = SimpleLinearModel(learning_rate=0.1, max_iterations=1000)
        metrics = model.train(X, y)
        
        assert model.is_trained
        assert metrics.accuracy > 0.95  # Should be nearly perfect
        
        # Test prediction
        X_test = np.array([[6], [7]])
        predictions = model.predict(X_test)
        
        # Should predict close to [12, 14]
        assert abs(predictions[0] - 12) < 1.0
        assert abs(predictions[1] - 14) < 1.0
    
    def test_convergence(self):
        """Test that model converges properly"""
        np.random.seed(42)
        X = np.random.randn(100, 1)
        y = 5 * X.flatten() + 2 + np.random.randn(100) * 0.05  # Low noise
        
        # Test with different iteration counts
        for max_iter in [100, 500, 2000]:
            model = SimpleLinearModel(learning_rate=0.01, max_iterations=max_iter)
            metrics = model.train(X, y)
            
            assert model.is_trained
            if max_iter >= 500:  # Should converge well with enough iterations
                assert metrics.accuracy > 0.9
                assert abs(model.weights[0] - 5.0) < 0.5
                assert abs(model.bias - 2.0) < 0.5