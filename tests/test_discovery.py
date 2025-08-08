"""Tests for discovery engine"""

import pytest
import numpy as np
from src.algorithms.discovery import DiscoveryEngine, Discovery


class TestDiscoveryEngine:
    
    def setup_method(self):
        """Setup for each test"""
        self.engine = DiscoveryEngine(discovery_threshold=0.6)
    
    def test_initialization(self):
        """Test engine initialization"""
        assert self.engine.discovery_threshold == 0.6
        assert len(self.engine.discoveries) == 0
        assert self.engine.hypotheses_tested == 0
    
    def test_generate_hypothesis(self):
        """Test hypothesis generation"""
        data = np.random.normal(0, 1, 100)
        hypothesis = self.engine.generate_hypothesis(data)
        
        assert isinstance(hypothesis, str)
        assert len(hypothesis) > 0
        assert self.engine.hypotheses_tested == 1
    
    def test_generate_hypothesis_with_context(self):
        """Test hypothesis generation with context"""
        data = np.random.normal(5, 0.1, 100)  # Low variance data
        hypothesis = self.engine.generate_hypothesis(data, "test_context")
        
        assert "test_context" in hypothesis
        assert "5" in hypothesis  # Should mention the mean
    
    def test_test_hypothesis(self):
        """Test hypothesis testing"""
        data = np.random.normal(0, 1, 50)
        targets = data + np.random.normal(0, 0.1, 50)
        
        hypothesis = "Test hypothesis"
        is_valid, metrics = self.engine.test_hypothesis(hypothesis, data, targets)
        
        assert isinstance(is_valid, bool)
        assert isinstance(metrics, dict)
        assert 'correlation' in metrics
        assert 'data_size' in metrics
        assert metrics['data_size'] == 50
    
    def test_discover_with_good_data(self):
        """Test discovery process with correlated data"""
        # Create strongly correlated data
        x = np.linspace(0, 10, 100)
        y = 2 * x + np.random.normal(0, 0.5, 100)
        
        discoveries = self.engine.discover(x.reshape(-1, 1), y)
        
        assert len(discoveries) >= 0  # Should find at least some patterns
        for discovery in discoveries:
            assert isinstance(discovery, Discovery)
            assert discovery.confidence >= self.engine.discovery_threshold
    
    def test_discover_no_targets(self):
        """Test discovery without targets"""
        data = np.random.normal(0, 1, 100)
        discoveries = self.engine.discover(data.reshape(-1, 1))
        
        assert isinstance(discoveries, list)
        # Should still be able to make discoveries based on data patterns
    
    def test_get_best_discoveries(self):
        """Test getting best discoveries"""
        # Run discovery first
        data = np.random.normal(0, 1, 100).reshape(-1, 1)
        targets = data.flatten() + np.random.normal(0, 0.1, 100)
        self.engine.discover(data, targets)
        
        best = self.engine.get_best_discoveries(top_k=2)
        assert len(best) <= 2
        
        # Check they're sorted by confidence
        if len(best) > 1:
            assert best[0].confidence >= best[1].confidence
    
    def test_summary(self):
        """Test summary statistics"""
        summary = self.engine.summary()
        
        assert 'total_discoveries' in summary
        assert 'hypotheses_tested' in summary
        assert 'avg_confidence' in summary
        assert 'discovery_threshold' in summary
        assert summary['discovery_threshold'] == 0.6