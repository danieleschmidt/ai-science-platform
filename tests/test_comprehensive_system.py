"""Comprehensive system tests for autonomous SDLC implementation"""

import pytest
import numpy as np
import asyncio
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import components to test
from src.research.autonomous_researcher import AutonomousResearcher, ResearchHypothesis
from src.algorithms.breakthrough_ml import AdaptiveMetaLearner, QuantumInspiredOptimizer, BreakthroughAlgorithmSuite
from src.models.adaptive_models import SelfOrganizingNeuralNetwork, AdaptiveEnsembleModel
from src.monitoring.advanced_monitoring import AdvancedMetricsCollector, IntelligentAlertSystem, ResearchHealthChecker
from src.security.advanced_security import SecurityManager, InputValidator, AuditLogger
from src.performance.hyperscale_system import HyperscaleComputeEngine, DistributedTaskQueue, AdaptiveWorkerPool


class TestAutonomousResearcher:
    """Test autonomous research capabilities"""
    
    @pytest.fixture
    def researcher(self):
        """Create test researcher instance"""
        return AutonomousResearcher(
            research_domain="test_science",
            significance_threshold=0.05,
            effect_size_threshold=0.3
        )
    
    def test_researcher_initialization(self, researcher):
        """Test researcher initialization"""
        assert researcher.research_domain == "test_science"
        assert researcher.significance_threshold == 0.05
        assert researcher.effect_size_threshold == 0.3
        assert len(researcher.hypotheses) == 0
        assert len(researcher.experiments) == 0
    
    def test_hypothesis_generation(self, researcher):
        """Test research hypothesis generation"""
        # Generate test data with clear trend
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        hypothesis = researcher.generate_research_hypothesis(data, context="linear_trend")
        
        assert hypothesis.id.startswith("hyp_")
        assert "trend" in hypothesis.title.lower()
        assert len(hypothesis.predictions) > 0
        assert len(hypothesis.methodology) > 0
        assert hypothesis.confidence_level > 0.5
        
        # Check that hypothesis was stored
        assert len(researcher.hypotheses) == 1
        assert researcher.hypotheses[0] == hypothesis
    
    def test_hypothesis_testing(self, researcher):
        """Test hypothesis testing functionality"""
        # Create test data
        treatment_data = np.random.normal(5, 1, 100)
        control_data = np.random.normal(3, 1, 100)
        
        # Generate hypothesis
        hypothesis = researcher.generate_research_hypothesis(treatment_data, "treatment_effect")
        
        # Test hypothesis
        result = researcher.test_hypothesis(hypothesis, treatment_data, control_data)
        
        assert result.hypothesis_id == hypothesis.id
        assert result.experiment_type == "comparative"
        assert 'mean_treatment' in result.metrics
        assert 'mean_control' in result.metrics
        assert result.p_value >= 0.0
        assert result.effect_size >= 0.0
        assert len(result.interpretation) > 0
        
        # Check that experiment was stored
        assert len(researcher.experiments) == 1
    
    def test_autonomous_research_pipeline(self, researcher):
        """Test full autonomous research pipeline"""
        # Create multiple test datasets
        datasets = [
            np.random.normal(0, 1, 50),
            np.random.exponential(2, 50),
            np.random.uniform(-1, 1, 50)
        ]
        
        research_questions = ["normal_distribution", "exponential_behavior", "uniform_pattern"]
        
        # Run autonomous research
        report = researcher.conduct_autonomous_research(datasets, research_questions)
        
        assert report['research_domain'] == "test_science"
        assert report['datasets_analyzed'] == 3
        assert len(report['hypotheses_generated']) == 3
        assert len(report['experiments_conducted']) <= 3  # Some might fail due to data size
        assert 'key_findings' in report
        assert 'statistical_summary' in report
        
        # Check that hypotheses were generated for each dataset
        assert len(researcher.hypotheses) == 3
    
    def test_research_summary(self, researcher):
        """Test research summary generation"""
        # Generate some research activity
        data = np.random.randn(100)
        hypothesis = researcher.generate_research_hypothesis(data)
        researcher.test_hypothesis(hypothesis, data[:50], data[50:])
        
        summary = researcher.get_research_summary()
        
        assert summary['domain'] == "test_science"
        assert summary['total_hypotheses'] == 1
        assert summary['total_experiments'] == 1
        assert 'average_effect_size' in summary
        assert 'research_quality_score' in summary
        assert 0.0 <= summary['research_quality_score'] <= 1.0


class TestBreakthroughAlgorithms:
    """Test breakthrough ML algorithms"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        return np.random.randn(100, 10)
    
    def test_adaptive_meta_learner(self, sample_data):
        """Test adaptive meta-learning algorithm"""
        learner = AdaptiveMetaLearner(adaptation_rate=0.1)
        
        result = learner.execute(sample_data)
        
        assert result.algorithm_name == "AdaptiveMetaLearner"
        assert 0.0 <= result.breakthrough_score <= 1.0
        assert len(result.novel_insights) > 0
        assert 'execution_time' in result.computational_efficiency
        assert 'reproducibility_metrics' in result.__dict__
        
        # Check that execution was recorded
        assert len(learner.execution_history) == 1
    
    def test_quantum_inspired_optimizer(self, sample_data):
        """Test quantum-inspired optimization"""
        optimizer = QuantumInspiredOptimizer(quantum_coherence=0.8)
        
        # Define simple objective function
        def objective_func(x):
            return -np.sum((x - 0.5)**2)  # Maximum at x = [0.5, 0.5, ...]
        
        result = optimizer.execute(sample_data, objective_function=objective_func)
        
        assert result.algorithm_name == "QuantumInspiredOptimizer"
        assert 0.0 <= result.breakthrough_score <= 1.0
        assert len(result.novel_insights) > 0
        assert 'quantum_advantage' in result.computational_efficiency
        assert 'coherence_utilization' in result.computational_efficiency
        
        # Check quantum-specific insights
        insights_text = ' '.join(result.novel_insights).lower()
        assert any(word in insights_text for word in ['quantum', 'superposition', 'coherence'])
    
    def test_breakthrough_algorithm_suite(self, sample_data):
        """Test comprehensive breakthrough algorithm suite"""
        suite = BreakthroughAlgorithmSuite()
        
        results = suite.run_comprehensive_analysis(sample_data)
        
        assert 'adaptive_meta_learner' in results
        assert 'quantum_optimizer' in results
        assert 'comparative_analysis' in results
        
        # Check comparative analysis
        comparative = results['comparative_analysis']
        if not comparative.get('error'):
            assert 'best_algorithm' in comparative
            assert 'breakthrough_scores' in comparative
            assert 'overall_assessment' in comparative


class TestAdaptiveModels:
    """Test adaptive model architectures"""
    
    @pytest.fixture
    def classification_data(self):
        """Create classification test data"""
        np.random.seed(42)
        X = np.random.randn(200, 5)
        y = (X.sum(axis=1) > 0).astype(int)
        return X, y
    
    @pytest.fixture
    def regression_data(self):
        """Create regression test data"""
        np.random.seed(42)
        X = np.random.randn(200, 3)
        y = X.sum(axis=1) + 0.1 * np.random.randn(200)
        return X, y
    
    def test_self_organizing_neural_network_classification(self, classification_data):
        """Test self-organizing neural network on classification"""
        X, y = classification_data
        
        model = SelfOrganizingNeuralNetwork(
            initial_hidden_size=16,
            growth_threshold=0.2,
            pruning_threshold=0.01
        )
        
        # Train model
        model.fit(X, y)
        
        # Test predictions
        predictions = model.predict(X)
        
        assert predictions.shape[0] == X.shape[0]
        assert len(model.performance_history) > 0
        
        # Check architecture summary
        summary = model.get_architecture_summary()
        assert summary['input_size'] == X.shape[1]
        assert summary['output_size'] == 1
        assert summary['total_parameters'] > 0
    
    def test_self_organizing_neural_network_adaptation(self, classification_data):
        """Test neural network adaptation capabilities"""
        X, y = classification_data
        X_train, X_adapt = X[:150], X[150:]
        y_train, y_adapt = y[:150], y[150:]
        
        model = SelfOrganizingNeuralNetwork(initial_hidden_size=8)
        
        # Initial training
        model.fit(X_train, y_train)
        initial_architecture = model.get_architecture_summary()
        
        # Adaptation
        adapted = model.adapt(X_adapt, y_adapt)
        
        if adapted:
            assert len(model.adaptation_history) > 0
            
        performance_summary = model.get_performance_summary()
        assert 'latest_accuracy' in performance_summary
        assert 'total_adaptations' in performance_summary
    
    def test_adaptive_ensemble_model(self, regression_data):
        """Test adaptive ensemble model"""
        X, y = regression_data
        
        ensemble = AdaptiveEnsembleModel(max_models=3)
        
        # Train ensemble
        ensemble.fit(X, y)
        
        # Test predictions
        predictions = ensemble.predict(X)
        
        assert predictions.shape[0] == X.shape[0]
        assert len(ensemble.base_models) > 0
        assert len(ensemble.model_weights) == len(ensemble.base_models)
        
        # Test adaptation
        X_new = np.random.randn(50, 3)
        y_new = X_new.sum(axis=1) + 0.1 * np.random.randn(50)
        
        adapted = ensemble.adapt(X_new, y_new)
        
        # Check ensemble summary
        summary = ensemble.get_ensemble_summary()
        assert summary['ensemble_size'] > 0
        assert 'diversity_score' in summary
        assert 'adaptation_efficiency' in summary


class TestAdvancedMonitoring:
    """Test advanced monitoring system"""
    
    @pytest.fixture
    def metrics_collector(self):
        """Create metrics collector for testing"""
        collector = AdvancedMetricsCollector(
            retention_hours=1,
            collection_interval_seconds=1,
            anomaly_detection=True
        )
        return collector
    
    def test_metrics_collection(self, metrics_collector):
        """Test metrics collection functionality"""
        # Register custom metric
        def test_metric():
            return 42.0
        
        metrics_collector.register_collector("test_metric", test_metric)
        
        # Manually collect metrics
        metrics_collector._collect_all_metrics()
        
        # Check metrics were recorded
        assert "test_metric" in metrics_collector.metrics
        assert len(metrics_collector.metrics["test_metric"]) > 0
        
        # Check metric summary
        summary = metrics_collector.get_metric_summary("test_metric")
        assert summary['mean'] == 42.0
        assert summary['count'] == 1
    
    def test_alert_system(self, metrics_collector):
        """Test intelligent alert system"""
        alert_system = IntelligentAlertSystem(metrics_collector)
        
        # Add test metric that will trigger alert
        def high_cpu_metric():
            return 95.0  # Above default CPU threshold
        
        metrics_collector.register_collector("cpu_usage", high_cpu_metric)
        
        # Collect metrics
        for _ in range(5):
            metrics_collector._collect_all_metrics()
        
        # Manually evaluate alert rules
        alert_system._check_alert_conditions()
        
        # Check if alerts were generated
        active_alerts = alert_system.get_active_alerts()
        alert_stats = alert_system.get_alert_statistics()
        
        assert isinstance(alert_stats['total_rules'], int)
        assert isinstance(alert_stats['enabled_rules'], int)
    
    def test_health_checker(self, metrics_collector):
        """Test research health checker"""
        alert_system = IntelligentAlertSystem(metrics_collector)
        health_checker = ResearchHealthChecker(metrics_collector, alert_system)
        
        # Run health check
        health_report = health_checker.run_comprehensive_health_check()
        
        assert health_report['overall_status'] in ['HEALTHY', 'WARNING', 'DEGRADED', 'CRITICAL']
        assert 'component_status' in health_report
        assert 'metrics_summary' in health_report
        assert 'recommendations' in health_report
        
        # Check individual components
        components = health_report['component_status']
        expected_components = [
            'system_resources', 'research_pipeline', 'data_quality',
            'model_performance', 'discovery_rate', 'storage_capacity'
        ]
        
        for component in expected_components:
            if component in components:
                assert 'status' in components[component]


class TestAdvancedSecurity:
    """Test advanced security system"""
    
    @pytest.fixture
    def security_manager(self):
        """Create security manager for testing"""
        return SecurityManager()
    
    def test_input_validation(self):
        """Test input validation and sanitization"""
        validator = InputValidator()
        
        # Test valid inputs
        valid_email = validator.validate_and_sanitize("test@example.com", "email")
        assert valid_email == "test@example.com"
        
        valid_string = validator.validate_and_sanitize("hello world", "safe_string")
        assert valid_string == "hello world"
        
        valid_int = validator.validate_and_sanitize("123", "int")
        assert valid_int == 123
        
        # Test dangerous input detection
        with pytest.raises(Exception):  # Should raise SecurityError
            validator.validate_and_sanitize("'; DROP TABLE users; --", "safe_string")
        
        with pytest.raises(Exception):  # Should raise SecurityError
            validator.validate_and_sanitize("<script>alert('xss')</script>", "safe_string")
    
    def test_user_authentication(self, security_manager):
        """Test user authentication system"""
        # Test successful authentication
        result = security_manager.authenticate_user("test_user", "validpassword123")
        
        assert result['success'] == True
        assert 'token' in result
        assert 'expires_at' in result
        assert 'scopes' in result
        
        # Test failed authentication
        result = security_manager.authenticate_user("test_user", "short")
        assert result['success'] == False
        assert result['reason'] == 'invalid_credentials'
    
    def test_authorization_system(self, security_manager):
        """Test authorization system"""
        # First authenticate to get token
        auth_result = security_manager.authenticate_user("test_user", "validpassword123")
        token = auth_result['token']
        
        # Test authorization
        authz_result = security_manager.authorize_action(
            token, "test_resource", "read"
        )
        
        assert authz_result['authorized'] == True
        assert authz_result['user_id'] == "test_user"
        
        # Test invalid token
        authz_result = security_manager.authorize_action(
            "invalid_token", "test_resource", "read"
        )
        
        assert authz_result['authorized'] == False
        assert authz_result['reason'] == 'invalid_token'
    
    def test_audit_logging(self):
        """Test security audit logging"""
        audit_logger = AuditLogger()
        
        # Log authentication attempt
        audit_logger.log_authentication_attempt("test_user", True, "127.0.0.1")
        
        # Log authorization check
        audit_logger.log_authorization_check("test_user", "resource", "action", True)
        
        # Get security events
        events = audit_logger.get_security_events(hours=24)
        assert len(events) >= 2
        
        # Get security summary
        summary = audit_logger.get_security_summary(hours=24)
        assert summary['total_events'] >= 2
        assert 'event_types' in summary
        assert 'outcomes' in summary


class TestHyperscaleSystem:
    """Test hyperscale computation system"""
    
    @pytest.fixture
    def task_queue(self):
        """Create task queue for testing"""
        return DistributedTaskQueue(max_queue_size=100)
    
    @pytest.mark.asyncio
    async def test_task_queue_operations(self, task_queue):
        """Test task queue operations"""
        from src.performance.hyperscale_system import ComputationTask
        
        # Create test task
        task = ComputationTask(
            task_id="test_task_1",
            task_type="matrix_computation",
            data=np.random.randn(10, 10),
            parameters={'operation': 'multiply'},
            priority=5
        )
        
        # Submit task
        success = await task_queue.submit_task(task)
        assert success == True
        
        # Get task statistics
        stats = task_queue.get_queue_stats()
        assert stats['total_queued'] == 1
        assert stats['total_queue_size'] == 1
    
    @pytest.mark.asyncio
    async def test_worker_pool_initialization(self):
        """Test worker pool initialization"""
        from src.performance.hyperscale_system import DistributedTaskQueue
        
        task_queue = DistributedTaskQueue(max_queue_size=50)
        worker_pool = AdaptiveWorkerPool(min_workers=1, max_workers=2)
        
        # Initialize worker pool
        await worker_pool.initialize(task_queue)
        
        # Check pool stats
        stats = worker_pool.get_pool_stats()
        assert stats['total_workers'] >= 1
        assert stats['active_workers'] >= 1
        
        # Shutdown
        await worker_pool.shutdown()
    
    @pytest.mark.asyncio
    async def test_hyperscale_compute_engine(self):
        """Test complete hyperscale compute engine"""
        engine = HyperscaleComputeEngine(min_workers=1, max_workers=2, queue_size=50)
        
        try:
            # Initialize engine
            await engine.initialize()
            
            # Submit test computation
            task_id = await engine.submit_computation(
                task_type="matrix_computation",
                data=np.random.randn(5, 5),
                parameters={'operation': 'multiply'},
                priority=5
            )
            
            assert task_id is not None
            assert task_id.startswith("matrix_computation_")
            
            # Get system status
            status = engine.get_system_status()
            assert status['system_info']['status'] == 'running'
            assert 'queue_status' in status
            assert 'worker_status' in status
            assert 'performance' in status
            
            # Wait briefly for task processing
            await asyncio.sleep(2)
            
        finally:
            # Shutdown engine
            await engine.shutdown()


class TestSystemIntegration:
    """Test system integration and end-to-end workflows"""
    
    def test_research_to_monitoring_integration(self):
        """Test integration between research and monitoring systems"""
        # Create researcher
        researcher = AutonomousResearcher("integration_test")
        
        # Create monitoring system
        metrics_collector = AdvancedMetricsCollector(retention_hours=1)
        
        # Register research metrics
        def discovery_count():
            return float(len(researcher.discoveries))
        
        def hypothesis_count():
            return float(len(researcher.hypotheses))
        
        metrics_collector.register_collector("research_discoveries", discovery_count)
        metrics_collector.register_collector("research_hypotheses", hypothesis_count)
        
        # Generate research activity
        data = np.random.randn(100)
        researcher.generate_research_hypothesis(data)
        researcher.discover(data)
        
        # Collect metrics
        metrics_collector._collect_all_metrics()
        
        # Verify metrics were captured
        discovery_summary = metrics_collector.get_metric_summary("research_discoveries")
        hypothesis_summary = metrics_collector.get_metric_summary("research_hypotheses")
        
        assert discovery_summary['count'] > 0
        assert hypothesis_summary['count'] > 0
    
    def test_security_integration(self):
        """Test security integration across systems"""
        security_manager = SecurityManager()
        
        # Simulate secure research workflow
        auth_result = security_manager.authenticate_user("researcher", "securepassword123")
        assert auth_result['success']
        
        token = auth_result['token']
        
        # Test authorization for different actions
        read_auth = security_manager.authorize_action(token, "research_data", "read")
        assert read_auth['authorized']
        
        write_auth = security_manager.authorize_action(token, "research_results", "create")
        assert write_auth['authorized']
        
        # Test secure data access
        test_data = "sensitive research findings"
        secure_access = security_manager.secure_data_access(
            "researcher", "research_results", "read", test_data
        )
        
        assert secure_access['success']
    
    @pytest.mark.asyncio
    async def test_full_system_workflow(self):
        """Test complete system workflow integration"""
        # Initialize all major components
        researcher = AutonomousResearcher("system_test")
        security_manager = SecurityManager()
        compute_engine = HyperscaleComputeEngine(min_workers=1, max_workers=2, queue_size=20)
        
        try:
            # Initialize compute engine
            await compute_engine.initialize()
            
            # Authenticate user
            auth_result = security_manager.authenticate_user("system_user", "systempassword123")
            assert auth_result['success']
            
            # Generate research hypothesis
            data = np.random.randn(50, 5)
            hypothesis = researcher.generate_research_hypothesis(data, "system_integration_test")
            
            # Submit computation tasks for research validation
            task_ids = []
            
            # Matrix computation task
            matrix_task_id = await compute_engine.submit_computation(
                task_type="matrix_computation",
                data=data,
                parameters={'operation': 'correlation'},
                priority=7
            )
            task_ids.append(matrix_task_id)
            
            # Data analysis task
            analysis_task_id = await compute_engine.submit_computation(
                task_type="data_analysis",
                data=data,
                parameters={'analysis_type': 'descriptive'},
                priority=6
            )
            task_ids.append(analysis_task_id)
            
            # Verify tasks were submitted
            assert len(task_ids) == 2
            
            # Get system status
            system_status = compute_engine.get_system_status()
            assert system_status['system_info']['status'] == 'running'
            
            # Wait for tasks to process
            await asyncio.sleep(3)
            
            # Test research hypothesis with computed results
            test_result = researcher.test_hypothesis(hypothesis, data[:25], data[25:])
            assert test_result.hypothesis_id == hypothesis.id
            
            # Get research summary
            research_summary = researcher.get_research_summary()
            assert research_summary['total_hypotheses'] >= 1
            assert research_summary['total_experiments'] >= 1
            
        finally:
            # Cleanup
            await compute_engine.shutdown()


class TestQualityGates:
    """Test quality gates and performance benchmarks"""
    
    def test_performance_benchmarks(self):
        """Test that system meets performance benchmarks"""
        # Test research speed
        researcher = AutonomousResearcher("benchmark_test")
        
        start_time = time.time()
        data = np.random.randn(1000)
        hypothesis = researcher.generate_research_hypothesis(data)
        hypothesis_time = time.time() - start_time
        
        assert hypothesis_time < 5.0, "Hypothesis generation too slow"
        
        start_time = time.time()
        result = researcher.test_hypothesis(hypothesis, data[:500], data[500:])
        test_time = time.time() - start_time
        
        assert test_time < 10.0, "Hypothesis testing too slow"
        
        # Test algorithm performance
        learner = AdaptiveMetaLearner()
        data = np.random.randn(500, 20)
        
        start_time = time.time()
        result = learner.execute(data)
        execution_time = time.time() - start_time
        
        assert execution_time < 30.0, "Algorithm execution too slow"
        assert result.breakthrough_score > 0.0, "Algorithm produced no breakthrough"
    
    def test_memory_usage(self):
        """Test memory usage stays within acceptable bounds"""
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large models and data
        model = SelfOrganizingNeuralNetwork(initial_hidden_size=128)
        data = np.random.randn(1000, 50)
        targets = np.random.randn(1000)
        
        model.fit(data, targets)
        
        # Adaptive ensemble
        ensemble = AdaptiveEnsembleModel(max_models=5)
        ensemble.fit(data, targets)
        
        # Check memory increase
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < 1000, f"Memory usage too high: {memory_increase:.1f} MB"
    
    def test_error_handling_robustness(self):
        """Test system robustness under error conditions"""
        # Test with invalid data
        researcher = AutonomousResearcher("error_test")
        
        # Empty data should be handled gracefully
        with pytest.raises(Exception):
            researcher.generate_research_hypothesis(np.array([]))
        
        # Test with mismatched data shapes
        data1 = np.random.randn(100)
        data2 = np.random.randn(50)  # Different size
        
        hypothesis = researcher.generate_research_hypothesis(data1)
        
        with pytest.raises(Exception):
            researcher.test_hypothesis(hypothesis, data1, data2)
        
        # Test model robustness
        model = SelfOrganizingNeuralNetwork()
        
        with pytest.raises(Exception):
            model.predict(np.random.randn(10, 5))  # Model not trained
    
    def test_security_compliance(self):
        """Test security compliance requirements"""
        security_manager = SecurityManager()
        
        # Test password policy enforcement
        weak_password_result = security_manager.authenticate_user("test", "123")
        assert weak_password_result['success'] == False
        
        # Test rate limiting
        for i in range(10):
            result = security_manager.authenticate_user(f"user_{i}", "password123")
            # Should handle rapid requests without crashing
            assert 'success' in result
        
        # Test input sanitization
        validator = InputValidator()
        
        # SQL injection attempt
        with pytest.raises(Exception):
            validator.validate_and_sanitize("'; DROP TABLE users; --", "safe_string")
        
        # XSS attempt
        with pytest.raises(Exception):
            validator.validate_and_sanitize("<script>alert('xss')</script>", "safe_string")
    
    @pytest.mark.asyncio
    async def test_scalability_limits(self):
        """Test system behavior at scale limits"""
        # Test task queue at capacity
        task_queue = DistributedTaskQueue(max_queue_size=10)
        
        from src.performance.hyperscale_system import ComputationTask
        
        # Fill queue to capacity
        for i in range(12):  # Try to exceed capacity
            task = ComputationTask(
                task_id=f"scale_test_{i}",
                task_type="matrix_computation",
                data=np.random.randn(5, 5),
                parameters={'operation': 'multiply'}
            )
            
            success = await task_queue.submit_task(task)
            if i < 10:
                assert success == True
            else:
                assert success == False  # Should reject when at capacity
        
        stats = task_queue.get_queue_stats()
        assert stats['total_queue_size'] == 10  # At capacity, not over


def run_comprehensive_tests():
    """Run all comprehensive system tests"""
    
    # Configure pytest to run with detailed output
    pytest_args = [
        __file__,
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--durations=10",  # Show 10 slowest tests
    ]
    
    print("🧪 Running Comprehensive System Tests...")
    print("=" * 80)
    
    # Run tests
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("\n✅ All tests passed! System meets quality gates.")
        print("🎉 SDLC Implementation verified and production-ready!")
    else:
        print(f"\n❌ Some tests failed. Exit code: {exit_code}")
        print("🔧 Please review and fix issues before production deployment.")
    
    return exit_code


if __name__ == "__main__":
    run_comprehensive_tests()