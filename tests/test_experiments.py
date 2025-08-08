"""Tests for experiment runner"""

import pytest
import numpy as np
import tempfile
import shutil
from src.experiments.runner import ExperimentRunner, ExperimentConfig, ExperimentResult


class TestExperimentRunner:
    
    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.runner = ExperimentRunner(results_dir=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup after each test"""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test runner initialization"""
        assert self.runner.results_dir.exists()
        assert len(self.runner.experiments) == 0
        assert len(self.runner.results) == 0
    
    def test_register_experiment(self):
        """Test experiment registration"""
        config = ExperimentConfig(
            name="test_exp",
            description="Test experiment",
            parameters={"param1": 1.0},
            metrics_to_track=["accuracy", "loss"],
            num_runs=3
        )
        
        self.runner.register_experiment(config)
        assert "test_exp" in self.runner.experiments
        assert self.runner.experiments["test_exp"] == config
    
    def test_run_experiment_success(self):
        """Test successful experiment execution"""
        config = ExperimentConfig(
            name="simple_test",
            description="Simple test experiment", 
            parameters={"multiplier": 2.0},
            metrics_to_track=["result"],
            num_runs=2,
            seed=42
        )
        
        self.runner.register_experiment(config)
        
        def simple_experiment(params):
            multiplier = params["multiplier"]
            data = params.get("data", np.array([1, 2, 3]))
            result = np.mean(data) * multiplier
            return {"result": result}
        
        results = self.runner.run_experiment("simple_test", simple_experiment)
        
        assert len(results) == 2
        assert all(r.success for r in results)
        assert all(r.config_name == "simple_test" for r in results)
        assert all("result" in r.metrics for r in results)
    
    def test_run_experiment_failure(self):
        """Test experiment execution with failures"""
        config = ExperimentConfig(
            name="failing_test",
            description="Failing test experiment",
            parameters={},
            metrics_to_track=["error"],
            num_runs=1
        )
        
        self.runner.register_experiment(config)
        
        def failing_experiment(params):
            raise ValueError("Intentional test failure")
        
        results = self.runner.run_experiment("failing_test", failing_experiment)
        
        assert len(results) == 1
        assert not results[0].success
        assert results[0].error_message == "Intentional test failure"
    
    def test_analyze_results(self):
        """Test results analysis"""
        config = ExperimentConfig(
            name="analysis_test",
            description="Test analysis",
            parameters={"base_value": 5.0},
            metrics_to_track=["output"],
            num_runs=3,
            seed=42
        )
        
        self.runner.register_experiment(config)
        
        def analysis_experiment(params):
            base = params["base_value"]
            noise = np.random.normal(0, 0.1)
            return {"output": base + noise}
        
        self.runner.run_experiment("analysis_test", analysis_experiment)
        analysis = self.runner.analyze_results("analysis_test")
        
        assert "experiment_name" in analysis
        assert analysis["experiment_name"] == "analysis_test"
        assert analysis["total_runs"] == 3
        assert analysis["successful_runs"] == 3
        assert analysis["success_rate"] == 1.0
        assert "metrics_summary" in analysis
        assert "output" in analysis["metrics_summary"]
    
    def test_compare_experiments(self):
        """Test experiment comparison"""
        # Setup two experiments
        for i, name in enumerate(["exp1", "exp2"]):
            config = ExperimentConfig(
                name=name,
                description=f"Experiment {i+1}",
                parameters={"multiplier": i + 1},
                metrics_to_track=["score"],
                num_runs=2,
                seed=42
            )
            self.runner.register_experiment(config)
            
            def experiment(params):
                mult = params["multiplier"]
                return {"score": mult * 10 + np.random.normal(0, 0.1)}
            
            self.runner.run_experiment(name, experiment)
        
        comparison = self.runner.compare_experiments(["exp1", "exp2"], "score")
        
        assert comparison["metric"] == "score"
        assert "experiments" in comparison
        assert "exp1" in comparison["experiments"]
        assert "exp2" in comparison["experiments"]
        assert "mean" in comparison["experiments"]["exp1"]
        assert "mean" in comparison["experiments"]["exp2"]
    
    def test_generate_report(self):
        """Test report generation"""
        config = ExperimentConfig(
            name="report_test",
            description="Test report generation",
            parameters={"value": 10},
            metrics_to_track=["metric1", "metric2"],
            num_runs=2
        )
        
        self.runner.register_experiment(config)
        
        def report_experiment(params):
            val = params["value"]
            return {
                "metric1": val * 0.1,
                "metric2": val * 0.2
            }
        
        self.runner.run_experiment("report_test", report_experiment)
        report = self.runner.generate_report("report_test")
        
        assert isinstance(report, str)
        assert "report_test" in report
        assert "metric1" in report
        assert "metric2" in report
        assert "Success Rate" in report