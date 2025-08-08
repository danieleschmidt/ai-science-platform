"""Experiment runner for systematic scientific validation"""

import numpy as np
import json
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    name: str
    description: str
    parameters: Dict[str, Any]
    metrics_to_track: List[str]
    num_runs: int = 3
    seed: Optional[int] = None


@dataclass  
class ExperimentResult:
    """Results from a single experiment run"""
    config_name: str
    run_id: int
    metrics: Dict[str, float]
    execution_time: float
    success: bool
    error_message: Optional[str] = None


class ExperimentRunner:
    """Systematic experiment execution and tracking"""
    
    def __init__(self, results_dir: str = "experiment_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.experiments = {}
        self.results = []
        logger.info(f"ExperimentRunner initialized with results dir: {self.results_dir}")
    
    def register_experiment(self, config: ExperimentConfig) -> None:
        """Register an experiment configuration"""
        self.experiments[config.name] = config
        logger.info(f"Registered experiment: {config.name}")
    
    def run_experiment(self, 
                      experiment_name: str, 
                      experiment_func: Callable[[Dict[str, Any]], Dict[str, float]],
                      data: Optional[np.ndarray] = None) -> List[ExperimentResult]:
        """Run a registered experiment multiple times"""
        
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment {experiment_name} not registered")
        
        config = self.experiments[experiment_name]
        results = []
        
        logger.info(f"Starting experiment: {experiment_name} ({config.num_runs} runs)")
        
        for run_id in range(config.num_runs):
            start_time = time.time()
            
            try:
                # Set seed for reproducibility
                if config.seed is not None:
                    np.random.seed(config.seed + run_id)
                
                # Prepare experiment parameters
                params = config.parameters.copy()
                if data is not None:
                    params['data'] = data
                
                # Run the experiment
                metrics = experiment_func(params)
                
                execution_time = time.time() - start_time
                
                result = ExperimentResult(
                    config_name=experiment_name,
                    run_id=run_id,
                    metrics=metrics,
                    execution_time=execution_time,
                    success=True
                )
                
                logger.info(f"Run {run_id + 1}/{config.num_runs} completed: {metrics}")
                
            except Exception as e:
                execution_time = time.time() - start_time
                result = ExperimentResult(
                    config_name=experiment_name,
                    run_id=run_id,
                    metrics={},
                    execution_time=execution_time,
                    success=False,
                    error_message=str(e)
                )
                
                logger.error(f"Run {run_id + 1}/{config.num_runs} failed: {e}")
            
            results.append(result)
            self.results.append(result)
        
        # Save results
        self._save_results(experiment_name, results)
        logger.info(f"Experiment {experiment_name} completed")
        
        return results
    
    def analyze_results(self, experiment_name: str) -> Dict[str, Any]:
        """Analyze results from an experiment"""
        exp_results = [r for r in self.results if r.config_name == experiment_name]
        
        if not exp_results:
            return {"error": f"No results found for experiment {experiment_name}"}
        
        successful_results = [r for r in exp_results if r.success]
        
        if not successful_results:
            return {"error": f"No successful runs for experiment {experiment_name}"}
        
        analysis = {
            "experiment_name": experiment_name,
            "total_runs": len(exp_results),
            "successful_runs": len(successful_results),
            "success_rate": len(successful_results) / len(exp_results),
            "avg_execution_time": np.mean([r.execution_time for r in successful_results]),
            "metrics_summary": {}
        }
        
        # Analyze metrics
        all_metrics = {}
        for result in successful_results:
            for metric, value in result.metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
        
        for metric, values in all_metrics.items():
            analysis["metrics_summary"][metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "values": values
            }
        
        return analysis
    
    def compare_experiments(self, experiment_names: List[str], metric: str) -> Dict[str, Any]:
        """Compare multiple experiments on a specific metric"""
        comparison = {
            "metric": metric,
            "experiments": {}
        }
        
        for exp_name in experiment_names:
            analysis = self.analyze_results(exp_name)
            
            if "error" in analysis:
                comparison["experiments"][exp_name] = {"error": analysis["error"]}
                continue
            
            if metric in analysis["metrics_summary"]:
                metric_data = analysis["metrics_summary"][metric]
                comparison["experiments"][exp_name] = {
                    "mean": metric_data["mean"],
                    "std": metric_data["std"],
                    "success_rate": analysis["success_rate"]
                }
            else:
                comparison["experiments"][exp_name] = {"error": f"Metric {metric} not found"}
        
        return comparison
    
    def generate_report(self, experiment_name: str) -> str:
        """Generate a formatted report for an experiment"""
        analysis = self.analyze_results(experiment_name)
        
        if "error" in analysis:
            return f"Report Error: {analysis['error']}"
        
        report = [
            f"# Experiment Report: {experiment_name}",
            "",
            f"**Total Runs:** {analysis['total_runs']}",
            f"**Successful Runs:** {analysis['successful_runs']}",
            f"**Success Rate:** {analysis['success_rate']:.2%}",
            f"**Average Execution Time:** {analysis['avg_execution_time']:.3f} seconds",
            "",
            "## Metrics Summary",
            ""
        ]
        
        for metric, summary in analysis["metrics_summary"].items():
            report.extend([
                f"### {metric}",
                f"- Mean: {summary['mean']:.6f}",
                f"- Std: {summary['std']:.6f}",
                f"- Range: [{summary['min']:.6f}, {summary['max']:.6f}]",
                ""
            ])
        
        return "\n".join(report)
    
    def _save_results(self, experiment_name: str, results: List[ExperimentResult]) -> None:
        """Save experiment results to JSON file"""
        filename = self.results_dir / f"{experiment_name}_results.json"
        
        results_data = {
            "experiment_name": experiment_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": [asdict(r) for r in results]
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved to {filename}")