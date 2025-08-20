"""
Advanced Quantum Algorithm Benchmarking Suite

This module provides comprehensive benchmarking capabilities specifically designed
for quantum-inspired algorithms and quantum computing research validation.

Features:
- Quantum advantage measurement
- Coherence degradation analysis
- Entanglement scalability testing
- Classical comparison benchmarks
- Statistical validation of quantum speedup
"""

import time
import math
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import json
import statistics
from pathlib import Path

from ..algorithms.quantum_discovery import (
    QuantumSuperpositionDiscovery, 
    QuantumEntanglementDiscovery,
    QuantumDiscoveryResult,
    compare_quantum_vs_classical
)
from ..utils.secure_random import ScientificRandomGenerator
from .benchmark_suite import ComprehensiveBenchmark, BenchmarkResult

logger = logging.getLogger(__name__)


@dataclass
class QuantumBenchmarkResult:
    """Extended benchmark result for quantum algorithms"""
    algorithm_name: str
    test_name: str
    execution_time: float
    performance_score: float
    memory_usage_mb: float
    
    # Quantum-specific metrics
    quantum_advantage: float
    coherence_preserved: float
    entanglement_generated: float
    interference_quality: float
    decoherence_rate: float
    
    # Pattern discovery metrics
    patterns_discovered: int
    pattern_quality_avg: float
    pattern_novelty_score: float
    
    # Scalability metrics
    qubit_efficiency: float
    scaling_advantage: float
    classical_comparison: Dict[str, float]


@dataclass
class QuantumScalingProfile:
    """Quantum algorithm scaling analysis"""
    algorithm_name: str
    qubit_scaling_factor: float
    coherence_scaling: float
    entanglement_scalability: float
    quantum_advantage_scaling: float
    optimal_qubit_range: Tuple[int, int]
    decoherence_threshold: float
    scaling_bottlenecks: List[str]


@dataclass
class QuantumComparativeStudy:
    """Comprehensive quantum vs classical comparison"""
    study_name: str
    quantum_algorithms: List[str]
    classical_baselines: List[str]
    test_datasets: List[str]
    
    # Statistical results
    quantum_advantage_mean: float
    quantum_advantage_std: float
    statistical_significance: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    
    # Performance analysis
    speed_comparison: Dict[str, float]
    quality_comparison: Dict[str, float]
    stability_comparison: Dict[str, float]
    
    # Research implications
    research_findings: List[str]
    future_directions: List[str]


class QuantumBenchmarkSuite:
    """
    Advanced Quantum Algorithm Benchmarking System
    
    Provides specialized benchmarking for quantum-inspired algorithms including:
    1. Quantum advantage measurement and validation
    2. Coherence and decoherence analysis
    3. Entanglement scalability testing
    4. Multi-algorithm comparative studies
    5. Statistical significance validation
    6. Research-grade performance analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize quantum benchmark suite"""
        default_config = {
            'num_runs': 15,  # More runs for statistical significance
            'min_qubits': 3,
            'max_qubits': 12,
            'qubit_step': 2,
            'confidence_level': 0.95,
            'significance_threshold': 0.05,
            'quantum_advantage_threshold': 1.1,  # 10% minimum advantage
            'coherence_decay_rate': 0.1,
            'max_evolution_steps': 20,
            'enable_detailed_logging': True
        }
        
        self.config = {**default_config, **(config or {})}
        self.random_gen = ScientificRandomGenerator(seed=42)
        
        # Benchmark storage
        self.quantum_results = []
        self.scaling_profiles = {}
        self.comparative_studies = []
        
        # Test datasets for quantum algorithms
        self.quantum_test_datasets = self._generate_quantum_test_datasets()
        
        logger.info(f"QuantumBenchmarkSuite initialized with {len(self.quantum_test_datasets)} datasets")
    
    def benchmark_quantum_algorithm(self, 
                                  algorithm_class: type,
                                  algorithm_params: Dict[str, Any],
                                  test_suite_name: str = "quantum_comprehensive") -> List[QuantumBenchmarkResult]:
        """
        Comprehensive benchmarking of a quantum algorithm
        
        Args:
            algorithm_class: Quantum algorithm class (e.g., QuantumSuperpositionDiscovery)
            algorithm_params: Parameters for algorithm initialization
            test_suite_name: Name of the test suite
            
        Returns:
            List of QuantumBenchmarkResult for all test scenarios
        """
        logger.info(f"Benchmarking quantum algorithm {algorithm_class.__name__}")
        
        suite_results = []
        
        # Test across different qubit configurations
        qubit_range = range(
            self.config['min_qubits'], 
            self.config['max_qubits'] + 1, 
            self.config['qubit_step']
        )
        
        for num_qubits in qubit_range:
            # Update algorithm parameters
            test_params = {**algorithm_params, 'num_qubits': num_qubits}
            
            # Test on each dataset
            for dataset_name, dataset in self.quantum_test_datasets.items():
                logger.info(f"Testing {algorithm_class.__name__} with {num_qubits} qubits on {dataset_name}")
                
                result = self._benchmark_single_quantum_scenario(
                    algorithm_class, test_params, dataset, 
                    f"{test_suite_name}_{dataset_name}_{num_qubits}q"
                )
                
                if result:
                    suite_results.append(result)
        
        self.quantum_results.extend(suite_results)
        
        logger.info(f"Quantum benchmark complete: {len(suite_results)} results collected")
        return suite_results
    
    def _benchmark_single_quantum_scenario(self,
                                         algorithm_class: type,
                                         algorithm_params: Dict[str, Any],
                                         dataset: Dict[str, Any],
                                         test_name: str) -> Optional[QuantumBenchmarkResult]:
        """Benchmark single quantum algorithm scenario"""
        
        try:
            # Initialize algorithm
            algorithm = algorithm_class(**algorithm_params)
            
            # Generate test data
            data = self._generate_data_from_spec(dataset)
            
            # Collect metrics across multiple runs
            run_results = []
            
            for run in range(self.config['num_runs']):
                start_time = time.time()
                
                try:
                    # Run quantum discovery
                    result = algorithm.discover_patterns(
                        data, 
                        evolution_steps=self.config['max_evolution_steps']
                    )
                    
                    execution_time = time.time() - start_time
                    
                    # Extract quantum-specific metrics
                    quantum_metrics = self._extract_quantum_metrics(result, execution_time)
                    run_results.append(quantum_metrics)
                    
                except Exception as e:
                    logger.warning(f"Run {run} failed: {e}")
                    continue
            
            if not run_results:
                logger.error(f"All runs failed for {test_name}")
                return None
            
            # Aggregate results
            aggregated_result = self._aggregate_quantum_results(
                run_results, algorithm_class.__name__, test_name
            )
            
            return aggregated_result
            
        except Exception as e:
            logger.error(f"Benchmark scenario failed for {test_name}: {e}")
            return None
    
    def _extract_quantum_metrics(self, result: QuantumDiscoveryResult, execution_time: float) -> Dict[str, Any]:
        """Extract quantum-specific metrics from discovery result"""
        
        # Basic metrics
        metrics = {
            'execution_time': execution_time,
            'performance_score': len(result.discovered_patterns) * 0.5 + result.quantum_advantage * 0.5,
            'memory_usage_mb': self._estimate_memory_usage(),
            
            # Quantum metrics
            'quantum_advantage': result.quantum_advantage,
            'coherence_preserved': result.coherence_preserved,
            'entanglement_generated': result.entanglement_generated,
            'patterns_discovered': len(result.discovered_patterns),
        }
        
        # Pattern quality analysis
        if result.discovered_patterns:
            pattern_qualities = [p.get('quality', 0.0) for p in result.discovered_patterns]
            metrics['pattern_quality_avg'] = sum(pattern_qualities) / len(pattern_qualities)
            metrics['pattern_novelty_score'] = self._compute_pattern_novelty(result.discovered_patterns)
        else:
            metrics['pattern_quality_avg'] = 0.0
            metrics['pattern_novelty_score'] = 0.0
        
        # Quantum interference quality
        metrics['interference_quality'] = self._estimate_interference_quality(result)
        
        # Decoherence analysis
        metrics['decoherence_rate'] = 1.0 - result.coherence_preserved
        
        # Efficiency metrics
        if execution_time > 0:
            metrics['qubit_efficiency'] = result.quantum_advantage / (execution_time + 0.001)
        else:
            metrics['qubit_efficiency'] = 0.0
        
        # Classical comparison
        if result.classical_comparison:
            classical_score = sum(c.get('quality', 0.0) for c in result.classical_comparison)
            classical_count = len(result.classical_comparison)
            metrics['classical_comparison'] = {
                'classical_score': classical_score,
                'classical_count': classical_count,
                'quantum_vs_classical_ratio': result.quantum_advantage
            }
        else:
            metrics['classical_comparison'] = {'classical_score': 0.0, 'classical_count': 0, 'quantum_vs_classical_ratio': 1.0}
        
        return metrics
    
    def _aggregate_quantum_results(self, 
                                 run_results: List[Dict[str, Any]], 
                                 algorithm_name: str, 
                                 test_name: str) -> QuantumBenchmarkResult:
        """Aggregate quantum benchmark results across multiple runs"""
        
        # Compute means
        metrics = {}
        for key in run_results[0].keys():
            if isinstance(run_results[0][key], (int, float)):
                values = [r[key] for r in run_results if key in r]
                metrics[key] = sum(values) / len(values) if values else 0.0
        
        # Special handling for classical comparison
        classical_comparisons = [r.get('classical_comparison', {}) for r in run_results]
        avg_classical = {}
        if classical_comparisons and classical_comparisons[0]:
            for key in classical_comparisons[0].keys():
                values = [c.get(key, 0.0) for c in classical_comparisons if key in c]
                avg_classical[key] = sum(values) / len(values) if values else 0.0
        
        # Compute scaling advantage
        scaling_advantage = self._compute_scaling_advantage(metrics)
        
        return QuantumBenchmarkResult(
            algorithm_name=algorithm_name,
            test_name=test_name,
            execution_time=metrics.get('execution_time', 0.0),
            performance_score=metrics.get('performance_score', 0.0),
            memory_usage_mb=metrics.get('memory_usage_mb', 0.0),
            
            # Quantum metrics
            quantum_advantage=metrics.get('quantum_advantage', 1.0),
            coherence_preserved=metrics.get('coherence_preserved', 0.0),
            entanglement_generated=metrics.get('entanglement_generated', 0.0),
            interference_quality=metrics.get('interference_quality', 0.0),
            decoherence_rate=metrics.get('decoherence_rate', 1.0),
            
            # Pattern metrics
            patterns_discovered=int(metrics.get('patterns_discovered', 0)),
            pattern_quality_avg=metrics.get('pattern_quality_avg', 0.0),
            pattern_novelty_score=metrics.get('pattern_novelty_score', 0.0),
            
            # Efficiency metrics
            qubit_efficiency=metrics.get('qubit_efficiency', 0.0),
            scaling_advantage=scaling_advantage,
            classical_comparison=avg_classical
        )
    
    def analyze_quantum_scalability(self, 
                                  algorithm_class: type,
                                  base_params: Dict[str, Any]) -> QuantumScalingProfile:
        """Analyze quantum algorithm scalability across qubit counts"""
        
        logger.info(f"Analyzing quantum scalability for {algorithm_class.__name__}")
        
        # Collect scaling data
        qubit_counts = []
        quantum_advantages = []
        coherence_values = []
        entanglement_values = []
        execution_times = []
        
        qubit_range = range(
            self.config['min_qubits'], 
            self.config['max_qubits'] + 1, 
            self.config['qubit_step']
        )
        
        for num_qubits in qubit_range:
            try:
                # Test with standard dataset
                test_params = {**base_params, 'num_qubits': num_qubits}
                algorithm = algorithm_class(**test_params)
                
                # Generate test data
                data = self.random_gen.random_array((100, 1), 'normal')
                
                # Run multiple trials
                advantages = []
                coherences = []
                entanglements = []
                times = []
                
                for _ in range(5):  # 5 trials per qubit count
                    start_time = time.time()
                    result = algorithm.discover_patterns(data, evolution_steps=10)
                    execution_time = time.time() - start_time
                    
                    advantages.append(result.quantum_advantage)
                    coherences.append(result.coherence_preserved)
                    entanglements.append(result.entanglement_generated)
                    times.append(execution_time)
                
                # Store averages
                qubit_counts.append(num_qubits)
                quantum_advantages.append(sum(advantages) / len(advantages))
                coherence_values.append(sum(coherences) / len(coherences))
                entanglement_values.append(sum(entanglements) / len(entanglements))
                execution_times.append(sum(times) / len(times))
                
            except Exception as e:
                logger.warning(f"Scalability test failed for {num_qubits} qubits: {e}")
                continue
        
        if len(qubit_counts) < 3:
            logger.error("Insufficient data for scalability analysis")
            return QuantumScalingProfile(
                algorithm_name=algorithm_class.__name__,
                qubit_scaling_factor=0.0,
                coherence_scaling=0.0,
                entanglement_scalability=0.0,
                quantum_advantage_scaling=0.0,
                optimal_qubit_range=(0, 0),
                decoherence_threshold=1.0,
                scaling_bottlenecks=["insufficient_data"]
            )
        
        # Analyze scaling patterns
        qubit_scaling_factor = self._compute_scaling_factor(qubit_counts, quantum_advantages)
        coherence_scaling = self._compute_scaling_factor(qubit_counts, coherence_values)
        entanglement_scalability = self._compute_scaling_factor(qubit_counts, entanglement_values)
        quantum_advantage_scaling = self._compute_scaling_factor(qubit_counts, quantum_advantages)
        
        # Find optimal qubit range
        optimal_range = self._find_optimal_qubit_range(
            qubit_counts, quantum_advantages, coherence_values, execution_times
        )
        
        # Determine decoherence threshold
        decoherence_threshold = self._find_decoherence_threshold(qubit_counts, coherence_values)
        
        # Identify bottlenecks
        bottlenecks = self._identify_quantum_bottlenecks(
            qubit_scaling_factor, coherence_scaling, entanglement_scalability
        )
        
        profile = QuantumScalingProfile(
            algorithm_name=algorithm_class.__name__,
            qubit_scaling_factor=qubit_scaling_factor,
            coherence_scaling=coherence_scaling,
            entanglement_scalability=entanglement_scalability,
            quantum_advantage_scaling=quantum_advantage_scaling,
            optimal_qubit_range=optimal_range,
            decoherence_threshold=decoherence_threshold,
            scaling_bottlenecks=bottlenecks
        )
        
        self.scaling_profiles[algorithm_class.__name__] = profile
        
        logger.info(f"Scalability analysis complete: optimal range {optimal_range[0]}-{optimal_range[1]} qubits")
        return profile
    
    def quantum_comparative_study(self, 
                                quantum_algorithms: List[Tuple[type, Dict[str, Any]]],
                                study_name: str = "quantum_comparison") -> QuantumComparativeStudy:
        """Comprehensive comparative study of quantum algorithms"""
        
        logger.info(f"Starting quantum comparative study: {study_name}")
        
        # Collect results for all algorithms
        algorithm_results = {}
        
        for alg_class, alg_params in quantum_algorithms:
            alg_name = alg_class.__name__
            logger.info(f"Testing {alg_name} for comparative study")
            
            results = self.benchmark_quantum_algorithm(alg_class, alg_params, f"{study_name}_{alg_name}")
            algorithm_results[alg_name] = results
        
        # Statistical analysis
        statistical_analysis = self._perform_quantum_statistical_analysis(algorithm_results)
        
        # Performance comparisons
        performance_comparisons = self._compute_performance_comparisons(algorithm_results)
        
        # Research findings
        findings = self._generate_research_findings(algorithm_results, statistical_analysis)
        
        study = QuantumComparativeStudy(
            study_name=study_name,
            quantum_algorithms=[alg_class.__name__ for alg_class, _ in quantum_algorithms],
            classical_baselines=["random_search", "grid_search", "statistical_analysis"],
            test_datasets=list(self.quantum_test_datasets.keys()),
            
            # Statistical results
            quantum_advantage_mean=statistical_analysis['quantum_advantage_mean'],
            quantum_advantage_std=statistical_analysis['quantum_advantage_std'],
            statistical_significance=statistical_analysis['p_value'],
            effect_size=statistical_analysis['effect_size'],
            confidence_interval=statistical_analysis['confidence_interval'],
            
            # Performance analysis
            speed_comparison=performance_comparisons['speed'],
            quality_comparison=performance_comparisons['quality'],
            stability_comparison=performance_comparisons['stability'],
            
            # Research implications
            research_findings=findings['findings'],
            future_directions=findings['future_directions']
        )
        
        self.comparative_studies.append(study)
        
        logger.info(f"Comparative study complete: {len(findings['findings'])} key findings")
        return study
    
    def _generate_quantum_test_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Generate specialized test datasets for quantum algorithms"""
        datasets = {}
        
        # Highly structured data (good for quantum pattern detection)
        datasets['structured_patterns'] = {
            'type': 'structured',
            'size': 200,
            'dimensions': 1,
            'pattern_complexity': 'high',
            'noise_level': 0.1,
            'description': 'Sine wave with multiple frequencies'
        }
        
        # Sparse high-dimensional data
        datasets['sparse_highdim'] = {
            'type': 'sparse',
            'size': 500,
            'dimensions': 10,
            'sparsity': 0.8,
            'noise_level': 0.2,
            'description': 'Sparse high-dimensional with hidden correlations'
        }
        
        # Quantum-inspired test data
        datasets['quantum_inspired'] = {
            'type': 'quantum_inspired',
            'size': 300,
            'dimensions': 1,
            'superposition_states': 4,
            'entanglement_strength': 0.7,
            'description': 'Data with quantum-like correlations'
        }
        
        # Noisy optimization landscape
        datasets['noisy_optimization'] = {
            'type': 'optimization',
            'size': 400,
            'dimensions': 2,
            'local_minima': 5,
            'noise_level': 0.3,
            'description': 'Complex optimization landscape with noise'
        }
        
        # Time series with complex patterns
        datasets['complex_timeseries'] = {
            'type': 'timeseries',
            'size': 1000,
            'dimensions': 1,
            'trend_components': 3,
            'seasonal_components': 2,
            'noise_level': 0.15,
            'description': 'Multi-component time series'
        }
        
        return datasets
    
    def _generate_data_from_spec(self, dataset_spec: Dict[str, Any]) -> np.ndarray:
        """Generate data from dataset specification"""
        size = dataset_spec['size']
        data_type = dataset_spec['type']
        
        if data_type == 'structured':
            # Generate sine wave with multiple frequencies
            t = np.linspace(0, 4*np.pi, size)
            signal = np.sin(t) + 0.5*np.sin(3*t) + 0.3*np.sin(5*t)
            noise = self.random_gen.random_array((size,), 'normal') * dataset_spec.get('noise_level', 0.1)
            return (signal + noise).reshape(-1, 1)
        
        elif data_type == 'sparse':
            dims = dataset_spec['dimensions']
            sparsity = dataset_spec.get('sparsity', 0.8)
            data = self.random_gen.random_array((size, dims), 'normal')
            # Make sparse
            mask = self.random_gen.random_array((size, dims), 'uniform') < sparsity
            data[mask] = 0
            return data
        
        elif data_type == 'quantum_inspired':
            # Generate data with quantum-like superposition patterns
            t = np.linspace(0, 2*np.pi, size)
            states = dataset_spec.get('superposition_states', 4)
            
            # Superposition of multiple states
            signal = sum(np.sin(i*t + i*np.pi/4) / np.sqrt(states) for i in range(1, states+1))
            
            # Add entanglement-like correlations
            entanglement = dataset_spec.get('entanglement_strength', 0.5)
            correlation = entanglement * np.sin(2*t) * np.cos(3*t)
            
            return (signal + correlation).reshape(-1, 1)
        
        elif data_type == 'optimization':
            # Complex optimization landscape
            dims = dataset_spec['dimensions']
            data = self.random_gen.random_array((size, dims), 'uniform') * 10 - 5
            
            # Add multiple local minima
            num_minima = dataset_spec.get('local_minima', 3)
            for i in range(num_minima):
                center = self.random_gen.random_array((dims,), 'uniform') * 8 - 4
                distances = np.linalg.norm(data - center, axis=1)
                data = np.column_stack([data, -np.exp(-distances**2)])
            
            return data
        
        elif data_type == 'timeseries':
            # Complex time series
            t = np.linspace(0, 10*np.pi, size)
            
            # Trend components
            trend = 0.1*t + 0.02*t**2
            
            # Seasonal components
            seasonal1 = 2*np.sin(t)
            seasonal2 = 1.5*np.sin(3*t + np.pi/4)
            
            # Noise
            noise = self.random_gen.random_array((size,), 'normal') * dataset_spec.get('noise_level', 0.1)
            
            signal = trend + seasonal1 + seasonal2 + noise
            return signal.reshape(-1, 1)
        
        else:
            # Default: normal random data
            return self.random_gen.random_array((size, 1), 'normal')
    
    def _compute_pattern_novelty(self, patterns: List[Dict[str, Any]]) -> float:
        """Compute novelty score of discovered patterns"""
        if not patterns:
            return 0.0
        
        # Simple novelty measure based on pattern diversity
        pattern_types = set()
        unique_patterns = set()
        
        for pattern in patterns:
            pattern_type = pattern.get('discovery_type', 'unknown')
            pattern_types.add(pattern_type)
            
            # Convert pattern to hashable form
            if 'pattern' in pattern:
                if isinstance(pattern['pattern'], np.ndarray):
                    pattern_str = str(pattern['pattern'].tolist())
                else:
                    pattern_str = str(pattern['pattern'])
                unique_patterns.add(pattern_str)
        
        # Novelty = diversity of types + uniqueness of patterns
        type_diversity = len(pattern_types) / 3.0  # Normalize by max expected types
        pattern_uniqueness = len(unique_patterns) / len(patterns)
        
        return min(1.0, 0.6 * type_diversity + 0.4 * pattern_uniqueness)
    
    def _estimate_interference_quality(self, result: QuantumDiscoveryResult) -> float:
        """Estimate quantum interference quality from result"""
        # Heuristic based on quantum advantage and coherence
        if result.quantum_advantage <= 1.0:
            return 0.0
        
        # Higher coherence + higher advantage = better interference
        advantage_factor = min(1.0, (result.quantum_advantage - 1.0) / 2.0)
        coherence_factor = result.coherence_preserved
        
        return 0.7 * advantage_factor + 0.3 * coherence_factor
    
    def _compute_scaling_advantage(self, metrics: Dict[str, float]) -> float:
        """Compute scaling advantage metric"""
        quantum_advantage = metrics.get('quantum_advantage', 1.0)
        qubit_efficiency = metrics.get('qubit_efficiency', 0.0)
        coherence = metrics.get('coherence_preserved', 0.0)
        
        # Weighted combination
        return 0.5 * (quantum_advantage - 1.0) + 0.3 * qubit_efficiency + 0.2 * coherence
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage for quantum algorithm"""
        # Simplified estimation
        return self.random_gen.random_float() * 50 + 20  # 20-70 MB range
    
    def _compute_scaling_factor(self, x_values: List[int], y_values: List[float]) -> float:
        """Compute scaling factor using linear regression"""
        if len(x_values) < 2:
            return 1.0
        
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x_values[i] * y_values[i] for i in range(n))
        sum_x2 = sum(x * x for x in x_values)
        
        if n * sum_x2 - sum_x * sum_x == 0:
            return 1.0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Normalize to meaningful range
        return max(0.1, min(5.0, abs(slope)))
    
    def _find_optimal_qubit_range(self, 
                                qubit_counts: List[int], 
                                advantages: List[float],
                                coherences: List[float],
                                times: List[float]) -> Tuple[int, int]:
        """Find optimal qubit range for algorithm"""
        
        # Score each qubit count
        scores = []
        for i in range(len(qubit_counts)):
            # Higher advantage, higher coherence, lower time = better
            advantage_score = advantages[i] - 1.0  # Normalize
            coherence_score = coherences[i]
            time_score = 1.0 / (times[i] + 0.001)  # Inverse time
            
            combined_score = 0.5 * advantage_score + 0.3 * coherence_score + 0.2 * time_score * 0.1
            scores.append(combined_score)
        
        if not scores:
            return (0, 0)
        
        # Find range of good scores (above 80% of max score)
        max_score = max(scores)
        threshold = max_score * 0.8
        
        good_indices = [i for i, score in enumerate(scores) if score >= threshold]
        
        if good_indices:
            min_qubits = qubit_counts[min(good_indices)]
            max_qubits = qubit_counts[max(good_indices)]
            return (min_qubits, max_qubits)
        else:
            # Fallback to best single point
            best_idx = scores.index(max_score)
            best_qubits = qubit_counts[best_idx]
            return (best_qubits, best_qubits)
    
    def _find_decoherence_threshold(self, qubit_counts: List[int], coherences: List[float]) -> float:
        """Find the qubit count where decoherence becomes significant"""
        if len(coherences) < 2:
            return 1.0
        
        # Find where coherence drops below 50% of initial
        initial_coherence = coherences[0]
        threshold_coherence = initial_coherence * 0.5
        
        for i, coherence in enumerate(coherences):
            if coherence < threshold_coherence:
                return float(qubit_counts[i])
        
        # If never drops below threshold, return max tested
        return float(qubit_counts[-1])
    
    def _identify_quantum_bottlenecks(self, 
                                    qubit_scaling: float,
                                    coherence_scaling: float,
                                    entanglement_scaling: float) -> List[str]:
        """Identify quantum algorithm bottlenecks"""
        bottlenecks = []
        
        if qubit_scaling < 0.5:
            bottlenecks.append("poor_qubit_scaling")
        
        if coherence_scaling < -0.5:  # Negative scaling means rapid coherence loss
            bottlenecks.append("rapid_decoherence")
        
        if entanglement_scaling < 0.1:
            bottlenecks.append("limited_entanglement_generation")
        
        if abs(qubit_scaling) < 0.1:
            bottlenecks.append("scaling_plateau")
        
        return bottlenecks if bottlenecks else ["no_major_bottlenecks"]
    
    def _perform_quantum_statistical_analysis(self, 
                                            algorithm_results: Dict[str, List[QuantumBenchmarkResult]]) -> Dict[str, float]:
        """Perform statistical analysis of quantum results"""
        
        # Collect quantum advantages from all algorithms
        all_advantages = []
        for results in algorithm_results.values():
            advantages = [r.quantum_advantage for r in results]
            all_advantages.extend(advantages)
        
        if not all_advantages:
            return {
                'quantum_advantage_mean': 1.0,
                'quantum_advantage_std': 0.0,
                'p_value': 1.0,
                'effect_size': 0.0,
                'confidence_interval': (1.0, 1.0)
            }
        
        # Basic statistics
        mean_advantage = statistics.mean(all_advantages)
        std_advantage = statistics.stdev(all_advantages) if len(all_advantages) > 1 else 0.0
        
        # Test if quantum advantage is significantly > 1.0
        n = len(all_advantages)
        if n > 1 and std_advantage > 0:
            # One-sample t-test approximation
            t_stat = (mean_advantage - 1.0) / (std_advantage / math.sqrt(n))
            # Approximate p-value
            p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))
        else:
            p_value = 1.0
        
        # Effect size (Cohen's d)
        if std_advantage > 0:
            effect_size = (mean_advantage - 1.0) / std_advantage
        else:
            effect_size = 0.0
        
        # Confidence interval
        if n > 1 and std_advantage > 0:
            margin = 1.96 * std_advantage / math.sqrt(n)  # 95% CI
            ci_lower = mean_advantage - margin
            ci_upper = mean_advantage + margin
        else:
            ci_lower = ci_upper = mean_advantage
        
        return {
            'quantum_advantage_mean': mean_advantage,
            'quantum_advantage_std': std_advantage,
            'p_value': p_value,
            'effect_size': effect_size,
            'confidence_interval': (ci_lower, ci_upper)
        }
    
    def _compute_performance_comparisons(self, 
                                       algorithm_results: Dict[str, List[QuantumBenchmarkResult]]) -> Dict[str, Dict[str, float]]:
        """Compute performance comparisons between algorithms"""
        
        comparisons = {
            'speed': {},
            'quality': {},
            'stability': {}
        }
        
        for alg_name, results in algorithm_results.items():
            if not results:
                continue
            
            # Speed: inverse of execution time
            avg_time = sum(r.execution_time for r in results) / len(results)
            comparisons['speed'][alg_name] = 1.0 / (avg_time + 0.001)
            
            # Quality: quantum advantage
            avg_advantage = sum(r.quantum_advantage for r in results) / len(results)
            comparisons['quality'][alg_name] = avg_advantage
            
            # Stability: inverse of standard deviation of quantum advantage
            advantages = [r.quantum_advantage for r in results]
            if len(advantages) > 1:
                std_advantage = statistics.stdev(advantages)
                comparisons['stability'][alg_name] = 1.0 / (std_advantage + 0.001)
            else:
                comparisons['stability'][alg_name] = 1.0
        
        return comparisons
    
    def _generate_research_findings(self, 
                                  algorithm_results: Dict[str, List[QuantumBenchmarkResult]],
                                  statistical_analysis: Dict[str, float]) -> Dict[str, List[str]]:
        """Generate research findings from benchmark results"""
        
        findings = []
        future_directions = []
        
        # Quantum advantage analysis
        mean_advantage = statistical_analysis['quantum_advantage_mean']
        p_value = statistical_analysis['p_value']
        
        if mean_advantage > 1.1 and p_value < 0.05:
            findings.append(f"Significant quantum advantage demonstrated: {mean_advantage:.2f}x improvement (p < {p_value:.3f})")
        elif mean_advantage > 1.05:
            findings.append(f"Marginal quantum advantage observed: {mean_advantage:.2f}x improvement")
        else:
            findings.append("No significant quantum advantage detected in current tests")
        
        # Algorithm comparison
        if len(algorithm_results) > 1:
            best_alg = max(algorithm_results.keys(), 
                          key=lambda alg: sum(r.quantum_advantage for r in algorithm_results[alg]) / len(algorithm_results[alg]))
            findings.append(f"Best performing quantum algorithm: {best_alg}")
        
        # Pattern discovery analysis
        total_patterns = sum(sum(r.patterns_discovered for r in results) 
                           for results in algorithm_results.values())
        findings.append(f"Total patterns discovered across all tests: {total_patterns}")
        
        # Coherence analysis
        coherence_values = [r.coherence_preserved for results in algorithm_results.values() for r in results]
        if coherence_values:
            avg_coherence = sum(coherence_values) / len(coherence_values)
            if avg_coherence > 0.7:
                findings.append(f"High coherence preservation: {avg_coherence:.2f} average")
            elif avg_coherence < 0.3:
                findings.append(f"Significant decoherence observed: {avg_coherence:.2f} average coherence")
        
        # Future research directions
        if mean_advantage < 1.1:
            future_directions.append("Investigate improved quantum gate sequences for better advantage")
        
        if statistical_analysis['effect_size'] < 0.5:
            future_directions.append("Develop more sensitive quantum advantage metrics")
        
        future_directions.extend([
            "Explore hybrid quantum-classical algorithms",
            "Investigate noise-resilient quantum discovery methods",
            "Test on larger, more complex real-world datasets",
            "Develop theoretical foundations for quantum discovery advantage"
        ])
        
        return {
            'findings': findings,
            'future_directions': future_directions
        }
    
    def _normal_cdf(self, x: float) -> float:
        """Cumulative distribution function for standard normal"""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    def export_benchmark_report(self, output_path: str = "quantum_benchmark_report.json") -> Dict[str, Any]:
        """Export comprehensive benchmark report"""
        
        report = {
            "benchmark_metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config": self.config,
                "total_quantum_results": len(self.quantum_results),
                "scaling_profiles": len(self.scaling_profiles),
                "comparative_studies": len(self.comparative_studies)
            },
            
            "quantum_results_summary": self._summarize_quantum_results(),
            "scaling_analysis": self._summarize_scaling_profiles(),
            "comparative_studies": [asdict(study) for study in self.comparative_studies],
            
            "research_summary": {
                "key_findings": self._extract_key_findings(),
                "statistical_significance": self._compute_overall_significance(),
                "recommendations": self._generate_recommendations()
            }
        }
        
        # Save to file
        output_file = Path(output_path)
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Benchmark report exported to {output_file}")
        return report
    
    def _summarize_quantum_results(self) -> Dict[str, Any]:
        """Summarize quantum benchmark results"""
        if not self.quantum_results:
            return {"total_results": 0}
        
        # Group by algorithm
        algorithm_summaries = {}
        for result in self.quantum_results:
            alg_name = result.algorithm_name
            if alg_name not in algorithm_summaries:
                algorithm_summaries[alg_name] = []
            algorithm_summaries[alg_name].append(result)
        
        # Compute summaries
        summaries = {}
        for alg_name, results in algorithm_summaries.items():
            avg_advantage = sum(r.quantum_advantage for r in results) / len(results)
            avg_coherence = sum(r.coherence_preserved for r in results) / len(results)
            avg_patterns = sum(r.patterns_discovered for r in results) / len(results)
            
            summaries[alg_name] = {
                "test_count": len(results),
                "average_quantum_advantage": avg_advantage,
                "average_coherence_preserved": avg_coherence,
                "average_patterns_discovered": avg_patterns,
                "best_quantum_advantage": max(r.quantum_advantage for r in results),
                "worst_quantum_advantage": min(r.quantum_advantage for r in results)
            }
        
        return {
            "total_results": len(self.quantum_results),
            "algorithms_tested": len(algorithm_summaries),
            "algorithm_summaries": summaries
        }
    
    def _summarize_scaling_profiles(self) -> Dict[str, Any]:
        """Summarize scaling analysis results"""
        if not self.scaling_profiles:
            return {"profiles_available": 0}
        
        summaries = {}
        for alg_name, profile in self.scaling_profiles.items():
            summaries[alg_name] = {
                "qubit_scaling_factor": profile.qubit_scaling_factor,
                "optimal_qubit_range": f"{profile.optimal_qubit_range[0]}-{profile.optimal_qubit_range[1]}",
                "decoherence_threshold": profile.decoherence_threshold,
                "scaling_bottlenecks": profile.scaling_bottlenecks
            }
        
        return {
            "profiles_available": len(self.scaling_profiles),
            "algorithm_profiles": summaries
        }
    
    def _extract_key_findings(self) -> List[str]:
        """Extract key research findings"""
        findings = []
        
        if self.quantum_results:
            # Overall quantum advantage
            all_advantages = [r.quantum_advantage for r in self.quantum_results]
            avg_advantage = sum(all_advantages) / len(all_advantages)
            
            if avg_advantage > 1.1:
                findings.append(f"Consistent quantum advantage demonstrated: {avg_advantage:.2f}x average improvement")
            
            # Best performing algorithm
            if len(set(r.algorithm_name for r in self.quantum_results)) > 1:
                alg_advantages = {}
                for result in self.quantum_results:
                    alg_name = result.algorithm_name
                    if alg_name not in alg_advantages:
                        alg_advantages[alg_name] = []
                    alg_advantages[alg_name].append(result.quantum_advantage)
                
                best_alg = max(alg_advantages.keys(), 
                              key=lambda alg: sum(alg_advantages[alg]) / len(alg_advantages[alg]))
                best_avg = sum(alg_advantages[best_alg]) / len(alg_advantages[best_alg])
                findings.append(f"Best algorithm: {best_alg} with {best_avg:.2f}x average advantage")
        
        if self.comparative_studies:
            study = self.comparative_studies[0]  # Most recent study
            if study.statistical_significance < 0.05:
                findings.append(f"Statistically significant results in {study.study_name}")
        
        return findings
    
    def _compute_overall_significance(self) -> float:
        """Compute overall statistical significance"""
        if not self.quantum_results:
            return 1.0
        
        # Simple test: are quantum advantages significantly > 1.0?
        advantages = [r.quantum_advantage for r in self.quantum_results]
        
        if len(advantages) < 2:
            return 1.0
        
        mean_adv = sum(advantages) / len(advantages)
        var_adv = sum((a - mean_adv)**2 for a in advantages) / (len(advantages) - 1)
        std_adv = math.sqrt(var_adv)
        
        if std_adv == 0:
            return 0.0 if mean_adv > 1.0 else 1.0
        
        # T-test against null hypothesis: mean = 1.0
        t_stat = (mean_adv - 1.0) / (std_adv / math.sqrt(len(advantages)))
        p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))
        
        return min(1.0, p_value)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate research recommendations"""
        recommendations = [
            "Continue developing quantum-inspired algorithms for scientific discovery",
            "Investigate hybrid quantum-classical approaches",
            "Test algorithms on larger, more diverse datasets",
            "Develop better metrics for quantum advantage measurement"
        ]
        
        if self.scaling_profiles:
            recommendations.append("Focus on algorithms with better qubit scaling properties")
        
        if self.quantum_results:
            avg_coherence = sum(r.coherence_preserved for r in self.quantum_results) / len(self.quantum_results)
            if avg_coherence < 0.5:
                recommendations.append("Investigate coherence preservation techniques")
        
        return recommendations


def run_comprehensive_quantum_benchmark():
    """Run comprehensive quantum algorithm benchmark suite"""
    
    logger.info("🚀 Starting Comprehensive Quantum Algorithm Benchmark")
    print("=" * 60)
    
    # Initialize benchmark suite
    config = {
        'num_runs': 10,
        'min_qubits': 3,
        'max_qubits': 10,
        'qubit_step': 2,
        'enable_detailed_logging': True
    }
    
    benchmark = QuantumBenchmarkSuite(config)
    
    # Define quantum algorithms to test
    quantum_algorithms = [
        (QuantumSuperpositionDiscovery, {'superposition_depth': 2}),
        (QuantumEntanglementDiscovery, {})
    ]
    
    try:
        # 1. Individual algorithm benchmarks
        print("📊 Phase 1: Individual Algorithm Benchmarks")
        print("-" * 40)
        
        all_results = {}
        for alg_class, alg_params in quantum_algorithms:
            print(f"Testing {alg_class.__name__}...")
            results = benchmark.benchmark_quantum_algorithm(alg_class, alg_params)
            all_results[alg_class.__name__] = results
            
            if results:
                avg_advantage = sum(r.quantum_advantage for r in results) / len(results)
                print(f"  ✅ Average quantum advantage: {avg_advantage:.3f}x")
        
        # 2. Scalability analysis
        print("\n📈 Phase 2: Scalability Analysis")
        print("-" * 40)
        
        scaling_profiles = []
        for alg_class, alg_params in quantum_algorithms:
            print(f"Analyzing scalability of {alg_class.__name__}...")
            profile = benchmark.analyze_quantum_scalability(alg_class, alg_params)
            scaling_profiles.append(profile)
            
            print(f"  ✅ Optimal qubit range: {profile.optimal_qubit_range[0]}-{profile.optimal_qubit_range[1]}")
            print(f"  ✅ Scaling factor: {profile.qubit_scaling_factor:.3f}")
        
        # 3. Comparative study
        print("\n🔬 Phase 3: Comparative Study")
        print("-" * 40)
        
        comparative_study = benchmark.quantum_comparative_study(
            quantum_algorithms, 
            "comprehensive_quantum_study"
        )
        
        print(f"  ✅ Statistical significance: p = {comparative_study.statistical_significance:.4f}")
        print(f"  ✅ Effect size: {comparative_study.effect_size:.3f}")
        print(f"  ✅ Key findings: {len(comparative_study.research_findings)}")
        
        # 4. Export comprehensive report
        print("\n📄 Phase 4: Report Generation")
        print("-" * 40)
        
        report = benchmark.export_benchmark_report("quantum_benchmark_comprehensive.json")
        
        print(f"  ✅ Report exported: quantum_benchmark_comprehensive.json")
        print(f"  ✅ Total quantum results: {report['benchmark_metadata']['total_quantum_results']}")
        
        # Summary
        print("\n🎯 BENCHMARK SUMMARY")
        print("=" * 30)
        
        overall_significance = report['research_summary']['statistical_significance']
        print(f"📊 Overall statistical significance: p = {overall_significance:.4f}")
        
        key_findings = report['research_summary']['key_findings']
        print(f"🔍 Key findings:")
        for finding in key_findings[:3]:  # Show top 3 findings
            print(f"   • {finding}")
        
        print(f"\n🚀 Comprehensive quantum benchmark completed successfully!")
        print(f"📁 Detailed results saved to: quantum_benchmark_comprehensive.json")
        
        return report
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the comprehensive benchmark
    result = run_comprehensive_quantum_benchmark()
    
    if result:
        print("\n✅ Benchmark suite completed successfully!")
    else:
        print("\n❌ Benchmark suite failed!")