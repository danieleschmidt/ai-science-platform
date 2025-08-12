"""
Comparative Analysis Framework
Advanced statistical comparison between bioneural fusion and baseline approaches
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass, asdict
import time
from scipy import stats
from collections import defaultdict

from ..algorithms.bioneural_pipeline import BioneuralOlfactoryPipeline, PipelineResult
from .baseline_models import BaselineModelSuite, BaselineResult
from ..utils.validation import ValidationMixin
from ..utils.error_handling import robust_execution, safe_array_operation

logger = logging.getLogger(__name__)


@dataclass
class ComparisonMetrics:
    """Metrics comparing bioneural vs baseline approaches"""
    processing_time_ratio: float
    quality_improvement: float
    feature_richness_ratio: float
    signal_preservation_ratio: float
    complexity_handling_ratio: float
    adaptation_benefit: float


@dataclass 
class StatisticalComparison:
    """Statistical test results between approaches"""
    metric_name: str
    bioneural_mean: float
    baseline_mean: float
    effect_size: float
    p_value: float
    confidence_interval: Tuple[float, float]
    statistical_significance: bool
    practical_significance: bool


@dataclass
class ComparativeAnalysisResult:
    """Complete comparative analysis result"""
    dataset_info: Dict[str, Any]
    bioneural_results: List[PipelineResult]
    baseline_results: Dict[str, List[BaselineResult]]
    comparison_metrics: Dict[str, ComparisonMetrics]
    statistical_tests: List[StatisticalComparison]
    summary_statistics: Dict[str, Any]
    recommendations: List[str]


class ComparativeAnalyzer(ValidationMixin):
    """
    Advanced Comparative Analysis Framework
    
    Performs comprehensive statistical comparison between the novel
    bioneural olfactory fusion approach and standard baseline methods.
    
    Key Features:
    1. Multi-metric comparative evaluation
    2. Statistical significance testing
    3. Effect size computation
    4. Performance profiling comparison
    5. Quality assessment comparison
    6. Scalability analysis
    """
    
    def __init__(self, signal_dim: int = 128, confidence_level: float = 0.95):
        self.signal_dim = self.validate_positive_int(signal_dim, "signal_dim")
        self.confidence_level = self.validate_probability(confidence_level, "confidence_level")
        self.alpha = 1.0 - confidence_level
        
        # Initialize comparison components
        self.bioneural_pipeline = BioneuralOlfactoryPipeline()
        self.baseline_suite = BaselineModelSuite(signal_dim=signal_dim)
        
        # Analysis statistics
        self.analysis_stats = {
            "comparisons_performed": 0,
            "datasets_analyzed": 0,
            "significant_improvements": 0,
            "total_metrics_compared": 0
        }
        
        logger.info(f"ComparativeAnalyzer initialized for {signal_dim}D signals with {confidence_level} confidence")
    
    @robust_execution(recovery_strategy='graceful_degradation')
    @safe_array_operation
    def analyze_dataset(self, 
                       test_signals: np.ndarray,
                       training_signals: Optional[np.ndarray] = None,
                       dataset_name: str = "unknown",
                       include_adaptation_analysis: bool = True) -> ComparativeAnalysisResult:
        """
        Perform comprehensive comparative analysis on dataset
        
        Args:
            test_signals: Test signals for evaluation
            training_signals: Training signals for models that need fitting
            dataset_name: Name of the dataset being analyzed
            include_adaptation_analysis: Whether to analyze adaptation benefits
            
        Returns:
            Complete comparative analysis results
        """
        logger.info(f"Starting comparative analysis on dataset '{dataset_name}'")
        start_time = time.time()
        
        if test_signals.ndim == 1:
            test_signals = test_signals.reshape(1, -1)
        
        # Validate inputs
        for i, signal in enumerate(test_signals):
            if signal.size == 0:
                raise ValueError(f"Test signal {i} is empty")
        
        # Dataset information
        dataset_info = {
            "name": dataset_name,
            "num_test_signals": test_signals.shape[0],
            "signal_dimension": test_signals.shape[1],
            "num_training_signals": training_signals.shape[0] if training_signals is not None else 0,
            "signal_statistics": self._compute_dataset_statistics(test_signals)
        }
        
        # Step 1: Fit baseline models if training data available
        if training_signals is not None:
            logger.info("Fitting baseline models on training data")
            self.baseline_suite.fit_models(training_signals)
        
        # Step 2: Process test signals with bioneural pipeline
        logger.info("Processing signals with bioneural pipeline")
        bioneural_results = []
        
        for i, signal in enumerate(test_signals):
            try:
                result = self.bioneural_pipeline.process(signal)
                bioneural_results.append(result)
            except Exception as e:
                logger.error(f"Bioneural processing failed for signal {i}: {e}")
        
        # Step 3: Process test signals with baseline suite
        logger.info("Processing signals with baseline models")
        baseline_results = defaultdict(list)
        
        for i, signal in enumerate(test_signals):
            signal_baseline_results = self.baseline_suite.process_signal(signal)
            for model_name, result in signal_baseline_results.items():
                baseline_results[model_name].append(result)
        
        # Step 4: Compute comparison metrics
        logger.info("Computing comparison metrics")
        comparison_metrics = self._compute_comparison_metrics(
            bioneural_results, baseline_results
        )
        
        # Step 5: Perform statistical tests
        logger.info("Performing statistical significance tests")
        statistical_tests = self._perform_statistical_tests(
            bioneural_results, baseline_results
        )
        
        # Step 6: Generate summary statistics
        summary_statistics = self._generate_summary_statistics(
            bioneural_results, baseline_results, statistical_tests
        )
        
        # Step 7: Generate recommendations
        recommendations = self._generate_recommendations(
            comparison_metrics, statistical_tests, summary_statistics
        )
        
        # Step 8: Adaptation analysis (if enabled)
        if include_adaptation_analysis and len(bioneural_results) > 1:
            adaptation_analysis = self._analyze_adaptation_benefits(bioneural_results)
            summary_statistics['adaptation_analysis'] = adaptation_analysis
        
        # Update analysis statistics
        self.analysis_stats["comparisons_performed"] += 1
        self.analysis_stats["datasets_analyzed"] += 1
        self.analysis_stats["significant_improvements"] += sum(1 for test in statistical_tests if test.statistical_significance)
        self.analysis_stats["total_metrics_compared"] += len(statistical_tests)
        
        analysis_time = time.time() - start_time
        
        result = ComparativeAnalysisResult(
            dataset_info=dataset_info,
            bioneural_results=bioneural_results,
            baseline_results=dict(baseline_results),
            comparison_metrics=comparison_metrics,
            statistical_tests=statistical_tests,
            summary_statistics=summary_statistics,
            recommendations=recommendations
        )
        
        logger.info(f"Comparative analysis complete in {analysis_time:.2f}s: {len(statistical_tests)} metrics compared")
        return result
    
    def _compute_dataset_statistics(self, signals: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive dataset statistics"""
        stats = {}
        
        # Signal-level statistics
        stats['mean_signal_mean'] = float(np.mean([np.mean(signal) for signal in signals]))
        stats['mean_signal_std'] = float(np.mean([np.std(signal) for signal in signals]))
        stats['mean_signal_norm'] = float(np.mean([np.linalg.norm(signal) for signal in signals]))
        
        # Dataset-level statistics
        all_values = signals.flatten()
        stats['global_mean'] = float(np.mean(all_values))
        stats['global_std'] = float(np.std(all_values))
        stats['global_min'] = float(np.min(all_values))
        stats['global_max'] = float(np.max(all_values))
        stats['global_skewness'] = float(stats.skew(all_values))
        stats['global_kurtosis'] = float(stats.kurtosis(all_values))
        
        # Diversity measures
        signal_correlations = []
        for i in range(len(signals)):
            for j in range(i+1, len(signals)):
                corr = np.corrcoef(signals[i], signals[j])[0, 1]
                if not np.isnan(corr):
                    signal_correlations.append(abs(corr))
        
        stats['mean_inter_signal_correlation'] = float(np.mean(signal_correlations)) if signal_correlations else 0.0
        stats['signal_diversity'] = float(1.0 - stats['mean_inter_signal_correlation'])
        
        return stats
    
    def _compute_comparison_metrics(self, 
                                  bioneural_results: List[PipelineResult],
                                  baseline_results: Dict[str, List[BaselineResult]]) -> Dict[str, ComparisonMetrics]:
        """Compute comparative metrics between approaches"""
        
        comparison_metrics = {}
        
        # Get bioneural metrics
        bioneural_times = [r.processing_time for r in bioneural_results]
        bioneural_qualities = [r.quality_metrics['overall_quality'] for r in bioneural_results]
        bioneural_richness = [r.quality_metrics['feature_richness'] for r in bioneural_results]
        bioneural_preservation = [r.quality_metrics['signal_preservation'] for r in bioneural_results]
        bioneural_complexity = [r.bioneural_result.pattern_complexity for r in bioneural_results]
        
        # Compare against each baseline model
        for model_name, results in baseline_results.items():
            if not results:
                continue
            
            baseline_times = [r.processing_time for r in results if np.isfinite(r.processing_time)]
            
            # Extract relevant metrics from baseline results
            baseline_qualities = []
            baseline_preservation = []
            baseline_complexities = []
            
            for result in results:
                # Estimate quality from available metrics (simplified)
                perf_metrics = result.performance_metrics
                
                # Quality estimation based on available metrics
                if 'signal_to_noise_ratio' in perf_metrics:
                    snr = perf_metrics['signal_to_noise_ratio']
                    quality = np.tanh(np.log10(max(snr, 1.0)) / 10.0)
                elif 'energy_preserved' in perf_metrics:
                    quality = perf_metrics['energy_preserved']
                elif 'reconstruction_error' in perf_metrics:
                    error = perf_metrics['reconstruction_error']
                    quality = 1.0 / (1.0 + error)
                else:
                    quality = 0.5  # Default neutral quality
                
                baseline_qualities.append(quality)
                
                # Preservation estimation
                if 'signal_to_noise_ratio' in perf_metrics:
                    preservation = np.tanh(np.log10(max(perf_metrics['signal_to_noise_ratio'], 1.0)) / 5.0)
                elif 'norm_preservation' in perf_metrics:
                    preservation = perf_metrics['norm_preservation']
                else:
                    preservation = 0.5
                
                baseline_preservation.append(preservation)
                
                # Complexity estimation (from entropy or sparsity measures)
                if 'wavelet_entropy' in perf_metrics:
                    complexity = perf_metrics['wavelet_entropy'] / 10.0  # Normalize
                elif 'sparsity' in perf_metrics:
                    complexity = 1.0 - perf_metrics['sparsity']
                else:
                    complexity = 0.5
                
                baseline_complexities.append(complexity)
            
            # Compute comparison ratios
            time_ratio = (np.mean(bioneural_times) / (np.mean(baseline_times) + 1e-10)) if baseline_times else float('inf')
            
            quality_improvement = np.mean(bioneural_qualities) - np.mean(baseline_qualities)
            
            richness_ratio = (np.mean(bioneural_richness) / (np.mean([0.5] * len(baseline_qualities)) + 1e-10))  # Baselines assumed to have neutral richness
            
            preservation_ratio = (np.mean(bioneural_preservation) / (np.mean(baseline_preservation) + 1e-10)) if baseline_preservation else 1.0
            
            complexity_ratio = (np.mean(bioneural_complexity) / (np.mean(baseline_complexities) + 1e-10)) if baseline_complexities else 1.0
            
            # Adaptation benefit (bioneural-specific)
            adaptation_benefit = self._estimate_adaptation_benefit(bioneural_results)
            
            comparison_metrics[model_name] = ComparisonMetrics(
                processing_time_ratio=float(time_ratio),
                quality_improvement=float(quality_improvement),
                feature_richness_ratio=float(richness_ratio),
                signal_preservation_ratio=float(preservation_ratio),
                complexity_handling_ratio=float(complexity_ratio),
                adaptation_benefit=float(adaptation_benefit)
            )
        
        return comparison_metrics
    
    def _estimate_adaptation_benefit(self, bioneural_results: List[PipelineResult]) -> float:
        """Estimate benefit from adaptation over the sequence"""
        if len(bioneural_results) < 2:
            return 0.0
        
        # Look at quality improvement over time
        qualities = [r.quality_metrics['overall_quality'] for r in bioneural_results]
        
        # Simple linear trend
        x = np.arange(len(qualities))
        slope, _, r_value, p_value, _ = stats.linregress(x, qualities)
        
        # Adaptation benefit is positive slope with good correlation
        if p_value < 0.05 and r_value > 0:
            return float(slope * len(qualities))  # Total improvement
        else:
            return 0.0
    
    def _perform_statistical_tests(self,
                                 bioneural_results: List[PipelineResult],
                                 baseline_results: Dict[str, List[BaselineResult]]) -> List[StatisticalComparison]:
        """Perform comprehensive statistical significance tests"""
        
        statistical_tests = []
        
        # Extract bioneural metrics
        bioneural_metrics = {
            'processing_time': [r.processing_time for r in bioneural_results],
            'overall_quality': [r.quality_metrics['overall_quality'] for r in bioneural_results],
            'feature_richness': [r.quality_metrics['feature_richness'] for r in bioneural_results],
            'signal_preservation': [r.quality_metrics['signal_preservation'] for r in bioneural_results],
            'pattern_complexity': [r.bioneural_result.pattern_complexity for r in bioneural_results],
            'fusion_confidence': [r.neural_fusion_result.fusion_confidence for r in bioneural_results]
        }
        
        # Test against each baseline model
        for model_name, results in baseline_results.items():
            if not results or len(results) < 3:  # Need minimum samples for tests
                continue
            
            # Extract baseline metrics (with appropriate mappings)
            baseline_times = [r.processing_time for r in results if np.isfinite(r.processing_time)]
            
            # Test processing time
            if baseline_times and len(bioneural_metrics['processing_time']) >= 3:
                test_result = self._perform_welch_t_test(
                    bioneural_metrics['processing_time'],
                    baseline_times,
                    f"processing_time_vs_{model_name}"
                )
                statistical_tests.append(test_result)
            
            # Test quality metrics (estimated for baselines)
            baseline_estimated_qualities = []
            for result in results:
                perf_metrics = result.performance_metrics
                if 'signal_to_noise_ratio' in perf_metrics:
                    quality = np.tanh(np.log10(max(perf_metrics['signal_to_noise_ratio'], 1.0)) / 10.0)
                elif 'energy_preserved' in perf_metrics:
                    quality = perf_metrics['energy_preserved']
                elif 'reconstruction_error' in perf_metrics:
                    error = perf_metrics['reconstruction_error']
                    quality = 1.0 / (1.0 + error)
                else:
                    quality = 0.5
                baseline_estimated_qualities.append(quality)
            
            if baseline_estimated_qualities:
                test_result = self._perform_welch_t_test(
                    bioneural_metrics['overall_quality'],
                    baseline_estimated_qualities,
                    f"overall_quality_vs_{model_name}"
                )
                statistical_tests.append(test_result)
        
        return statistical_tests
    
    def _perform_welch_t_test(self, sample1: List[float], sample2: List[float], 
                             metric_name: str) -> StatisticalComparison:
        """Perform Welch's t-test between two samples"""
        
        # Convert to arrays and remove invalid values
        arr1 = np.array([x for x in sample1 if np.isfinite(x)])
        arr2 = np.array([x for x in sample2 if np.isfinite(x)])
        
        if len(arr1) < 2 or len(arr2) < 2:
            # Not enough samples for test
            return StatisticalComparison(
                metric_name=metric_name,
                bioneural_mean=float(np.mean(arr1)) if len(arr1) > 0 else 0.0,
                baseline_mean=float(np.mean(arr2)) if len(arr2) > 0 else 0.0,
                effect_size=0.0,
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                statistical_significance=False,
                practical_significance=False
            )
        
        # Perform Welch's t-test (unequal variances)
        t_stat, p_value = stats.ttest_ind(arr1, arr2, equal_var=False)
        
        # Compute effect size (Cohen's d)
        mean1, mean2 = np.mean(arr1), np.mean(arr2)
        pooled_std = np.sqrt(((len(arr1) - 1) * np.var(arr1, ddof=1) + 
                             (len(arr2) - 1) * np.var(arr2, ddof=1)) / 
                            (len(arr1) + len(arr2) - 2))
        
        effect_size = (mean1 - mean2) / (pooled_std + 1e-10)
        
        # Confidence interval for mean difference
        se_diff = pooled_std * np.sqrt(1/len(arr1) + 1/len(arr2))
        df = len(arr1) + len(arr2) - 2
        t_critical = stats.t.ppf(1 - self.alpha/2, df)
        
        mean_diff = mean1 - mean2
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff
        
        # Significance tests
        statistical_significance = p_value < self.alpha
        practical_significance = abs(effect_size) > 0.5  # Medium effect size threshold
        
        return StatisticalComparison(
            metric_name=metric_name,
            bioneural_mean=float(mean1),
            baseline_mean=float(mean2),
            effect_size=float(effect_size),
            p_value=float(p_value),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            statistical_significance=statistical_significance,
            practical_significance=practical_significance
        )
    
    def _generate_summary_statistics(self,
                                   bioneural_results: List[PipelineResult],
                                   baseline_results: Dict[str, List[BaselineResult]],
                                   statistical_tests: List[StatisticalComparison]) -> Dict[str, Any]:
        """Generate comprehensive summary statistics"""
        
        summary = {}
        
        # Bioneural performance summary
        bioneural_times = [r.processing_time for r in bioneural_results]
        bioneural_qualities = [r.quality_metrics['overall_quality'] for r in bioneural_results]
        
        summary['bioneural_performance'] = {
            'mean_processing_time': float(np.mean(bioneural_times)),
            'std_processing_time': float(np.std(bioneural_times)),
            'mean_quality_score': float(np.mean(bioneural_qualities)),
            'std_quality_score': float(np.std(bioneural_qualities)),
            'num_successful_processes': len(bioneural_results)
        }
        
        # Baseline performance summary
        summary['baseline_performance'] = {}
        for model_name, results in baseline_results.items():
            if results:
                times = [r.processing_time for r in results if np.isfinite(r.processing_time)]
                summary['baseline_performance'][model_name] = {
                    'mean_processing_time': float(np.mean(times)) if times else float('inf'),
                    'std_processing_time': float(np.std(times)) if times else 0.0,
                    'num_successful_processes': len([r for r in results if np.isfinite(r.processing_time)])
                }
        
        # Statistical test summary
        significant_tests = [test for test in statistical_tests if test.statistical_significance]
        practically_significant_tests = [test for test in statistical_tests if test.practical_significance]
        
        summary['statistical_significance'] = {
            'total_tests_performed': len(statistical_tests),
            'statistically_significant': len(significant_tests),
            'practically_significant': len(practically_significant_tests),
            'significance_rate': float(len(significant_tests) / (len(statistical_tests) + 1e-10)),
            'effect_sizes': [float(test.effect_size) for test in statistical_tests],
            'mean_effect_size': float(np.mean([test.effect_size for test in statistical_tests]))
        }
        
        # Performance rankings
        if baseline_results:
            all_models = list(baseline_results.keys())
            time_rankings = {}
            
            for model_name in all_models:
                results = baseline_results[model_name]
                times = [r.processing_time for r in results if np.isfinite(r.processing_time)]
                if times:
                    time_rankings[model_name] = np.mean(times)
            
            # Add bioneural
            time_rankings['bioneural_fusion'] = np.mean(bioneural_times)
            
            # Sort by processing time
            sorted_models = sorted(time_rankings.items(), key=lambda x: x[1])
            
            summary['performance_rankings'] = {
                'by_speed': [{'model': model, 'avg_time': float(time)} for model, time in sorted_models],
                'bioneural_rank': [i for i, (model, _) in enumerate(sorted_models) if model == 'bioneural_fusion'][0] + 1,
                'total_models': len(sorted_models)
            }
        
        return summary
    
    def _analyze_adaptation_benefits(self, bioneural_results: List[PipelineResult]) -> Dict[str, Any]:
        """Analyze adaptation benefits over processing sequence"""
        
        if len(bioneural_results) < 3:
            return {"insufficient_data": "Need at least 3 samples for adaptation analysis"}
        
        # Extract quality metrics over time
        qualities = [r.quality_metrics['overall_quality'] for r in bioneural_results]
        confidences = [r.bioneural_result.confidence_score for r in bioneural_results]
        processing_times = [r.processing_time for r in bioneural_results]
        
        # Trend analysis
        x = np.arange(len(qualities))
        
        # Quality trend
        quality_slope, quality_intercept, quality_r, quality_p, _ = stats.linregress(x, qualities)
        
        # Confidence trend
        conf_slope, conf_intercept, conf_r, conf_p, _ = stats.linregress(x, confidences)
        
        # Processing time trend (should decrease or stabilize with adaptation)
        time_slope, time_intercept, time_r, time_p, _ = stats.linregress(x, processing_times)
        
        adaptation_analysis = {
            'quality_trend': {
                'slope': float(quality_slope),
                'r_squared': float(quality_r ** 2),
                'p_value': float(quality_p),
                'improvement_per_signal': float(quality_slope),
                'total_improvement': float(quality_slope * len(qualities)),
                'is_improving': quality_slope > 0 and quality_p < 0.05
            },
            'confidence_trend': {
                'slope': float(conf_slope),
                'r_squared': float(conf_r ** 2),
                'p_value': float(conf_p),
                'is_improving': conf_slope > 0 and conf_p < 0.05
            },
            'processing_time_trend': {
                'slope': float(time_slope),
                'r_squared': float(time_r ** 2),
                'p_value': float(time_p),
                'is_stabilizing': abs(time_slope) < 0.001 or (time_slope < 0 and time_p < 0.05)
            },
            'overall_adaptation_benefit': float(
                (1.0 if quality_slope > 0 and quality_p < 0.05 else 0.0) +
                (1.0 if conf_slope > 0 and conf_p < 0.05 else 0.0) +
                (1.0 if time_slope < 0 and time_p < 0.05 else 0.0)
            ) / 3.0
        }
        
        return adaptation_analysis
    
    def _generate_recommendations(self,
                                comparison_metrics: Dict[str, ComparisonMetrics],
                                statistical_tests: List[StatisticalComparison],
                                summary_statistics: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        
        recommendations = []
        
        # Performance recommendations
        bioneural_rank = summary_statistics.get('performance_rankings', {}).get('bioneural_rank', 0)
        total_models = summary_statistics.get('performance_rankings', {}).get('total_models', 1)
        
        if bioneural_rank <= total_models // 3:
            recommendations.append("✅ Bioneural fusion shows competitive processing speed among top performers")
        elif bioneural_rank > 2 * total_models // 3:
            recommendations.append("⚠️ Consider optimizing bioneural fusion for better processing speed")
        
        # Quality recommendations
        significant_quality_improvements = [
            test for test in statistical_tests 
            if 'quality' in test.metric_name and test.statistical_significance and test.effect_size > 0
        ]
        
        if len(significant_quality_improvements) > len(comparison_metrics) // 2:
            recommendations.append("🎯 Bioneural fusion demonstrates statistically significant quality improvements")
        
        # Effect size recommendations
        large_effects = [test for test in statistical_tests if abs(test.effect_size) > 0.8]
        if large_effects:
            recommendations.append(f"💪 Found {len(large_effects)} metrics with large effect sizes (Cohen's d > 0.8)")
        
        # Adaptation recommendations
        adaptation_analysis = summary_statistics.get('adaptation_analysis', {})
        if adaptation_analysis.get('overall_adaptation_benefit', 0) > 0.5:
            recommendations.append("🔄 Adaptation mechanism provides substantial benefits - consider increasing adaptation strength")
        
        # Statistical power recommendations
        significance_rate = summary_statistics.get('statistical_significance', {}).get('significance_rate', 0)
        if significance_rate > 0.7:
            recommendations.append("📊 Strong statistical evidence supports bioneural fusion superiority")
        elif significance_rate < 0.3:
            recommendations.append("🔬 Consider larger sample sizes for more robust statistical validation")
        
        # Practical significance
        practical_tests = [test for test in statistical_tests if test.practical_significance]
        if len(practical_tests) > len(statistical_tests) // 2:
            recommendations.append("⚡ Bioneural improvements are both statistically and practically significant")
        
        return recommendations
    
    def export_analysis_report(self, result: ComparativeAnalysisResult) -> Dict[str, Any]:
        """Export comprehensive analysis report"""
        
        report = {
            "analysis_metadata": {
                "dataset_name": result.dataset_info["name"],
                "analysis_timestamp": time.time(),
                "analyzer_config": {
                    "signal_dim": self.signal_dim,
                    "confidence_level": self.confidence_level
                }
            },
            "dataset_summary": result.dataset_info,
            "performance_comparison": {},
            "statistical_analysis": {},
            "recommendations": result.recommendations,
            "detailed_results": {}
        }
        
        # Performance comparison summary
        for model_name, metrics in result.comparison_metrics.items():
            report["performance_comparison"][model_name] = asdict(metrics)
        
        # Statistical analysis summary
        report["statistical_analysis"] = {
            "significant_tests": [
                {
                    "metric": test.metric_name,
                    "effect_size": test.effect_size,
                    "p_value": test.p_value,
                    "bioneural_better": test.bioneural_mean > test.baseline_mean
                }
                for test in result.statistical_tests if test.statistical_significance
            ],
            "summary_statistics": result.summary_statistics
        }
        
        # Detailed results (subset for export)
        report["detailed_results"] = {
            "num_bioneural_results": len(result.bioneural_results),
            "num_baseline_models": len(result.baseline_results),
            "bioneural_avg_quality": np.mean([r.quality_metrics['overall_quality'] for r in result.bioneural_results]),
            "bioneural_avg_time": np.mean([r.processing_time for r in result.bioneural_results])
        }
        
        return report
    
    def summary(self) -> Dict[str, Any]:
        """Get comprehensive analyzer summary"""
        return {
            "analyzer_type": "ComparativeAnalyzer",
            "configuration": {
                "signal_dimension": self.signal_dim,
                "confidence_level": self.confidence_level,
                "alpha": self.alpha
            },
            "analysis_statistics": self.analysis_stats.copy(),
            "components": {
                "bioneural_pipeline": self.bioneural_pipeline.summary(),
                "baseline_suite": self.baseline_suite.summary()
            }
        }