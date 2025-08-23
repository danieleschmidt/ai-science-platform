"""Autonomous Research Agent for Scientific Discovery"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path

from ..algorithms.discovery import DiscoveryEngine, Discovery
from ..utils.validation import ValidationMixin
from ..utils.error_handling import robust_execution, DiscoveryError

logger = logging.getLogger(__name__)


@dataclass
class ResearchHypothesis:
    """Structured research hypothesis with testable predictions"""
    id: str
    title: str
    description: str
    predictions: List[str]
    methodology: Dict[str, Any]
    expected_outcomes: Dict[str, Any]
    confidence_level: float = 0.5
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "proposed"  # proposed, testing, validated, rejected


@dataclass
class ExperimentResult:
    """Results from hypothesis testing experiment"""
    hypothesis_id: str
    experiment_type: str
    metrics: Dict[str, float]
    statistical_significance: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    raw_data: Optional[Dict[str, Any]] = None
    interpretation: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class AutonomousResearcher(ValidationMixin):
    """AI agent for autonomous scientific research and hypothesis generation"""
    
    def __init__(self, 
                 research_domain: str = "general",
                 significance_threshold: float = 0.05,
                 effect_size_threshold: float = 0.3):
        """
        Initialize autonomous researcher
        
        Args:
            research_domain: Domain of research focus
            significance_threshold: Statistical significance threshold (p-value)
            effect_size_threshold: Minimum effect size for practical significance
        """
        self.research_domain = research_domain
        self.significance_threshold = significance_threshold
        self.effect_size_threshold = effect_size_threshold
        
        self.hypotheses: List[ResearchHypothesis] = []
        self.experiments: List[ExperimentResult] = []
        self.discovery_engine = DiscoveryEngine(discovery_threshold=0.75)
        
        logger.info(f"Autonomous researcher initialized for domain: {research_domain}")
    
    @robust_execution(recovery_strategy='graceful_degradation')
    def generate_research_hypothesis(self, 
                                   data: np.ndarray,
                                   context: str = "",
                                   prior_knowledge: Optional[Dict[str, Any]] = None) -> ResearchHypothesis:
        """Generate structured research hypothesis from data patterns"""
        
        if data.size == 0:
            raise DiscoveryError("Cannot generate hypothesis from empty data")
        
        # Analyze data patterns
        patterns = self._analyze_data_patterns(data)
        
        # Generate hypothesis based on patterns
        hypothesis_id = f"hyp_{len(self.hypotheses) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create structured hypothesis
        if patterns['trend_strength'] > 0.7:
            title = f"Strong {patterns['trend_type']} trend in {self.research_domain}"
            description = f"Data exhibits significant {patterns['trend_type']} pattern with strength {patterns['trend_strength']:.3f}"
            predictions = [
                f"Future observations will continue {patterns['trend_type']} pattern",
                f"Effect size will be >= {self.effect_size_threshold}",
                f"Statistical significance will be p < {self.significance_threshold}"
            ]
        elif patterns['clustering_score'] > 0.6:
            title = f"Distinct clustering patterns in {self.research_domain} data"
            description = f"Data shows {patterns['n_clusters']} distinct clusters with separation score {patterns['clustering_score']:.3f}"
            predictions = [
                f"Data naturally separates into {patterns['n_clusters']} groups",
                f"Inter-cluster distance will be significantly different from intra-cluster",
                f"Classification accuracy will exceed 80%"
            ]
        else:
            title = f"Complex patterns in {self.research_domain} requiring further analysis"
            description = f"Data exhibits complex behavior with multiple underlying factors"
            predictions = [
                "Non-linear relationships exist between variables",
                "Standard linear models will show poor fit (R² < 0.5)",
                "Advanced modeling approaches will reveal hidden patterns"
            ]
        
        # Define methodology
        methodology = {
            "data_analysis": ["descriptive_statistics", "pattern_recognition"],
            "statistical_tests": self._recommend_statistical_tests(patterns),
            "validation_approach": "cross_validation" if data.shape[0] > 100 else "bootstrap",
            "sample_size": data.shape[0],
            "required_replications": 3
        }
        
        # Expected outcomes
        expected_outcomes = {
            "primary_metric": patterns.get('primary_metric', 'effect_size'),
            "expected_effect_size": patterns.get('expected_effect', 0.5),
            "statistical_power": 0.8,
            "confidence_level": 0.95
        }
        
        hypothesis = ResearchHypothesis(
            id=hypothesis_id,
            title=title,
            description=description,
            predictions=predictions,
            methodology=methodology,
            expected_outcomes=expected_outcomes,
            confidence_level=patterns.get('confidence', 0.6)
        )
        
        self.hypotheses.append(hypothesis)
        logger.info(f"Generated research hypothesis: {title}")
        
        return hypothesis
    
    @robust_execution(recovery_strategy='graceful_degradation')
    def test_hypothesis(self, 
                       hypothesis: ResearchHypothesis,
                       test_data: np.ndarray,
                       control_data: Optional[np.ndarray] = None) -> ExperimentResult:
        """Test research hypothesis with rigorous experimental design"""
        
        logger.info(f"Testing hypothesis: {hypothesis.title}")
        
        # Prepare experimental design
        experiment_type = self._determine_experiment_type(hypothesis, test_data, control_data)
        
        # Run statistical analysis
        metrics = {}
        
        if experiment_type == "comparative":
            if control_data is None:
                raise ValueError("Control data required for comparative experiments")
            
            # Two-sample comparison
            from scipy import stats
            
            # Calculate test statistics
            statistic, p_value = stats.ttest_ind(test_data, control_data)
            effect_size = self._calculate_cohens_d(test_data, control_data)
            
            # Confidence interval for effect size
            ci_lower, ci_upper = self._bootstrap_ci(test_data, control_data)
            
            metrics = {
                "mean_treatment": float(np.mean(test_data)),
                "mean_control": float(np.mean(control_data)),
                "std_treatment": float(np.std(test_data)),
                "std_control": float(np.std(control_data)),
                "sample_size_treatment": len(test_data),
                "sample_size_control": len(control_data)
            }
            
        else:
            # Single sample analysis
            from scipy import stats
            
            # One-sample t-test against theoretical mean
            theoretical_mean = hypothesis.expected_outcomes.get('expected_mean', 0)
            statistic, p_value = stats.ttest_1samp(test_data, theoretical_mean)
            effect_size = abs(np.mean(test_data) - theoretical_mean) / np.std(test_data)
            
            # Bootstrap confidence interval
            bootstrap_means = []
            for _ in range(1000):
                bootstrap_sample = np.random.choice(test_data, len(test_data), replace=True)
                bootstrap_means.append(np.mean(bootstrap_sample))
            
            ci_lower, ci_upper = np.percentile(bootstrap_means, [2.5, 97.5])
            
            metrics = {
                "observed_mean": float(np.mean(test_data)),
                "theoretical_mean": theoretical_mean,
                "std": float(np.std(test_data)),
                "sample_size": len(test_data)
            }
        
        # Determine statistical significance
        is_significant = p_value < self.significance_threshold
        is_practically_significant = effect_size > self.effect_size_threshold
        
        # Generate interpretation
        interpretation = self._interpret_results(
            hypothesis, p_value, effect_size, is_significant, is_practically_significant
        )
        
        # Create experiment result
        result = ExperimentResult(
            hypothesis_id=hypothesis.id,
            experiment_type=experiment_type,
            metrics=metrics,
            statistical_significance=float(is_significant),
            p_value=float(p_value),
            effect_size=float(effect_size),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            interpretation=interpretation
        )
        
        self.experiments.append(result)
        
        # Update hypothesis status
        if is_significant and is_practically_significant:
            hypothesis.status = "validated"
        elif p_value > 0.1:  # Clearly non-significant
            hypothesis.status = "rejected"
        else:
            hypothesis.status = "inconclusive"
        
        logger.info(f"Hypothesis test complete: {hypothesis.status}, p={p_value:.4f}, effect_size={effect_size:.3f}")
        
        return result
    
    def conduct_autonomous_research(self, 
                                  datasets: List[np.ndarray],
                                  research_questions: Optional[List[str]] = None) -> Dict[str, Any]:
        """Conduct fully autonomous research across multiple datasets"""
        
        logger.info(f"Starting autonomous research on {len(datasets)} datasets")
        
        research_report = {
            "research_domain": self.research_domain,
            "start_time": datetime.now().isoformat(),
            "datasets_analyzed": len(datasets),
            "hypotheses_generated": [],
            "experiments_conducted": [],
            "key_findings": [],
            "statistical_summary": {}
        }
        
        # Generate hypotheses for each dataset
        for i, data in enumerate(datasets):
            context = f"dataset_{i+1}"
            if research_questions and i < len(research_questions):
                context = f"{research_questions[i]}_dataset_{i+1}"
            
            hypothesis = self.generate_research_hypothesis(data, context)
            research_report["hypotheses_generated"].append({
                "id": hypothesis.id,
                "title": hypothesis.title,
                "confidence": hypothesis.confidence_level
            })
            
            # Test hypothesis with train/test split
            if len(data) > 20:  # Sufficient data for splitting
                split_point = len(data) // 2
                train_data = data[:split_point]
                test_data = data[split_point:]
                
                result = self.test_hypothesis(hypothesis, test_data, train_data)
                research_report["experiments_conducted"].append({
                    "hypothesis_id": result.hypothesis_id,
                    "p_value": result.p_value,
                    "effect_size": result.effect_size,
                    "significant": result.statistical_significance > 0.5
                })
        
        # Synthesize key findings
        validated_hypotheses = [h for h in self.hypotheses if h.status == "validated"]
        research_report["key_findings"] = [
            f"Validated {len(validated_hypotheses)} of {len(self.hypotheses)} hypotheses",
            f"Average effect size: {np.mean([e.effect_size for e in self.experiments]):.3f}",
            f"Research domain: {self.research_domain} shows consistent patterns"
        ]
        
        # Statistical summary
        if self.experiments:
            research_report["statistical_summary"] = {
                "mean_p_value": float(np.mean([e.p_value for e in self.experiments])),
                "mean_effect_size": float(np.mean([e.effect_size for e in self.experiments])),
                "significant_results": sum(1 for e in self.experiments if e.p_value < self.significance_threshold),
                "total_experiments": len(self.experiments)
            }
        
        research_report["end_time"] = datetime.now().isoformat()
        
        logger.info("Autonomous research completed successfully")
        return research_report
    
    def _analyze_data_patterns(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze data for patterns and characteristics"""
        patterns = {}
        
        # Basic statistics
        patterns['mean'] = float(np.mean(data))
        patterns['std'] = float(np.std(data))
        patterns['skew'] = float(self._calculate_skewness(data))
        
        # Trend analysis
        if data.ndim == 1:
            x = np.arange(len(data))
            correlation = np.corrcoef(x, data)[0, 1]
            patterns['trend_strength'] = abs(correlation)
            patterns['trend_type'] = "increasing" if correlation > 0 else "decreasing"
        else:
            patterns['trend_strength'] = 0.0
            patterns['trend_type'] = "none"
        
        # Clustering analysis (simplified)
        patterns['clustering_score'] = self._estimate_clustering(data)
        patterns['n_clusters'] = self._estimate_cluster_count(data)
        
        # Confidence estimation
        patterns['confidence'] = min(0.9, 0.5 + patterns['trend_strength'] * 0.4)
        
        return patterns
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _estimate_clustering(self, data: np.ndarray) -> float:
        """Estimate how clustered the data is"""
        if data.ndim == 1:
            # Use histogram-based approach
            hist, _ = np.histogram(data, bins=min(10, len(data)//3))
            # Measure concentration
            return float(np.std(hist) / (np.mean(hist) + 1e-6))
        else:
            # For multi-dimensional, use simple variance-based measure
            return float(1.0 / (1.0 + np.mean(np.var(data, axis=0))))
    
    def _estimate_cluster_count(self, data: np.ndarray) -> int:
        """Estimate number of natural clusters"""
        if data.ndim == 1 and len(data) > 10:
            # Simple approach using histogram peaks
            hist, _ = np.histogram(data, bins=min(10, len(data)//5))
            peaks = sum(1 for i in range(1, len(hist)-1) 
                       if hist[i] > hist[i-1] and hist[i] > hist[i+1])
            return max(2, min(5, peaks))
        return 2
    
    def _recommend_statistical_tests(self, patterns: Dict[str, Any]) -> List[str]:
        """Recommend appropriate statistical tests based on data patterns"""
        tests = ["descriptive_statistics"]
        
        if patterns.get('trend_strength', 0) > 0.5:
            tests.extend(["correlation_analysis", "trend_test"])
        
        if patterns.get('clustering_score', 0) > 0.5:
            tests.extend(["cluster_analysis", "anova"])
        
        tests.extend(["normality_test", "hypothesis_test"])
        return tests
    
    def _determine_experiment_type(self, 
                                  hypothesis: ResearchHypothesis,
                                  test_data: np.ndarray,
                                  control_data: Optional[np.ndarray]) -> str:
        """Determine appropriate experiment type"""
        if control_data is not None:
            return "comparative"
        elif "trend" in hypothesis.title.lower():
            return "trend_analysis"
        elif "cluster" in hypothesis.title.lower():
            return "clustering_analysis"
        else:
            return "descriptive"
    
    def _calculate_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size"""
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        
        # Pooled standard deviation
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
        
        return (mean1 - mean2) / pooled_std
    
    def _bootstrap_ci(self, group1: np.ndarray, group2: np.ndarray, 
                     n_bootstrap: int = 1000) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for mean difference"""
        bootstrap_diffs = []
        
        for _ in range(n_bootstrap):
            boot1 = np.random.choice(group1, len(group1), replace=True)
            boot2 = np.random.choice(group2, len(group2), replace=True)
            bootstrap_diffs.append(np.mean(boot1) - np.mean(boot2))
        
        ci_lower, ci_upper = np.percentile(bootstrap_diffs, [2.5, 97.5])
        return ci_lower, ci_upper
    
    def _interpret_results(self, 
                          hypothesis: ResearchHypothesis,
                          p_value: float,
                          effect_size: float,
                          is_significant: bool,
                          is_practically_significant: bool) -> str:
        """Generate interpretation of experimental results"""
        
        interpretation = f"Hypothesis '{hypothesis.title}' "
        
        if is_significant and is_practically_significant:
            interpretation += f"is SUPPORTED by strong evidence (p={p_value:.4f}, effect_size={effect_size:.3f}). "
            interpretation += "Both statistical and practical significance achieved."
        elif is_significant and not is_practically_significant:
            interpretation += f"shows statistical significance (p={p_value:.4f}) but small effect size ({effect_size:.3f}). "
            interpretation += "Results may not be practically meaningful."
        elif is_practically_significant and not is_significant:
            interpretation += f"shows large effect size ({effect_size:.3f}) but lacks statistical significance (p={p_value:.4f}). "
            interpretation += "May require larger sample size."
        else:
            interpretation += f"is NOT SUPPORTED by the evidence (p={p_value:.4f}, effect_size={effect_size:.3f}). "
            interpretation += "Both statistical and practical significance are lacking."
        
        return interpretation
    
    def get_research_summary(self) -> Dict[str, Any]:
        """Get comprehensive research summary"""
        return {
            "domain": self.research_domain,
            "total_hypotheses": len(self.hypotheses),
            "total_experiments": len(self.experiments),
            "validated_hypotheses": sum(1 for h in self.hypotheses if h.status == "validated"),
            "rejected_hypotheses": sum(1 for h in self.hypotheses if h.status == "rejected"),
            "average_effect_size": np.mean([e.effect_size for e in self.experiments]) if self.experiments else 0.0,
            "significant_results": sum(1 for e in self.experiments if e.p_value < self.significance_threshold),
            "research_quality_score": self._calculate_research_quality_score()
        }
    
    def _calculate_research_quality_score(self) -> float:
        """Calculate overall research quality score (0-1)"""
        if not self.experiments:
            return 0.0
        
        # Factors: statistical rigor, effect sizes, replication
        stat_rigor = sum(1 for e in self.experiments if e.p_value < 0.05) / len(self.experiments)
        effect_quality = sum(1 for e in self.experiments if e.effect_size > 0.3) / len(self.experiments)
        hypothesis_quality = sum(1 for h in self.hypotheses if h.confidence_level > 0.7) / len(self.hypotheses)
        
        return (stat_rigor + effect_quality + hypothesis_quality) / 3.0