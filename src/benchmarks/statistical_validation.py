"""
Statistical Validation Framework
Rigorous statistical testing for research reproducibility and validation
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import logging
from dataclasses import dataclass, asdict
import time
from scipy import stats
from scipy.stats import mannwhitneyu, wilcoxon, kruskal, friedmanchisquare
from scipy.stats import bootstrap
import warnings

from ..algorithms.bioneural_pipeline import BioneuralOlfactoryPipeline, PipelineResult, PipelineConfig
from ..utils.validation import ValidationMixin
from ..utils.error_handling import robust_execution, safe_array_operation

logger = logging.getLogger(__name__)


@dataclass
class HypothesisTest:
    """Result of a statistical hypothesis test"""
    test_name: str
    null_hypothesis: str
    alternative_hypothesis: str
    test_statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    degrees_of_freedom: Optional[int]
    statistical_significance: bool
    practical_significance: bool
    interpretation: str


@dataclass
class PowerAnalysis:
    """Statistical power analysis result"""
    effect_size: float
    sample_size: int
    alpha: float
    power: float
    required_sample_size: int
    minimum_detectable_effect: float


@dataclass
class ReproducibilityTest:
    """Reproducibility validation result"""
    test_type: str
    original_results: List[float]
    replicated_results: List[float]
    correlation: float
    mean_absolute_error: float
    relative_error: float
    reproducibility_score: float
    is_reproducible: bool


@dataclass
class StatisticalValidationResult:
    """Complete statistical validation result"""
    validation_metadata: Dict[str, Any]
    hypothesis_tests: List[HypothesisTest]
    power_analyses: List[PowerAnalysis]
    reproducibility_tests: List[ReproducibilityTest]
    multiple_comparison_corrections: Dict[str, Any]
    bootstrap_results: Dict[str, Any]
    cross_validation_results: Dict[str, Any]
    summary_conclusions: List[str]


class StatisticalValidator(ValidationMixin):
    """
    Comprehensive Statistical Validation Framework
    
    Provides rigorous statistical testing for research validation including:
    1. Hypothesis testing with multiple test corrections
    2. Power analysis and sample size calculations
    3. Effect size computations with confidence intervals
    4. Reproducibility testing across runs
    5. Bootstrap confidence intervals
    6. Cross-validation statistical assessments
    7. Non-parametric alternatives for robust testing
    
    Research Standards Compliance:
    - Multiple comparison corrections (Bonferroni, FDR)
    - Effect size reporting (Cohen's d, eta-squared)
    - Power analysis for adequacy assessment
    - Reproducibility validation across multiple runs
    - Confidence interval reporting
    """
    
    def __init__(self, 
                 alpha: float = 0.05,
                 power_threshold: float = 0.8,
                 effect_size_threshold: float = 0.5,
                 n_bootstrap_samples: int = 1000,
                 random_seed: int = 42):
        """
        Initialize statistical validator
        
        Args:
            alpha: Type I error rate (significance level)
            power_threshold: Minimum acceptable statistical power
            effect_size_threshold: Minimum meaningful effect size
            n_bootstrap_samples: Number of bootstrap samples
            random_seed: Random seed for reproducibility
        """
        self.alpha = self.validate_probability(alpha, "alpha")
        self.power_threshold = self.validate_probability(power_threshold, "power_threshold") 
        self.effect_size_threshold = self.validate_positive_float(effect_size_threshold, "effect_size_threshold")
        self.n_bootstrap_samples = self.validate_positive_int(n_bootstrap_samples, "n_bootstrap_samples")
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Validation statistics
        self.validation_stats = {
            "validations_performed": 0,
            "tests_conducted": 0,
            "significant_results": 0,
            "reproducible_results": 0
        }
        
        logger.info(f"StatisticalValidator initialized: α={alpha}, power≥{power_threshold}, effect_size≥{effect_size_threshold}")
    
    @robust_execution(recovery_strategy='graceful_degradation')
    @safe_array_operation 
    def validate_research_claims(self,
                               experimental_data: List[Dict[str, Any]],
                               control_data: Optional[List[Dict[str, Any]]] = None,
                               research_hypotheses: Optional[List[str]] = None,
                               validation_name: str = "research_validation") -> StatisticalValidationResult:
        """
        Perform comprehensive statistical validation of research claims
        
        Args:
            experimental_data: Experimental condition results
            control_data: Control condition results (if available)
            research_hypotheses: List of research hypotheses to test
            validation_name: Name for this validation
            
        Returns:
            Complete statistical validation results
        """
        logger.info(f"Starting statistical validation: '{validation_name}'")
        start_time = time.time()
        
        validation_metadata = {
            "validation_name": validation_name,
            "start_time": start_time,
            "alpha": self.alpha,
            "power_threshold": self.power_threshold,
            "effect_size_threshold": self.effect_size_threshold,
            "n_experimental": len(experimental_data),
            "n_control": len(control_data) if control_data else 0,
            "hypotheses": research_hypotheses or []
        }
        
        # Step 1: Hypothesis Testing
        logger.info("Performing hypothesis tests")
        hypothesis_tests = self._perform_hypothesis_tests(experimental_data, control_data)
        
        # Step 2: Power Analysis
        logger.info("Conducting power analysis")
        power_analyses = self._conduct_power_analysis(experimental_data, control_data)
        
        # Step 3: Reproducibility Testing
        logger.info("Testing reproducibility")
        reproducibility_tests = self._test_reproducibility(experimental_data)
        
        # Step 4: Multiple Comparison Corrections
        logger.info("Applying multiple comparison corrections")
        corrected_results = self._apply_multiple_comparison_corrections(hypothesis_tests)
        
        # Step 5: Bootstrap Confidence Intervals
        logger.info("Computing bootstrap confidence intervals")
        bootstrap_results = self._compute_bootstrap_intervals(experimental_data, control_data)
        
        # Step 6: Cross-Validation Assessment
        logger.info("Performing cross-validation assessment")
        cv_results = self._perform_cross_validation_assessment(experimental_data)
        
        # Step 7: Generate Summary Conclusions
        summary_conclusions = self._generate_statistical_conclusions(
            hypothesis_tests, power_analyses, reproducibility_tests, corrected_results
        )
        
        # Update validation statistics
        self.validation_stats["validations_performed"] += 1
        self.validation_stats["tests_conducted"] += len(hypothesis_tests)
        self.validation_stats["significant_results"] += sum(1 for test in hypothesis_tests if test.statistical_significance)
        self.validation_stats["reproducible_results"] += sum(1 for test in reproducibility_tests if test.is_reproducible)
        
        validation_time = time.time() - start_time
        validation_metadata["completion_time"] = validation_time
        
        result = StatisticalValidationResult(
            validation_metadata=validation_metadata,
            hypothesis_tests=hypothesis_tests,
            power_analyses=power_analyses,
            reproducibility_tests=reproducibility_tests,
            multiple_comparison_corrections=corrected_results,
            bootstrap_results=bootstrap_results,
            cross_validation_results=cv_results,
            summary_conclusions=summary_conclusions
        )
        
        logger.info(f"Statistical validation complete in {validation_time:.2f}s: {len(hypothesis_tests)} tests, {sum(1 for t in hypothesis_tests if t.statistical_significance)} significant")
        return result
    
    def _perform_hypothesis_tests(self,
                                experimental_data: List[Dict[str, Any]],
                                control_data: Optional[List[Dict[str, Any]]]) -> List[HypothesisTest]:
        """Perform comprehensive hypothesis testing"""
        
        tests = []
        
        # Extract key metrics for testing
        exp_metrics = self._extract_metrics_for_testing(experimental_data)
        
        if control_data:
            ctrl_metrics = self._extract_metrics_for_testing(control_data)
            
            # Perform between-group tests
            for metric_name in exp_metrics.keys():
                if metric_name in ctrl_metrics:
                    exp_values = exp_metrics[metric_name]
                    ctrl_values = ctrl_metrics[metric_name]
                    
                    # Parametric t-test
                    t_test = self._welch_t_test(exp_values, ctrl_values, metric_name)
                    tests.append(t_test)
                    
                    # Non-parametric Mann-Whitney U test
                    mw_test = self._mann_whitney_test(exp_values, ctrl_values, metric_name)
                    tests.append(mw_test)
                    
                    # Effect size test
                    effect_test = self._effect_size_test(exp_values, ctrl_values, metric_name)
                    tests.append(effect_test)
        
        else:
            # Single-sample tests against theoretical values
            for metric_name, values in exp_metrics.items():
                # Test against neutral/baseline values
                if 'quality' in metric_name.lower():
                    theoretical_value = 0.5  # Neutral quality
                elif 'time' in metric_name.lower():
                    theoretical_value = np.median(values)  # Median as baseline
                else:
                    theoretical_value = 0.0  # Zero baseline
                
                one_sample_test = self._one_sample_test(values, theoretical_value, metric_name)
                tests.append(one_sample_test)
        
        # Within-subjects tests (if applicable)
        if len(experimental_data) > 1:
            within_tests = self._perform_within_subjects_tests(experimental_data)
            tests.extend(within_tests)
        
        return tests
    
    def _extract_metrics_for_testing(self, data: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """Extract numerical metrics from data for statistical testing"""
        metrics = {}
        
        for item in data:
            # Handle PipelineResult objects
            if hasattr(item, 'quality_metrics'):
                for key, value in item.quality_metrics.items():
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        if key not in metrics:
                            metrics[key] = []
                        metrics[key].append(float(value))
                
                # Add processing time
                if hasattr(item, 'processing_time') and np.isfinite(item.processing_time):
                    if 'processing_time' not in metrics:
                        metrics['processing_time'] = []
                    metrics['processing_time'].append(float(item.processing_time))
            
            # Handle dictionary format
            elif isinstance(item, dict):
                for key, value in item.items():
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        if key not in metrics:
                            metrics[key] = []
                        metrics[key].append(float(value))
                    elif isinstance(value, dict):
                        # Nested metrics
                        for nested_key, nested_value in value.items():
                            if isinstance(nested_value, (int, float)) and np.isfinite(nested_value):
                                combined_key = f"{key}_{nested_key}"
                                if combined_key not in metrics:
                                    metrics[combined_key] = []
                                metrics[combined_key].append(float(nested_value))
        
        # Filter out metrics with insufficient data
        filtered_metrics = {k: v for k, v in metrics.items() if len(v) >= 3}
        
        return filtered_metrics
    
    def _welch_t_test(self, sample1: List[float], sample2: List[float], metric_name: str) -> HypothesisTest:
        """Perform Welch's t-test (unequal variances)"""
        
        arr1, arr2 = np.array(sample1), np.array(sample2)
        
        # Perform test
        t_stat, p_value = stats.ttest_ind(arr1, arr2, equal_var=False)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(arr1) - 1) * np.var(arr1, ddof=1) + 
                             (len(arr2) - 1) * np.var(arr2, ddof=1)) / 
                            (len(arr1) + len(arr2) - 2))
        cohens_d = (np.mean(arr1) - np.mean(arr2)) / (pooled_std + 1e-10)
        
        # Confidence interval for mean difference
        se_diff = pooled_std * np.sqrt(1/len(arr1) + 1/len(arr2))
        df = len(arr1) + len(arr2) - 2
        t_critical = stats.t.ppf(1 - self.alpha/2, df)
        
        mean_diff = np.mean(arr1) - np.mean(arr2)
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff
        
        # Interpretations
        if p_value < self.alpha and abs(cohens_d) > self.effect_size_threshold:
            interpretation = f"Significant difference with {self._interpret_effect_size(cohens_d)} effect"
        elif p_value < self.alpha:
            interpretation = "Statistically significant but small effect size"
        else:
            interpretation = "No significant difference detected"
        
        return HypothesisTest(
            test_name=f"Welch_t_test_{metric_name}",
            null_hypothesis="No difference in means between groups",
            alternative_hypothesis="Significant difference in means between groups",
            test_statistic=float(t_stat),
            p_value=float(p_value),
            effect_size=float(cohens_d),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            degrees_of_freedom=int(df),
            statistical_significance=p_value < self.alpha,
            practical_significance=abs(cohens_d) > self.effect_size_threshold,
            interpretation=interpretation
        )
    
    def _mann_whitney_test(self, sample1: List[float], sample2: List[float], metric_name: str) -> HypothesisTest:
        """Perform Mann-Whitney U test (non-parametric)"""
        
        arr1, arr2 = np.array(sample1), np.array(sample2)
        
        # Perform test
        u_stat, p_value = mannwhitneyu(arr1, arr2, alternative='two-sided')
        
        # Effect size (r = Z / sqrt(N))
        n1, n2 = len(arr1), len(arr2)
        total_n = n1 + n2
        
        # Convert U to Z-score approximation
        mean_u = n1 * n2 / 2
        std_u = np.sqrt(n1 * n2 * (total_n + 1) / 12)
        z_score = (u_stat - mean_u) / (std_u + 1e-10)
        
        effect_size_r = abs(z_score) / np.sqrt(total_n)
        
        # Interpretation
        if p_value < self.alpha and effect_size_r > 0.3:  # Medium effect for r
            interpretation = f"Significant difference with {self._interpret_effect_size_r(effect_size_r)} effect"
        elif p_value < self.alpha:
            interpretation = "Statistically significant but small effect size"
        else:
            interpretation = "No significant difference detected"
        
        return HypothesisTest(
            test_name=f"Mann_Whitney_U_{metric_name}",
            null_hypothesis="No difference in distributions between groups", 
            alternative_hypothesis="Significant difference in distributions between groups",
            test_statistic=float(u_stat),
            p_value=float(p_value),
            effect_size=float(effect_size_r),
            confidence_interval=(0.0, 0.0),  # Not directly available for Mann-Whitney
            degrees_of_freedom=None,
            statistical_significance=p_value < self.alpha,
            practical_significance=effect_size_r > 0.3,
            interpretation=interpretation
        )
    
    def _effect_size_test(self, sample1: List[float], sample2: List[float], metric_name: str) -> HypothesisTest:
        """Test specifically for meaningful effect size"""
        
        arr1, arr2 = np.array(sample1), np.array(sample2)
        
        # Calculate Cohen's d
        pooled_std = np.sqrt(((len(arr1) - 1) * np.var(arr1, ddof=1) + 
                             (len(arr2) - 1) * np.var(arr2, ddof=1)) / 
                            (len(arr1) + len(arr2) - 2))
        cohens_d = (np.mean(arr1) - np.mean(arr2)) / (pooled_std + 1e-10)
        
        # Bootstrap confidence interval for effect size
        def bootstrap_cohens_d(x, y):
            n_x, n_y = len(x), len(y)
            boot_x = np.random.choice(x, n_x, replace=True)
            boot_y = np.random.choice(y, n_y, replace=True)
            pooled_std_boot = np.sqrt(((n_x - 1) * np.var(boot_x, ddof=1) + 
                                      (n_y - 1) * np.var(boot_y, ddof=1)) / 
                                     (n_x + n_y - 2))
            return (np.mean(boot_x) - np.mean(boot_y)) / (pooled_std_boot + 1e-10)
        
        bootstrap_effects = [bootstrap_cohens_d(arr1, arr2) for _ in range(1000)]
        ci_lower = np.percentile(bootstrap_effects, 2.5)
        ci_upper = np.percentile(bootstrap_effects, 97.5)
        
        # Test if effect size is meaningfully different from zero
        # Pseudo p-value: proportion of bootstrap samples with effect size < threshold
        meaningful_threshold = self.effect_size_threshold
        p_value_approx = np.mean(np.abs(bootstrap_effects) < meaningful_threshold)
        
        interpretation = f"Effect size Cohen's d = {cohens_d:.3f} ({self._interpret_effect_size(cohens_d)})"
        
        return HypothesisTest(
            test_name=f"Effect_Size_Test_{metric_name}",
            null_hypothesis=f"Effect size is less than {meaningful_threshold}",
            alternative_hypothesis=f"Effect size is greater than or equal to {meaningful_threshold}",
            test_statistic=float(abs(cohens_d)),
            p_value=float(p_value_approx),
            effect_size=float(cohens_d),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            degrees_of_freedom=None,
            statistical_significance=abs(cohens_d) > meaningful_threshold,
            practical_significance=abs(cohens_d) > self.effect_size_threshold,
            interpretation=interpretation
        )
    
    def _one_sample_test(self, sample: List[float], theoretical_value: float, metric_name: str) -> HypothesisTest:
        """Perform one-sample t-test against theoretical value"""
        
        arr = np.array(sample)
        
        # Perform test
        t_stat, p_value = stats.ttest_1samp(arr, theoretical_value)
        
        # Effect size (Cohen's d for one sample)
        cohens_d = (np.mean(arr) - theoretical_value) / (np.std(arr, ddof=1) + 1e-10)
        
        # Confidence interval for mean
        se_mean = stats.sem(arr)
        df = len(arr) - 1
        t_critical = stats.t.ppf(1 - self.alpha/2, df)
        
        mean_val = np.mean(arr)
        ci_lower = mean_val - t_critical * se_mean
        ci_upper = mean_val + t_critical * se_mean
        
        interpretation = f"Sample mean {mean_val:.3f} vs theoretical {theoretical_value:.3f}"
        
        return HypothesisTest(
            test_name=f"One_Sample_t_test_{metric_name}",
            null_hypothesis=f"Population mean equals {theoretical_value}",
            alternative_hypothesis=f"Population mean differs from {theoretical_value}",
            test_statistic=float(t_stat),
            p_value=float(p_value),
            effect_size=float(cohens_d),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            degrees_of_freedom=int(df),
            statistical_significance=p_value < self.alpha,
            practical_significance=abs(cohens_d) > self.effect_size_threshold,
            interpretation=interpretation
        )
    
    def _perform_within_subjects_tests(self, data: List[Dict[str, Any]]) -> List[HypothesisTest]:
        """Perform within-subjects analysis (e.g., improvement over time)"""
        
        tests = []
        
        # Extract time-series metrics
        metrics = self._extract_metrics_for_testing(data)
        
        for metric_name, values in metrics.items():
            if len(values) >= 4:  # Need sufficient data for trend analysis
                # Test for linear trend
                x = np.arange(len(values))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                
                # Effect size for trend (r-squared)
                r_squared = r_value ** 2
                
                interpretation = f"{'Positive' if slope > 0 else 'Negative'} trend with R² = {r_squared:.3f}"
                
                trend_test = HypothesisTest(
                    test_name=f"Trend_Analysis_{metric_name}",
                    null_hypothesis="No linear trend over time",
                    alternative_hypothesis="Significant linear trend over time",
                    test_statistic=float(slope / (std_err + 1e-10)),
                    p_value=float(p_value),
                    effect_size=float(r_squared),
                    confidence_interval=(
                        float(slope - 1.96 * std_err),
                        float(slope + 1.96 * std_err)
                    ),
                    degrees_of_freedom=int(len(values) - 2),
                    statistical_significance=p_value < self.alpha,
                    practical_significance=r_squared > 0.1,  # 10% variance explained
                    interpretation=interpretation
                )
                tests.append(trend_test)
        
        return tests
    
    def _conduct_power_analysis(self,
                              experimental_data: List[Dict[str, Any]],
                              control_data: Optional[List[Dict[str, Any]]]) -> List[PowerAnalysis]:
        """Conduct statistical power analysis"""
        
        power_analyses = []
        
        exp_metrics = self._extract_metrics_for_testing(experimental_data)
        
        for metric_name, exp_values in exp_metrics.items():
            if len(exp_values) < 3:
                continue
            
            # Estimate effect size and power
            if control_data:
                ctrl_metrics = self._extract_metrics_for_testing(control_data)
                if metric_name in ctrl_metrics:
                    ctrl_values = ctrl_metrics[metric_name]
                    
                    # Calculate observed effect size
                    pooled_std = np.sqrt(((len(exp_values) - 1) * np.var(exp_values, ddof=1) + 
                                         (len(ctrl_values) - 1) * np.var(ctrl_values, ddof=1)) / 
                                        (len(exp_values) + len(ctrl_values) - 2))
                    
                    observed_effect_size = abs(np.mean(exp_values) - np.mean(ctrl_values)) / (pooled_std + 1e-10)
                    
                    current_n = len(exp_values) + len(ctrl_values)
                    
                    # Estimate power using approximation
                    # For t-test: power ≈ 1 - β where β depends on effect size and sample size
                    delta = observed_effect_size * np.sqrt(current_n / 2)
                    estimated_power = float(1 - stats.norm.cdf(stats.norm.ppf(1 - self.alpha/2) - delta) - 
                                          stats.norm.cdf(stats.norm.ppf(self.alpha/2) - delta))
                    
                    # Calculate required sample size for desired power
                    # Simplified calculation
                    if observed_effect_size > 0:
                        required_n = int(2 * ((stats.norm.ppf(1 - self.alpha/2) + stats.norm.ppf(self.power_threshold)) / observed_effect_size) ** 2)
                    else:
                        required_n = current_n
                    
                    # Minimum detectable effect with current sample size
                    min_detectable_effect = (stats.norm.ppf(1 - self.alpha/2) + stats.norm.ppf(self.power_threshold)) / np.sqrt(current_n / 2)
                    
                    power_analysis = PowerAnalysis(
                        effect_size=observed_effect_size,
                        sample_size=current_n,
                        alpha=self.alpha,
                        power=estimated_power,
                        required_sample_size=required_n,
                        minimum_detectable_effect=float(min_detectable_effect)
                    )
                    
                    power_analyses.append(power_analysis)
        
        return power_analyses
    
    def _test_reproducibility(self, data: List[Dict[str, Any]]) -> List[ReproducibilityTest]:
        """Test reproducibility by comparing subsets of results"""
        
        reproducibility_tests = []
        
        if len(data) < 6:  # Need minimum data for split
            return reproducibility_tests
        
        metrics = self._extract_metrics_for_testing(data)
        
        for metric_name, values in metrics.items():
            if len(values) >= 6:
                # Split into two halves
                mid = len(values) // 2
                first_half = values[:mid]
                second_half = values[mid:mid+len(first_half)]  # Ensure equal lengths
                
                # Compute reproducibility metrics
                correlation = np.corrcoef(first_half, second_half)[0, 1] if not np.isnan(np.corrcoef(first_half, second_half)[0, 1]) else 0.0
                mae = np.mean(np.abs(np.array(first_half) - np.array(second_half)))
                relative_error = mae / (np.mean(np.abs(values)) + 1e-10)
                
                # Reproducibility score (combination of correlation and low error)
                reproducibility_score = correlation * (1 - min(relative_error, 1.0))
                
                # Threshold for reproducibility
                is_reproducible = correlation > 0.7 and relative_error < 0.2
                
                reproducibility_test = ReproducibilityTest(
                    test_type=f"Split_Half_Reproducibility_{metric_name}",
                    original_results=first_half,
                    replicated_results=second_half,
                    correlation=float(correlation),
                    mean_absolute_error=float(mae),
                    relative_error=float(relative_error),
                    reproducibility_score=float(reproducibility_score),
                    is_reproducible=is_reproducible
                )
                
                reproducibility_tests.append(reproducibility_test)
        
        return reproducibility_tests
    
    def _apply_multiple_comparison_corrections(self, tests: List[HypothesisTest]) -> Dict[str, Any]:
        """Apply multiple comparison corrections"""
        
        p_values = [test.p_value for test in tests]
        
        corrections = {}
        
        # Bonferroni correction
        bonferroni_alpha = self.alpha / len(p_values) if p_values else self.alpha
        bonferroni_significant = [p <= bonferroni_alpha for p in p_values]
        
        corrections['bonferroni'] = {
            'corrected_alpha': bonferroni_alpha,
            'significant_tests': sum(bonferroni_significant),
            'significant_indices': [i for i, sig in enumerate(bonferroni_significant) if sig]
        }
        
        # False Discovery Rate (Benjamini-Hochberg)
        if p_values:
            sorted_indices = np.argsort(p_values)
            sorted_p_values = np.array(p_values)[sorted_indices]
            
            # BH critical values
            bh_critical = [(i + 1) / len(p_values) * self.alpha for i in range(len(p_values))]
            
            # Find largest k such that P(k) <= (k/m) * alpha
            significant_bh = []
            for i in range(len(sorted_p_values) - 1, -1, -1):
                if sorted_p_values[i] <= bh_critical[i]:
                    significant_bh = list(range(i + 1))
                    break
            
            bh_significant_original = [sorted_indices[i] for i in significant_bh]
            
            corrections['benjamini_hochberg'] = {
                'significant_tests': len(significant_bh),
                'significant_indices': bh_significant_original,
                'critical_values': bh_critical
            }
        
        corrections['summary'] = {
            'total_tests': len(tests),
            'uncorrected_significant': sum(1 for p in p_values if p < self.alpha),
            'bonferroni_significant': corrections['bonferroni']['significant_tests'],
            'bh_significant': corrections.get('benjamini_hochberg', {}).get('significant_tests', 0)
        }
        
        return corrections
    
    def _compute_bootstrap_intervals(self,
                                   experimental_data: List[Dict[str, Any]],
                                   control_data: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Compute bootstrap confidence intervals"""
        
        bootstrap_results = {}
        
        exp_metrics = self._extract_metrics_for_testing(experimental_data)
        
        for metric_name, values in exp_metrics.items():
            if len(values) >= 10:  # Need sufficient data for bootstrap
                
                # Bootstrap mean
                bootstrap_means = []
                for _ in range(self.n_bootstrap_samples):
                    bootstrap_sample = np.random.choice(values, len(values), replace=True)
                    bootstrap_means.append(np.mean(bootstrap_sample))
                
                # Confidence intervals
                ci_lower = np.percentile(bootstrap_means, 2.5)
                ci_upper = np.percentile(bootstrap_means, 97.5)
                
                bootstrap_results[metric_name] = {
                    'mean': np.mean(values),
                    'bootstrap_mean': np.mean(bootstrap_means),
                    'bootstrap_std': np.std(bootstrap_means),
                    'confidence_interval': (float(ci_lower), float(ci_upper)),
                    'n_bootstrap_samples': self.n_bootstrap_samples
                }
        
        return bootstrap_results
    
    def _perform_cross_validation_assessment(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform cross-validation statistical assessment"""
        
        if len(data) < 10:
            return {"insufficient_data": "Need at least 10 samples for cross-validation"}
        
        # Simple k-fold approach for metric stability assessment
        k = 5
        fold_size = len(data) // k
        
        cv_results = {}
        metrics = self._extract_metrics_for_testing(data)
        
        for metric_name, values in metrics.items():
            fold_means = []
            fold_stds = []
            
            for fold in range(k):
                start_idx = fold * fold_size
                end_idx = start_idx + fold_size if fold < k - 1 else len(values)
                
                fold_data = values[start_idx:end_idx]
                if fold_data:
                    fold_means.append(np.mean(fold_data))
                    fold_stds.append(np.std(fold_data))
            
            if fold_means:
                cv_results[metric_name] = {
                    'fold_means': fold_means,
                    'fold_stds': fold_stds,
                    'overall_mean': np.mean(fold_means),
                    'between_fold_variance': np.var(fold_means),
                    'stability_score': 1.0 / (1.0 + np.var(fold_means))  # Higher = more stable
                }
        
        return cv_results
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _interpret_effect_size_r(self, r: float) -> str:
        """Interpret r effect size"""
        if r < 0.1:
            return "negligible"
        elif r < 0.3:
            return "small"
        elif r < 0.5:
            return "medium"
        else:
            return "large"
    
    def _generate_statistical_conclusions(self,
                                        hypothesis_tests: List[HypothesisTest],
                                        power_analyses: List[PowerAnalysis],
                                        reproducibility_tests: List[ReproducibilityTest],
                                        corrections: Dict[str, Any]) -> List[str]:
        """Generate statistical conclusions and recommendations"""
        
        conclusions = []
        
        # Significance summary
        total_tests = len(hypothesis_tests)
        significant_tests = sum(1 for test in hypothesis_tests if test.statistical_significance)
        practically_significant = sum(1 for test in hypothesis_tests if test.practical_significance)
        
        if total_tests > 0:
            conclusions.append(f"📊 Statistical Testing: {significant_tests}/{total_tests} tests significant, {practically_significant}/{total_tests} practically significant")
        
        # Effect sizes
        large_effects = sum(1 for test in hypothesis_tests if abs(test.effect_size) > 0.8)
        if large_effects > 0:
            conclusions.append(f"💪 Large Effect Sizes: {large_effects} tests showed large effect sizes (|d| > 0.8)")
        
        # Multiple comparisons
        if corrections['summary']['total_tests'] > 1:
            uncorrected = corrections['summary']['uncorrected_significant']
            bonferroni = corrections['summary']['bonferroni_significant']
            bh = corrections['summary']['bh_significant']
            
            conclusions.append(f"🔬 Multiple Comparison Corrections: {uncorrected} uncorrected → {bonferroni} Bonferroni → {bh} FDR significant")
        
        # Power analysis
        low_power_tests = sum(1 for analysis in power_analyses if analysis.power < self.power_threshold)
        if low_power_tests > 0:
            conclusions.append(f"⚡ Power Analysis: {low_power_tests} tests may be underpowered (power < {self.power_threshold})")
        
        # Reproducibility
        reproducible_tests = sum(1 for test in reproducibility_tests if test.is_reproducible)
        if reproducibility_tests:
            conclusions.append(f"🔄 Reproducibility: {reproducible_tests}/{len(reproducibility_tests)} metrics showed good reproducibility")
        
        # Overall assessment
        if significant_tests > total_tests * 0.5 and practically_significant > total_tests * 0.3:
            conclusions.append("✅ Strong statistical evidence supports research claims with meaningful effect sizes")
        elif significant_tests > total_tests * 0.3:
            conclusions.append("✅ Moderate statistical evidence supports research claims")
        else:
            conclusions.append("⚠️ Limited statistical evidence - consider larger sample sizes or effect size analysis")
        
        return conclusions
    
    def summary(self) -> Dict[str, Any]:
        """Get comprehensive validator summary"""
        return {
            "validator_type": "StatisticalValidator",
            "configuration": {
                "alpha": self.alpha,
                "power_threshold": self.power_threshold,
                "effect_size_threshold": self.effect_size_threshold,
                "n_bootstrap_samples": self.n_bootstrap_samples
            },
            "validation_statistics": self.validation_stats.copy(),
            "supported_tests": [
                "Welch's t-test",
                "Mann-Whitney U test", 
                "One-sample t-test",
                "Effect size testing",
                "Trend analysis",
                "Power analysis",
                "Reproducibility testing",
                "Bootstrap confidence intervals",
                "Multiple comparison corrections"
            ]
        }