"""Autonomous Hypothesis Validation Engine"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from abc import ABC, abstractmethod

from ..utils.error_handling import robust_execution, DiscoveryError
from ..utils.validation import ValidationMixin

logger = logging.getLogger(__name__)


class ValidationMethod(Enum):
    """Available validation methods"""
    STATISTICAL_TEST = "statistical_test"
    CROSS_VALIDATION = "cross_validation"
    BOOTSTRAP_SAMPLING = "bootstrap_sampling"
    BAYESIAN_ANALYSIS = "bayesian_analysis"
    SYNTHETIC_EXPERIMENT = "synthetic_experiment"
    ADVERSARIAL_TESTING = "adversarial_testing"


@dataclass
class ExperimentDesign:
    """Structured experimental design for hypothesis testing"""
    design_id: str
    hypothesis_id: str
    method: ValidationMethod
    parameters: Dict[str, Any]
    sample_size: int
    expected_power: float
    expected_effect_size: float
    control_conditions: List[str]
    experimental_conditions: List[str]
    confounding_variables: List[str] = field(default_factory=list)
    randomization_scheme: str = "simple"
    blinding_level: str = "none"  # none, single, double
    
    def get_statistical_power(self, alpha: float = 0.05) -> float:
        """Calculate expected statistical power"""
        # Simplified power calculation
        base_power = self.expected_power
        
        # Adjust for sample size
        size_factor = min(1.0, self.sample_size / 100.0)
        
        # Adjust for effect size
        effect_factor = min(1.0, self.expected_effect_size * 2.0)
        
        return base_power * size_factor * effect_factor


@dataclass
class ValidationResult:
    """Result from hypothesis validation experiment"""
    result_id: str
    hypothesis_id: str
    design_id: str
    method: ValidationMethod
    
    # Statistical results
    test_statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    statistical_power: float
    
    # Validation outcome
    hypothesis_supported: bool
    evidence_strength: float  # 0-1 scale
    uncertainty_level: float  # 0-1 scale
    
    # Meta-information
    sample_size_used: int
    execution_time: float
    computational_cost: float
    
    # Detailed results
    raw_results: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    assumptions_met: Dict[str, bool] = field(default_factory=dict)
    
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def get_quality_score(self) -> float:
        """Compute overall quality score for validation"""
        scores = [
            self.evidence_strength,
            1.0 - self.uncertainty_level,
            self.statistical_power,
            min(1.0, self.sample_size_used / 100.0),  # Sample size adequacy
            sum(self.assumptions_met.values()) / max(1, len(self.assumptions_met))  # Assumption satisfaction
        ]
        return np.mean(scores)


class ExperimentValidator(ABC):
    """Abstract base class for experiment validators"""
    
    @abstractmethod
    def validate_hypothesis(self,
                          hypothesis_claim: str,
                          data: np.ndarray,
                          design: ExperimentDesign) -> ValidationResult:
        """Validate hypothesis using specific method"""
        pass
    
    @abstractmethod
    def check_assumptions(self, data: np.ndarray, design: ExperimentDesign) -> Dict[str, bool]:
        """Check method-specific assumptions"""
        pass


class StatisticalTestValidator(ExperimentValidator, ValidationMixin):
    """Statistical hypothesis testing validator"""
    
    def validate_hypothesis(self,
                          hypothesis_claim: str,
                          data: np.ndarray,
                          design: ExperimentDesign) -> ValidationResult:
        """Validate using statistical tests"""
        
        start_time = datetime.now()
        
        # Determine appropriate statistical test
        test_type = self._determine_test_type(hypothesis_claim, data, design)
        
        # Perform the test
        test_result = self._perform_statistical_test(data, test_type, design)
        
        # Check assumptions
        assumptions_met = self.check_assumptions(data, design)
        
        # Calculate effect size
        effect_size = self._calculate_effect_size(data, test_type)
        
        # Estimate statistical power
        statistical_power = self._estimate_statistical_power(data, effect_size, design)
        
        # Determine support for hypothesis
        alpha = design.parameters.get('alpha', 0.05)
        hypothesis_supported = test_result['p_value'] < alpha
        
        # Calculate evidence strength
        evidence_strength = self._calculate_evidence_strength(
            test_result['p_value'], effect_size, statistical_power
        )
        
        # Calculate uncertainty
        uncertainty_level = self._calculate_uncertainty_level(
            test_result['p_value'], statistical_power, assumptions_met
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return ValidationResult(
            result_id=f"stat_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            hypothesis_id=design.hypothesis_id,
            design_id=design.design_id,
            method=ValidationMethod.STATISTICAL_TEST,
            test_statistic=test_result['statistic'],
            p_value=test_result['p_value'],
            confidence_interval=test_result['confidence_interval'],
            effect_size=effect_size,
            statistical_power=statistical_power,
            hypothesis_supported=hypothesis_supported,
            evidence_strength=evidence_strength,
            uncertainty_level=uncertainty_level,
            sample_size_used=len(data),
            execution_time=execution_time,
            computational_cost=execution_time * 0.1,  # Simple cost estimate
            raw_results=test_result,
            assumptions_met=assumptions_met
        )
    
    def check_assumptions(self, data: np.ndarray, design: ExperimentDesign) -> Dict[str, bool]:
        """Check statistical test assumptions"""
        assumptions = {}
        
        if data.size == 0:
            return {'sufficient_data': False}
        
        flat_data = data.flatten()
        
        # Normality assumption (Shapiro-Wilk approximation)
        assumptions['normality'] = self._test_normality(flat_data)
        
        # Independence assumption (simplified check)
        assumptions['independence'] = self._test_independence(flat_data)
        
        # Homoscedasticity (equal variances)
        if data.ndim > 1 and data.shape[1] > 1:
            assumptions['homoscedasticity'] = self._test_homoscedasticity(data)
        else:
            assumptions['homoscedasticity'] = True
        
        # Sufficient sample size
        assumptions['sufficient_sample_size'] = len(flat_data) >= 30
        
        return assumptions
    
    def _determine_test_type(self, hypothesis_claim: str, data: np.ndarray, design: ExperimentDesign) -> str:
        """Determine appropriate statistical test"""
        
        claim_lower = hypothesis_claim.lower()
        
        # T-test indicators
        if any(word in claim_lower for word in ['mean', 'average', 'difference']):
            if data.ndim == 1 or data.shape[1] == 1:
                return 'one_sample_ttest'
            elif data.shape[1] == 2:
                return 'two_sample_ttest'
            else:
                return 'anova'
        
        # Correlation test indicators  
        elif any(word in claim_lower for word in ['correlation', 'relationship', 'association']):
            return 'correlation_test'
        
        # Chi-square indicators
        elif any(word in claim_lower for word in ['proportion', 'frequency', 'categorical']):
            return 'chi_square'
        
        # Regression indicators
        elif any(word in claim_lower for word in ['predict', 'model', 'regression']):
            return 'regression_test'
        
        # Default to t-test
        else:
            return 'one_sample_ttest'
    
    def _perform_statistical_test(self, data: np.ndarray, test_type: str, design: ExperimentDesign) -> Dict[str, Any]:
        """Perform the specified statistical test"""
        
        if data.size == 0:
            return {'statistic': 0.0, 'p_value': 1.0, 'confidence_interval': (0.0, 0.0)}
        
        if test_type == 'one_sample_ttest':
            return self._one_sample_ttest(data.flatten(), design)
        
        elif test_type == 'two_sample_ttest':
            if data.ndim == 1:
                # Split data in half for demonstration
                mid = len(data) // 2
                group1, group2 = data[:mid], data[mid:]
            else:
                group1, group2 = data[:, 0], data[:, 1]
            return self._two_sample_ttest(group1, group2, design)
        
        elif test_type == 'correlation_test':
            return self._correlation_test(data, design)
        
        elif test_type == 'anova':
            return self._anova_test(data, design)
        
        elif test_type == 'chi_square':
            return self._chi_square_test(data, design)
        
        elif test_type == 'regression_test':
            return self._regression_test(data, design)
        
        else:
            # Fallback to one-sample t-test
            return self._one_sample_ttest(data.flatten(), design)
    
    def _one_sample_ttest(self, data: np.ndarray, design: ExperimentDesign) -> Dict[str, Any]:
        """Perform one-sample t-test"""
        
        if len(data) < 2:
            return {'statistic': 0.0, 'p_value': 1.0, 'confidence_interval': (0.0, 0.0)}
        
        mu = design.parameters.get('null_mean', 0.0)
        n = len(data)
        
        # Calculate t-statistic
        sample_mean = np.mean(data)
        sample_std = np.std(data, ddof=1)
        
        if sample_std == 0:
            t_stat = np.inf if sample_mean != mu else 0.0
            p_value = 0.0 if sample_mean != mu else 1.0
        else:
            se = sample_std / np.sqrt(n)
            t_stat = (sample_mean - mu) / se
            
            # Approximate p-value using normal distribution (for large n)
            if n > 30:
                p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))
            else:
                # Rough approximation for small samples
                p_value = 2 * (1 - self._t_cdf(abs(t_stat), n - 1))
        
        # Confidence interval
        alpha = design.parameters.get('alpha', 0.05)
        t_critical = self._t_critical(alpha / 2, n - 1)
        margin_of_error = t_critical * sample_std / np.sqrt(n) if sample_std > 0 else 0
        
        ci_lower = sample_mean - margin_of_error
        ci_upper = sample_mean + margin_of_error
        
        return {
            'statistic': t_stat,
            'p_value': p_value,
            'confidence_interval': (ci_lower, ci_upper),
            'sample_mean': sample_mean,
            'null_mean': mu,
            'degrees_of_freedom': n - 1
        }
    
    def _two_sample_ttest(self, group1: np.ndarray, group2: np.ndarray, design: ExperimentDesign) -> Dict[str, Any]:
        """Perform two-sample t-test"""
        
        if len(group1) < 2 or len(group2) < 2:
            return {'statistic': 0.0, 'p_value': 1.0, 'confidence_interval': (0.0, 0.0)}
        
        # Welch's t-test (unequal variances)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        n1, n2 = len(group1), len(group2)
        
        # Handle zero variance case
        if var1 == 0 and var2 == 0:
            t_stat = np.inf if mean1 != mean2 else 0.0
            p_value = 0.0 if mean1 != mean2 else 1.0
        else:
            se_diff = np.sqrt(var1/n1 + var2/n2)
            t_stat = (mean1 - mean2) / se_diff if se_diff > 0 else 0.0
            
            # Welch-Satterthwaite degrees of freedom
            if var1 > 0 and var2 > 0:
                dof = ((var1/n1 + var2/n2)**2) / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
            else:
                dof = min(n1-1, n2-1)
            
            # Approximate p-value
            p_value = 2 * (1 - self._t_cdf(abs(t_stat), dof))
        
        # Confidence interval for difference in means
        alpha = design.parameters.get('alpha', 0.05)
        t_critical = self._t_critical(alpha / 2, dof)
        margin_of_error = t_critical * se_diff if var1 > 0 or var2 > 0 else 0
        
        mean_diff = mean1 - mean2
        ci_lower = mean_diff - margin_of_error
        ci_upper = mean_diff + margin_of_error
        
        return {
            'statistic': t_stat,
            'p_value': p_value,
            'confidence_interval': (ci_lower, ci_upper),
            'mean_difference': mean_diff,
            'degrees_of_freedom': dof
        }
    
    def _correlation_test(self, data: np.ndarray, design: ExperimentDesign) -> Dict[str, Any]:
        """Perform correlation test"""
        
        if data.ndim == 1 or data.shape[1] < 2:
            return {'statistic': 0.0, 'p_value': 1.0, 'confidence_interval': (0.0, 0.0)}
        
        # Pearson correlation between first two columns
        x, y = data[:, 0], data[:, 1]
        n = len(x)
        
        if n < 3:
            return {'statistic': 0.0, 'p_value': 1.0, 'confidence_interval': (0.0, 0.0)}
        
        # Calculate correlation coefficient
        r = np.corrcoef(x, y)[0, 1]
        
        if np.isnan(r):
            r = 0.0
        
        # Test statistic for correlation
        if abs(r) == 1.0:
            t_stat = np.inf
            p_value = 0.0
        else:
            t_stat = r * np.sqrt((n - 2) / (1 - r**2))
            p_value = 2 * (1 - self._t_cdf(abs(t_stat), n - 2))
        
        # Fisher's z-transformation for confidence interval
        alpha = design.parameters.get('alpha', 0.05)
        z_critical = self._normal_critical(alpha / 2)
        
        if abs(r) < 0.999:  # Avoid division by zero
            z_r = 0.5 * np.log((1 + r) / (1 - r))  # Fisher's z
            se_z = 1 / np.sqrt(n - 3)
            
            z_lower = z_r - z_critical * se_z
            z_upper = z_r + z_critical * se_z
            
            # Transform back to correlation scale
            r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
            r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        else:
            r_lower, r_upper = r, r
        
        return {
            'statistic': t_stat,
            'p_value': p_value,
            'confidence_interval': (r_lower, r_upper),
            'correlation_coefficient': r,
            'degrees_of_freedom': n - 2
        }
    
    def _anova_test(self, data: np.ndarray, design: ExperimentDesign) -> Dict[str, Any]:
        """Perform one-way ANOVA"""
        
        if data.ndim == 1 or data.shape[1] < 3:
            return {'statistic': 0.0, 'p_value': 1.0, 'confidence_interval': (0.0, 0.0)}
        
        # Treat each column as a group
        groups = [data[:, i] for i in range(data.shape[1])]
        k = len(groups)  # number of groups
        n_total = sum(len(group) for group in groups)
        
        if n_total <= k:
            return {'statistic': 0.0, 'p_value': 1.0, 'confidence_interval': (0.0, 0.0)}
        
        # Calculate group means and overall mean
        group_means = [np.mean(group) for group in groups]
        overall_mean = np.mean(np.concatenate(groups))
        
        # Between-group sum of squares (SSB)
        ssb = sum(len(groups[i]) * (group_means[i] - overall_mean)**2 for i in range(k))
        
        # Within-group sum of squares (SSW)
        ssw = sum(sum((x - group_means[i])**2 for x in groups[i]) for i in range(k))
        
        # Degrees of freedom
        df_between = k - 1
        df_within = n_total - k
        
        # Mean squares
        msb = ssb / df_between if df_between > 0 else 0
        msw = ssw / df_within if df_within > 0 else 0
        
        # F-statistic
        f_stat = msb / msw if msw > 0 else 0
        
        # Approximate p-value (simplified)
        p_value = 1 - self._f_cdf(f_stat, df_between, df_within) if f_stat > 0 else 1.0
        
        return {
            'statistic': f_stat,
            'p_value': p_value,
            'confidence_interval': (0.0, f_stat),  # Simplified CI
            'f_statistic': f_stat,
            'df_between': df_between,
            'df_within': df_within,
            'sum_squares_between': ssb,
            'sum_squares_within': ssw
        }
    
    def _chi_square_test(self, data: np.ndarray, design: ExperimentDesign) -> Dict[str, Any]:
        """Perform chi-square goodness of fit test"""
        
        flat_data = data.flatten()
        
        if len(flat_data) < 5:
            return {'statistic': 0.0, 'p_value': 1.0, 'confidence_interval': (0.0, 0.0)}
        
        # Create frequency table (simple binning)
        bins = min(10, len(np.unique(flat_data)))
        observed, bin_edges = np.histogram(flat_data, bins=bins)
        
        # Expected frequencies (uniform distribution)
        expected = np.full_like(observed, len(flat_data) / bins, dtype=float)
        
        # Chi-square statistic
        chi2_stat = np.sum((observed - expected)**2 / expected)
        
        # Degrees of freedom
        df = bins - 1
        
        # Approximate p-value
        p_value = 1 - self._chi2_cdf(chi2_stat, df)
        
        return {
            'statistic': chi2_stat,
            'p_value': p_value,
            'confidence_interval': (0.0, chi2_stat),
            'degrees_of_freedom': df,
            'observed_frequencies': observed.tolist(),
            'expected_frequencies': expected.tolist()
        }
    
    def _regression_test(self, data: np.ndarray, design: ExperimentDesign) -> Dict[str, Any]:
        """Perform regression significance test"""
        
        if data.ndim == 1 or data.shape[1] < 2:
            return {'statistic': 0.0, 'p_value': 1.0, 'confidence_interval': (0.0, 0.0)}
        
        # Simple linear regression: y ~ x
        x, y = data[:, 0], data[:, 1]
        n = len(x)
        
        if n < 3:
            return {'statistic': 0.0, 'p_value': 1.0, 'confidence_interval': (0.0, 0.0)}
        
        # Calculate regression coefficients
        x_mean, y_mean = np.mean(x), np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean)**2)
        
        if denominator == 0:
            return {'statistic': 0.0, 'p_value': 1.0, 'confidence_interval': (0.0, 0.0)}
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # Predicted values and residuals
        y_pred = slope * x + intercept
        residuals = y - y_pred
        
        # Standard error of slope
        sse = np.sum(residuals**2)
        mse = sse / (n - 2)
        se_slope = np.sqrt(mse / denominator)
        
        # t-test for slope
        t_stat = slope / se_slope if se_slope > 0 else 0
        p_value = 2 * (1 - self._t_cdf(abs(t_stat), n - 2))
        
        # Confidence interval for slope
        alpha = design.parameters.get('alpha', 0.05)
        t_critical = self._t_critical(alpha / 2, n - 2)
        margin_of_error = t_critical * se_slope
        
        ci_lower = slope - margin_of_error
        ci_upper = slope + margin_of_error
        
        # R-squared
        ss_tot = np.sum((y - y_mean)**2)
        r_squared = 1 - (sse / ss_tot) if ss_tot > 0 else 0
        
        return {
            'statistic': t_stat,
            'p_value': p_value,
            'confidence_interval': (ci_lower, ci_upper),
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'degrees_of_freedom': n - 2
        }
    
    def _calculate_effect_size(self, data: np.ndarray, test_type: str) -> float:
        """Calculate effect size for the test"""
        
        if data.size == 0:
            return 0.0
        
        flat_data = data.flatten()
        
        if test_type in ['one_sample_ttest', 'two_sample_ttest']:
            # Cohen's d
            if test_type == 'one_sample_ttest':
                # Effect size = (mean - mu) / std
                mean_diff = abs(np.mean(flat_data))
                std_dev = np.std(flat_data, ddof=1)
                return mean_diff / std_dev if std_dev > 0 else 0.0
            else:
                # Two groups: split data
                mid = len(flat_data) // 2
                group1, group2 = flat_data[:mid], flat_data[mid:]
                
                mean_diff = abs(np.mean(group1) - np.mean(group2))
                pooled_std = np.sqrt((np.var(group1, ddof=1) + np.var(group2, ddof=1)) / 2)
                return mean_diff / pooled_std if pooled_std > 0 else 0.0
        
        elif test_type == 'correlation_test':
            # Correlation coefficient as effect size
            if data.ndim > 1 and data.shape[1] >= 2:
                return abs(np.corrcoef(data[:, 0], data[:, 1])[0, 1])
            return 0.0
        
        else:
            # Generic effect size based on coefficient of variation
            mean_val = np.mean(flat_data)
            std_val = np.std(flat_data)
            return std_val / abs(mean_val) if mean_val != 0 else 0.0
    
    def _estimate_statistical_power(self, data: np.ndarray, effect_size: float, design: ExperimentDesign) -> float:
        """Estimate statistical power"""
        
        n = len(data.flatten())
        alpha = design.parameters.get('alpha', 0.05)
        
        # Simplified power calculation
        # Power increases with sample size and effect size, decreases with alpha
        base_power = min(0.95, effect_size * np.sqrt(n) * 0.1)
        
        # Adjust for alpha level
        alpha_adjustment = (0.05 / alpha) if alpha > 0 else 1.0
        
        power = base_power * alpha_adjustment
        return min(0.99, max(0.05, power))
    
    def _calculate_evidence_strength(self, p_value: float, effect_size: float, power: float) -> float:
        """Calculate overall evidence strength"""
        
        # Combine p-value, effect size, and power
        p_component = max(0, 1 - p_value * 10)  # Strong evidence when p < 0.1
        effect_component = min(1, effect_size)
        power_component = power
        
        return np.mean([p_component, effect_component, power_component])
    
    def _calculate_uncertainty_level(self, p_value: float, power: float, assumptions_met: Dict[str, bool]) -> float:
        """Calculate uncertainty level"""
        
        # Higher uncertainty when:
        # - p-value is close to threshold
        # - low statistical power
        # - assumptions violated
        
        p_uncertainty = abs(p_value - 0.05) if p_value < 0.1 else 0.0
        power_uncertainty = 1 - power
        assumption_violations = 1 - (sum(assumptions_met.values()) / max(1, len(assumptions_met)))
        
        return np.mean([p_uncertainty, power_uncertainty, assumption_violations])
    
    def _test_normality(self, data: np.ndarray) -> bool:
        """Simple normality test"""
        
        if len(data) < 3:
            return False
        
        # Check skewness and kurtosis
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        if std_val == 0:
            return True  # Constant data is "normal"
        
        # Simplified skewness test
        skewness = np.mean(((data - mean_val) / std_val) ** 3)
        
        # Simplified kurtosis test  
        kurtosis = np.mean(((data - mean_val) / std_val) ** 4) - 3
        
        # Rough normality check
        return abs(skewness) < 1.0 and abs(kurtosis) < 1.0
    
    def _test_independence(self, data: np.ndarray) -> bool:
        """Simple independence test"""
        
        if len(data) < 10:
            return True  # Assume independence for small samples
        
        # Durbin-Watson approximation: check for autocorrelation
        diffs = np.diff(data)
        dw_stat = np.sum(diffs**2) / np.sum(data[1:]**2) if np.sum(data[1:]**2) > 0 else 2.0
        
        # DW stat around 2 indicates no autocorrelation
        return 1.5 < dw_stat < 2.5
    
    def _test_homoscedasticity(self, data: np.ndarray) -> bool:
        """Simple homoscedasticity test"""
        
        if data.shape[1] < 2:
            return True
        
        # Compare variances across groups
        variances = [np.var(data[:, i]) for i in range(data.shape[1])]
        max_var = max(variances)
        min_var = min(variances)
        
        # Levene's test approximation
        return max_var / min_var < 4.0 if min_var > 0 else True
    
    # Statistical distribution functions (simplified approximations)
    
    def _normal_cdf(self, x: float) -> float:
        """Approximate standard normal CDF"""
        return 0.5 * (1 + self._erf(x / np.sqrt(2)))
    
    def _erf(self, x: float) -> float:
        """Approximate error function"""
        # Abramowitz and Stegun approximation
        a1 =  0.254829592
        a2 = -0.284496736
        a3 =  1.421413741
        a4 = -1.453152027
        a5 =  1.061405429
        p  =  0.3275911
        
        sign = 1 if x >= 0 else -1
        x = abs(x)
        
        # A&S formula 7.1.26
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
        
        return sign * y
    
    def _t_cdf(self, t: float, df: int) -> float:
        """Approximate t-distribution CDF"""
        # For large df, t-distribution approaches normal
        if df > 30:
            return self._normal_cdf(t)
        
        # Simplified approximation for small df
        return 0.5 + 0.5 * np.tanh(t / 2)
    
    def _t_critical(self, alpha: float, df: int) -> float:
        """Approximate t-critical value"""
        # Common critical values
        if df > 30:
            if alpha <= 0.001:
                return 3.29
            elif alpha <= 0.01:
                return 2.58
            elif alpha <= 0.025:
                return 1.96
            elif alpha <= 0.05:
                return 1.64
            else:
                return 1.0
        
        # Rough adjustment for small df
        adjustment = 1 + 2.0 / df
        return self._t_critical(alpha, 1000) * adjustment
    
    def _normal_critical(self, alpha: float) -> float:
        """Normal distribution critical value"""
        return self._t_critical(alpha, 1000)
    
    def _chi2_cdf(self, chi2: float, df: int) -> float:
        """Approximate chi-square CDF"""
        if df <= 0:
            return 0.0
        
        # Rough approximation
        return min(1.0, chi2 / (2 * df))
    
    def _f_cdf(self, f: float, df1: int, df2: int) -> float:
        """Approximate F-distribution CDF"""
        if f <= 0:
            return 0.0
        
        # Very rough approximation
        return min(1.0, f / (1 + f))


class AutonomousHypothesisValidator(ValidationMixin):
    """Main autonomous hypothesis validation engine"""
    
    def __init__(self,
                 default_alpha: float = 0.05,
                 min_effect_size: float = 0.2,
                 min_power: float = 0.8):
        """
        Initialize autonomous validator
        
        Args:
            default_alpha: Default significance level
            min_effect_size: Minimum meaningful effect size
            min_power: Minimum acceptable statistical power
        """
        self.default_alpha = default_alpha
        self.min_effect_size = min_effect_size
        self.min_power = min_power
        
        # Initialize validators
        self.validators = {
            ValidationMethod.STATISTICAL_TEST: StatisticalTestValidator(),
            # Additional validators would be implemented here
        }
        
        self.validation_history: List[ValidationResult] = []
        
        logger.info("AutonomousHypothesisValidator initialized")
    
    @robust_execution(recovery_strategy='partial_recovery')
    def design_experiment(self,
                         hypothesis_claim: str,
                         available_data: np.ndarray,
                         domain: str = "general",
                         constraints: Optional[Dict[str, Any]] = None) -> ExperimentDesign:
        """
        Automatically design experiment for hypothesis validation
        
        Args:
            hypothesis_claim: The hypothesis to test
            available_data: Available data for the experiment
            domain: Scientific domain
            constraints: Any experimental constraints
            
        Returns:
            ExperimentDesign optimized for the hypothesis
        """
        
        logger.info(f"Designing experiment for hypothesis: {hypothesis_claim[:100]}...")
        
        if constraints is None:
            constraints = {}
        
        # Determine validation method
        method = self._select_validation_method(hypothesis_claim, available_data)
        
        # Estimate required sample size
        sample_size = self._estimate_sample_size(hypothesis_claim, available_data, method)
        
        # Determine experimental conditions
        conditions = self._determine_conditions(hypothesis_claim, domain)
        
        # Identify potential confounders
        confounders = self._identify_confounders(hypothesis_claim, domain)
        
        # Set up parameters
        parameters = {
            'alpha': constraints.get('alpha', self.default_alpha),
            'power': constraints.get('power', self.min_power),
            'effect_size': constraints.get('effect_size', self.min_effect_size)
        }
        
        # Estimate expected effect size
        expected_effect = self._estimate_expected_effect_size(hypothesis_claim, available_data)
        
        design = ExperimentDesign(
            design_id=f"exp_design_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            hypothesis_id=f"hyp_{hash(hypothesis_claim) % 10000}",
            method=method,
            parameters=parameters,
            sample_size=sample_size,
            expected_power=self.min_power,
            expected_effect_size=expected_effect,
            control_conditions=conditions['control'],
            experimental_conditions=conditions['experimental'],
            confounding_variables=confounders
        )
        
        logger.info(f"Designed {method.value} experiment with sample size {sample_size}")
        return design
    
    def validate_hypothesis(self,
                           hypothesis_claim: str,
                           data: np.ndarray,
                           design: Optional[ExperimentDesign] = None,
                           method: Optional[ValidationMethod] = None) -> ValidationResult:
        """
        Validate hypothesis using the specified or designed experiment
        
        Args:
            hypothesis_claim: Hypothesis to validate
            data: Data for validation
            design: Pre-designed experiment (optional)
            method: Specific validation method (optional)
            
        Returns:
            ValidationResult with comprehensive validation outcome
        """
        
        logger.info(f"Validating hypothesis: {hypothesis_claim[:100]}...")
        
        # Design experiment if not provided
        if design is None:
            design = self.design_experiment(hypothesis_claim, data)
        
        # Override method if specified
        if method is not None:
            design.method = method
        
        # Get appropriate validator
        validator = self.validators.get(design.method)
        if validator is None:
            raise DiscoveryError(f"No validator available for method: {design.method}")
        
        # Perform validation
        result = validator.validate_hypothesis(hypothesis_claim, data, design)
        
        # Add to history
        self.validation_history.append(result)
        
        # Log result
        logger.info(f"Validation complete: {'SUPPORTED' if result.hypothesis_supported else 'NOT SUPPORTED'} "
                   f"(p={result.p_value:.4f}, effect={result.effect_size:.3f})")
        
        return result
    
    def batch_validate_hypotheses(self,
                                hypotheses: List[Tuple[str, np.ndarray]],
                                domain: str = "general") -> List[ValidationResult]:
        """
        Validate multiple hypotheses in batch
        
        Args:
            hypotheses: List of (hypothesis_claim, data) tuples
            domain: Scientific domain
            
        Returns:
            List of ValidationResult objects
        """
        
        logger.info(f"Batch validating {len(hypotheses)} hypotheses")
        
        results = []
        
        for i, (claim, data) in enumerate(hypotheses):
            try:
                logger.info(f"Validating hypothesis {i+1}/{len(hypotheses)}")
                result = self.validate_hypothesis(claim, data)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to validate hypothesis {i+1}: {e}")
                # Create failed result
                failed_result = ValidationResult(
                    result_id=f"failed_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    hypothesis_id=f"hyp_{i}",
                    design_id="failed",
                    method=ValidationMethod.STATISTICAL_TEST,
                    test_statistic=0.0,
                    p_value=1.0,
                    confidence_interval=(0.0, 0.0),
                    effect_size=0.0,
                    statistical_power=0.0,
                    hypothesis_supported=False,
                    evidence_strength=0.0,
                    uncertainty_level=1.0,
                    sample_size_used=0,
                    execution_time=0.0,
                    computational_cost=0.0
                )
                results.append(failed_result)
        
        # Summary statistics
        supported_count = sum(1 for r in results if r.hypothesis_supported)
        avg_p_value = np.mean([r.p_value for r in results])
        avg_effect_size = np.mean([r.effect_size for r in results])
        
        logger.info(f"Batch validation complete: {supported_count}/{len(hypotheses)} supported, "
                   f"avg p-value: {avg_p_value:.4f}, avg effect size: {avg_effect_size:.3f}")
        
        return results
    
    def _select_validation_method(self, hypothesis_claim: str, data: np.ndarray) -> ValidationMethod:
        """Select appropriate validation method based on hypothesis and data"""
        
        # For now, default to statistical testing
        # In a full implementation, this would analyze the hypothesis claim
        # and data characteristics to choose the best method
        
        return ValidationMethod.STATISTICAL_TEST
    
    def _estimate_sample_size(self, hypothesis_claim: str, available_data: np.ndarray, method: ValidationMethod) -> int:
        """Estimate required sample size for adequate power"""
        
        # Simple heuristic based on available data
        available_size = len(available_data.flatten())
        
        # Minimum sample sizes by method
        min_sizes = {
            ValidationMethod.STATISTICAL_TEST: 30,
            ValidationMethod.CROSS_VALIDATION: 50,
            ValidationMethod.BOOTSTRAP_SAMPLING: 100,
        }
        
        min_size = min_sizes.get(method, 30)
        
        # Use available data size, but ensure minimum
        return max(min_size, min(available_size, 1000))
    
    def _determine_conditions(self, hypothesis_claim: str, domain: str) -> Dict[str, List[str]]:
        """Determine experimental and control conditions"""
        
        # Simple heuristic based on claim analysis
        claim_lower = hypothesis_claim.lower()
        
        control = ["baseline", "no_treatment"]
        experimental = ["treatment", "intervention"]
        
        # Domain-specific adjustments
        if "drug" in claim_lower or "medication" in claim_lower:
            control.append("placebo")
            experimental.append("active_drug")
        
        elif "therapy" in claim_lower or "treatment" in claim_lower:
            control.append("standard_care")
            experimental.append("new_therapy")
        
        elif "algorithm" in claim_lower or "method" in claim_lower:
            control.append("baseline_method")
            experimental.append("new_algorithm")
        
        return {"control": control, "experimental": experimental}
    
    def _identify_confounders(self, hypothesis_claim: str, domain: str) -> List[str]:
        """Identify potential confounding variables"""
        
        # Common confounders by domain
        domain_confounders = {
            "medical": ["age", "sex", "weight", "comorbidities"],
            "psychology": ["age", "education", "socioeconomic_status"],
            "machine_learning": ["dataset_size", "feature_quality", "hyperparameters"],
            "physics": ["temperature", "pressure", "measurement_error"],
            "general": ["time", "measurement_error", "sampling_bias"]
        }
        
        return domain_confounders.get(domain, domain_confounders["general"])
    
    def _estimate_expected_effect_size(self, hypothesis_claim: str, data: np.ndarray) -> float:
        """Estimate expected effect size from preliminary data analysis"""
        
        if data.size == 0:
            return self.min_effect_size
        
        # Simple heuristic: use coefficient of variation as proxy for effect size
        flat_data = data.flatten()
        mean_val = np.mean(flat_data)
        std_val = np.std(flat_data)
        
        if mean_val == 0:
            return self.min_effect_size
        
        cv = std_val / abs(mean_val)
        
        # Map CV to effect size (rough heuristic)
        effect_size = min(1.0, cv * 2.0)
        
        return max(self.min_effect_size, effect_size)
    
    def generate_validation_report(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        
        if not results:
            return {"error": "No validation results provided"}
        
        # Summary statistics
        supported_count = sum(1 for r in results if r.hypothesis_supported)
        total_count = len(results)
        
        p_values = [r.p_value for r in results]
        effect_sizes = [r.effect_size for r in results]
        quality_scores = [r.get_quality_score() for r in results]
        
        report = {
            "summary": {
                "total_hypotheses": total_count,
                "supported_hypotheses": supported_count,
                "support_rate": supported_count / total_count,
                "average_p_value": np.mean(p_values),
                "median_p_value": np.median(p_values),
                "average_effect_size": np.mean(effect_sizes),
                "median_effect_size": np.median(effect_sizes),
                "average_quality_score": np.mean(quality_scores)
            },
            "distribution_analysis": {
                "p_value_distribution": {
                    "significant_001": sum(1 for p in p_values if p < 0.001),
                    "significant_01": sum(1 for p in p_values if p < 0.01),
                    "significant_05": sum(1 for p in p_values if p < 0.05),
                    "not_significant": sum(1 for p in p_values if p >= 0.05)
                },
                "effect_size_distribution": {
                    "small_effect": sum(1 for e in effect_sizes if e < 0.2),
                    "medium_effect": sum(1 for e in effect_sizes if 0.2 <= e < 0.5),
                    "large_effect": sum(1 for e in effect_sizes if e >= 0.5)
                }
            },
            "quality_assessment": {
                "high_quality": sum(1 for q in quality_scores if q > 0.8),
                "medium_quality": sum(1 for q in quality_scores if 0.5 <= q <= 0.8),
                "low_quality": sum(1 for q in quality_scores if q < 0.5)
            },
            "methodological_breakdown": {
                method.value: sum(1 for r in results if r.method == method)
                for method in ValidationMethod
            },
            "recommendations": self._generate_recommendations(results)
        }
        
        return report
    
    def _generate_recommendations(self, results: List[ValidationResult]) -> List[str]:
        """Generate recommendations based on validation results"""
        
        recommendations = []
        
        # Sample size recommendations
        small_samples = [r for r in results if r.sample_size_used < 50]
        if len(small_samples) > len(results) * 0.3:
            recommendations.append("Consider increasing sample sizes for more reliable results")
        
        # Power recommendations
        low_power = [r for r in results if r.statistical_power < 0.8]
        if len(low_power) > len(results) * 0.3:
            recommendations.append("Many tests have low statistical power; consider power analysis for future studies")
        
        # Effect size recommendations
        effect_sizes = [r.effect_size for r in results]
        if np.mean(effect_sizes) < 0.2:
            recommendations.append("Consider practical significance in addition to statistical significance")
        
        # Multiple comparisons
        if len(results) > 10:
            recommendations.append("Consider multiple comparison corrections for large-scale hypothesis testing")
        
        # Quality recommendations
        quality_scores = [r.get_quality_score() for r in results]
        if np.mean(quality_scores) < 0.6:
            recommendations.append("Focus on improving overall study design and execution quality")
        
        return recommendations