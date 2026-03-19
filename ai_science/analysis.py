"""
Result analysis: effect size, p-value, confidence interval, verdict.

Implements:
  - Cohen's d (pooled SD)
  - Two-sample Welch t-test (no scipy) using the Abramowitz & Stegun
    approximation for the t-distribution CDF
  - 95% confidence interval for the mean difference
  - Supports/refutes verdict with strength characterisation
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import List, Sequence

from .hypothesis import Hypothesis
from .experiment import ExperimentDesign


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """Raw numeric data from a completed experiment."""
    design_id: str
    control_values: List[float]
    treatment_values: List[float]
    metadata: dict = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Statistical analysis of an experiment result."""
    id: str
    experiment_result_id: str
    hypothesis_id: str
    cohens_d: float
    p_value: float
    ci_lower: float
    ci_upper: float
    verdict: str              # "supports" | "refutes" | "inconclusive"
    effect_strength: str      # "negligible" | "small" | "medium" | "large"
    summary: str

    def __str__(self) -> str:
        import math as _math
        d_str = (
            f"{self.cohens_d:+.3f}"
            if _math.isfinite(self.cohens_d)
            else ("+∞" if self.cohens_d > 0 else "-∞")
        )
        ci_str = (
            f"[{self.ci_lower:.3f}, {self.ci_upper:.3f}]"
            if _math.isfinite(self.ci_lower) and _math.isfinite(self.ci_upper)
            else "[n/a]"
        )
        return (
            f"AnalysisResult({self.id})\n"
            f"  Cohen's d : {d_str}  ({self.effect_strength})\n"
            f"  p-value   : {self.p_value:.4f}\n"
            f"  95% CI    : {ci_str}\n"
            f"  Verdict   : {self.verdict.upper()}\n"
            f"  Summary   : {self.summary}"
        )


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class ResultAnalyzer:
    """
    Analyzes experiment results against the originating hypothesis.

    All statistics are computed using stdlib only (math module).
    """

    ALPHA = 0.05

    def analyze(
        self,
        result: ExperimentResult,
        hypothesis: Hypothesis,
        design: ExperimentDesign,
    ) -> AnalysisResult:
        c = result.control_values
        t = result.treatment_values

        mean_diff = self._mean(t) - self._mean(c)
        d         = self._cohens_d(c, t)
        p         = self._welch_t_test(c, t)
        ci_lo, ci_hi = self._confidence_interval(c, t)
        strength  = self._effect_strength(d)
        verdict   = self._verdict(p, d, hypothesis, design)
        summary   = self._build_summary(hypothesis, mean_diff, d, p, ci_lo, ci_hi, verdict)

        return AnalysisResult(
            id=str(uuid.uuid4())[:8],
            experiment_result_id=result.design_id,
            hypothesis_id=hypothesis.id,
            cohens_d=d,
            p_value=p,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            verdict=verdict,
            effect_strength=strength,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _mean(xs: Sequence[float]) -> float:
        return sum(xs) / len(xs)

    @staticmethod
    def _variance(xs: Sequence[float]) -> float:
        m = sum(xs) / len(xs)
        return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)

    def _std(self, xs: Sequence[float]) -> float:
        return math.sqrt(self._variance(xs))

    def _cohens_d(self, c: Sequence[float], t: Sequence[float]) -> float:
        """Pooled-SD Cohen's d."""
        nc, nt = len(c), len(t)
        sc, st = self._std(c), self._std(t)
        pooled_sd = math.sqrt(((nc - 1) * sc**2 + (nt - 1) * st**2) / (nc + nt - 2))
        diff = self._mean(t) - self._mean(c)
        if pooled_sd == 0:
            # Zero within-group variance: infinite effect if means differ, else 0
            return math.copysign(math.inf, diff) if diff != 0 else 0.0
        return diff / pooled_sd

    def _welch_df(self, vc: float, nc: int, vt: float, nt: int) -> float:
        """Welch–Satterthwaite degrees of freedom, guarded against zero variance."""
        term_c = vc / nc
        term_t = vt / nt
        numerator = (term_c + term_t) ** 2
        denom = term_c ** 2 / (nc - 1) + term_t ** 2 / (nt - 1)
        if denom == 0:
            # Both groups have zero variance → degenerate, use large df
            return float(nc + nt - 2)
        return numerator / denom

    def _welch_t_test(self, c: Sequence[float], t: Sequence[float]) -> float:
        """Two-sided Welch t-test; returns p-value."""
        nc, nt = len(c), len(t)
        vc, vt = self._variance(c), self._variance(t)
        se = math.sqrt(vc / nc + vt / nt)
        if se == 0:
            # Zero variance in both groups — if means differ, p→0; else p=1
            return 0.0 if self._mean(t) != self._mean(c) else 1.0
        t_stat = (self._mean(t) - self._mean(c)) / se
        df = self._welch_df(vc, nc, vt, nt)
        p = 2 * self._t_cdf_upper(abs(t_stat), df)
        return min(1.0, max(0.0, p))

    def _confidence_interval(
        self, c: Sequence[float], t: Sequence[float], alpha: float = 0.05
    ) -> tuple[float, float]:
        """95% CI for mean difference (treatment − control) using Welch's SE."""
        nc, nt = len(c), len(t)
        vc, vt = self._variance(c), self._variance(t)
        se = math.sqrt(vc / nc + vt / nt)
        diff = self._mean(t) - self._mean(c)
        if se == 0:
            return diff, diff
        df = self._welch_df(vc, nc, vt, nt)
        t_crit = self._t_inv(1 - alpha / 2, df)
        return diff - t_crit * se, diff + t_crit * se

    # ------------------------------------------------------------------
    # t-distribution helpers (no scipy)
    # ------------------------------------------------------------------

    @staticmethod
    def _t_cdf_upper(t: float, df: float) -> float:
        """
        Upper-tail probability P(T > t) for t-distribution with *df* degrees
        of freedom, via the regularized incomplete beta function.
        """
        x = df / (df + t * t)
        # regularized incomplete beta I_x(df/2, 1/2)
        # Using continued-fraction expansion (Lentz's method)
        a, b = df / 2, 0.5
        return 0.5 * ResultAnalyzer._betainc(x, a, b)

    @staticmethod
    def _betainc(x: float, a: float, b: float) -> float:
        """
        Regularized incomplete beta I_x(a,b) via continued fractions.
        Accurate for x in (0,1), a,b > 0.
        """
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        # Use the symmetry relation when x > (a+1)/(a+b+2)
        if x > (a + 1) / (a + b + 2):
            return 1.0 - ResultAnalyzer._betainc(1 - x, b, a)

        lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a

        # Lentz continued fraction
        TINY = 1e-30
        f = TINY
        C, D = f, 0.0
        for m in range(200):
            for step in range(2):
                if step == 0:
                    d = m * (b - m) * x / ((a + 2*m - 1) * (a + 2*m))
                else:
                    d = -(a + m) * (a + b + m) * x / ((a + 2*m) * (a + 2*m + 1))
                D = 1.0 + d * D
                if abs(D) < TINY:
                    D = TINY
                C = 1.0 + d / C
                if abs(C) < TINY:
                    C = TINY
                D = 1.0 / D
                delta = C * D
                f *= delta
                if abs(delta - 1.0) < 1e-10:
                    break
        return front * f

    @staticmethod
    def _t_inv(p: float, df: float) -> float:
        """
        Inverse t-distribution CDF (upper quantile) via bisection.
        Returns t such that P(T ≤ t) = p.
        """
        lo, hi = 0.0, 100.0
        for _ in range(60):
            mid = (lo + hi) / 2
            cdf = 1 - ResultAnalyzer._t_cdf_upper(mid, df)
            if cdf < p:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------

    @staticmethod
    def _effect_strength(d: float) -> str:
        ad = abs(d)
        if math.isinf(ad):
            return "large"
        if ad < 0.20:
            return "negligible"
        if ad < 0.50:
            return "small"
        if ad < 0.80:
            return "medium"
        return "large"

    def _verdict(
        self,
        p: float,
        d: float,
        h: Hypothesis,
        design: ExperimentDesign,
    ) -> str:
        significant = p < self.ALPHA
        adequate_effect = abs(d) >= 0.2
        if significant and adequate_effect:
            # Check sign matches causal direction (positive d → treatment > control)
            return "supports"
        if significant and not adequate_effect:
            return "inconclusive"
        return "refutes"

    @staticmethod
    def _build_summary(
        h: Hypothesis,
        mean_diff: float,
        d: float,
        p: float,
        ci_lo: float,
        ci_hi: float,
        verdict: str,
    ) -> str:
        sig = "significant" if p < 0.05 else "non-significant"
        direction = "increase" if mean_diff > 0 else "decrease"
        d_str  = f"{d:+.3f}" if math.isfinite(d)  else f"{'+' if d > 0 else '-'}∞"
        ci_str = (
            f"[{ci_lo:.3f},{ci_hi:.3f}]"
            if math.isfinite(ci_lo) and math.isfinite(ci_hi)
            else "[n/a]"
        )
        return (
            f"The experiment found a {sig} {direction} in the outcome "
            f"(Δmean={mean_diff:+.3f}, d={d_str}, p={p:.4f}, "
            f"95%CI={ci_str}). "
            f"This {verdict} the hypothesis: \"{h.statement}\""
        )
