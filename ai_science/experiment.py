"""
Experiment design from a hypothesis.

Power analysis uses the standard formula for two-sample t-test:
    n ≈ 2 * ((z_α/2 + z_β) / δ)²
where δ = expected effect size (Cohen's d), z_α/2 = 1.96, z_β = 0.84 (80% power).
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import List

from .hypothesis import Hypothesis


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ExperimentDesign:
    """A concrete experimental design derived from a hypothesis."""

    id: str
    hypothesis_id: str
    control_group: str
    treatment_group: str
    measurements: List[str]
    sample_size_per_group: int
    confounds_to_control: List[str]
    expected_effect_size: float   # Cohen's d
    alpha: float = 0.05
    power: float = 0.80

    def __str__(self) -> str:
        return (
            f"ExperimentDesign({self.id})\n"
            f"  Control:   {self.control_group}\n"
            f"  Treatment: {self.treatment_group}\n"
            f"  N/group:   {self.sample_size_per_group}\n"
            f"  Measurements: {', '.join(self.measurements)}\n"
            f"  Confounds:    {', '.join(self.confounds_to_control)}\n"
            f"  Expected d:   {self.expected_effect_size:.2f}"
        )


# ---------------------------------------------------------------------------
# Designer
# ---------------------------------------------------------------------------

class ExperimentDesigner:
    """
    Translates a Hypothesis into a concrete ExperimentDesign.

    Strategy:
      - Parses key variables from the hypothesis statement.
      - Selects measurements appropriate for the hypothesis template.
      - Estimates a plausible effect size based on confidence.
      - Computes required sample size via power analysis (two-sample t-test).
      - Suggests confounds based on the domain.
    """

    # Common confounds by scientific domain
    _DOMAIN_CONFOUNDS = {
        "biology":    ["age", "sex", "body_weight", "batch_effects", "circadian_rhythm"],
        "psychology": ["age", "gender", "socioeconomic_status", "prior_exposure", "experimenter_bias"],
        "physics":    ["temperature", "humidity", "instrument_calibration", "vibration"],
        "chemistry":  ["pH", "temperature", "purity", "reaction_time", "catalyst_concentration"],
        "general":    ["baseline_variability", "measurement_noise", "time_of_day", "operator_effect"],
    }

    def design(self, hypothesis: Hypothesis, alpha: float = 0.05, power: float = 0.80) -> ExperimentDesign:
        """Return an ExperimentDesign for *hypothesis*."""
        expected_d = self._estimate_effect_size(hypothesis.confidence)
        n = self._power_analysis(expected_d, alpha=alpha, power=power)
        confounds = self._select_confounds(hypothesis.domain)
        measurements = self._select_measurements(hypothesis)
        control, treatment = self._define_groups(hypothesis)

        return ExperimentDesign(
            id=str(uuid.uuid4())[:8],
            hypothesis_id=hypothesis.id,
            control_group=control,
            treatment_group=treatment,
            measurements=measurements,
            sample_size_per_group=n,
            confounds_to_control=confounds,
            expected_effect_size=expected_d,
            alpha=alpha,
            power=power,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _estimate_effect_size(self, confidence: float) -> float:
        """
        Higher hypothesis confidence → we expect a stronger signal.
        Map confidence [0,1] → Cohen's d in [0.2, 0.8].
        """
        return 0.2 + confidence * 0.6

    def _power_analysis(self, d: float, alpha: float = 0.05, power: float = 0.80) -> int:
        """
        Two-sample t-test power analysis.
        n = 2 * ((z_α/2 + z_β) / d)²
        Returns n per group (minimum 10).
        """
        if d <= 0:
            return 100  # fallback
        z_alpha_2 = self._inv_normal(1 - alpha / 2)
        z_beta    = self._inv_normal(power)
        n = 2 * ((z_alpha_2 + z_beta) / d) ** 2
        return max(10, math.ceil(n))

    @staticmethod
    def _inv_normal(p: float) -> float:
        """
        Approximation of the inverse normal CDF (Beasley-Springer-Moro).
        Good to ~4 decimal places for p in (0.001, 0.999).
        """
        # Rational approximation
        a = [0, -3.969683028665376e+01,  2.209460984245205e+02,
             -2.759285104469687e+02,  1.383577518672690e+02,
             -3.066479806614716e+01,  2.506628277459239e+00]
        b = [0, -5.447609879822406e+01,  1.615858368580409e+02,
             -1.556989798598866e+02,  6.680131188771972e+01,
             -1.328068155288572e+01]
        c = [0, -7.784894002430293e-03, -3.223964580411365e-01,
             -2.400758277161838e+00, -2.549732539343734e+00,
              4.374664141464968e+00,  2.938163982698783e+00]
        d_ = [0,  7.784695709041462e-03,  3.224671290700398e-01,
               2.445134137142996e+00,  3.754408661907416e+00]
        p_low, p_high = 0.02425, 1 - 0.02425
        if p_low <= p <= p_high:
            q = p - 0.5
            r = q * q
            return (q * (((((a[1]*r + a[2])*r + a[3])*r + a[4])*r + a[5])*r + a[6]) /
                    (((((b[1]*r + b[2])*r + b[3])*r + b[4])*r + b[5])*r + 1))
        elif p < p_low:
            q = math.sqrt(-2 * math.log(p))
            return (((((c[1]*q + c[2])*q + c[3])*q + c[4])*q + c[5])*q + c[6]) / \
                   ((((d_[1]*q + d_[2])*q + d_[3])*q + d_[4])*q + 1)
        else:
            q = math.sqrt(-2 * math.log(1 - p))
            return -(((((c[1]*q + c[2])*q + c[3])*q + c[4])*q + c[5])*q + c[6]) / \
                    ((((d_[1]*q + d_[2])*q + d_[3])*q + d_[4])*q + 1)

    def _select_confounds(self, domain: str) -> List[str]:
        domain_key = domain.lower() if domain.lower() in self._DOMAIN_CONFOUNDS else "general"
        pool = self._DOMAIN_CONFOUNDS[domain_key]
        # Return up to 3 confounds
        return pool[:3]

    def _select_measurements(self, h: Hypothesis) -> List[str]:
        base = [
            "primary_outcome_continuous",
            "secondary_outcome_binary",
            "process_measure",
        ]
        if h.template == "mechanistic":
            base.append("biomarker_intermediate")
        elif h.template == "comparative":
            base.append("subgroup_stratification_variable")
        return base

    def _define_groups(self, h: Hypothesis) -> tuple[str, str]:
        if h.template == "comparative":
            words = h.statement.split()
            # Try to extract group names from hypothesis text
            if "between" in words:
                idx = words.index("between")
                if idx + 2 < len(words):
                    return words[idx + 1].rstrip(","), words[idx + 2].rstrip(".")
        return "control_no_intervention", "treatment_intervention"
