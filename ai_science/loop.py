"""
ScientificLoop — orchestrates the hypothesis → experiment → analysis → refine cycle.

Each round:
  1. Generate (or refine) a hypothesis from available observations
  2. Design an experiment
  3. Run the experiment (simulate or real data)
  4. Analyze results
  5. Update observations with findings; refine if inconclusive/refuted

The loop terminates when:
  - max_rounds is reached, OR
  - a hypothesis is definitively supported, OR
  - no further refinement is possible
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .hypothesis import Hypothesis, HypothesisGenerator
from .experiment import ExperimentDesign, ExperimentDesigner
from .analysis import AnalysisResult, ExperimentResult, ResultAnalyzer


# ---------------------------------------------------------------------------
# Round record
# ---------------------------------------------------------------------------

@dataclass
class DiscoveryRound:
    round_number: int
    hypothesis: Hypothesis
    design: ExperimentDesign
    result: ExperimentResult
    analysis: AnalysisResult

    def summary_line(self) -> str:
        return (
            f"Round {self.round_number}: [{self.hypothesis.template}] "
            f"p={self.analysis.p_value:.4f}  d={self.analysis.cohens_d:+.3f}  "
            f"→ {self.analysis.verdict.upper()}"
        )


# ---------------------------------------------------------------------------
# Simulator type
# ---------------------------------------------------------------------------

# Signature: (design, hypothesis, rng) → ExperimentResult
SimulatorFn = Callable[[ExperimentDesign, Hypothesis, random.Random], ExperimentResult]


# ---------------------------------------------------------------------------
# ScientificLoop
# ---------------------------------------------------------------------------

class ScientificLoop:
    """
    Drives the iterative scientific discovery process.

    Parameters
    ----------
    domain        : scientific domain (e.g. "biology", "psychology")
    max_rounds    : hard cap on iterations
    simulator_fn  : callable that produces ExperimentResult from a design.
                    If None, uses the built-in synthetic data simulator.
    seed          : RNG seed for reproducibility
    """

    def __init__(
        self,
        domain: str = "general",
        max_rounds: int = 5,
        simulator_fn: Optional[SimulatorFn] = None,
        seed: int = 42,
    ):
        self.domain = domain
        self.max_rounds = max_rounds
        self._rng = random.Random(seed)
        self._simulator = simulator_fn or self._default_simulator

        self._generator = HypothesisGenerator(domain=domain, seed=seed)
        self._designer  = ExperimentDesigner()
        self._analyzer  = ResultAnalyzer()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, initial_observations: dict) -> List[DiscoveryRound]:
        """
        Execute the discovery loop.

        Returns a list of DiscoveryRound objects, one per completed round.
        """
        observations = dict(initial_observations)
        rounds: List[DiscoveryRound] = []
        hypothesis: Optional[Hypothesis] = None

        for round_num in range(1, self.max_rounds + 1):
            # 1. Generate or refine hypothesis
            if hypothesis is None:
                hyps = self._generator.generate(observations, n=1)
                hypothesis = hyps[0]
            else:
                hypothesis = self._generator.refine(hypothesis, observations)

            # 2. Design experiment
            design = self._designer.design(hypothesis)

            # 3. Run experiment
            result = self._simulator(design, hypothesis, self._rng)

            # 4. Analyze
            analysis = self._analyzer.analyze(result, hypothesis, design)

            # 5. Record round
            dr = DiscoveryRound(
                round_number=round_num,
                hypothesis=hypothesis,
                design=design,
                result=result,
                analysis=analysis,
            )
            rounds.append(dr)

            # 6. Update observations with findings
            observations["last_p_value"]    = analysis.p_value
            observations["last_cohens_d"]   = analysis.cohens_d
            observations["last_verdict"]    = analysis.verdict
            observations["last_mean_diff"]  = (
                sum(result.treatment_values) / len(result.treatment_values)
                - sum(result.control_values) / len(result.control_values)
            )

            # 7. Stop if definitive support found
            if analysis.verdict == "supports":
                break

            # If refuted entirely, reset hypothesis selection
            if analysis.verdict == "refutes" and round_num < self.max_rounds:
                hypothesis = None

        return rounds

    # ------------------------------------------------------------------
    # Default synthetic data simulator
    # ------------------------------------------------------------------

    def _default_simulator(
        self,
        design: ExperimentDesign,
        hypothesis: Hypothesis,
        rng: random.Random,
    ) -> ExperimentResult:
        """
        Generates synthetic normally distributed data.

        The true effect size is drawn from N(expected_d * 0.8, 0.15)
        to simulate imperfect knowledge of the true effect.
        """
        n = design.sample_size_per_group
        expected_d = design.expected_effect_size

        # True effect is uncertain — drawn around expected
        true_d = rng.gauss(expected_d * 0.8, 0.15)

        # Control group: N(0, 1)
        control   = [rng.gauss(0.0, 1.0) for _ in range(n)]
        # Treatment group: N(true_d, 1)
        treatment = [rng.gauss(true_d, 1.0) for _ in range(n)]

        return ExperimentResult(
            design_id=design.id,
            control_values=control,
            treatment_values=treatment,
            metadata={
                "simulated": True,
                "true_effect_size": true_d,
                "n_per_group": n,
            },
        )
