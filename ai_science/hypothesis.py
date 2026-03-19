"""
Hypothesis generation via template-based abductive reasoning.

Abduction: given observations, infer the most plausible explanation.
Four templates model the main modes of scientific reasoning:
  1. Correlation   — two variables co-vary
  2. Causal        — one variable drives another via a mechanism
  3. Mechanistic   — proposes an underlying process
  4. Comparative   — contrasts behaviour across subgroups/conditions
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Hypothesis:
    """A falsifiable scientific hypothesis."""

    id: str
    statement: str
    domain: str
    confidence: float                        # 0.0 – 1.0
    testable_predictions: List[str]
    falsification_criteria: List[str]
    template: str                            # which template produced this
    generation: int = 1                      # refinement round
    parent_id: str | None = None            # set when refined from another

    def __str__(self) -> str:
        return (
            f"[{self.template.upper()}] {self.statement}\n"
            f"  Confidence: {self.confidence:.2f}\n"
            f"  Predictions: {'; '.join(self.testable_predictions)}\n"
            f"  Falsified if: {'; '.join(self.falsification_criteria)}"
        )


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class HypothesisGenerator:
    """
    Generates scientific hypotheses from a set of observations using
    template-based abductive reasoning.

    Each template encodes a different explanatory structure:
      - correlation   : X and Y co-vary systematically
      - causal        : X causes Y through some mechanism
      - mechanistic   : proposes the underlying process
      - comparative   : effect differs between groups/conditions
    """

    TEMPLATES = ("correlation", "causal", "mechanistic", "comparative")

    def __init__(self, domain: str = "general", seed: int | None = None):
        self.domain = domain
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        observations: dict,
        n: int = 1,
        templates: list[str] | None = None,
    ) -> List[Hypothesis]:
        """Return *n* hypotheses derived from *observations*."""
        chosen_templates = templates or list(self.TEMPLATES)
        selected = self._rng.choices(chosen_templates, k=n)
        return [self._build(obs=observations, template=t) for t in selected]

    def refine(self, parent: Hypothesis, observations: dict) -> Hypothesis:
        """Produce a refined hypothesis informed by a previous one."""
        refined = self._build(obs=observations, template=parent.template)
        # Boost confidence slightly if we're refining (we know more now)
        refined.confidence = min(1.0, parent.confidence + self._rng.uniform(0.02, 0.12))
        refined.generation = parent.generation + 1
        refined.parent_id = parent.id
        return refined

    # ------------------------------------------------------------------
    # Template implementations
    # ------------------------------------------------------------------

    def _build(self, obs: dict, template: str) -> Hypothesis:
        builders = {
            "correlation":  self._correlation,
            "causal":       self._causal,
            "mechanistic":  self._mechanistic,
            "comparative":  self._comparative,
        }
        return builders[template](obs)

    def _extract_vars(self, obs: dict):
        """Pull numeric variables out of observations for naming."""
        numeric = {k: v for k, v in obs.items() if isinstance(v, (int, float))}
        keys = list(numeric.keys())
        if len(keys) >= 2:
            return keys[0], keys[1], numeric
        if keys:
            return keys[0], "outcome", numeric
        return "X", "Y", {}

    def _correlation(self, obs: dict) -> Hypothesis:
        x, y, nums = self._extract_vars(obs)
        direction = "positively" if self._rng.random() > 0.5 else "negatively"
        conf = self._rng.uniform(0.45, 0.75)
        return Hypothesis(
            id=str(uuid.uuid4())[:8],
            statement=f"In {self.domain}, {x} and {y} are {direction} correlated.",
            domain=self.domain,
            confidence=conf,
            testable_predictions=[
                f"Pearson r({x},{y}) will be {'>' if direction == 'positively' else '<'} 0 (p < 0.05).",
                f"Scatter plot of {x} vs {y} will show a {'positive' if direction == 'positively' else 'negative'} trend.",
            ],
            falsification_criteria=[
                f"r({x},{y}) is not significantly different from 0.",
                f"The relationship reverses under replication.",
            ],
            template="correlation",
        )

    def _causal(self, obs: dict) -> Hypothesis:
        x, y, nums = self._extract_vars(obs)
        conf = self._rng.uniform(0.35, 0.65)
        return Hypothesis(
            id=str(uuid.uuid4())[:8],
            statement=f"Increasing {x} causally increases {y} in {self.domain}.",
            domain=self.domain,
            confidence=conf,
            testable_predictions=[
                f"Manipulating {x} (RCT) will produce a statistically significant change in {y}.",
                f"Effect of {x} on {y} persists after controlling for confounders.",
            ],
            falsification_criteria=[
                f"A controlled experiment finds no significant {y} difference across {x} levels.",
                f"Effect disappears when mediating variable is held constant.",
            ],
            template="causal",
        )

    def _mechanistic(self, obs: dict) -> Hypothesis:
        x, y, nums = self._extract_vars(obs)
        mechanisms = [
            "feedback inhibition", "resource competition", "signal transduction",
            "metabolic regulation", "allosteric modulation", "gene expression",
        ]
        mech = self._rng.choice(mechanisms)
        conf = self._rng.uniform(0.25, 0.55)
        return Hypothesis(
            id=str(uuid.uuid4())[:8],
            statement=(
                f"The relationship between {x} and {y} in {self.domain} "
                f"is mediated by {mech}."
            ),
            domain=self.domain,
            confidence=conf,
            testable_predictions=[
                f"Blocking {mech} will attenuate the {x}→{y} effect.",
                f"Intermediate products of {mech} will be detectable when {x} is varied.",
            ],
            falsification_criteria=[
                f"Inhibiting {mech} has no effect on the {x}→{y} relationship.",
                f"No intermediate products are found.",
            ],
            template="mechanistic",
        )

    def _comparative(self, obs: dict) -> Hypothesis:
        x, y, nums = self._extract_vars(obs)
        groups = obs.get("groups", ["group_A", "group_B"])
        if isinstance(groups, list) and len(groups) >= 2:
            g1, g2 = groups[0], groups[1]
        else:
            g1, g2 = "control", "treatment"
        conf = self._rng.uniform(0.40, 0.70)
        return Hypothesis(
            id=str(uuid.uuid4())[:8],
            statement=(
                f"The effect of {x} on {y} in {self.domain} "
                f"differs significantly between {g1} and {g2}."
            ),
            domain=self.domain,
            confidence=conf,
            testable_predictions=[
                f"Mean {y} in {g1} will differ from {g2} by ≥ 0.5 SD.",
                f"ANOVA interaction term ({x} × group) will be significant.",
            ],
            falsification_criteria=[
                f"No significant group × {x} interaction is found.",
                f"Effect size (η²) is < 0.01.",
            ],
            template="comparative",
        )
