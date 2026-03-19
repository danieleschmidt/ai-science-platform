"""AI Science Platform — hypothesis-driven scientific discovery engine."""

from .hypothesis import Hypothesis, HypothesisGenerator
from .experiment import ExperimentDesign, ExperimentDesigner
from .analysis import ExperimentResult, AnalysisResult, ResultAnalyzer
from .loop import ScientificLoop

__all__ = [
    "Hypothesis",
    "HypothesisGenerator",
    "ExperimentDesign",
    "ExperimentDesigner",
    "ExperimentResult",
    "AnalysisResult",
    "ResultAnalyzer",
    "ScientificLoop",
]
