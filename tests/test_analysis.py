"""Tests for result analysis."""
import math
import unittest
from ai_science.hypothesis import HypothesisGenerator
from ai_science.experiment import ExperimentDesigner
from ai_science.analysis import ExperimentResult, ResultAnalyzer, AnalysisResult


class TestResultAnalyzer(unittest.TestCase):

    def setUp(self):
        self.obs = {"x": 1.0, "y": 2.0}
        gen = HypothesisGenerator(domain="general", seed=7)
        self.h = gen.generate(self.obs, n=1, templates=["causal"])[0]
        designer = ExperimentDesigner()
        self.design = designer.design(self.h)
        self.analyzer = ResultAnalyzer()

    def _make_result(self, control, treatment):
        return ExperimentResult(
            design_id=self.design.id,
            control_values=control,
            treatment_values=treatment,
        )

    def test_analyze_returns_analysis_result(self):
        c = [0.0] * 30
        t = [1.0] * 30
        r = self._make_result(c, t)
        a = self.analyzer.analyze(r, self.h, self.design)
        self.assertIsInstance(a, AnalysisResult)

    def test_large_effect_supports(self):
        """Clearly different groups should support the hypothesis."""
        import random
        rng = random.Random(1)
        c = [rng.gauss(0, 1) for _ in range(50)]
        t = [rng.gauss(2, 1) for _ in range(50)]    # d ≈ 2
        r = self._make_result(c, t)
        a = self.analyzer.analyze(r, self.h, self.design)
        self.assertEqual(a.verdict, "supports")
        self.assertLess(a.p_value, 0.05)
        self.assertGreater(abs(a.cohens_d), 0.5)

    def test_no_effect_refutes(self):
        """Identical groups should refute."""
        import random
        rng = random.Random(2)
        c = [rng.gauss(0, 1) for _ in range(50)]
        t = [rng.gauss(0, 1) for _ in range(50)]    # d ≈ 0
        r = self._make_result(c, t)
        a = self.analyzer.analyze(r, self.h, self.design)
        # p should be > 0.05 most of the time with seed=2
        self.assertIn(a.verdict, ("refutes", "inconclusive"))

    def test_cohens_d_sign(self):
        """treatment > control → d > 0."""
        c = [0.0] * 30
        t = [1.5] * 30
        r = self._make_result(c, t)
        a = self.analyzer.analyze(r, self.h, self.design)
        self.assertGreater(a.cohens_d, 0)

    def test_confidence_interval_contains_true_diff(self):
        """95% CI should typically contain the true difference."""
        import random
        rng = random.Random(99)
        true_diff = 1.0
        c = [rng.gauss(0.0, 1.0) for _ in range(100)]
        t = [rng.gauss(true_diff, 1.0) for _ in range(100)]
        r = self._make_result(c, t)
        a = self.analyzer.analyze(r, self.h, self.design)
        self.assertLessEqual(a.ci_lower, true_diff)
        self.assertGreaterEqual(a.ci_upper, true_diff)

    def test_p_value_in_range(self):
        import random
        rng = random.Random(5)
        c = [rng.gauss(0, 1) for _ in range(30)]
        t = [rng.gauss(0.5, 1) for _ in range(30)]
        r = self._make_result(c, t)
        a = self.analyzer.analyze(r, self.h, self.design)
        self.assertGreaterEqual(a.p_value, 0.0)
        self.assertLessEqual(a.p_value, 1.0)

    def test_effect_strength_labels(self):
        analyzer = ResultAnalyzer()
        self.assertEqual(analyzer._effect_strength(0.1), "negligible")
        self.assertEqual(analyzer._effect_strength(0.3), "small")
        self.assertEqual(analyzer._effect_strength(0.6), "medium")
        self.assertEqual(analyzer._effect_strength(1.0), "large")

    def test_str_representation(self):
        c = [0.0] * 20
        t = [1.0] * 20
        r = self._make_result(c, t)
        a = self.analyzer.analyze(r, self.h, self.design)
        s = str(a)
        self.assertIn("Cohen's d", s)
        self.assertIn("p-value", s)
        self.assertIn("95% CI", s)
        self.assertIn("Verdict", s)

    def test_betainc_edge_cases(self):
        from ai_science.analysis import ResultAnalyzer
        self.assertAlmostEqual(ResultAnalyzer._betainc(0.0, 2, 3), 0.0)
        self.assertAlmostEqual(ResultAnalyzer._betainc(1.0, 2, 3), 1.0)


if __name__ == "__main__":
    unittest.main()
