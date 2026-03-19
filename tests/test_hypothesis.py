"""Tests for hypothesis generation."""
import unittest
from ai_science.hypothesis import Hypothesis, HypothesisGenerator


class TestHypothesisGenerator(unittest.TestCase):

    def setUp(self):
        self.obs = {"temperature": 37.0, "growth_rate": 1.2, "groups": ["A", "B"]}
        self.gen = HypothesisGenerator(domain="biology", seed=42)

    def test_generate_returns_correct_count(self):
        hyps = self.gen.generate(self.obs, n=4)
        self.assertEqual(len(hyps), 4)

    def test_all_templates_producible(self):
        results = {}
        for template in HypothesisGenerator.TEMPLATES:
            hyps = self.gen.generate(self.obs, n=1, templates=[template])
            self.assertEqual(len(hyps), 1)
            results[template] = hyps[0]

        for t, h in results.items():
            self.assertEqual(h.template, t)
            self.assertIsInstance(h.statement, str)
            self.assertGreater(len(h.statement), 10)

    def test_hypothesis_confidence_range(self):
        hyps = self.gen.generate(self.obs, n=20)
        for h in hyps:
            self.assertGreaterEqual(h.confidence, 0.0)
            self.assertLessEqual(h.confidence, 1.0)

    def test_hypothesis_has_predictions_and_criteria(self):
        hyps = self.gen.generate(self.obs, n=4)
        for h in hyps:
            self.assertGreater(len(h.testable_predictions), 0)
            self.assertGreater(len(h.falsification_criteria), 0)

    def test_refine_increments_generation(self):
        parent = self.gen.generate(self.obs, n=1)[0]
        child  = self.gen.refine(parent, self.obs)
        self.assertEqual(child.generation, parent.generation + 1)
        self.assertEqual(child.parent_id, parent.id)

    def test_refine_increases_confidence(self):
        parent = self.gen.generate(self.obs, n=1)[0]
        parent.confidence = 0.4   # fix for determinism
        child  = self.gen.refine(parent, self.obs)
        self.assertGreater(child.confidence, parent.confidence)

    def test_hypothesis_ids_are_unique(self):
        hyps = self.gen.generate(self.obs, n=10)
        ids = [h.id for h in hyps]
        self.assertEqual(len(set(ids)), len(ids))

    def test_str_representation(self):
        h = self.gen.generate(self.obs, n=1)[0]
        s = str(h)
        self.assertIn("Confidence", s)
        self.assertIn("Predictions", s)
        self.assertIn("Falsified", s)


if __name__ == "__main__":
    unittest.main()
