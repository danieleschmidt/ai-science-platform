"""Tests for experiment design."""
import unittest
from ai_science.hypothesis import HypothesisGenerator
from ai_science.experiment import ExperimentDesigner, ExperimentDesign


class TestExperimentDesigner(unittest.TestCase):

    def setUp(self):
        self.obs = {"dose": 10.0, "response": 0.85, "groups": ["placebo", "drug"]}
        self.gen = HypothesisGenerator(domain="biology", seed=0)
        self.designer = ExperimentDesigner()

    def _make_hypothesis(self, template):
        return self.gen.generate(self.obs, n=1, templates=[template])[0]

    def test_design_returns_experiment_design(self):
        h = self._make_hypothesis("causal")
        d = self.designer.design(h)
        self.assertIsInstance(d, ExperimentDesign)

    def test_sample_size_positive(self):
        for template in ("correlation", "causal", "mechanistic", "comparative"):
            h = self._make_hypothesis(template)
            d = self.designer.design(h)
            self.assertGreater(d.sample_size_per_group, 0)

    def test_higher_confidence_lower_n(self):
        """Higher confidence → larger effect size → fewer subjects needed."""
        obs = dict(self.obs)
        gen = HypothesisGenerator(domain="biology", seed=1)
        h_low  = gen.generate(obs, n=1, templates=["causal"])[0]
        h_low.confidence = 0.1
        h_high = gen.generate(obs, n=1, templates=["causal"])[0]
        h_high.confidence = 0.9

        d_low  = self.designer.design(h_low)
        d_high = self.designer.design(h_high)

        # Higher confidence → larger expected effect → fewer subjects
        self.assertGreater(d_low.sample_size_per_group, d_high.sample_size_per_group)

    def test_confounds_not_empty(self):
        h = self._make_hypothesis("causal")
        d = self.designer.design(h)
        self.assertGreater(len(d.confounds_to_control), 0)

    def test_measurements_not_empty(self):
        h = self._make_hypothesis("mechanistic")
        d = self.designer.design(h)
        self.assertGreater(len(d.measurements), 0)
        # mechanistic should add biomarker measurement
        self.assertTrue(any("biomarker" in m for m in d.measurements))

    def test_hypothesis_id_propagated(self):
        h = self._make_hypothesis("correlation")
        d = self.designer.design(h)
        self.assertEqual(d.hypothesis_id, h.id)

    def test_str_representation(self):
        h = self._make_hypothesis("causal")
        d = self.designer.design(h)
        s = str(d)
        self.assertIn("Control", s)
        self.assertIn("Treatment", s)
        self.assertIn("N/group", s)


if __name__ == "__main__":
    unittest.main()
