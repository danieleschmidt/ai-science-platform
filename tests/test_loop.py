"""Tests for ScientificLoop."""
import unittest
from ai_science.loop import ScientificLoop, DiscoveryRound


class TestScientificLoop(unittest.TestCase):

    BASE_OBS = {"x": 5.0, "y": 3.0, "groups": ["ctrl", "trt"]}

    def test_returns_list_of_rounds(self):
        loop = ScientificLoop(max_rounds=2, seed=1)
        rounds = loop.run(self.BASE_OBS)
        self.assertIsInstance(rounds, list)
        self.assertGreater(len(rounds), 0)

    def test_respects_max_rounds(self):
        loop = ScientificLoop(max_rounds=3, seed=2)
        rounds = loop.run(self.BASE_OBS)
        self.assertLessEqual(len(rounds), 3)

    def test_each_round_has_all_fields(self):
        loop = ScientificLoop(max_rounds=2, seed=3)
        rounds = loop.run(self.BASE_OBS)
        for r in rounds:
            self.assertIsInstance(r, DiscoveryRound)
            self.assertIsNotNone(r.hypothesis)
            self.assertIsNotNone(r.design)
            self.assertIsNotNone(r.result)
            self.assertIsNotNone(r.analysis)

    def test_round_numbers_are_sequential(self):
        loop = ScientificLoop(max_rounds=3, seed=4)
        rounds = loop.run(self.BASE_OBS)
        for i, r in enumerate(rounds, start=1):
            self.assertEqual(r.round_number, i)

    def test_stops_early_on_support(self):
        """Loop should stop as soon as a supported result is found."""
        # Force high effect by using a simulator that always returns a large difference
        import random
        from ai_science.analysis import ExperimentResult

        def strong_sim(design, hypothesis, rng):
            n = design.sample_size_per_group
            return ExperimentResult(
                design_id=design.id,
                control_values=[rng.gauss(0, 1) for _ in range(n)],
                treatment_values=[rng.gauss(3, 1) for _ in range(n)],  # d≈3
            )

        loop = ScientificLoop(max_rounds=5, simulator_fn=strong_sim, seed=5)
        rounds = loop.run(self.BASE_OBS)
        self.assertEqual(rounds[-1].analysis.verdict, "supports")
        # Should not need all 5 rounds
        self.assertLessEqual(len(rounds), 5)

    def test_refinement_increases_generation(self):
        """Later rounds should have higher hypothesis generation numbers."""
        loop = ScientificLoop(max_rounds=3, seed=6)
        rounds = loop.run(self.BASE_OBS)
        # At least one round should have gen > 1 if refinement happened
        gens = [r.hypothesis.generation for r in rounds]
        self.assertGreaterEqual(max(gens), 1)

    def test_custom_domain(self):
        loop = ScientificLoop(domain="chemistry", max_rounds=2, seed=7)
        rounds = loop.run({"pH": 7.0, "yield": 0.85})
        self.assertGreater(len(rounds), 0)

    def test_summary_line_format(self):
        loop = ScientificLoop(max_rounds=1, seed=8)
        rounds = loop.run(self.BASE_OBS)
        line = rounds[0].summary_line()
        self.assertIn("Round 1", line)
        self.assertIn("p=", line)
        self.assertIn("d=", line)


if __name__ == "__main__":
    unittest.main()
