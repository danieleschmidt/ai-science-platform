# AI Science Platform

> AI-accelerated scientific discovery: hypothesis generation → experiment design → statistical analysis → iterative refinement.

## What it does

The platform orchestrates the core loop of empirical science programmatically:

```
Observations → Hypothesis → Experiment Design → Run → Analyze → Refine → ...
```

Each component does real work:

| Component | What it does |
|---|---|
| `HypothesisGenerator` | Abductive reasoning from observations using 4 scientific templates |
| `ExperimentDesigner` | Designs controlled experiments with power-analysis-based sample sizing |
| `ResultAnalyzer` | Computes Cohen's d, Welch t-test p-value, 95% CI — no scipy needed |
| `ScientificLoop` | Orchestrates the full discovery cycle with iterative refinement |

## Quick start

```bash
python demo.py
```

Output:
```
Round 1: [correlation] p=0.0000  d=+0.592  → SUPPORTS
✓ Converged on a SUPPORTED hypothesis in 1 round(s).
```

## Usage

```python
from ai_science import ScientificLoop

observations = {
    "treatment_dose": 10.0,
    "biomarker_level": 4.2,
    "groups": ["placebo", "drug"],
}

loop = ScientificLoop(domain="biology", max_rounds=5, seed=42)
rounds = loop.run(observations)

for r in rounds:
    print(r.summary_line())
    # Round 1: [causal] p=0.0032  d=+0.481  → SUPPORTS
```

### Use your own data

Plug in a custom simulator that returns real experimental data:

```python
from ai_science.analysis import ExperimentResult

def my_experiment(design, hypothesis, rng):
    # ... run your actual experiment or fetch data ...
    return ExperimentResult(
        design_id=design.id,
        control_values=[...],
        treatment_values=[...],
    )

loop = ScientificLoop(simulator_fn=my_experiment)
```

## Hypothesis templates

The generator uses four abductive reasoning templates:

- **Correlation** — two variables co-vary systematically
- **Causal** — one variable drives another (proposes an RCT)
- **Mechanistic** — proposes an underlying biological/chemical process
- **Comparative** — effect differs between defined subgroups

Each hypothesis includes:
- `testable_predictions` — concrete, measurable predictions
- `falsification_criteria` — explicit conditions that would refute it

## Statistics (stdlib only)

- **Cohen's d** — pooled-SD effect size
- **Welch's t-test** — two-sample, unequal variance, no scipy
- **95% CI** — Welch–Satterthwaite degrees of freedom
- **Power analysis** — Beasley-Springer-Moro inverse normal approximation
- **t-distribution CDF** — regularized incomplete beta via Lentz continued fractions

## Tests

```bash
python -m pytest tests/ -v
# 32 passed
```

## Project structure

```
ai_science/
  __init__.py       — public API
  hypothesis.py     — Hypothesis dataclass + HypothesisGenerator
  experiment.py     — ExperimentDesign dataclass + ExperimentDesigner
  analysis.py       — ExperimentResult, AnalysisResult, ResultAnalyzer
  loop.py           — ScientificLoop + DiscoveryRound
tests/
  test_hypothesis.py
  test_experiment.py
  test_analysis.py
  test_loop.py
demo.py             — 3-round discovery loop example
```

## Design philosophy

- **No external dependencies** — stdlib only (math, random, dataclasses, uuid)
- **Bring your own data** — the simulator is a pluggable function
- **Reproducible** — seed the RNG for deterministic runs
- **Falsifiable** — every hypothesis carries explicit falsification criteria

## License

MIT
