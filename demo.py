#!/usr/bin/env python3
"""
AI Science Platform — Demo
Runs a 3-round scientific discovery loop on a synthetic dataset
exploring the effect of a training intervention on cognitive performance.
"""

from ai_science import ScientificLoop

def main():
    print("=" * 65)
    print("  AI Science Platform — Hypothesis-Driven Discovery Demo")
    print("=" * 65)

    # Initial observations from exploratory data collection
    observations = {
        "training_hours":      8.3,     # mean training hours/week
        "cognitive_score":     72.4,    # mean outcome score
        "groups":              ["control_group", "training_group"],
        "n_participants":      120,
        "prior_correlation_r": 0.31,
        "notes": "Pilot study suggests positive trend but underpowered",
    }

    print("\nInitial Observations:")
    for k, v in observations.items():
        if k != "notes":
            print(f"  {k}: {v}")
    print(f"  notes: {observations['notes']}")

    loop = ScientificLoop(
        domain="psychology",
        max_rounds=3,
        seed=2026,
    )

    print("\nStarting discovery loop (max 3 rounds)...\n")
    rounds = loop.run(observations)

    for dr in rounds:
        print("-" * 65)
        print(f"\n{'▶ ROUND ' + str(dr.round_number):^65}")
        print()
        print("HYPOTHESIS")
        print(f"  {dr.hypothesis}")
        print()
        print("EXPERIMENT DESIGN")
        print(f"  {dr.design}")
        print()
        print("ANALYSIS")
        print(f"  {dr.analysis}")
        print()

    print("=" * 65)
    print("DISCOVERY SUMMARY")
    print("=" * 65)
    for dr in rounds:
        print(f"  {dr.summary_line()}")

    final = rounds[-1].analysis
    print()
    if final.verdict == "supports":
        print(f"✓ Converged on a SUPPORTED hypothesis in {len(rounds)} round(s).")
        print(f"  Effect: {final.effect_strength} (d={final.cohens_d:+.3f})")
    elif final.verdict == "refutes":
        print(f"✗ Hypothesis REFUTED after {len(rounds)} round(s).")
        print("  Recommendation: revisit assumptions, gather new observations.")
    else:
        print(f"~ INCONCLUSIVE after {len(rounds)} round(s). More data needed.")
    print()

if __name__ == "__main__":
    main()
