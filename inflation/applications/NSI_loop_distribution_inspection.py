"""
NSI_loop_distribution_inspection.py
--------------------------------------------------------------------
Compare the current NSI 3-loop distribution against the target
3-loop distribution

    P(a,b,c) = 1/8 * (1 + (ab + bc + ac) * (sqrt(2) - 1))

where the target variables use outcomes in {-1, +1}, identified with
the current binary labels by

    0 -> -1
    1 -> +1
--------------------------------------------------------------------
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
import sys

import sympy as sp

# Ensure repo root is on sys.path so "import inflation" works when running directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inflation.distributions import NSIPRDistribution


def generate_loop_events(n: int) -> list[tuple[int, ...]]:
    return list(product((0, 1), repeat=n))


def bits_to_target_pm1(event: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(-1 if bit == 0 else 1 for bit in event)


def target_three_loop_probability(event: tuple[int, int, int]) -> sp.Expr:
    a, b, c = (sp.Integer(value) for value in bits_to_target_pm1(event))
    return sp.simplify(
        (sp.Integer(1) + (a * b + b * c + a * c) * (sp.sqrt(2) - 1))
        / 8
    )


def compute_correlator(
    events: list[tuple[int, ...]],
    probabilities: list[sp.Expr],
    indices: tuple[int, ...],
) -> sp.Expr:
    total = sp.Integer(0)
    for event, probability in zip(events, probabilities):
        total += ((-1) ** sum(event[index] for index in indices)) * probability
    return sp.simplify(total)


def compute_three_loop_correlators(
    events: list[tuple[int, int, int]],
    probabilities: list[sp.Expr],
) -> dict[str, sp.Expr]:
    return {
        "A": compute_correlator(events, probabilities, (0,)),
        "B": compute_correlator(events, probabilities, (1,)),
        "C": compute_correlator(events, probabilities, (2,)),
        "AB": compute_correlator(events, probabilities, (0, 1)),
        "BC": compute_correlator(events, probabilities, (1, 2)),
        "AC": compute_correlator(events, probabilities, (0, 2)),
        "ABC": compute_correlator(events, probabilities, (0, 1, 2)),
    }


if __name__ == "__main__":
    n = 3
    digits = 32
    distribution = NSIPRDistribution()

    events = generate_loop_events(n)

    nsi_probabilities = [
        sp.simplify(distribution.prob_event_loop(event))
        for event in events
    ]
    target_probabilities = [
        target_three_loop_probability(event)
        for event in events
    ]

    nsi_correlators = compute_three_loop_correlators(events, nsi_probabilities)
    target_correlators = compute_three_loop_correlators(events, target_probabilities)

    print("Target 3-loop formula:")
    print("P(a,b,c) = 1/8 * (1 + (ab + bc + ac) * (sqrt(2) - 1))")
    print("")
    print("Outcome identification:")
    print("0 -> -1")
    print("1 -> +1")
    print("")
    print("Events:")
    print(events)
    print("")
    print("NSI 3-loop probability vector:")
    print(nsi_probabilities)
    print("")
    print("NSI 3-loop probability vector (approx):")
    print([sp.N(probability, digits) for probability in nsi_probabilities])
    print("")
    print("Target 3-loop probability vector:")
    print(target_probabilities)
    print("")
    print("Target 3-loop probability vector (approx):")
    print([sp.N(probability, digits) for probability in target_probabilities])
    print("")
    print("NSI correlators:")
    for name, value in nsi_correlators.items():
        print(f"<{name}> = {value}    approx={sp.N(value, digits)}")
    print("")
    print("Target correlators:")
    for name, value in target_correlators.items():
        print(f"<{name}> = {value}    approx={sp.N(value, digits)}")
