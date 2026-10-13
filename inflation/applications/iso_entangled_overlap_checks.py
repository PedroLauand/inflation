"""Exact context checks for the proposed coarse-grained measurement scan.

These cases check overlap with classical triangle models, without solving
an inflation or implementing the general iso-entangled parameter scan.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
import sys

import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from inflation.distributions.ejm import EJMDistribution


EVENTS = tuple(product((0, 1), repeat=3))


def maxent_binary_ranks():
    bell = sp.Matrix([0, 1, -1, 0]) / sp.sqrt(2)
    wire = bell * bell.H
    local = sp.Matrix(2, 2, lambda i, j: sum(wire[2*i+t, 2*j+t] for t in range(2)))
    assert local == sp.eye(2) / 2
    party_input = sp.kronecker_product(local, local)
    assert party_input == sp.eye(4) / 4
    cases = []
    for rank in (1, 2, 3):
        effect = sp.diag(*([1] * rank + [0] * (4-rank)))
        p0 = sp.trace(party_input * effect)
        e1 = 2*p0 - 1  # bit 0 is +1, bit 1 is -1
        assert e1 == sp.Rational(rank, 2) - 1
        cases.append({"rank": rank, "p0": str(p0), "E1": str(e1)})
    # The result depends only on trace(effect), so holds for every rotated
    # rank-r projector, not only the diagonal representatives used here.
    v = sp.symbols("v", real=True)
    intersections = []
    for rank in (1, 2, 3):
        root = sp.solve(sp.Eq(v/3, sp.Rational(rank, 2)-1), v)[0]
        if 0 <= root <= 1:
            intersections.append({"rank": rank, "visibility": str(root)})
    assert intersections == [{"rank": 2, "visibility": "0"}]
    return {"case": "maxent-binary-ranks", "marginals": cases,
            "noisy_W_intersections_in_unit_interval": intersections}


def balanced_ejm():
    def response(outcome, left, right):
        return sp.Rational(int(outcome == left) + int(outcome == right), 2)

    classical = {}
    for a, b, c in EVENTS:
        classical[a, b, c] = sum(
            response(a, beta, gamma) * response(b, gamma, alpha)
            * response(c, alpha, beta) / 8
            for alpha, beta, gamma in EVENTS
        )
    assert sum(classical.values()) == 1
    results = []
    for partner in (1, 2, 3):
        first = [0, partner]
        partition = [first, [j for j in range(4) if j not in first]]
        fragment = EJMDistribution(coarsen=partition)
        quantum = {event: fragment.prob_event_loop(event) for event in EVENTS}
        assert quantum == classical
        moments = [sp.simplify(sum((-1)**sum(event[i] for i in subset) * p
                                  for event, p in quantum.items()))
                   for subset in ((0,), (0, 1), (0, 1, 2))]
        assert moments == [0, sp.Rational(1, 4), 0]
        results.append({"partition": partition,
                        "E1_E2_E3": [str(value) for value in moments],
                        "triangle_matches_symmetric_classical_model": True})
    return {"case": "balanced-ejm", "partitions": results,
            "classical_model": "Independent fair wire bits; each party independently chooses one incoming bit uniformly.",
            "scope": "triangle statistics only; no claim about other loop lengths",
            "probabilities": {"".join(map(str, event)): str(p)
                              for event, p in classical.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("maxent-binary-ranks", "balanced-ejm", "all"))
    args = parser.parse_args()
    checks = {"maxent-binary-ranks": maxent_binary_ranks, "balanced-ejm": balanced_ejm}
    selected = checks if args.case == "all" else {args.case: checks[args.case]}
    print(json.dumps([check() for check in selected.values()], indent=2))
