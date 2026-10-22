"""Extract and exhaustively verify a triangle/pentagon classical witness.

Standalone alternative to PrepLP: enumerate all binary assignments to the
20 directed edges on five source labels. Only cyclic outcome relabelings
are identified; no data-dependent symmetry or LPI constraint is used.
"""
from __future__ import annotations

import itertools
import argparse
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as sp
from numba import njit
from scipy.optimize import Bounds, LinearConstraint, milp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from inflation.distributions.nsi_pr import NSIPRDistribution  # noqa: E402

COEFFICIENTS = [0, 0, 2, 0, 0, 1, -1, 2, 0, 2, -1, 1]
BOUND = 0


def rotate(word):
    return {word[i:] + word[:i] for i in range(len(word))}


def build_columns():
    edges = list(itertools.permutations(range(5), 2))
    edge_index = {edge: i for i, edge in enumerate(edges)}
    assignments = np.arange(1 << len(edges), dtype=np.uint32)
    columns = []
    labels = []
    targets = []
    provider = NSIPRDistribution()
    cycle_counts = []
    for k in (3, 5):
        words = list(itertools.product(range(2), repeat=k))
        reps = sorted({min(rotate(word)) for word in words})
        word_to_row = {word: reps.index(min(rotate(word))) for word in words}
        lookup = np.array([word_to_row[word] for word in words], dtype=np.uint8)
        cycles = [c for c in itertools.permutations(range(5), k) if c[0] == min(c)]
        cycle_counts.append(len(cycles))
        histogram = np.zeros((len(assignments), len(reps)), dtype=np.uint8)
        for cycle in cycles:
            code = np.zeros(len(assignments), dtype=np.uint8)
            for i in range(k):
                bit = edge_index[(cycle[i], cycle[(i + 1) % k])]
                code = 2 * code + ((assignments >> bit) & 1).astype(np.uint8)
            histogram[assignments, lookup[code]] += 1
        columns.append(histogram)
        labels.extend(["".join(map(str, word)) for word in reps])
        targets.extend([sp.simplify(sum(provider.prob_event_loop(w) for w in rotate(word)))
                        for word in reps])
    counts = np.concatenate(columns, axis=1)
    return counts, labels, targets, cycle_counts


@njit
def check_direct(edge_cycles, cycle_lengths, word_weights):
    """Independent integer check without histogram construction/compression."""
    minimum = 2 ** 30
    tight = 0
    for assignment in range(1 << 20):
        value = 0
        for j in range(len(cycle_lengths)):
            word = 0
            for k in range(cycle_lengths[j]):
                word = 2 * word + ((assignment >> edge_cycles[j, k]) & 1)
            value += word_weights[j, word]
        minimum = min(minimum, value)
        if value == 0:
            tight += 1
    return minimum, tight


def direct_verification():
    edges = list(itertools.permutations(range(5), 2))
    cycles = []
    lengths = []
    weights = []
    offset = 0
    for k, multiplier in ((3, 6), (5, 5)):
        words = list(itertools.product(range(2), repeat=k))
        representatives = sorted({min(rotate(word)) for word in words})
        for cycle in itertools.permutations(range(5), k):
            if cycle[0] != min(cycle):
                continue
            cycles.append([edges.index((cycle[i], cycle[(i + 1) % k]))
                           for i in range(k)] + [0] * (5 - k))
            lengths.append(k)
            weights.append([multiplier * COEFFICIENTS[offset + representatives.index(min(rotate(word)))]
                            for word in words] + [0] * (32 - 2 ** k))
        offset += len(representatives)
    return check_direct(np.array(cycles), np.array(lengths), np.array(weights))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", action="store_true", help="Rediscover small integer coefficients")
    args = parser.parse_args()
    counts, labels, targets, cycle_counts = build_columns()
    unique = np.unique(counts, axis=0)
    denominators = np.array([cycle_counts[0]] * 4 + [cycle_counts[1]] * 8)
    vertices = unique / denominators
    target = np.array([float(t) for t in targets])
    # Seek a small integer inequality w.vertex >= alpha. The floating-point
    # target is used only for discovery; acceptance below uses exact arithmetic.
    size = len(labels) + 1
    homogeneous = np.c_[vertices, -np.ones(len(vertices))]
    identity = np.eye(size)
    matrix = np.vstack([
        np.c_[homogeneous, np.zeros_like(homogeneous)],
        np.r_[target, -1.0, np.zeros(size)][None, :],
        np.c_[identity, -identity],
        np.c_[-identity, -identity],
    ])
    lower = np.r_[np.zeros(len(vertices)), -np.inf, np.full(2 * size, -np.inf)]
    upper = np.r_[np.full(len(vertices), np.inf), -0.05, np.zeros(2 * size)]
    low_bounds = np.r_[np.full(size, -30), np.zeros(size)]
    high_bounds = np.full(2 * size, 30)
    # Remove the two normalization freedoms by setting one coefficient per
    # loop to zero. This does not impose a symmetry on the probability data.
    low_bounds[[0, 4]] = 0
    high_bounds[[0, 4]] = 0
    coefficients = COEFFICIENTS
    bound = BOUND
    search_status = "not requested; verify recorded integer coefficients"
    if args.search:
        result = milp(np.r_[np.zeros(size), np.ones(size)],
                      integrality=np.r_[np.ones(size), np.zeros(size)],
                      bounds=Bounds(low_bounds, high_bounds),
                      constraints=LinearConstraint(matrix, lower, upper),
                      options={"time_limit": 45})
        if result.x is None:
            raise RuntimeError(result.message)
        rational = [Fraction(float(x)).limit_denominator(10000) for x in result.x[:size]]
        scale = math.lcm(*(x.denominator for x in rational))
        coefficients = [int(x * scale) for x in rational[:-1]]
        bound = int(rational[-1] * scale)
        search_status = result.message
    # Integer exhaustive check; these are the original assignments, without
    # relying on the unique-column calculation or on a solver tolerance.
    integer_scale = math.lcm(*cycle_counts)
    edge_weights = np.array([c * integer_scale // d
                            for c, d in zip(coefficients, denominators)], dtype=np.int64)
    slack = counts.astype(np.int64) @ edge_weights - integer_scale * bound
    exact_value = sp.simplify(sum(c * t for c, t in zip(coefficients, targets)))
    direct_minimum, direct_tight = direct_verification()
    report = {
        "case": "nsi-triangle-pentagon-five-source-witness",
        "assignments_checked": len(counts),
        "unique_histograms": len(unique),
        "cycle_counts": cycle_counts,
        "labels": labels,
        "coefficients": coefficients,
        "bound": bound,
        "targets": [str(t) for t in targets],
        "exact_value": str(exact_value),
        "exact_violation": str(sp.simplify(bound - exact_value)),
        "minimum_integer_slack": int(slack.min()),
        "tight_assignments": int(np.count_nonzero(slack == 0)),
        "integer_slack_scale": integer_scale,
        "independent_direct_minimum_integer_slack": int(direct_minimum),
        "independent_direct_tight_assignments": int(direct_tight),
        "independent_check_applies_to": "recorded COEFFICIENTS, BOUND",
        "solver": "scipy.optimize.milp (HiGHS), 45 second search limit",
        "search_status": search_status,
        "versions": {"numpy": np.__version__, "sympy": sp.__version__,
                     "scipy": __import__("scipy").__version__,
                     "python": sys.version.split()[0]},
    }
    if slack.min() < 0 or not bool(exact_value < bound):
        raise AssertionError("Candidate failed exact validity or violation check")
    if direct_minimum < 0:
        raise AssertionError("Independent direct check failed")
    if coefficients == COEFFICIENTS and (int(slack.min()), int(np.count_nonzero(slack == 0))) != (direct_minimum, direct_tight):
        raise AssertionError("Two exact checks disagree")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
