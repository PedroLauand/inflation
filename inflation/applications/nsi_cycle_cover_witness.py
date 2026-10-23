"""Extract and exhaustively verify a full-cycle-cover classical witness.

Generalizes nsi_cycle_witness.py from the signatures (3),(5) to every cycle
cover of five source copies with cycle lengths at least two:
(2),(3),(4),(5),(2,2),(3,2). Product signatures supply products of loop
tables, so the witness is quadratic in P_2,P_3 and linear in P_4,P_5.
Only cyclic outcome relabelings are identified; no data-dependent symmetry
or LPI constraint is used. Verification is exact integer arithmetic over
all 2**20 binary assignments to the 20 ordered slots.
"""
from __future__ import annotations

import argparse
import itertools
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

N = 5
EDGES = list(itertools.permutations(range(N), 2))
EDGE_INDEX = {edge: i for i, edge in enumerate(EDGES)}
FULL = ((2,), (3,), (4,), (5,), (2, 2), (3, 2))
SINGLE = ((2,), (3,), (4,), (5,))

# Recorded coefficients, one per label in the order printed by the script.
# Filled in after a --search run; verified without any optimizer.
RECORDED = {
    "full": {"coefficients": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0], "bound": 0},
    "single": {"coefficients": [0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, -1, 0], "bound": 0},
}


def rotate(word):
    return {word[i:] + word[:i] for i in range(len(word))}


def orbit_reps(k):
    words = list(itertools.product(range(2), repeat=k))
    reps = sorted({min(rotate(w)) for w in words})
    return reps, {w: reps.index(min(rotate(w))) for w in words}


def directed_cycles(labels, k):
    """Directed k-cycles on the given labels, one per cyclic rotation."""
    return [c for c in itertools.permutations(labels, k) if c[0] == min(c)]


def covers(signature):
    """Cycle covers with this signature; each cover is a tuple of directed cycles."""
    if len(signature) == 1:
        return [(c,) for c in directed_cycles(range(N), signature[0])]
    k1, k2 = signature
    out = []
    for labels1 in itertools.combinations(range(N), k1):
        rest = [x for x in range(N) if x not in labels1]
        for labels2 in itertools.combinations(rest, k2):
            if k1 == k2 and labels1 > labels2:
                continue  # unordered pair of equal-length cycles
            for c1 in directed_cycles(labels1, k1):
                for c2 in directed_cycles(labels2, k2):
                    out.append((c1, c2))
    return out


def classes(signature):
    """Outcome classes of a cover: tuples of orbit representatives, unordered when lengths agree."""
    if len(signature) == 1:
        reps, _ = orbit_reps(signature[0])
        return [(r,) for r in reps]
    k1, k2 = signature
    reps1, _ = orbit_reps(k1)
    reps2, _ = orbit_reps(k2)
    if k1 == k2:
        return [(a, b) for a in reps1 for b in reps2 if a <= b]
    return [(a, b) for a in reps1 for b in reps2]


def class_of(signature, words):
    maps = [orbit_reps(k)[1] for k in signature]
    reps = tuple(orbit_reps(k)[0][maps[i][w]] for i, (k, w) in enumerate(zip(signature, words)))
    if len(signature) == 2 and signature[0] == signature[1]:
        reps = tuple(sorted(reps))
    return reps


def targets_for(signature, provider):
    """Exact NSI value of each class: orbit probabilities and their products."""
    def R(word):
        return sp.nsimplify(sp.simplify(sum(provider.prob_event_loop(v) for v in rotate(word))))
    out = []
    for cls in classes(signature):
        if len(cls) == 1:
            out.append(R(cls[0]))
        elif signature[0] == signature[1] and cls[0] != cls[1]:
            out.append(sp.simplify(2 * R(cls[0]) * R(cls[1])))
        else:
            out.append(sp.simplify(R(cls[0]) * R(cls[1])))
    return out


def build(signatures):
    assignments = np.arange(1 << len(EDGES), dtype=np.uint32)
    provider = NSIPRDistribution()
    blocks, labels, targets, denominators, sig_of_row = [], [], [], [], []
    cover_lists = []
    for sig in signatures:
        cls = classes(sig)
        cls_index = {c: i for i, c in enumerate(cls)}
        cvs = covers(sig)
        cover_lists.append(cvs)
        hist = np.zeros((len(assignments), len(cls)), dtype=np.uint8)
        # lookup from (code1, code2) to class index
        k = list(sig) + [0] * (2 - len(sig))
        table = np.zeros((1 << k[0], 1 << max(k[1], 0)), dtype=np.uint8)
        for w1 in itertools.product(range(2), repeat=k[0]):
            c1 = int("".join(map(str, w1)), 2)
            if len(sig) == 1:
                table[c1, 0] = cls_index[class_of(sig, (w1,))]
            else:
                for w2 in itertools.product(range(2), repeat=k[1]):
                    c2 = int("".join(map(str, w2)), 2)
                    table[c1, c2] = cls_index[class_of(sig, (w1, w2))]
        for cover in cvs:
            codes = []
            for cycle in cover:
                code = np.zeros(len(assignments), dtype=np.uint8)
                for i in range(len(cycle)):
                    bit = EDGE_INDEX[(cycle[i], cycle[(i + 1) % len(cycle)])]
                    code = 2 * code + ((assignments >> bit) & 1).astype(np.uint8)
                codes.append(code)
            if len(codes) == 1:
                codes.append(np.zeros(len(assignments), dtype=np.uint8))
            hist[assignments, table[codes[0], codes[1]]] += 1
        blocks.append(hist)
        name = "".join(map(str, sig))
        labels.extend([f"{name}:" + "|".join("".join(map(str, w)) for w in c) for c in cls])
        targets.extend(targets_for(sig, provider))
        denominators.extend([len(cvs)] * len(cls))
        sig_of_row.extend([sig] * len(cls))
    counts = np.concatenate(blocks, axis=1)
    return counts, labels, targets, np.array(denominators), sig_of_row, cover_lists


@njit
def check_direct(cover_edges, cover_lengths, cover_ncycles, cover_weights, scaled_bound):
    """Independent integer check without histogram construction/compression."""
    minimum = 2 ** 40
    tight = 0
    for assignment in range(1 << 20):
        value = 0
        for j in range(cover_edges.shape[0]):
            code = 0
            for c in range(2):  # a missing second cycle has length 0 and word 0
                word = 0
                for k in range(cover_lengths[j, c]):
                    word = 2 * word + ((assignment >> cover_edges[j, c, k]) & 1)
                code = code * 32 + word
            value += cover_weights[j, code]
        value -= scaled_bound
        if value < minimum:
            minimum = value
        if value == 0:
            tight += 1
    return minimum, tight


def direct_verification(signatures, labels, coefficients, denominators, integer_scale, bound):
    cover_edges, cover_lengths, cover_ncycles, cover_weights = [], [], [], []
    row = 0
    for sig in signatures:
        cls = classes(sig)
        cls_index = {c: i for i, c in enumerate(cls)}
        cvs = covers(sig)
        multiplier = integer_scale // len(cvs)
        weights = np.zeros(1024, dtype=np.int64)
        k = list(sig) + [0] * (2 - len(sig))
        for w1 in itertools.product(range(2), repeat=k[0]):
            c1 = int("".join(map(str, w1)), 2)
            if len(sig) == 1:
                weights[c1 * 32 + 0] = multiplier * coefficients[row + cls_index[class_of(sig, (w1,))]]
            else:
                for w2 in itertools.product(range(2), repeat=k[1]):
                    c2 = int("".join(map(str, w2)), 2)
                    weights[c1 * 32 + c2] = multiplier * coefficients[row + cls_index[class_of(sig, (w1, w2))]]
        for cover in cvs:
            edges = np.zeros((2, 5), dtype=np.int64)
            lengths = np.zeros(2, dtype=np.int64)
            for c, cycle in enumerate(cover):
                lengths[c] = len(cycle)
                for i in range(len(cycle)):
                    edges[c, i] = EDGE_INDEX[(cycle[i], cycle[(i + 1) % len(cycle)])]
            cover_edges.append(edges)
            cover_lengths.append(lengths)
            cover_ncycles.append(len(cover))
            cover_weights.append(weights.copy())
        row += len(cls)
    return check_direct(np.array(cover_edges), np.array(cover_lengths),
                        np.array(cover_ncycles), np.array(cover_weights), integer_scale * bound)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("full", "single"), default="full")
    parser.add_argument("--search", action="store_true", help="Rediscover small integer coefficients")
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--max-coefficient", type=int, default=30)
    args = parser.parse_args()
    signatures = FULL if args.family == "full" else SINGLE
    counts, labels, targets, denominators, sig_of_row, cover_lists = build(signatures)
    unique = np.unique(counts, axis=0)
    vertices = unique / denominators
    target = np.array([float(t) for t in targets])
    size = len(labels) + 1
    recorded = RECORDED[args.family]
    coefficients, bound = recorded["coefficients"], recorded["bound"]
    search_status = "not requested; verify recorded integer coefficients"
    if args.search or coefficients is None:
        homogeneous = np.c_[vertices, -np.ones(len(vertices))]
        identity = np.eye(size)
        matrix = np.vstack([
            np.c_[homogeneous, np.zeros_like(homogeneous)],
            np.r_[target, -1.0, np.zeros(size)][None, :],
            np.c_[identity, -identity],
            np.c_[-identity, -identity],
        ])
        lower = np.r_[np.zeros(len(vertices)), -np.inf, np.full(2 * size, -np.inf)]
        upper = np.r_[np.full(len(vertices), np.inf), -args.margin, np.zeros(2 * size)]
        low_bounds = np.r_[np.full(size, -args.max_coefficient), np.zeros(size)]
        high_bounds = np.full(2 * size, args.max_coefficient)
        # One normalization freedom per signature: its frequencies sum to one
        # for every assignment and its targets sum to one. Fix the first row
        # of each signature to zero; this imposes nothing on the data.
        first_rows = [i for i in range(len(labels)) if i == 0 or sig_of_row[i] != sig_of_row[i - 1]]
        low_bounds[first_rows] = 0
        high_bounds[first_rows] = 0
        result = milp(np.r_[np.zeros(size), np.ones(size)],
                      integrality=np.r_[np.ones(size), np.zeros(size)],
                      bounds=Bounds(low_bounds, high_bounds),
                      constraints=LinearConstraint(matrix, lower, upper),
                      options={"time_limit": args.time_limit})
        if result.x is None:
            raise RuntimeError(result.message)
        rational = [Fraction(float(x)).limit_denominator(10000) for x in result.x[:size]]
        scale = math.lcm(*(x.denominator for x in rational))
        coefficients = [int(x * scale) for x in rational[:-1]]
        bound = int(rational[-1] * scale)
        search_status = result.message
    integer_scale = math.lcm(*(len(c) for c in cover_lists))
    edge_weights = np.array([c * integer_scale // d for c, d in zip(coefficients, denominators)], dtype=np.int64)
    slack = counts.astype(np.int64) @ edge_weights - integer_scale * bound
    exact_value = sp.simplify(sum(c * t for c, t in zip(coefficients, targets)))
    direct_minimum, direct_tight = direct_verification(signatures, labels, coefficients, denominators, integer_scale, bound)
    report = {
        "case": f"nsi-{args.family}-cycle-cover-five-source-witness",
        "signatures": [list(s) for s in signatures],
        "assignments_checked": len(counts),
        "unique_histograms": len(unique),
        "cover_counts": [len(c) for c in cover_lists],
        "labels": labels,
        "coefficients": coefficients,
        "bound": bound,
        "nonzero_terms": {l: c for l, c in zip(labels, coefficients) if c},
        "targets": [str(t) for t in targets],
        "exact_value": str(exact_value),
        "exact_violation": str(sp.simplify(bound - exact_value)),
        "float_violation": float(bound - exact_value),
        "minimum_integer_slack": int(slack.min()),
        "tight_assignments": int(np.count_nonzero(slack == 0)),
        "integer_slack_scale": integer_scale,
        "independent_direct_minimum_integer_slack": int(direct_minimum),
        "independent_direct_tight_assignments": int(direct_tight),
        "solver": f"scipy.optimize.milp (HiGHS), {args.time_limit:g} second search limit, margin {args.margin}",
        "search_status": search_status,
        "versions": {"numpy": np.__version__, "sympy": sp.__version__,
                     "scipy": __import__("scipy").__version__,
                     "numba": __import__("numba").__version__,
                     "python": sys.version.split()[0]},
    }
    if slack.min() < 0 or not bool(exact_value < bound):
        raise AssertionError("Candidate failed exact validity or violation check")
    if direct_minimum < 0:
        raise AssertionError("Independent direct check failed")
    if (int(slack.min()), int(np.count_nonzero(slack == 0))) != (direct_minimum, direct_tight):
        raise AssertionError("Two exact checks disagree")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
