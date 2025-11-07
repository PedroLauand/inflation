# inflation_pipeline.py
from typing import Dict, List, Tuple
from collections import defaultdict
from math import prod

# --- your existing modules (as-is) ---
from EJM_distribution import loop_prob_event
from EJM_utils import iterate_global_events_containing_marginal
import Group_tools.py as gt
'''from Group_tools.py import (
    build_sympy_group,
    prepare_group_chain,
    canonical_leximin_coset_chain,
)'''
from marginal_generator.py import generate_minimal_marginal_events


# ---------------------------
# Small helpers (hashable keys)
# ---------------------------
def serialize_marginal(marginal: List[List[int]]) -> Tuple[Tuple[int, ...], ...]:
    """Hashable key for [[1,i,j,0,a], ...]."""
    return tuple(tuple(row) for row in marginal)

def serialize_evt(evt: List[int]) -> Tuple[int, ...]:
    """Hashable key for a compact global event (length n^2)."""
    return tuple(evt)


# ---------------------------
# Read structure from a marginal
# ---------------------------
def marginal_to_J_and_a(marginal: List[List[int]]) -> Tuple[List[int], List[int]]:
    """
    From [[1,i,j,0,a], ...] with i=1..n each once, return:
      J (1-based one-line) with J[i-1] = j
      a with a[i-1] = a_i
    """
    n = len(marginal)
    J = [0] * n
    a = [0] * n
    for _, i, j, _, ai in marginal:
        J[i - 1] = j
        a[i - 1] = ai
    return J, a

def permutation_cycles_1based(J: List[int]) -> List[List[int]]:
    """Disjoint cycles of a 1-based one-line permutation J; fixed points as [i]."""
    n = len(J)
    seen = [False] * (n + 1)
    cycles: List[List[int]] = []
    for s in range(1, n + 1):
        if not seen[s]:
            cur = s
            cyc = []
            while not seen[cur]:
                seen[cur] = True
                cyc.append(cur)
                cur = J[cur - 1]
            cycles.append(cyc)
    return cycles


# ---------------------------
# Factorized value via loop_prob_event
# ---------------------------
def factorized_marginal_value(marginal: List[List[int]]) -> float:
    """
    Value(marginal) = ∏_cycles loop_prob_event(outcomes_along_cycle).
    Cycles are from the permutation J encoded by the marginal.
    """
    J, a = marginal_to_J_and_a(marginal)
    parts = []
    for cyc in permutation_cycles_1based(J):
        outcomes_tuple = tuple(a[i - 1] for i in cyc)
        parts.append(loop_prob_event(outcomes_tuple))
    return float(prod(parts))


# ---------------------------
# Count canonical reps of global extensions (under group G)
# ---------------------------
def global_extension_reps_counts(
    n: int,
    outcomes: int,
    marginal: List[List[int]],
    precomp: Dict,
) -> Dict[Tuple[int, ...], int]:
    """
    Iterate all global extensions of 'marginal', canonicalize each under G (precomp),
    and return {canonical_rep_evt : count}.
    """
    counts: Dict[Tuple[int, ...], int] = defaultdict(int)
    for evt in iterate_global_events_containing_marginal(n, outcomes, marginal):
        rep_evt, _ = gt.canonical_leximin_coset_chain(evt, outcomes, precomp=precomp)
        counts[serialize_evt(rep_evt)] += 1
    return dict(counts)


# ---------------------------
# Orchestration helpers
# ---------------------------
def build_group_and_precomp(raw_G: List[List[int]], N: int):
    """
    From raw permutations on N positions (0- or 1-based):
    build SymPy group and precompute the stabilizer chain.
    Returns (G, precomp_dict).
    """
    G = gt.build_sympy_group(raw_G, N)
    precomp = gt.prepare_group_chain(G, N)
    return G, precomp

def generate_all_marginals(n: int, outcomes: int) -> List[List[List[int]]]:
    """
    Minimal marginal list, reduced by:
      - S_n conjugacy (operator structure),
      - outcome relabeling classes (equality patterns).
    Exact format: [[1,i,j,0,a], ...].
    """
    return generate_minimal_marginal_events(n, outcomes)

def process_marginal(
    n: int,
    outcomes: int,
    marginal: List[List[int]],
    precomp: Dict,
) -> List[Dict]:
    """
    Build one entry: [e1, e2], where
      e1 = { marginal : value }
      e2 = { canonical_global_rep : count }
    """
    e1 = {serialize_marginal(marginal): factorized_marginal_value(marginal)}
    e2 = global_extension_reps_counts(n, outcomes, marginal, precomp)
    return [e1, e2]

def run_pipeline(
    n: int,
    outcomes: int,
    raw_G: List[List[int]],
) -> List[List[Dict]]:
    """
    Full pipeline:
      1) build group + precompute chain,
      2) generate minimal marginals,
      3) produce [ {marginal:value}, {rep:count} ] per marginal.
    """
    N = (n * n) * outcomes
    _G, precomp = gt.build_group_and_precomp(raw_G, N)

    results: List[List[Dict]] = []
    for marginal in generate_all_marginals(n, outcomes):
        results.append(process_marginal(n, outcomes, marginal, precomp))
    return results
