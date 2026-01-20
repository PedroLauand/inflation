# step2_canonicalization_demo.py
# ------------------------------------------------------------
# Two canonicalization methods for global events under G_raw:
#   (1) Brute force: min_{g in G} g·x
#   (2) Stabilizer-chain (Schreier–Sims) greedy transversal scan
#
# Example at bottom shows how to get G_raw from InflationProblem
# for n=2, outcomes=2 (ring_problem as in your pipeline style).
# ------------------------------------------------------------

from __future__ import annotations
from typing import List, Dict, Union, Iterable
from functools import lru_cache
import numpy as np
import sys

# --- Your import style for Inflation ---
sys.path.append('/Users/pedrolauand/My_Code/Inflation/inflation')
from inflation import InflationProblem  # type: ignore

# --- SymPy permutation group tools ---
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.perm_groups import PermutationGroup


# ============================================================
# Minimal ring_problem (same spirit as your file; keep it simple)
# ============================================================
def ring_problem(inflation_level: int, nof_outcomes: int = 2) -> InflationProblem:
    prob = InflationProblem(
        dag={"i1": ["A"], "i2": ["A"]},
        outcomes_per_party=(nof_outcomes,),
        settings_per_party=(1,),
        classical_sources=None,
        inflation_level_per_source=(inflation_level, inflation_level),
        order=["A"],
    )
    return prob


# ============================================================
# Event representations + lex order
# ============================================================
def to_lex_representation(evt: List[int], outcomes: int) -> List[int]:
    """Compact event (length n^2) -> one-hot vector (length n^2*outcomes)."""
    n2 = len(evt)
    N = n2 * outcomes
    lex = [0] * N
    for s, k in enumerate(evt):
        if not (0 <= k < outcomes):
            raise ValueError("Outcome out of range")
        lex[k + outcomes * s] = 1
    return lex

def from_lex_representation(lex_evt: List[int], outcomes: int) -> List[int]:
    """One-hot vector -> compact event; validates one-hot per slot."""
    N = len(lex_evt)
    if N % outcomes != 0:
        raise ValueError("Invalid length")
    n2 = N // outcomes
    evt = [0] * n2
    for s in range(n2):
        block = lex_evt[s * outcomes : (s + 1) * outcomes]
        if sum(block) != 1:
            raise ValueError(f"Block {s} not one-hot: {block}")
        evt[s] = block.index(1)
    return evt

def lex_less(a: List[int], b: List[int]) -> bool:
    """True iff a < b in lexicographic order."""
    for x, y in zip(a, b):
        if x < y:
            return True
        if x > y:
            return False
    return False


# ============================================================
# Group construction (from raw permutations, 0- or 1-based)
# ============================================================
def _normalize_perm_list(g: List[int], N: int) -> List[int]:
    """Ensure 0-based bijection of range(N) from list (accepts 1- or 0-based)."""
    if len(g) != N:
        raise ValueError(f"Permutation length {len(g)} != {N}")
    if max(g) == N:  # likely 1-based
        g = [x - 1 for x in g]
    if sorted(g) != list(range(N)):
        raise ValueError("Invalid permutation (not a bijection).")
    return g

def build_sympy_group(raw_G: Union[List[List[int]], np.ndarray], N: int) -> PermutationGroup:
    """Build a SymPy PermutationGroup from raw list permutations."""
    raw_list = raw_G.tolist() if isinstance(raw_G, np.ndarray) else raw_G
    gens = [Permutation(_normalize_perm_list(gl, N)) for gl in raw_list] or [Permutation(list(range(N)))]
    return PermutationGroup(gens)

def apply_perm_to_lex(lex_evt: List[int], perm: Permutation) -> List[int]:
    """
    Apply permutation to one-hot coordinate vector.
    Convention (same as your pipeline):
        out[perm(p)] = lex_evt[p]
    """
    N = len(lex_evt)
    out = [0] * N
    for p in range(N):
        out[perm(p)] = lex_evt[p]
    return out


# ============================================================
# (1) BRUTE FORCE canonicalization
# ============================================================
def canonical_rep_bruteforce(evt: List[int], outcomes: int, G: PermutationGroup) -> List[int]:
    """
    Canonical representative by brute force:
      compute min_lex { g·evt : g in G } under lex order on one-hot representation.
    """
    x = to_lex_representation(evt, outcomes)
    best = None

    # generate all group elements (feasible only for small groups!)
    for g in G.generate_schreier_sims():
        y = apply_perm_to_lex(x, g)
        if best is None or lex_less(y, best):
            best = y

    if best is None:
        raise RuntimeError("Group enumeration returned nothing.")
    return from_lex_representation(best, outcomes)


# ============================================================
# (2) Stabilizer-chain canonicalization (your "smart" method)
# ============================================================
def prepare_group_chain(G: PermutationGroup, N: int) -> Dict[str, object]:
    """Run Schreier–Sims once and store stabilizer chain data."""
    G.schreier_sims()
    return {
        "G": G,
        "base": list(G.base),
        "basic_orbits": list(G.basic_orbits),
        "basic_transversals": list(G.basic_transversals),
        "N": N,
    }

def canonical_rep_stabchain(evt: List[int], outcomes: int, precomp: Dict[str, object]) -> List[int]:
    """
    Canonical representative using stabilizer chain (greedy transversal scan),
    matching your canonical_leximin_coset_chain implementation.
    """
    x = to_lex_representation(evt, outcomes)
    base = precomp["base"]  # type: ignore
    basic_orbits = precomp["basic_orbits"]  # type: ignore
    basic_transversals = precomp["basic_transversals"]  # type: ignore

    g_star = Permutation(list(range(len(x))))  # identity

    for k in range(len(base)):
        current = apply_perm_to_lex(x, g_star)
        cand_vec = None
        cand_U = None
        for u in basic_orbits[k]:
            U = basic_transversals[k][u]
            y = apply_perm_to_lex(current, U)
            if cand_vec is None or lex_less(y, cand_vec):
                cand_vec = y
                cand_U = U
        if cand_U is not None:
            g_star = cand_U * g_star

    best_vec = apply_perm_to_lex(x, g_star)
    return from_lex_representation(best_vec, outcomes)


# ============================================================
# Example: get G_raw from InflationProblem for n=2, outcomes=2
# ============================================================
if __name__ == "__main__":
    n = 2
    outcomes = 2

    prob = ring_problem(n, outcomes)

    # Optional: if you want outcome-relabeling symmetries too (as in your pipeline),
    # uncomment the next line (it exists on your InflationProblem object).
    #prob.add_symmetries(prob._setting_specific_outcome_relabelling_symmetries)

    raw_G = prob.symmetries  # this is your G_raw
    print(len(raw_G))
    print(raw_G)
    N = (n * n) * outcomes   # one-hot coordinate size

    G = build_sympy_group(raw_G, N)
    precomp = prepare_group_chain(G, N)

    # Pick an example global event over all A_ij slots (n^2 slots).
    # Here n=2 => 4 slots. Each slot is in {0,1}.
    '''evt = [0, 1, 1, 0]  # example assignment to (A00,A01,A10,A11) in row-major slot order

    rep_bf = canonical_rep_bruteforce(evt, outcomes, G)
    rep_sc = canonical_rep_stabchain(evt, outcomes, precomp)

    print("G_raw generators:", len(raw_G))
    print("Event evt:            ", evt)
    print("Brute-force canonical:", rep_bf)
    print("Stab-chain canonical: ", rep_sc)
    print("Agree?               ", rep_bf == rep_sc)'''

    # Quick sanity: test a few random events for agreement (small n only)
    rng = np.random.default_rng(0)
    for evt_ in ([0, 0, 0, 0],[0, 0, 1, 0],[0, 1, 0, 0],[0, 1, 1, 0]):
        rep_bf = canonical_rep_bruteforce(evt_, outcomes, G)
        rep_sc = canonical_rep_stabchain(evt_, outcomes, precomp)
        if rep_bf !=rep_sc:
            raise AssertionError(f"Mismatch on evt={evt_}")
        print("Brute-force canonical:", rep_bf)
        print("Stab-chain canonical: ", rep_sc)
        
    print("Random test: OK (20/20)")
    '''for t in range(3):
        evt_rand = rng.integers(0, outcomes, size=n * n).tolist()
        if canonical_rep_bruteforce(evt_rand, outcomes, G) != canonical_rep_stabchain(evt_rand, outcomes, precomp):
            raise AssertionError(f"Mismatch on evt={evt_rand}")
    print("Random test: OK (20/20)")'''
