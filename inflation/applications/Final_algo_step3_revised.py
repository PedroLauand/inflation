# step3_register_rhs.py
# ------------------------------------------------------------
# Step 3:
#   Given a marginal event (list of [1,i,j,0,a]) and a symmetry group G_raw,
#   enumerate all global extensions, canonicalize each extension to its
#   lex-min orbit representative, and tally coefficients (multiplicities).
#
# Example at bottom: marginal (A_00=0, A_11=0) for n=2, outcomes=2.
# ------------------------------------------------------------

from __future__ import annotations
from typing import List, Dict, Tuple, Union, Any
from collections import Counter
from itertools import product
import numpy as np
import sys

# --- your Inflation import style ---
sys.path.append('/Users/pedrolauand/My_Code/Inflation/inflation')
from inflation import InflationProblem  # type: ignore

# --- SymPy group tools ---
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.perm_groups import PermutationGroup


# ============================================================
# Inflation problem (minimal; adapt if you need your modified version)
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
# Global extension enumeration (from your pipeline)
# ============================================================
def iterate_global_events_containing_marginal(
    n: int,
    outcomes: int,
    marginal: List[List[int]],
):
    """
    Yield all compact global events (length n^2, row-major over (i,j)) that extend the marginal.
    marginal format: [[1,i,j,0,a], ...] with i,j 1-based.
    """
    fixed: Dict[int, int] = {}
    for (_one, i, j, _zero, a) in marginal:
        si = (i - 1) * n + (j - 1)  # slot index for A_{i,j}
        if si in fixed and fixed[si] != a:
            return  # inconsistent
        fixed[si] = a

    Nslots = n * n
    remaining = [s for s in range(Nslots) if s not in fixed]

    for combo in product(range(outcomes), repeat=len(remaining)):
        evt = [0] * Nslots
        for s, a in fixed.items():
            evt[s] = a
        for s, a in zip(remaining, combo):
            evt[s] = a
        yield evt


# ============================================================
# One-hot lex representation + lex compare
# ============================================================
def to_lex_representation(evt: List[int], outcomes: int) -> List[int]:
    """Compact event (length n^2) -> one-hot vector (length n^2*outcomes)."""
    n2 = len(evt)
    N = n2 * outcomes
    lex = [0] * N
    for s, k in enumerate(evt):
        lex[k + outcomes * s] = 1
    return lex

def from_lex_representation(lex_evt: List[int], outcomes: int) -> List[int]:
    """One-hot vector -> compact event."""
    N = len(lex_evt)
    n2 = N // outcomes
    evt = [0] * n2
    for s in range(n2):
        block = lex_evt[s * outcomes : (s + 1) * outcomes]
        evt[s] = block.index(1)
    return evt

def lex_less(a: List[int], b: List[int]) -> bool:
    for x, y in zip(a, b):
        if x < y:
            return True
        if x > y:
            return False
    return False


# ============================================================
# Group build + action (same convention as your pipeline)
# ============================================================
def _normalize_perm_list(g: List[int], N: int) -> List[int]:
    if len(g) != N:
        raise ValueError(f"Permutation length {len(g)} != {N}")
    if max(g) == N:  # 1-based
        g = [x - 1 for x in g]
    if sorted(g) != list(range(N)):
        raise ValueError("Invalid permutation (not a bijection).")
    return g

def build_sympy_group(raw_G: Union[List[List[int]], np.ndarray], N: int) -> PermutationGroup:
    raw_list = raw_G.tolist() if isinstance(raw_G, np.ndarray) else raw_G
    gens = [Permutation(_normalize_perm_list(gl, N)) for gl in raw_list] or [Permutation(list(range(N)))]
    return PermutationGroup(gens)

def apply_perm_to_lex(lex_evt: List[int], perm: Permutation) -> List[int]:
    """out[perm(p)] = lex_evt[p]"""
    N = len(lex_evt)
    out = [0] * N
    for p in range(N):
        out[perm(p)] = lex_evt[p]
    return out


# ============================================================
# Step 2 canonicalizers (smart + brute)
# ============================================================
def prepare_group_chain(G: PermutationGroup, N: int) -> Dict[str, object]:
    G.schreier_sims()
    return {
        "G": G,
        "base": list(G.base),
        "basic_orbits": list(G.basic_orbits),
        "basic_transversals": list(G.basic_transversals),
        "N": N,
    }

def canonical_rep_stabchain(evt: List[int], outcomes: int, precomp: Dict[str, object]) -> List[int]:
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

def canonical_rep_bruteforce(evt: List[int], outcomes: int, G: PermutationGroup) -> List[int]:
    x = to_lex_representation(evt, outcomes)
    best = None
    for g in G.generate_schreier_sims():
        y = apply_perm_to_lex(x, g)
        if best is None or lex_less(y, best):
            best = y
    if best is None:
        raise RuntimeError("Group enumeration returned nothing.")
    return from_lex_representation(best, outcomes)


# ============================================================
# Step 3: build RHS coefficients for one marginal
# ============================================================
def step3_rhs_coeffs_for_marginal(
    n: int,
    outcomes: int,
    marginal: List[List[int]],
    *,
    G: PermutationGroup,
    precomp: Dict[str, object],
    method: str = "stabchain",  # "stabchain" or "bruteforce"
) -> Counter[Tuple[int, ...]]:
    """
    Returns a Counter mapping canonical global event reps (tuple length n^2) -> coefficient.
    """
    counts: Counter[Tuple[int, ...]] = Counter()

    for evt in iterate_global_events_containing_marginal(n, outcomes, marginal):
        print("evt:",evt)
        if method == "stabchain":
            rep = canonical_rep_stabchain(evt, outcomes, precomp)
            print("rep:",rep)
        elif method == "bruteforce":
            rep = canonical_rep_bruteforce(evt, outcomes, G)
        else:
            raise ValueError("method must be 'stabchain' or 'bruteforce'")
        counts[tuple(rep)] += 1

    return counts


# ============================================================
# Example: marginal (A_00=0, A_11=0) with n=2, outcomes=2
# ============================================================
if __name__ == "__main__":
    n = 3
    outcomes = 2

    prob = ring_problem(n, outcomes)

    # If you want the same symmetry enrichment as your pipeline:
    #prob.add_symmetries(prob._setting_specific_outcome_relabelling_symmetries)

    raw_G = prob.symmetries
    N = (n * n) * outcomes
    G = build_sympy_group(raw_G, N)
    precomp = prepare_group_chain(G, N)

    # Marginal event: (A_00=0, A_11=0)
    # In your [1,i,j,0,a] convention with 1-based i,j:
    marginal = [
        [1, 1, 2, 0, 0],  # A_{1,2}=0  (A_01 in 0-based)
        [1, 2, 1, 0, 0],  # A_{2,1}=0  (A_10 in 0-based)
        [1, 3, 3, 0, 0],  # A_{3,3}=0  (A_22 in 0-based)
    ]

    rhs = step3_rhs_coeffs_for_marginal(
        n, outcomes, marginal,
        G=G, precomp=precomp, method="stabchain"
    )

    # Print as an equation over Q events in compact row-major order:
    # compact evt for n=2 is [A00, A01, A10, A11]
    print("\nMarginal: (A01=0, A10=0, A22=0)")
    print("RHS coefficients (rep -> coeff):")
    print(len(raw_G))
    for rep, coeff in sorted(rhs.items()):
        print(f"  {coeff} * Q{rep}")

    # Optional sanity check: compare with brute force for this small case
    #rhs_bf = step3_rhs_coeffs_for_marginal(
       # n, outcomes, marginal,
        #G=G, precomp=precomp, method="bruteforce"
    #)
   # print("\nBrute force agrees with stabchain?", rhs_bf == rhs)
