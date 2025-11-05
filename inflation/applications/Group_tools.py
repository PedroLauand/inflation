"""
Canonical leximinization of global events under a permutation group acting on
the one-hot (lexicographic) representation.

This file separates:
  (1) Group construction & preprocessing (build once),
  (2) Canonicalization routines (reuse many times).

SymPy version agnostic:
- We never pass a 'base' argument to schreier_sims (older SymPy didn't support it).
- We rely on the attributes exposed on the group after schreier_sims():
    G.base, G.basic_orbits, G.basic_transversals
"""

from typing import Iterable, List, Tuple, Dict, Any

from sympy.combinatorics.permutations import Permutation # type: ignore
from sympy.combinatorics.perm_groups import PermutationGroup # pyright: ignore[reportMissingModuleSource]


# =========================================================
# Event <-> One-hot (lex) helpers
# =========================================================

def to_lex_representation(evt: List[int], outcomes: int) -> List[int]:
    """
    Compact global event -> one-hot lex representation.

    evt: length n^2; evt[s] is the chosen outcome k for operator slot s (row-major over (i,j)).
    outcomes: number of outcomes m.

    Returns a length N = n^2 * m one-hot vector laid out in blocks of size m per operator slot:
      [A^(1,1)=0, ..., A^(1,1)=m-1, A^(1,2)=0, ..., ..., A^(n,n)=m-1]
    """
    n2 = len(evt)
    N = n2 * outcomes
    lex = [0] * N
    for s, k in enumerate(evt):
        if not (0 <= k < outcomes):
            raise ValueError(f"Outcome {k} at slot {s} out of 0..{outcomes-1}")
        lex[k + outcomes * s] = 1
    return lex


def from_lex_representation(lex_evt: List[int], outcomes: int) -> List[int]:
    """
    One-hot lex representation -> compact global event.
    Validates one-hotness in each block.
    """
    N = len(lex_evt)
    if N % outcomes != 0:
        raise ValueError("Length of one-hot vector must be divisible by outcomes")
    n2 = N // outcomes
    evt = [0] * n2
    for s in range(n2):
        block = lex_evt[s * outcomes : (s + 1) * outcomes]
        if sum(block) != 1:
            raise ValueError(f"Block {s} is not one-hot: {block}")
        evt[s] = block.index(1)
    return evt


def lex_less(a: List[int], b: List[int]) -> bool:
    """Return True iff a < b in lexicographic order (short-circuit compare)."""
    for x, y in zip(a, b):
        if x < y:
            return True
        if x > y:
            return False
    return False  # equal


# =========================================================
# Group plumbing (build once, reuse many times)
# =========================================================

def _normalize_perm_list(g: List[int], N: int) -> List[int]:
    """
    Make a raw permutation list 0-based, length N, bijection of range(N).
    Accepts either 0-based [0..N-1] or 1-based [1..N].
    """
    if len(g) != N:
        raise ValueError(f"Permutation length {len(g)} != {N}")
    if max(g) == N:  # looks 1-based
        g = [x - 1 for x in g]
    if sorted(g) != list(range(N)):
        raise ValueError("Invalid permutation (not a bijection of 0..N-1)")
    return g


def _perm_list_to_sympy(g: List[int], N: int) -> Permutation:
    """Convert a 0- or 1-based mapping list to a SymPy Permutation on 0..N-1."""
    g0 = _normalize_perm_list(g, N)
    return Permutation(g0)


def build_sympy_group(G_lists: Iterable[List[int]], N: int) -> PermutationGroup:
    """
    Build a SymPy PermutationGroup from raw permutations (lists).
    You may pass either a generating set or the full list of elements.
    """
    gens = [_perm_list_to_sympy(gl, N) for gl in G_lists]
    if not gens:  # trivial group with identity
        gens = [Permutation(list(range(N)))]
    return PermutationGroup(gens)


def prepare_group_chain(G: PermutationGroup, N: int) -> Dict[str, Any]:
    """
    Run Schreier–Sims ONCE to populate stabilizer-chain data for G.
    Uses SymPy's default base (no 'base=' kwarg).

    Returns a dict with the precomputed structures you can pass to the leximin function:
      {
        "base": List[int],                                 # chain base (coordinate order used)
        "basic_orbits": List[List[int]],                   # per-level orbits
        "basic_transversals": List[Dict[int, Permutation]] # per-level coset reps: point -> perm
      }
    """
    # Compute stabilizer chain with SymPy's default base
    G.schreier_sims()

    # Extract and return the precomputed data (older/newer SymPy expose these on G)
    return {
        "base": list(G.base),
        "basic_orbits": list(G.basic_orbits),
        "basic_transversals": list(G.basic_transversals), # type: ignore
    }


def perm_to_list(p: Permutation, N: int) -> List[int]:
    """SymPy Permutation -> 0-based mapping list on 0..N-1."""
    return [p(i) for i in range(N)]


def apply_perm_to_lex(lex_evt: List[int], perm: Permutation) -> List[int]:
    """Apply a permutation to a one-hot vector: out[perm(p)] = lex_evt[p]."""
    N = len(lex_evt)
    out = [0] * N
    for p in range(N):
        out[perm(p)] = lex_evt[p]
    return out


# =========================================================
# Canonical leximin — (A) naive orbit scan
# =========================================================

def canonical_naive_leximin(
    evt: List[int],
    outcomes: int,
    G: PermutationGroup,
) -> Tuple[List[int], Permutation]:
    """
    Lexicographically minimal image of 'evt' under the full group G,
    comparing in one-hot space. Enumerates the whole group
    (feasible only if |G| is small).
    """
    x = to_lex_representation(evt, outcomes)
    N = len(x)

    best_vec = x
    best_g = Permutation(list(range(N)))  # identity

    for g in G.generate_schreier_sims():
        y = apply_perm_to_lex(x, g)
        if lex_less(y, best_vec):
            best_vec = y
            best_g = g

    rep_evt = from_lex_representation(best_vec, outcomes)
    return rep_evt, best_g


# =========================================================
# Canonical leximin — (B) prefix-stabilizer with precomputed coset transversals
# =========================================================

def canonical_leximin_coset_chain(
    evt: List[int],
    outcomes: int,
    *,
    precomp: Dict[str, Any],
) -> Tuple[List[int], Permutation]:
    """
    Lexicographic canonical representative using a precomputed stabilizer chain.

    Parameters
    ----------
    evt : list[int]
        Compact global event (length n^2).
    outcomes : int
        Number of outcomes m.
    precomp : dict
        Output of prepare_group_chain(...), containing:
          - "base": List[int]
          - "basic_orbits": List[List[int]]
          - "basic_transversals": List[Dict[int,Permutation]]

    Returns
    -------
    (rep_evt, g_star):
        rep_evt : list[int]     # compact representative event
        g_star  : Permutation   # achieving permutation (product of chosen transversals)

    Notes
    -----
    - This function does not call schreier_sims() and does not touch the group.
      It is safe/cheap to call repeatedly in a hot loop.
    - Canonicalization follows the order given by precomp["base"] (the group's chain base).
    """
    x = to_lex_representation(evt, outcomes)
    N = len(x)

    base = precomp["base"]
    basic_orbits = precomp["basic_orbits"]
    basic_transversals = precomp["basic_transversals"]

    # Witness permutation and current best image
    g_star = Permutation(list(range(N)))     # identity
    best_image = x                           # current g_star · x

    # Walk the chain: one representative per left coset at each level
    levels = len(base)
    for t in range(levels):
        orbit_t = basic_orbits[t]
        transv_t = basic_transversals[t]

        cand_vec = None
        cand_U = None

        for u in orbit_t:
            U = transv_t[u]                       # transversal fixing previous base points
            y = apply_perm_to_lex(best_image, U)  # type: ignore # candidate image
            if (cand_vec is None) or lex_less(y, cand_vec):
                cand_vec = y
                cand_U = U

        if cand_U is not None:
            g_star = cand_U * g_star
            best_image = cand_vec

    rep_evt = from_lex_representation(best_image, outcomes) # type: ignore
    return rep_evt, g_star


# =========================================================
# Example usage & quick test (optional)
# =========================================================
'''if __name__ == "__main__":
    # Example: n=2 (n^2=4 ops), outcomes=2 -> one-hot length N=8
    n ,outcomes = 2,2
    N = (n * n) * outcomes

    # Example compact global event
    evt = [0, 1, 1, 0]

    # Small group on N=8 coordinates (0-based permutations)
    raw_G = [
        [0, 1, 2, 3, 4, 5, 6, 7],        # identity
        [1, 0, 3, 2, 5, 4, 7, 6],        # swap within each outcome block
        [6, 7, 4, 5, 2, 3, 0, 1],        # reverse block order
        [7, 6, 5, 4, 3, 2, 1, 0],        # full reversal
    ]

    # Build group ONCE and precompute chain data ONCE
    G = build_sympy_group(raw_G, N)
    precomp = prepare_group_chain(G, N)
    print("Chain base (group-chosen):", precomp["base"])

    # Naive (for small groups)
    rep_naive, g_naive = canonical_naive_leximin(evt, outcomes, G)
    print("Naive   representative:", rep_naive)
    print("Naive   achieving perm:", perm_to_list(g_naive, N))

    # Coset-chain (hot path; no recomputation)
    rep_chain, g_chain = canonical_leximin_coset_chain(evt, outcomes, precomp=precomp)
    print("Chain   representative:", rep_chain)
    print("Chain   achieving perm:", perm_to_list(g_chain, N))'''
