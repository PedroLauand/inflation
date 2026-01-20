# revised_step1_polytypes.py
# -------------------------------------------------------------------
# Step 1 (revised):
#   Given n (inflation level), enumerate polynomial-types ("polytypes")
#   as cycle-structures (integer partitions of n, padded with zeros),
#   and for each polytype list ALL marginals (variable-sets) that realize it.
#
# Representation of a marginal variable-set:
#   A marginal is a list of monomials in the lex-style you use:
#       [1, i, j, 0, a_fixed]
#   meaning the variable A_{i,j} (i,j are 1-based) with fixed setting=0
#   and fixed outcome a_fixed (outcome patterns ignored for now).
#
# Output:
#   Dict[decomposition_tuple, List[marginal]]
#   where:
#     - decomposition_tuple has length n, e.g. (2,1,0) or (1,1,1)
#     - each marginal is List[List[int]] of shape (n, 5)
#
# Optional:
#   If you pass an InflationProblem `prob`, you can also request output in
#   Inflation "names" (via prob.mon_to_lexrepr and prob._lexrepr_to_names).
# -------------------------------------------------------------------

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
from itertools import permutations
from collections import defaultdict


def _cycle_partition_of_perm_1based(J: Tuple[int, ...]) -> List[int]:
    """
    J is a 1-line permutation on {1..n} given as a tuple of length n with values in 1..n.
    Returns cycle lengths sorted in nonincreasing order (an integer partition of n).
    """
    n = len(J)
    seen = [False] * (n + 1)
    parts: List[int] = []
    for start in range(1, n + 1):
        if seen[start]:
            continue
        v = start
        L = 0
        while not seen[v]:
            seen[v] = True
            v = J[v - 1]
            L += 1
        parts.append(L)
    parts.sort(reverse=True)
    return parts


def step1_polytype_variables(
    n: int,
    *,
    a_fixed: int = 0,
    prob: Optional[Any] = None,
    return_names: bool = False,
) -> Dict[Tuple[int, ...], List[Any]]:
    """
    Step 1:
    Group marginal variable-sets by polytype (cycle structure).

    Args:
      n: inflation level (number of source-copies per side).
      a_fixed: fixed outcome label to attach in each monomial [1,i,j,0,a_fixed].
               (We keep outcomes fixed for now as requested.)
      prob: optional InflationProblem. If provided and return_names=True,
            each marginal is returned as a list of Inflation 'names' instead of monomials.
      return_names: if True, uses Inflation naming:
            names = list(prob._lexrepr_to_names[prob.mon_to_lexrepr(marginal)])

    Returns:
      A dict mapping:
        decomposition (length-n tuple, e.g. (2,1,0))  ->  list of marginals
      where each marginal is either:
        - List[List[int]] with entries [1,i,J(i),0,a_fixed] for i=1..n, or
        - List[str] (Inflation names), if return_names=True and prob is provided.
    """
    if n <= 0:
        return {}

    if return_names and prob is None:
        raise ValueError("return_names=True requires prob=InflationProblem(...)")

    out: Dict[Tuple[int, ...], List[Any]] = defaultdict(list)

    # Enumerate ALL permutations J (this lists all marginals belonging to each polytype).
    for J in permutations(range(1, n + 1)):
        parts = _cycle_partition_of_perm_1based(J)
        decomp = tuple(parts + [0] * (n - len(parts)))  # pad with zeros to match your "2+0" style

        marginal_monomials = [[1, i, J[i - 1], 0, a_fixed] for i in range(1, n + 1)]

        if return_names:
            lexrepr = prob.mon_to_lexrepr(marginal_monomials)
            names = list(prob._lexrepr_to_names[lexrepr])
            out[decomp].append(names)
        else:
            out[decomp].append(marginal_monomials)

    return dict(out)


# -------------------------
# Example usage (without Inflation names)
# -------------------------
def _pretty_Aij(monomial: List[int]) -> str:
    # monomial = [1,i,j,0,a]
    _, i, j, _, _ = monomial
    return f"A_{{{i},{j}}}"


if __name__ == "__main__":
    for n in (1,4):
        d = step1_polytype_variables(n, a_fixed=0, prob=None, return_names=False)

        print(f"\n=== n={n} ===")
        for decomp in sorted(d.keys(), reverse=True):
            decomp_str = "+".join(map(str, decomp))
            print(f"Polytype {decomp_str}  (#marginals={len(d[decomp])})")

            # show a few marginals
            for marg in d[decomp][:min(6, len(d[decomp]))]:
                vars_str = "[" + ", ".join(_pretty_Aij(m) for m in marg) + "]"
                print("  ", vars_str)
