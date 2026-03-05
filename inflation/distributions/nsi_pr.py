# -*- coding: utf-8 -*-
"""
Infinite-precision (sympy) implementation of the Bancal–Gisin saturating construction.

Implements:
- Line correlators E_n via recursion Eq. (12)
- Loop correlators E^o_n via recursion Eq. (21)
- Probabilities:
    P_line([x1,...,xn])   open chain (no wrap)
    P_loop([x1,...,xn])   cycle (wrap), with full-set correlator = E^o_n

All internal arithmetic uses sympy exact types; conversion to float happens only at return,
and you can request exact sympy return values by setting as_float=False.

xi must be ±1 (Python int or sympy Integer).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List, Tuple

import sympy as sp


# ---------------------------------------------------------------------------
# Correlators (sympy exact)
# ---------------------------------------------------------------------------

# Fixed inputs from the construction
E1 = sp.Integer(0)
E2 = sp.sqrt(2) - 1
E2o = sp.Integer(1)

# Saturating branch for E3 (see discussion around Eq. (6)/(7) in the paper)
E3 = 3 - 2 * sp.sqrt(2)


@lru_cache(None)
def E_line(n: int) -> sp.Expr:
    """
    Line correlator E_n for n >= 1 using recursion Eq. (12):
      E_{n+1} = -E_n + E_{n-1} + (1 - E2) E_{n-2}

    Base:
      E1, E2, E3 fixed above.
    """
    if n < 1:
        raise ValueError("E_line(n): n must be >= 1")
    if n == 1:
        return E1
    if n == 2:
        return E2
    if n == 3:
        return E3

    # iterative build (exact sympy arithmetic)
    E = {1: E1, 2: E2, 3: E3}
    for k in range(3, n):
        E[k + 1] = -E[k] + E[k - 1] + (1 - E2) * E[k - 2]
    return sp.simplify(E[n])


@lru_cache(None)
def E_loop(n: int) -> sp.Expr:
    """
    Loop correlator E^o_n for n >= 1.

    Base (consistent with construction):
      E^o_1 = 0
      E^o_2 = 1

    Recursion Eq. (21) (paper notation):
      E^o_{n+1} = (sqrt2)^(n-1)
                - sum_{k=1}^{n-2} E_k * k * (sqrt2)^(n-k-2)
                - n E_n - (n-1) E_{n-1}
    (valid for n >= 2)

    We implement E^o_n by setting m=n-1 in that formula (so m >= 2 <=> n >= 3).
    """
    if n < 1:
        raise ValueError("E_loop(n): n must be >= 1")
    if n == 1:
        return sp.Integer(0)
    if n == 2:
        return E2o
    if n == 3:
        # Apply Eq (21) with n=2 (sum empty):
        # E^o_3 = (sqrt2)^(1) - 2 E2 - 1 E1
        return sp.simplify(sp.sqrt(2) - 2 * E_line(2) - E_line(1))

    m = n - 1  # corresponds to "n" in Eq. (21)
    term0 = (sp.sqrt(2) ** (m - 1))

    s = sp.Integer(0)
    for k in range(1, m - 1):  # k = 1,...,m-2
        s += E_line(k) * sp.Integer(k) * (sp.sqrt(2) ** (m - k - 2))

    expr = term0 - s - sp.Integer(m) * E_line(m) - sp.Integer(m - 1) * E_line(m - 1)
    return sp.simplify(expr)


# ---------------------------------------------------------------------------
# Subset decomposition helpers
# ---------------------------------------------------------------------------

def _runs_on_cycle(n: int, subset_bits: int) -> List[int]:
    """
    For a cycle C_n, subset_bits encodes S ⊆ {0,...,n-1}.
    Return lengths of maximal contiguous runs of 1s on the cycle.
    """
    if subset_bits == 0:
        return []
    if subset_bits == (1 << n) - 1:
        return [n]

    b = [(subset_bits >> i) & 1 for i in range(n)]

    runs: List[int] = []
    i = 0
    while i < n:
        if b[i] == 0:
            i += 1
            continue
        j = i
        while j < n and b[j] == 1:
            j += 1
        runs.append(j - i)
        i = j

    # wrap merge: first+last run if boundary bits are 1
    if b[0] == 1 and b[-1] == 1 and len(runs) >= 2:
        runs[0] += runs[-1]
        runs.pop()

    return runs


def _runs_on_line(n: int, subset_bits: int) -> List[int]:
    """
    For a line (open chain) of length n, subset_bits encodes S ⊆ {0,...,n-1}.
    Return lengths of maximal contiguous runs of 1s (no wrap).
    """
    if subset_bits == 0:
        return []

    b = [(subset_bits >> i) & 1 for i in range(n)]
    runs: List[int] = []
    i = 0
    while i < n:
        if b[i] == 0:
            i += 1
            continue
        j = i
        while j < n and b[j] == 1:
            j += 1
        runs.append(j - i)
        i = j
    return runs


def _validate_x_list(x_iter: Iterable[int]) -> List[sp.Integer]:
    """
    Convert iterable to a list of sympy Integers; enforce xi ∈ {±1}.
    """
    x_list = list(x_iter)
    if len(x_list) == 0:
        raise ValueError("Need at least one outcome.")

    out: List[sp.Integer] = []
    for v in x_list:
        vv = sp.Integer(v)
        if vv not in (sp.Integer(-1), sp.Integer(1)):
            raise ValueError("Each xi must be +1 or -1.")
        out.append(vv)
    return out


def _subset_products(x: List[sp.Integer]) -> List[sp.Integer]:
    """
    Precompute prod_{i in S} x_i for all subsets S as sympy Integers.
    """
    n = len(x)
    prod_x: List[sp.Integer] = [sp.Integer(1)] * (1 << n)
    for s in range(1, 1 << n):
        lsb = s & -s
        i = (lsb.bit_length() - 1)
        prod_x[s] = prod_x[s ^ lsb] * x[i]
    return prod_x


# ---------------------------------------------------------------------------
# Probabilities
# ---------------------------------------------------------------------------

def P_loop(x_iter: Iterable[int], as_float: bool = True):
    """
    Probability for loop (cycle) network:
        P_loop([x1,...,xn]) with xi ∈ {±1}

    Uses correlator expansion:
      p(x) = 2^{-n} * Σ_{S⊆[n]} (∏_{i∈S} x_i) * <∏_{i∈S} X_i>

    For a cycle:
      - if S = ∅: correlator = 1
      - if S = full set: correlator = E^o_n
      - else: correlator factors over contiguous runs; each run length r contributes E_r
    """
    x = _validate_x_list(x_iter)
    n = len(x)
    full_mask = (1 << n) - 1

    prod_x = _subset_products(x)

    total = sp.Integer(0)
    for s in range(0, 1 << n):
        if s == 0:
            ES = sp.Integer(1)
        elif s == full_mask:
            ES = E_loop(n)
        else:
            runs = _runs_on_cycle(n, s)
            ES = sp.Integer(1)
            for r in runs:
                ES *= E_line(r)
        total += prod_x[s] * ES

    p = sp.simplify(total / (sp.Integer(2) ** n))
    return float(p.evalf()) if as_float else p


def P_line(x_iter: Iterable[int], as_float: bool = True):
    """
    Probability for open chain (no wrap):
        P_line([x1,...,xn]) with xi ∈ {±1}

    Same correlator expansion, but subsets decompose into linear contiguous runs only,
    and there is NO special full-set loop correlator; the full-set correlator is E_n.
    """
    x = _validate_x_list(x_iter)
    n = len(x)

    prod_x = _subset_products(x)

    total = sp.Integer(0)
    for s in range(0, 1 << n):
        if s == 0:
            ES = sp.Integer(1)
        else:
            runs = _runs_on_line(n, s)
            ES = sp.Integer(1)
            for r in runs:
                ES *= E_line(r)
        total += prod_x[s] * ES

    p = sp.simplify(total / (sp.Integer(2) ** n))
    return float(p.evalf()) if as_float else p

def prob_event_loop(bits_iter: Iterable[int], as_float: bool = True):
    """
    Alias for P_loop with 0/1 outcomes:
      0 -> +1, 1 -> -1
    """
    bits = list(bits_iter)
    x_pm = [(+1 if int(b) == 0 else -1) for b in bits]
    return P_loop(x_pm, as_float=as_float)


def prob_event_line(bits_iter: Iterable[int], as_float: bool = True):
    """
    Alias for P_line with 0/1 outcomes:
      0 -> +1, 1 -> -1
    """
    bits = list(bits_iter)
    x_pm = [(+1 if int(b) == 0 else -1) for b in bits]
    return P_line(x_pm, as_float=as_float)

# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import itertools

    ZERO = sp.Integer(0)

    def all_bitstrings(n: int):
        for x in itertools.product((0, 1), repeat=n):
            yield x

    # ---- pretty printers ----
    def fmt_bits(bits):
        return "".join(str(b) for b in bits)

    def fmt_prob(p):
        # print exact sympy + a numeric approximation
        return f"{p}  (~{sp.N(p, 4)})"

    for n in range(1,4):
        print(f"--- LOOP n={n} ---")
        for bits in all_bitstrings(n):
            p = prob_event_loop(bits, as_float=False)
            print(f"  x={fmt_bits(bits)}  P_loop={fmt_prob(p)}")
            assert sp.simplify(p) >= ZERO, f"Negative P_loop for n={n}, x={bits}: {p}"
        print(f"--- LINE n={n} ---")
        for bits in all_bitstrings(n):
            p = prob_event_line(bits, as_float=False)
            print(f"  x={fmt_bits(bits)}  P_line={fmt_prob(p)}")
            assert sp.simplify(p) >= ZERO, f"Negative P_line for n={n}, x={bits}: {p}"
        print()

