# -*- coding: utf-8 -*-
"""
Minimal sympy-exact probability evaluators with hard cutoffs:

Non-negotiable correlators:
  E1 = 0
  E2 = sqrt(2) - 1
  E2o = 1
And we FIX:
  E3o = 0   (to enforce global bit-flip invariance for n=3 loop)

Implemented:
  - _p_line up to n<=2 (±1 inputs)
  - _p_loop up to n<=3 (±1 inputs)

Also provided 0/1 aliases:
  - prob_event_line(bits)  0->+1, 1->-1
  - prob_event_loop(bits)

For n beyond limits: raises NotImplementedError.
"""

from __future__ import annotations

from typing import Iterable, List

import sympy as sp

# ---------------------------------------------------------------------------
# Fixed correlators (non-negotiable) + fixed E3o
# ---------------------------------------------------------------------------

E1 = sp.Integer(0)
E2 = sp.sqrt(2) - 1
E2o = sp.Integer(1)
E3o = sp.Integer(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pm_list(x_iter: Iterable[int]) -> List[sp.Integer]:
    x = list(x_iter)
    if len(x) == 0:
        raise ValueError("Need at least one outcome.")
    out: List[sp.Integer] = []
    for v in x:
        vv = sp.Integer(v)
        if vv not in (sp.Integer(-1), sp.Integer(1)):
            raise ValueError("Each xi must be +1 or -1.")
        out.append(vv)
    return out


def _bits_to_pm(bits_iter: Iterable[int]) -> List[int]:
    bits = list(bits_iter)
    return [(+1 if int(b) == 0 else -1) for b in bits]


# ---------------------------------------------------------------------------
# Probabilities (±1 interface)
# ---------------------------------------------------------------------------

def _p_line(x_iter: Iterable[int], *, as_float: bool = True):
    """
    Open chain probabilities for n<=2 only.

    n=1:
      p(x1) = 1/2

    n=2:
      p(x1,x2) = 1/4 * (1 + E2 x1 x2)
    """
    x = _pm_list(x_iter)
    n = len(x)

    if n > 2:
        raise NotImplementedError("_p_line implemented only up to n<=2.")

    if n == 1:
        p = sp.Rational(1, 2)
    else:
        x1, x2 = x
        p = sp.simplify(sp.Rational(1, 4) * (1 + E2 * x1 * x2))

    return float(p.evalf()) if as_float else p


def _p_loop(x_iter: Iterable[int], *, as_float: bool = True):
    """
    Cycle probabilities for n<=3 only.

    n=1:
      p(x1) = 1/2

    n=2:
      p(x1,x2) = 1/4 * (1 + E2 x1 x2)

    n=3:
      p(x1,x2,x3) = 1/8 * ( 1 + E2*(x1x2 + x2x3 + x3x1) + E3o*(x1x2x3) )
      with E3o fixed to 0 (global flip invariant).
    """
    x = _pm_list(x_iter)
    n = len(x)

    if n > 3:
        raise NotImplementedError("_p_loop implemented only up to n<=3.")

    if n == 1:
        p = sp.Rational(1, 2)
    elif n == 2:
        x1, x2 = x
        p = sp.simplify(sp.Rational(1, 4) * (1 + E2 * x1 * x2))
    else:
        x1, x2, x3 = x
        p = sp.simplify(
            sp.Rational(1, 8)
            * (1 + E2 * (x1 * x2 + x2 * x3 + x3 * x1) + E3o * (x1 * x2 * x3))
        )

    return float(p.evalf()) if as_float else p


# ---------------------------------------------------------------------------
# 0/1 aliases
# ---------------------------------------------------------------------------

def prob_event_line(bits_iter: Iterable[int], *, as_float: bool = True):
    return _p_line(_bits_to_pm(bits_iter), as_float=as_float)


def prob_event_loop(bits_iter: Iterable[int], *, as_float: bool = True):
    return _p_loop(_bits_to_pm(bits_iter), as_float=as_float)


__all__ = [
    "E1", "E2", "E2o", "E3o",
    "_p_line", "_p_loop",
    "prob_event_line", "prob_event_loop",
]


# ---------------------------------------------------------------------------
# Tests (implemented sizes only)
# ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     import itertools

#     ZERO = sp.Integer(0)

#     def all_bitstrings(n: int):
#         for x in itertools.product((0, 1), repeat=n):
#             yield x

#     # Nonnegativity tests for implemented functions only:
#     # - line: n=1,2
#     # - loop: n=1,2,3
#     for n in (1, 2):
#         for x in all_bitstrings(n):
#             p = prob_event_line(x, as_float=False)
#             assert sp.simplify(p) >= ZERO, f"Negative P_line for n={n}, x={x}: {p}"

#     for n in (1, 2, 3):
#         for x in all_bitstrings(n):
#             p = prob_event_loop(x, as_float=False)
#             assert sp.simplify(p) >= ZERO, f"Negative P_loop for n={n}, x={x}: {p}"

#     # Bit-flip invariance for implemented sizes only.
#     for n in (1, 2):
#         for x in all_bitstrings(n):
#             x_flip = tuple(1 - v for v in x)
#             p = prob_event_line(x, as_float=False)
#             p_flip = prob_event_line(x_flip, as_float=False)
#             assert sp.simplify(p - p_flip) == 0, f"Flip failed (line) n={n}, x={x}: {p} vs {p_flip}"

#     for n in (1, 2, 3):
#         for x in all_bitstrings(n):
#             x_flip = tuple(1 - v for v in x)
#             p = prob_event_loop(x, as_float=False)
#             p_flip = prob_event_loop(x_flip, as_float=False)
#             assert sp.simplify(p - p_flip) == 0, f"Flip failed (loop) n={n}, x={x}: {p} vs {p_flip}"

#     print("All implemented-size tests passed (E3o fixed to 0).")

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
        return f"{p}  (~{sp.N(p, 20)})"

    print("=== Enumerating implemented probabilities (exact; plus numeric approx) ===\n")

    # Line: n=1,2
    for n in (1, 2):
        print(f"--- LINE n={n} ---")
        for bits in all_bitstrings(n):
            p = prob_event_line(bits, as_float=False)
            print(f"  x={fmt_bits(bits)}  P_line={fmt_prob(p)}")
            assert sp.simplify(p) >= ZERO, f"Negative P_line for n={n}, x={bits}: {p}"
        print()

    # Loop: n=1,2,3
    for n in (1, 2, 3):
        print(f"--- LOOP n={n} ---")
        for bits in all_bitstrings(n):
            p = prob_event_loop(bits, as_float=False)
            print(f"  x={fmt_bits(bits)}  P_loop={fmt_prob(p)}")
            assert sp.simplify(p) >= ZERO, f"Negative P_loop for n={n}, x={bits}: {p}"
        print()

    print("=== Bit-flip invariance checks (implemented sizes only) ===\n")

    # Bit-flip invariance: line n=1,2
    for n in (1, 2):
        print(f"--- LINE flip check n={n} ---")
        for bits in all_bitstrings(n):
            bits_flip = tuple(1 - b for b in bits)
            p = prob_event_line(bits, as_float=False)
            p_flip = prob_event_line(bits_flip, as_float=False)
            print(
                f"  x={fmt_bits(bits)} -> {fmt_bits(bits_flip)} | "
                f"P={fmt_prob(p)} ; P_flip={fmt_prob(p_flip)}"
            )
            assert sp.simplify(p - p_flip) == 0, (
                f"Flip failed (line) n={n}, x={bits}: {p} vs {p_flip}"
            )
        print()

    # Bit-flip invariance: loop n=1,2,3
    for n in (1, 2, 3):
        print(f"--- LOOP flip check n={n} ---")
        for bits in all_bitstrings(n):
            bits_flip = tuple(1 - b for b in bits)
            p = prob_event_loop(bits, as_float=False)
            p_flip = prob_event_loop(bits_flip, as_float=False)
            print(
                f"  x={fmt_bits(bits)} -> {fmt_bits(bits_flip)} | "
                f"P={fmt_prob(p)} ; P_flip={fmt_prob(p_flip)}"
            )
            assert sp.simplify(p - p_flip) == 0, (
                f"Flip failed (loop) n={n}, x={bits}: {p} vs {p_flip}"
            )
        print()

    print("All implemented-size tests passed (E3o fixed to 0).")