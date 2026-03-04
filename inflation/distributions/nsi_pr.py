from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import sympy as sp

from ._common import parse_outcomes

_U = sp.sqrt(5 + 4 * sp.sqrt(2))
_TERM_1 = sp.simplify(2 / (_U - 1))
_TERM_2 = sp.simplify(-2 / (_U + 1))


@lru_cache(maxsize=None)
def _expec_line(n: int) -> sp.Expr:
    if n < 0:
        raise ValueError("n must be >= 0.")
    return sp.simplify(((_TERM_1 ** (n - 1)) - (_TERM_2 ** (n - 1))) / _U)


@lru_cache(maxsize=None)
def _expec_loop(n: int) -> sp.Expr:
    if n < 0:
        raise ValueError("n must be >= 0.")
    if n == 0:
        return sp.Integer(1)
    if n == 1:
        return sp.Integer(0)
    return sp.simplify((_TERM_1 ** n) + (_TERM_2 ** n))


@lru_cache(maxsize=None)
def _allzero_line(n: int) -> sp.Expr:
    if n < 0:
        raise ValueError("n must be >= 0.")
    return sp.simplify(
        sum(sp.binomial(n, k) * _expec_line(k) for k in range(0, n + 1)) / (2**n)
    )


@lru_cache(maxsize=None)
def _allzero_loop(n: int) -> sp.Expr:
    if n < 0:
        raise ValueError("n must be >= 0.")
    return sp.simplify(
        sum(
            sp.binomial(n, k) * (_expec_line(k) if k < n else _expec_loop(n))
            for k in range(0, n + 1)
        )
        / (2**n)
    )


@lru_cache(maxsize=None)
def _prob_line_by_counts(n: int, h: int) -> sp.Expr:
    if not (0 <= h <= n):
        raise ValueError("h must satisfy 0 <= h <= n.")
    total = sp.Integer(0)
    for t in range(0, h + 1):
        total += ((-1) ** t) * sp.binomial(h, t) * _allzero_line(n - h + t)
    return sp.simplify(total)


@lru_cache(maxsize=None)
def _prob_loop_by_counts(n: int, h: int) -> sp.Expr:
    if not (0 <= h <= n):
        raise ValueError("h must satisfy 0 <= h <= n.")
    if h == 0:
        return _allzero_loop(n)

    total = sp.Integer(0)
    for t in range(0, h):
        total += ((-1) ** t) * sp.binomial(h, t) * _allzero_line(n - h + t)
    total += ((-1) ** h) * _allzero_loop(n)
    return sp.simplify(total)


def prob_event_line(outcomes: Iterable[int]) -> float:
    """
    NSI-PR probability on an open chain for a binary event list.

    This distribution is orderless: the value depends only on (n, h),
    where n is event length and h is the number of ones.
    """
    event = parse_outcomes(outcomes, max_outcome=1)
    n = len(event)
    h = sum(event)
    return float(_prob_line_by_counts(n, h))


def prob_event_loop(outcomes: Iterable[int]) -> float:
    """
    NSI-PR probability on a ring for a binary event list.

    Full-set term uses the loop all-zero probability, strict subsets
    use line all-zero probabilities.
    """
    event = parse_outcomes(outcomes, max_outcome=1)
    n = len(event)
    h = sum(event)
    return float(_prob_loop_by_counts(n, h))


__all__ = ["prob_event_loop", "prob_event_line"]
