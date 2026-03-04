from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence

import sympy as sp

from ._common import (
    NativeCoarsen,
    cyclic_canonical,
    expand_coarse_event,
    line_from_loop,
    normalize_coarsen,
    parse_outcomes,
)

_I = sp.I

_EJM_NATIVE = (
    sp.Matrix([[-1 - _I, 0], [-2 * _I, -1 + _I]]),
    sp.Matrix([[1 - _I, 2 * _I], [0, 1 + _I]]),
    sp.Matrix([[-1 + _I, 2 * _I], [0, -1 - _I]]),
    sp.Matrix([[1 + _I, 0], [-2 * _I, 1 - _I]]),
)
_PSI = sp.Matrix([[0, 1], [-1, 0]])


@lru_cache(maxsize=None)
def _native_M() -> tuple[sp.Matrix, ...]:
    return tuple(mat * _PSI for mat in _EJM_NATIVE)


@lru_cache(maxsize=None)
def _native_loop_prob_cached(canonical_native_event: tuple[int, ...]) -> sp.Expr:
    M = _native_M()
    pmat = sp.eye(2)
    for x in canonical_native_event:
        pmat = pmat * M[x]
    amp = sp.trace(pmat)
    prob = sp.simplify((amp * sp.conjugate(amp)) / (sp.Integer(16) ** len(canonical_native_event)))
    return sp.simplify(prob)


@lru_cache(maxsize=None)
def _coarsened_loop_prob_cached(
    event: tuple[int, ...],
    coarsen_key: NativeCoarsen,
) -> sp.Expr:
    total = sp.Integer(0)
    for native_event in expand_coarse_event(event, coarsen_key):
        total += _native_loop_prob_cached(cyclic_canonical(native_event))
    return sp.simplify(total)


@lru_cache(maxsize=None)
def _coarsened_line_prob_cached(
    event: tuple[int, ...],
    coarsen_key: NativeCoarsen,
) -> sp.Expr:
    alphabet = len(coarsen_key)
    return line_from_loop(
        event,
        loop_prob_fn=lambda ext: _coarsened_loop_prob_cached(
            cyclic_canonical(ext),
            coarsen_key,
        ),
        alphabet_size=alphabet,
    )


def prob_event_loop(
    outcomes: Iterable[int],
    *,
    coarsen: Sequence[Sequence[int]] | None = None,
) -> float:
    """
    Probability P[a1,...,an] on an n-site ring for the EJM distribution.

    Parameters
    ----------
    outcomes
        Event outcome list in the coarsened alphabet.
    coarsen
        Strict partition of native outcomes {0,1,2,3}. Group order defines
        the coarse labels. Example: [[0, 1], [2], [3]].
    """
    coarsen_key = normalize_coarsen(coarsen, native_outcomes=4)
    event = cyclic_canonical(parse_outcomes(outcomes, max_outcome=len(coarsen_key) - 1))
    return float(_coarsened_loop_prob_cached(event, coarsen_key))


def prob_event_line(
    outcomes: Iterable[int],
    *,
    coarsen: Sequence[Sequence[int]] | None = None,
) -> float:
    """
    Probability P[a1,...,an] on an n-site open chain for the EJM distribution.
    """
    coarsen_key = normalize_coarsen(coarsen, native_outcomes=4)
    event = parse_outcomes(outcomes, max_outcome=len(coarsen_key) - 1)
    return float(_coarsened_line_prob_cached(event, coarsen_key))


__all__ = ["prob_event_loop", "prob_event_line"]
