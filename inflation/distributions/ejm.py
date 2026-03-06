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
    mats = _native_M()
    product_mat = sp.eye(2)
    for x in canonical_native_event:
        product_mat = product_mat * mats[x]
    amp = sp.trace(product_mat)
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
    return line_from_loop(
        event,
        loop_prob_fn=lambda ext: _coarsened_loop_prob_cached(cyclic_canonical(ext), coarsen_key),
        alphabet_size=len(coarsen_key),
    )


class EJMDistribution:
    """Exact EJM distribution with optional coarse-graining set at construction."""

    def __init__(self, *, coarsen: Sequence[Sequence[int]] | None = None) -> None:
        self._coarsen_key = normalize_coarsen(coarsen, native_outcomes=4)

    @property
    def nof_outcomes(self) -> int:
        return len(self._coarsen_key)

    def prob_event_loop(self, outcomes: Iterable[int]) -> sp.Expr:
        event = cyclic_canonical(parse_outcomes(outcomes, max_outcome=self.nof_outcomes - 1))
        return _coarsened_loop_prob_cached(event, self._coarsen_key)

    def prob_event_line(self, outcomes: Iterable[int]) -> sp.Expr:
        event = parse_outcomes(outcomes, max_outcome=self.nof_outcomes - 1)
        return _coarsened_line_prob_cached(event, self._coarsen_key)


__all__ = ["EJMDistribution"]
