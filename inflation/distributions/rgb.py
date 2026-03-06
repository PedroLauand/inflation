from __future__ import annotations

from functools import lru_cache
from numbers import Integral, Real
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

DEFAULT_U = sp.sqrt(sp.Rational(9, 10))
DEFAULT_LAMBDA0 = sp.sqrt(sp.Rational(1, 2))


def _as_exact_param(name: str, value: object) -> sp.Expr:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric or a SymPy expression.")
    if isinstance(value, sp.Basic):
        expr = sp.simplify(value)
    elif isinstance(value, Integral):
        expr = sp.Integer(int(value))
    elif isinstance(value, Real):
        expr = sp.Rational(str(float(value)))
    else:
        raise TypeError(f"{name} must be numeric or a SymPy expression.")
    numeric = sp.N(expr)
    if numeric.is_real is False:
        raise ValueError(f"{name} must be real.")
    try:
        numeric_float = float(numeric)
    except TypeError as exc:
        raise ValueError(f"{name} must evaluate to a numeric value in [0, 1].") from exc
    if numeric_float < 0.0 or numeric_float > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return expr


@lru_cache(maxsize=None)
def _native_M(
    u_exact: sp.Expr,
    lambda0_exact: sp.Expr,
) -> tuple[sp.Matrix, ...]:
    v = sp.sqrt(1 - u_exact * u_exact)
    lambda1 = sp.sqrt(1 - lambda0_exact * lambda0_exact)
    measurement = (
        sp.Matrix([[1, 0], [0, 0]]),
        sp.Matrix([[0, v], [u_exact, 0]]),
        sp.Matrix([[0, u_exact], [-v, 0]]),
        sp.Matrix([[0, 0], [0, 1]]),
    )
    state = sp.Matrix([[0, lambda0_exact], [lambda1, 0]])
    return tuple(mat * state for mat in measurement)


@lru_cache(maxsize=None)
def _native_loop_prob_cached(
    canonical_native_event: tuple[int, ...],
    u_exact: sp.Expr,
    lambda0_exact: sp.Expr,
) -> sp.Expr:
    mats = _native_M(u_exact, lambda0_exact)
    product_mat = sp.eye(2)
    for x in canonical_native_event:
        product_mat = product_mat * mats[x]
    amp = sp.trace(product_mat)
    return sp.simplify(amp * sp.conjugate(amp))


@lru_cache(maxsize=None)
def _coarsened_loop_prob_cached(
    event: tuple[int, ...],
    coarsen_key: NativeCoarsen,
    u_exact: sp.Expr,
    lambda0_exact: sp.Expr,
) -> sp.Expr:
    total = sp.Integer(0)
    for native_event in expand_coarse_event(event, coarsen_key):
        total += _native_loop_prob_cached(cyclic_canonical(native_event), u_exact, lambda0_exact)
    return sp.simplify(total)


@lru_cache(maxsize=None)
def _coarsened_line_prob_cached(
    event: tuple[int, ...],
    coarsen_key: NativeCoarsen,
    u_exact: sp.Expr,
    lambda0_exact: sp.Expr,
) -> sp.Expr:
    return line_from_loop(
        event,
        loop_prob_fn=lambda ext: _coarsened_loop_prob_cached(
            cyclic_canonical(ext),
            coarsen_key,
            u_exact,
            lambda0_exact,
        ),
        alphabet_size=len(coarsen_key),
    )


class RGBDistribution:
    """Exact RGB distribution with constructor-bound parameters and coarse-graining."""

    def __init__(
        self,
        *,
        u: object = DEFAULT_U,
        lambda0: object = DEFAULT_LAMBDA0,
        coarsen: Sequence[Sequence[int]] | None = None,
    ) -> None:
        self._u_exact = _as_exact_param("u", u)
        self._lambda0_exact = _as_exact_param("lambda0", lambda0)
        self._coarsen_key = normalize_coarsen(coarsen, native_outcomes=4)

    @property
    def nof_outcomes(self) -> int:
        return len(self._coarsen_key)

    def prob_event_loop(self, outcomes: Iterable[int]) -> sp.Expr:
        event = cyclic_canonical(parse_outcomes(outcomes, max_outcome=self.nof_outcomes - 1))
        return _coarsened_loop_prob_cached(event, self._coarsen_key, self._u_exact, self._lambda0_exact)

    def prob_event_line(self, outcomes: Iterable[int]) -> sp.Expr:
        event = parse_outcomes(outcomes, max_outcome=self.nof_outcomes - 1)
        return _coarsened_line_prob_cached(event, self._coarsen_key, self._u_exact, self._lambda0_exact)


__all__ = ["RGBDistribution", "DEFAULT_U", "DEFAULT_LAMBDA0"]
