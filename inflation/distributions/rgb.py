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

_DEFAULT_U = sp.sqrt(sp.Rational(9, 10))
_DEFAULT_LAMBDA0 = sp.sqrt(sp.Rational(1, 2))


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
    num = sp.N(expr)
    if num.is_real is False:
        raise ValueError(f"{name} must be real.")
    try:
        num_float = float(num)
    except TypeError as exc:
        raise ValueError(f"{name} must evaluate to a numeric value in [0, 1].") from exc
    if num_float < 0.0 or num_float > 1.0:
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
    M = _native_M(u_exact, lambda0_exact)
    pmat = sp.eye(2)
    for x in canonical_native_event:
        pmat = pmat * M[x]
    amp = sp.trace(pmat)
    # RGB uses normalized operators/state, so no global 16^{-n} scaling is needed.
    prob = sp.simplify(amp * sp.conjugate(amp))
    return sp.simplify(prob)


@lru_cache(maxsize=None)
def _coarsened_loop_prob_cached(
    event: tuple[int, ...],
    coarsen_key: NativeCoarsen,
    u_exact: sp.Expr,
    lambda0_exact: sp.Expr,
) -> sp.Expr:
    total = sp.Integer(0)
    for native_event in expand_coarse_event(event, coarsen_key):
        total += _native_loop_prob_cached(
            cyclic_canonical(native_event),
            u_exact,
            lambda0_exact,
        )
    return sp.simplify(total)


@lru_cache(maxsize=None)
def _coarsened_line_prob_cached(
    event: tuple[int, ...],
    coarsen_key: NativeCoarsen,
    u_exact: sp.Expr,
    lambda0_exact: sp.Expr,
) -> sp.Expr:
    alphabet = len(coarsen_key)
    return line_from_loop(
        event,
        loop_prob_fn=lambda ext: _coarsened_loop_prob_cached(
            cyclic_canonical(ext), coarsen_key, u_exact, lambda0_exact
        ),
        alphabet_size=alphabet,
    )


def prob_event_loop(
    outcomes: Iterable[int],
    *,
    u: object = _DEFAULT_U,
    lambda0: object = _DEFAULT_LAMBDA0,
    coarsen: Sequence[Sequence[int]] | None = None,
) -> float:
    """
    Probability P[a1,...,an] on an n-site ring for the RGB distribution.

    Parameters
    ----------
    outcomes
        Event outcome list in the coarsened alphabet.
    u
        RGB measurement parameter.
    lambda0
        State parameter in lambda0|01> + lambda1|10>.
    coarsen
        Strict partition of native outcomes {0,1,2,3}. Group order defines
        the coarse labels. Example: [[0, 1], [2], [3]].
    """
    coarsen_key = normalize_coarsen(coarsen, native_outcomes=4)
    event = cyclic_canonical(parse_outcomes(outcomes, max_outcome=len(coarsen_key) - 1))
    u_exact = _as_exact_param("u", u)
    lambda0_exact = _as_exact_param("lambda0", lambda0)
    return float(_coarsened_loop_prob_cached(event, coarsen_key, u_exact, lambda0_exact))


def prob_event_line(
    outcomes: Iterable[int],
    *,
    u: object = _DEFAULT_U,
    lambda0: object = _DEFAULT_LAMBDA0,
    coarsen: Sequence[Sequence[int]] | None = None,
) -> float:
    """
    Probability P[a1,...,an] on an n-site open chain for the RGB distribution.
    """
    coarsen_key = normalize_coarsen(coarsen, native_outcomes=4)
    event = parse_outcomes(outcomes, max_outcome=len(coarsen_key) - 1)
    u_exact = _as_exact_param("u", u)
    lambda0_exact = _as_exact_param("lambda0", lambda0)
    return float(_coarsened_line_prob_cached(event, coarsen_key, u_exact, lambda0_exact))


__all__ = ["prob_event_loop", "prob_event_line"]
