from __future__ import annotations

from typing import Iterable, Callable, List, Sequence, Union
from math import ldexp, sqrt
import numpy as np


def default_ejm_measurement() -> np.ndarray:
    """Default EJM measurement basis (4 outcomes)."""
    e0_un = np.array([[-1 - 1j, 0,     -2j,  -1 + 1j]], dtype=np.complex128).reshape(2, 2)
    e1_un = np.array([[ 1 - 1j, 2j,     0,    1 + 1j]], dtype=np.complex128).reshape(2, 2)
    e2_un = np.array([[-1 + 1j, 2j,     0,   -1 - 1j]], dtype=np.complex128).reshape(2, 2)
    e3_un = np.array([[ 1 + 1j, 0,     -2j,   1 - 1j]], dtype=np.complex128).reshape(2, 2)
    return np.stack([e0_un, e1_un, e2_un, e3_un])  # (4,2,2)


def default_state() -> np.ndarray:
    """Default state used in the original code."""
    return np.array([[0, 1, -1, 0]], dtype=np.complex128).reshape(2, 2)


def build_loop_prob_fn(measurement: np.ndarray, state: np.ndarray) -> Callable[[Iterable[int]], float]:
    """
    Build P[a1,...,an] for an n-site ring using a chosen measurement and state.
    This keeps the original formula but makes the inputs explicit.
    """
    M = np.einsum('aij,jk->aik', measurement, state, optimize=True)

    def loop_prob_event(outcomes: Iterable[int]) -> float:
        a = tuple(int(x) for x in outcomes)
        if not a:
            raise ValueError("Provide at least one outcome.")
        if any((x < 0 or x > 3) for x in a):
            raise ValueError("Outcomes must be in {0,1,2,3}.")
        Pmat = np.eye(2, dtype=np.complex128)
        for x in a:
            Pmat = Pmat @ M[x]
        amp = np.trace(Pmat)
        prob = (amp.real * amp.real + amp.imag * amp.imag) * ldexp(1.0, -4 * len(a))  # 16^{-n}
        return float(prob)

    return loop_prob_event


DEFAULT_EVENT_PROB = build_loop_prob_fn(default_ejm_measurement(), default_state())


def _parse_coarse_groups(
    groups: Sequence[Union[str, int, Iterable[int]]],
) -> List[List[int]]:
    """
    Normalize coarse-graining groups into lists of outcome indices.
    Examples:
      ["01", "2", "3"] -> [[0,1],[2],[3]]
      [[0,1], [2], [3]] -> same
      [0, 1, 2, 3] -> [[0],[1],[2],[3]]
    """
    out: List[List[int]] = []
    for g in groups:
        if isinstance(g, int):
            out.append([g])
        elif isinstance(g, str):
            digits = [int(ch) for ch in g if ch.isdigit()]
            if not digits:
                raise ValueError(f"Empty coarse group from string: {g!r}")
            out.append(digits)
        else:
            gg = [int(x) for x in g]
            if not gg:
                raise ValueError(f"Empty coarse group: {g!r}")
            out.append(gg)
    return out


def coarse_ejm(
    groups: Sequence[Union[str, int, Iterable[int]]],
    *,
    measurement: np.ndarray | None = None,
) -> np.ndarray:
    """
    Coarse-grain EJM outcomes by summing measurement elements in each group.
    Example usage:
      coarse_ejm(["01", "2", "3"])  -> outcomes {0,1} merged, 2, 3 separate
      coarse_ejm(["01", "23"])      -> outcomes {0,1} and {2,3}
      coarse_ejm([[0,1], [2], [3]]) -> same as first example
    """
    base = default_ejm_measurement() if measurement is None else measurement
    if base.ndim != 3:
        raise ValueError("measurement must have shape (outcomes, d, d)")
    coarse_groups = _parse_coarse_groups(groups)
    max_idx = max(max(g) for g in coarse_groups)
    if max_idx >= base.shape[0]:
        raise ValueError("coarse groups reference outcome outside measurement range")
    coarse = []
    for g in coarse_groups:
        acc = np.zeros_like(base[0])
        for idx in g:
            acc = acc + base[idx]
        coarse.append(acc)
    return np.stack(coarse)


def rgb4_measurement(u: float) -> np.ndarray:
    """
    Measurement with outcomes:
      |00>, v|01>+u|10>, u|01>-v|10>, |11>,
    where v = sqrt(1 - u^2).
    """
    if not (0.0 <= u <= 1.0):
        raise ValueError("u must be in [0, 1].")
    v = sqrt(1.0 - u * u)
    e0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128).reshape(2, 2)
    e1 = np.array([0.0, v,   u,   0.0], dtype=np.complex128).reshape(2, 2)
    e2 = np.array([0.0, u,  -v,   0.0], dtype=np.complex128).reshape(2, 2)
    e3 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.complex128).reshape(2, 2)
    return np.stack([e0, e1, e2, e3])

def coarse_rgb(
    groups: Sequence[Union[str, int, Iterable[int]]],
    *,
    u: float,
    measurement: np.ndarray | None = None,
) -> np.ndarray:
    """
    Coarse-grain RGB4 outcomes by summing measurement elements in each group.
    Example usage:
      coarse_rgb(["01", "2", "3"], u=0.7)
      coarse_rgb(["01", "23"], u=0.7)
    """
    base = rgb4_measurement(u) if measurement is None else measurement
    if base.ndim != 3:
        raise ValueError("measurement must have shape (outcomes, d, d)")
    coarse_groups = _parse_coarse_groups(groups)
    max_idx = max(max(g) for g in coarse_groups)
    if max_idx >= base.shape[0]:
        raise ValueError("coarse groups reference outcome outside measurement range")
    coarse = []
    for g in coarse_groups:
        acc = np.zeros_like(base[0])
        for idx in g:
            acc = acc + base[idx]
        coarse.append(acc)
    return np.stack(coarse)


def rgb4_state(lambda0: float) -> np.ndarray:
    """
    State: lambda0|01> + lambda1|10>, where lambda1 = sqrt(1 - lambda0^2).
    """
    if not (0.0 <= lambda0 <= 1.0):
        raise ValueError("lambda0 must be in [0, 1].")
    lambda1 = sqrt(1.0 - lambda0 * lambda0)
    return np.array([0.0, lambda0, lambda1, 0.0], dtype=np.complex128).reshape(2, 2)
