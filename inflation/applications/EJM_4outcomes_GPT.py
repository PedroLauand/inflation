from __future__ import annotations
from math import ldexp
import numpy as np
from functools import lru_cache
from typing import Dict, Tuple

# ---------------------------
# Unnormalized operators/effects (no 1/sqrt(8))
# ---------------------------
e0_un = np.array([[-1 - 1j, 0,     -2j,  -1 + 1j]], dtype=np.complex128).reshape(2, 2)
e1_un = np.array([[ 1 - 1j, 2j,     0,    1 + 1j]], dtype=np.complex128).reshape(2, 2)
e2_un = np.array([[-1 + 1j, 2j,     0,   -1 - 1j]], dtype=np.complex128).reshape(2, 2)
e3_un = np.array([[ 1 + 1j, 0,     -2j,   1 - 1j]], dtype=np.complex128).reshape(2, 2)
ejm = np.stack([e0_un, e1_un, e2_un, e3_un])  # (4, 2, 2), complex128

# ---------------------------
# Unnormalized edge state (no 1/sqrt(2))
# ---------------------------
psi = np.array([[0, 1, -1, 0]], dtype=np.complex128).reshape(2, 2)  # (2, 2), complex128

@lru_cache(maxsize=None)
def loop_n(n: int) -> np.ndarray:
    """
    Vectorized probability table P[a0,...,a_{n-1}] for an n-site ring, using UNNORMALIZED psi/ejm.

    - Each site i chooses outcome a_i ∈ {0,1,2,3}.
    - Form M[a] = ejm[a] @ psi  (shape: (4, 2, 2)).
    - Amplitude tensor A[a0,...,a_{n-1}] = Tr( M[a0] @ M[a1] @ ... @ M[a_{n-1}] ).
    - Return P = |A|^2 * 16^{-n} as float64 with shape (4,)*n.

    Notes:
    - Uses np.einsum with a generated equation (fast, optimize=True).
    - Falls back to einops.einsum for n>26 (multi-character index tokens).

    Parameters
    ----------
    n : int
        Number of sites (>= 3 recommended).

    Returns
    -------
    np.ndarray
        Probability tensor of shape (4,)*n, dtype float64.
    """

    # Precompute M[a] = ejm[a] @ psi (broadcasted over the outcome axis).
    # Result: M has shape (4, 2, 2), with axes (outcome, left-index, right-index).
    M = np.einsum('aij,jk->aik', ejm, psi, optimize=True)

    # Build the ring contraction for amplitudes, vectorized over all outcomes.
    # For n<=26 we use single-letter indices for speed via np.einsum.
    assert n <= 26, "Number of sites exceeds 26, the maximum NumPy can handle."
    outs = [chr(ord('a') + i) for i in range(n)]      # outcome labels
    ins  = [chr(ord('A') + i) for i in range(n)]      # internal link labels
    terms = [f"{outs[i]}{ins[i]}{ins[(i+1)%n]}" for i in range(n)]
    eq = ",".join(terms) + "->" + "".join(outs)
    # Contract n copies of M around the ring; output shape (4,)*n
    A = np.einsum(eq, *([M] * n), optimize=True)

    # Probabilities: |A|^2, then apply the SINGLE global correction 16^{-n}.
    # Use ldexp for exact power-of-two scaling: 16^{-n} = 2^{-4n}.
    prob_corr: float = ldexp(1.0, -4 * n)
    P = (A.real * A.real + A.imag * A.imag) * prob_corr
    return P.astype(np.float64, copy=False)

@lru_cache(maxsize=None)
def line_n(n: int) -> np.ndarray:
    """
    Vectorized probability table P[a0,...,a_{n-1}] for an n-site line, using UNNORMALIZED psi/ejm.

    - Each site i chooses outcome a_i ∈ {0,1,2,3}.
    - Form M[a] = ejm[a] @ psi  (shape: (4, 2, 2)).
    - Amplitude tensor A[a0,...,a_{n-1}] = Tr( M[a0] @ M[a1] @ ... @ M[a_{n-1}] ).
    - Return P = |A|^2 * 16^{-n} as float64 with shape (4,)*n.

    Notes:
    - Uses np.einsum with a generated equation (fast, optimize=True).
    - Falls back to einops.einsum for n>26 (multi-character index tokens).

    Parameters
    ----------
    n : int
        Number of sites (>= 3 recommended).

    Returns
    -------
    np.ndarray
        Probability tensor of shape (4,)*n, dtype float64.
    """
    return loop_n(n+1).sum(axis=0)

def _loop_key(outcome: Tuple[int, ...]) -> str:
    """P[ A^{1,2}=a1 A^{2,3}=a2 ... A^{n,1}=an ]"""
    n = len(outcome)
    parts = [f"A^{{{i},{(i % n) + 1}}}={outcome[i-1]}" for i in range(1, n + 1)]
    return "P[" + " ".join(parts) + "]"
def _line_key(outcome: Tuple[int, ...]) -> str:
    """P[ A^{1,2}=a1 A^{2,3}=a2 ... A^{n,n+1}=an ]"""
    n = len(outcome)
    parts = [f"A^{{{i},{i+1}}}={outcome[i-1]}" for i in range(1, n + 1)]
    return "P[" + " ".join(parts) + "]"

def loop_dict(n: int) -> Dict[str, float]:
    return {_loop_key(event): probability
            for event, probability in np.ndenumerate(loop_n(n))}
def line_dict(n: int) -> Dict[str, float]:
    return {_line_key(event): probability
            for event, probability in np.ndenumerate(loop_n(n))}

def build_values(n: int, include_lines: bool = True, include_loops: bool = True) -> Dict[str, float]:
    """
    Construct a dict of probabilities with keys matching your original string format
    for all k=1..n loops and (optionally) lines.

    Requires:
      - loop_n(k): cached function returning shape (oA,)*k tensor
      - line_n(k): cached function returning shape (oA,)*k tensor
    """
    if not include_lines and not include_loops:
        return {}

    values: Dict[str, float] = {}
    for k in range(1, n):
        if include_loops:
            values.update(loop_dict(k))
        if include_lines:
            values.update(line_dict(k))
    if include_loops:
        values.update(loop_dict(n))
    return values

if __name__ == "__main__":
    from EJM_4outcomes import loop_3, loop_4
    print(np.isclose(loop_n(3), loop_3()).all())
    print(np.isclose(loop_n(4), loop_4()).all())
    for k, v in build_values(4).items():
        print(f"{k}: {v}")


