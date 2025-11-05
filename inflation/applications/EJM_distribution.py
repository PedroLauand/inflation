# scalar_loop_prob.py
from __future__ import annotations
from math import ldexp
from functools import lru_cache
from typing import Iterable
import numpy as np

# ---------- Unnormalized effects (4 outcomes) ----------
e0_un = np.array([[-1 - 1j, 0,     -2j,  -1 + 1j]], dtype=np.complex128).reshape(2, 2)
e1_un = np.array([[ 1 - 1j, 2j,     0,    1 + 1j]], dtype=np.complex128).reshape(2, 2)
e2_un = np.array([[-1 + 1j, 2j,     0,   -1 - 1j]], dtype=np.complex128).reshape(2, 2)
e3_un = np.array([[ 1 + 1j, 0,     -2j,   1 - 1j]], dtype=np.complex128).reshape(2, 2)
ejm = np.stack([e0_un, e1_un, e2_un, e3_un])  # shape (4,2,2)

# ---------- Unnormalized 2x2 edge state ----------
psi = np.array([[0, 1, -1, 0]], dtype=np.complex128).reshape(2, 2)

@lru_cache(maxsize=None)
def _M() -> np.ndarray:
    """Precompute M[a] = ejm[a] @ psi, shape (4,2,2), complex128."""
    return np.einsum('aij,jk->aik', ejm, psi, optimize=True)

def loop_prob_event(outcomes: Iterable[int]) -> float:
    """
    Probability P[a1,...,an] on an n-site RING (n = len(outcomes)),
    where outcomes are integers in {0,1,2,3} ordered as A^{1,2}, A^{2,3}, ..., A^{n,1}.

    Uses unnormalized objects and applies the global factor 16^{-n}.
    """
    a = tuple(int(x) for x in outcomes)
    if not a:
        raise ValueError("Provide at least one outcome.")
    if any((x < 0 or x > 3) for x in a):
        raise ValueError("Outcomes must be in {0,1,2,3}.")
    M = _M()
    # Sequential 2x2 product around the loop, then trace
    Pmat = np.eye(2, dtype=np.complex128)
    for x in a:
        Pmat = Pmat @ M[x]
    amp = np.trace(Pmat)
    prob = (amp.real * amp.real + amp.imag * amp.imag) * ldexp(1.0, -4 * len(a))  # 16^{-n}
    return float(prob)

# ---- tiny demo ----
if __name__ == "__main__":
    # Example: n=3, outcomes (0,0,0)
    print(loop_prob_event([0, 0, 0]))
    
