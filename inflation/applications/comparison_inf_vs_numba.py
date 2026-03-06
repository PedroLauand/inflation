"""
comparison_inf_vs_numba.py
--------------------------------------------------------------------
Collect linear systems A x = b for:
  1) The postquantum proof setup (InflationSDP with classical sources), and
  2) The symmetric inflation LP construction (numba/symmetric_inflation_test).

We export sparse matrices and vectors to a compressed NPZ file and print
summary stats for quick inspection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import sys
import numpy as np
from scipy.sparse import coo_array

# Ensure repo root is on sys.path so "import inflation" works when running directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inflation import InflationProblem, InflationSDP
from inflation.applications.ring_utils import build_off_diagonal_ring_problem
from inflation.applications.symmetric_inflation_test import build_eq_system, inf_problem
from inflation.distributions import NSIPRDistribution


# --------------------------------------------------------------------
# Postquantum proof: build A x = b from SDP equalities + known moments
# --------------------------------------------------------------------

def _ring_problem_classical(inflation_level: int, nof_outcomes: int = 2) -> InflationProblem:
    return build_off_diagonal_ring_problem(
        inflation_level,
        nof_outcomes,
        classical_sources="all",
    )


def _collect_postquantum_system() -> Tuple[coo_array, np.ndarray, np.ndarray]:
    prob = _ring_problem_classical(3, 2)
    distribution = NSIPRDistribution()
    sdp = InflationSDP(prob, verbose=0, include_all_outcomes=False)
    sdp.generate_relaxation("physical2")

    values = {
        "P[A^{1,2}=0]": float(distribution.prob_event_line([0])),
        "P[A^{1,2}=0 A^{2,1}=0]": float(distribution.prob_event_loop([0, 0])),
        "P[A^{1,2}=0 A^{2,3}=0]": float(distribution.prob_event_line([0, 0])),
        "P[A^{1,2}=0 A^{2,3}=0 A^{3,1}=0]": float(distribution.prob_event_loop([0, 0, 0])),
    }
    sdp.update_values(values=values, only_specified_values=False)

    moments = list(sdp.moments)
    if sdp.Constant_Term not in moments:
        moments.append(sdp.Constant_Term)
    moment_names = np.asarray([mon.name for mon in moments], dtype=str)
    moment_index = {mon: i for i, mon in enumerate(moments)}

    eqs = sdp.moment_equalities()
    n_eq = len(eqs)

    # Start with all equality constraints (sum coeff * moment = 0)
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    b = np.zeros(n_eq, dtype=float)

    for r, eq in enumerate(eqs):
        for mon, coeff in eq.items():
            if mon not in moment_index:
                continue
            rows.append(r)
            cols.append(moment_index[mon])
            data.append(float(coeff))

    # Add known moment assignments: moment = value
    known = dict(sdp.known_moments)
    known[sdp.Constant_Term] = 1.0
    known_items = list(known.items())

    if known_items:
        base = n_eq
        b_known = np.zeros(len(known_items), dtype=float)
        for i, (mon, val) in enumerate(known_items):
            if mon not in moment_index:
                continue
            rows.append(base + i)
            cols.append(moment_index[mon])
            data.append(1.0)
            b_known[i] = float(val)
        b = np.concatenate([b, b_known])

    A = coo_array((data, (rows, cols)), shape=(b.size, len(moments)))
    A.sum_duplicates()
    return A, b, moment_names


# --------------------------------------------------------------------
# Symmetric inflation: build A x = b from numba/symmetric_inflation_test
# --------------------------------------------------------------------

def _collect_symmetric_inflation_system() -> Tuple[coo_array, np.ndarray, np.ndarray]:
    n, outcomes = 3, 2
    prob = inf_problem(n, outcomes)

    sqrt2 = np.sqrt(2.0)
    E_line = {
        1: 0.0,
        2: sqrt2 - 1.0,
        3: 3.0 - 2.0 * sqrt2,
    }
    E_loop = {
        1: 0.0,
        2: 1.0,
        3: 2.0 - sqrt2,
    }

    _row_labels, col_keys, A_eq, b_eq = build_eq_system(prob, E_line, E_loop, show_progress=False)
    var_names = np.asarray([f"q[{k}]" for k in col_keys], dtype=str)
    return A_eq.tocoo(), b_eq, var_names


# --------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------

def _pack_npz(prefix: str, A: coo_array, b: np.ndarray, var_names: np.ndarray) -> dict:
    return {
        f"{prefix}_row": A.row,
        f"{prefix}_col": A.col,
        f"{prefix}_data": A.data,
        f"{prefix}_shape": np.asarray(A.shape, dtype=np.int64),
        f"{prefix}_b": b,
        f"{prefix}_var_names": var_names,
    }


def _print_summary(tag: str, A: coo_array, b: np.ndarray) -> None:
    print(f"\n[{tag}] A shape: {A.shape}, nnz={A.nnz}")
    if b.size:
        print(f"[{tag}] b length: {b.size}, min/max={b.min():.6g}/{b.max():.6g}")
    else:
        print(f"[{tag}] b length: 0")


if __name__ == "__main__":
    A_sdp, b_sdp, names_sdp = _collect_postquantum_system()
    A_inf, b_inf, names_inf = _collect_symmetric_inflation_system()

    _print_summary("postquantum_sdp", A_sdp, b_sdp)
    _print_summary("symmetric_inflation", A_inf, b_inf)

    out_path = Path(__file__).resolve().parent / "cache" / "comparison_inf_vs_numba.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {}
    payload.update(_pack_npz("postquantum", A_sdp, b_sdp, names_sdp))
    payload.update(_pack_npz("symmetric", A_inf, b_inf, names_inf))
    np.savez_compressed(out_path, **payload)

    print(f"\nWrote: {out_path}")
