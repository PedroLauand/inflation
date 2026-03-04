"""
symmetric_inflation_test.py
--------------------------------------------------------------------
Scenario utilities for building symmetrized LP equalities from
correlator inputs.

This script intentionally keeps only scenario-specific logic:
  - correlator -> loop-event probability map
  - RHS construction from those correlators
  - equality/LP assembly wrappers for MOSEK

Core marginal generation and global-extension canonicalization are
imported from Final_algo_numba.
--------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import sys

import numpy as np
from scipy.sparse import coo_array

# Ensure repo root is on sys.path so "import inflation" works when running directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inflation import InflationProblem
from inflation.applications.Final_algo_numba import (
    build_lhs_from_marginals,
    factorized_marginal_value,
    generate_canonical_marginals,
    ring_problem,
)


def inf_problem(inflation_level: int, nof_outcomes: int = 2) -> InflationProblem:
    """
    Build the ring inflation problem used in this scenario.

    Outcome-relabelling symmetries are controlled by the caller through
    modifications on the returned InflationProblem (if desired).
    """
    return ring_problem(inflation_level, nof_outcomes)


def build_symmetrized_lhs(
    prob: InflationProblem,
    *,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, coo_array]:
    """
    Build raw LHS matrix N for constraints N q' = rhs.

    Column 0 is a reserved sentinel; columns 1.. correspond to canonical
    global-event representatives.
    """
    marginals = generate_canonical_marginals(prob, show_progress=show_progress)
    return build_lhs_from_marginals(prob, marginals, show_progress=show_progress)


def loop_prob_from_correlators(
    outcomes: Tuple[int, ...] | List[int],
    E_line: dict[int, float],
    E_loop: dict[int, float],
) -> float:
    """
    Probability for a loop of length m = len(outcomes) with binary outcomes {0,1},
    using line correlators E_k (k < m) and loop correlator E^o_m.

    Outcomes are mapped to x in {+1,-1} via x = (-1)^a (a in {0,1}).
    """
    m = len(outcomes)
    if m == 0:
        raise ValueError("outcomes must be non-empty")
    if any(o not in (0, 1) for o in outcomes):
        raise ValueError("this RHS demo assumes binary outcomes {0,1}")

    x = [1 if o == 0 else -1 for o in outcomes]
    total = 1.0

    # Connected k-body correlators around the loop (k=2..m-1).
    for k in range(2, m):
        if k not in E_line:
            raise KeyError(f"missing E_line[{k}] for loop length {m}")
        seg_sum = 0.0
        for start in range(m):
            prod = 1
            for t in range(k):
                prod *= x[(start + t) % m]
            seg_sum += prod
        total += E_line[k] * seg_sum

    # Full loop correlator.
    if m not in E_loop:
        raise KeyError(f"missing E_loop[{m}] for loop length {m}")
    prod_all = 1
    for val in x:
        prod_all *= val
    total += E_loop[m] * prod_all

    return total / (2**m)


def build_rhs_from_correlators(
    prob: InflationProblem,
    E_line: dict[int, float],
    E_loop: dict[int, float],
    *,
    marginals: List[List[List[int]]] | None = None,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Build RHS vector for the canonical marginals using loop correlators.
    """
    outcomes = prob.outcomes_per_party[0]
    if outcomes != 2:
        raise NotImplementedError("RHS demo currently assumes binary outcomes")

    if marginals is None:
        marginals = generate_canonical_marginals(prob, show_progress=show_progress)

    def event_prob_fn(cycle_outcomes: Tuple[int, ...] | List[int]) -> float:
        return loop_prob_from_correlators(cycle_outcomes, E_line, E_loop)

    rhs = np.empty(len(marginals), dtype=float)
    for idx, marginal in enumerate(marginals):
        rhs[idx] = factorized_marginal_value(marginal, event_prob_fn)
    return rhs


def build_eq_system(
    prob: InflationProblem,
    E_line: dict[int, float],
    E_loop: dict[int, float],
    *,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, coo_array, np.ndarray]:
    """
    Build the equality system A_eq @ q = b_eq for the LP.
    """
    marginals = generate_canonical_marginals(prob, show_progress=show_progress)
    row_labels, col_keys, lhs_raw = build_lhs_from_marginals(
        prob,
        marginals,
        show_progress=show_progress,
    )
    b_eq = build_rhs_from_correlators(
        prob,
        E_line,
        E_loop,
        marginals=marginals,
        show_progress=False,
    )

    # Convert raw LHS to 0-based column indexing by dropping sentinel column 0.
    lhs_coo = lhs_raw.tocoo()
    if lhs_coo.nnz == 0:
        A_eq = coo_array(([], ([], [])), shape=(lhs_raw.shape[0], 0))
    else:
        A_eq = coo_array(
            (lhs_coo.data, (lhs_coo.row, lhs_coo.col - 1)),
            shape=(lhs_raw.shape[0], lhs_raw.shape[1] - 1),
        )
        A_eq.sum_duplicates()

    return row_labels, col_keys, A_eq, b_eq


def build_mosek_lp(
    prob: InflationProblem,
    E_line: dict[int, float],
    E_loop: dict[int, float],
    *,
    add_normalization: bool = True,
    show_progress: bool = True,
) -> Tuple[np.ndarray, coo_array, coo_array, np.ndarray]:
    """
    Assemble the LP in the format expected by solveLP_sparse (MOSEK backend).
    """
    _row_labels, col_keys, A_eq, b_eq = build_eq_system(
        prob, E_line, E_loop, show_progress=show_progress
    )

    # Number of symmetrized global-event variables (q).
    n_vars = A_eq.shape[1]
    const_col = n_vars  # last column is the constant variable c

    # Base equalities: A_eq q - b_eq * c = 0.
    A = A_eq.tocoo()
    base_rows = A.row.astype(np.int64, copy=False)
    base_cols = A.col.astype(np.int64, copy=False)
    base_data = A.data.astype(float, copy=False)

    rhs_rows = np.arange(A_eq.shape[0], dtype=np.int64)
    rhs_cols = np.full(A_eq.shape[0], const_col, dtype=np.int64)
    rhs_data = -b_eq.astype(float, copy=False)

    rows = [base_rows, rhs_rows]
    cols = [base_cols, rhs_cols]
    data = [base_data, rhs_data]

    # Optional normalization row: sum_i q_i - c = 0.
    if add_normalization:
        norm_row = A_eq.shape[0]
        norm_rows = np.full(n_vars, norm_row, dtype=np.int64)
        norm_cols = np.arange(n_vars, dtype=np.int64)
        norm_data = np.ones(n_vars, dtype=float)
        norm_c_row = np.array([norm_row], dtype=np.int64)
        norm_c_col = np.array([const_col], dtype=np.int64)
        norm_c_data = np.array([-1.0], dtype=float)
        rows.extend([norm_rows, norm_c_row])
        cols.extend([norm_cols, norm_c_col])
        data.extend([norm_data, norm_c_data])

    all_rows = np.concatenate(rows)
    all_cols = np.concatenate(cols)
    all_data = np.concatenate(data)
    n_rows = A_eq.shape[0] + (1 if add_normalization else 0)
    equalities = coo_array(
        (all_data, (all_rows, all_cols)),
        shape=(n_rows, n_vars + 1),
    )
    equalities.sum_duplicates()

    # "known_vars" encodes c = 1 (the constant variable).
    known_vars = coo_array(
        (np.array([1.0]), (np.array([0], dtype=np.int64), np.array([const_col], dtype=np.int64))),
        shape=(1, n_vars + 1),
    )

    variable_names = np.asarray([f"q[{k}]" for k in col_keys] + ["const_1"], dtype=str)
    return variable_names, known_vars, equalities, b_eq


def solve_lp_mosek(
    prob: InflationProblem,
    E_line: dict[int, float],
    E_loop: dict[int, float],
    *,
    add_normalization: bool = True,
    show_progress: bool = True,
    verbose: int = 1,
) -> dict:
    """
    Solve the feasibility LP using the MOSEK-backed solver.
    """
    from inflation.lp.lp_utils import solveLP_sparse

    variable_names, known_vars, equalities, _b_eq = build_mosek_lp(
        prob,
        E_line,
        E_loop,
        add_normalization=add_normalization,
        show_progress=show_progress,
    )

    objective = coo_array(([], ([], [])), shape=(1, equalities.shape[1]))
    return solveLP_sparse(
        objective=objective,
        known_vars=known_vars,
        equalities=equalities,
        default_non_negative=True,
        variables=variable_names,
        verbose=verbose,
    )


if __name__ == "__main__":
    n, outcomes = 3, 2
    include_outcome_relabel_symmetries = False

    prob = inf_problem(n, outcomes)
    if include_outcome_relabel_symmetries:
        prob.add_symmetries(prob._setting_specific_outcome_relabelling_symmetries)
    print(prob)

    # Correlators from Table I of arXiv:2102.03597 (Bancal & Gisin, 2021).
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

    _row_labels, _col_keys, A_eq, b_eq = build_eq_system(
        prob, E_line, E_loop, show_progress=True
    )
    print("\nLHS matrix summary:")
    print(f"  rows (marginals): {A_eq.shape[0]}")
    print(f"  cols (symmetrized globals): {A_eq.shape[1]}")
    print(f"  nnz: {A_eq.nnz}")
    print("\nRHS vector summary:")
    print(f"  length: {b_eq.size}")
    print(f"  min/max: {b_eq.min():.6g} / {b_eq.max():.6g}")

    solution = solve_lp_mosek(
        prob,
        E_line,
        E_loop,
        add_normalization=True,
        show_progress=False,
        verbose=1,
    )
    print("\nMOSEK status:", solution["status"])
