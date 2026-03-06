"""
EJM_numba_test.py
--------------------------------------------------------------------
Test the ring inflation LP pipeline for n=4, outcomes=4 using the
EJM loop distribution.
--------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import mosek
from scipy.sparse import coo_array

# Ensure repo root is on sys.path so "import inflation" works when running directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inflation.applications.Final_algo_numba import PrepLP, ring_problem
from inflation.distributions.ejm import prob_event_loop as ejm_prob_event_loop
from inflation.lp.lp_utils import solveLP_sparse


def _min_index_dtype(n_constraints: int, n_variables: int, n_known: int) -> np.dtype:
    """
    Match solveLP_sparse index sizing: max over (constraints, variables, 2*known).
    Returns the smallest signed integer dtype that can hold that max index.
    """
    max_size = max(n_constraints, n_variables, n_known * 2, 1)
    max_index = max_size - 1
    if max_index <= np.iinfo(np.int8).max:
        return np.dtype(np.int8)
    if max_index <= np.iinfo(np.int16).max:
        return np.dtype(np.int16)
    if max_index <= np.iinfo(np.int32).max:
        return np.dtype(np.int32)
    return np.dtype(np.int64)


def main(*, n: int, outcomes: int) -> None:
    prob = ring_problem(n, outcomes)
    print("done with prob")

    prep = PrepLP(
        prob,
        event_prob_fn=ejm_prob_event_loop,
        auto_discover_symmetries=True,
        compress_rows_under_discovered_group=True,
    )
    variable_names = prep.variable_names
    known_vars_coo_vec = prep.known_vars_coo_vec
    inflation_matrix = prep.inflation_matrix
    min_dtype = _min_index_dtype(
        n_constraints=inflation_matrix.shape[0] + known_vars_coo_vec.nnz,
        n_variables=inflation_matrix.shape[1],
        n_known=int(known_vars_coo_vec.nnz),
    )
    print(
        "Index dtype (min signed):",
        min_dtype.name,
        "| constraints:", inflation_matrix.shape[0] + known_vars_coo_vec.nnz,
        "| variables:", inflation_matrix.shape[1],
        "| known:", int(known_vars_coo_vec.nnz),
    )

    nof_all_LP_vars = inflation_matrix.shape[1]
    solverparameters = {
        mosek.iparam.optimizer: mosek.optimizertype.intpnt,
    }
    solution = solveLP_sparse(
        objective=coo_array(([], ([], [])), shape=(1, nof_all_LP_vars)),
        known_vars=known_vars_coo_vec,
        equalities=inflation_matrix,
        default_non_negative=True,
        variables=variable_names,
        verbose=True,
        solverparameters=solverparameters,
    )
    print(solution["status"])


if __name__ == "__main__":
    n, outcomes = 4, 4
    main(n=n, outcomes=outcomes)
