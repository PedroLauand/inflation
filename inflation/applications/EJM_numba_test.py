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

import mosek
from scipy.sparse import coo_array

# Ensure repo root is on sys.path so "import inflation" works when running directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inflation.applications.Final_algo_numba import PrepLP
from inflation.distributions import EJMDistribution
from inflation.lp.lp_utils import solveLP_sparse


def main(*, n: int) -> None:
    distribution = EJMDistribution()
    prep = PrepLP(
        n,
        distribution,
        cache_name=None,
        auto_discover_symmetries=True,
        compress_rows_under_discovered_group=True,
    )
    variable_names = prep.variable_names
    known_vars = prep.known_vars
    inflation_matrix = prep.inflation_matrix
    print(
        "Index dtype (min signed):",
        prep.min_dtype.name,
        "| constraints:", inflation_matrix.shape[0] + known_vars.nnz,
        "| variables:", inflation_matrix.shape[1],
        "| known:", int(known_vars.nnz),
    )

    nof_all_LP_vars = inflation_matrix.shape[1]
    solverparameters = {
        mosek.iparam.optimizer: mosek.optimizertype.intpnt,
    }
    solution = solveLP_sparse(
        objective=coo_array(([], ([], [])), shape=(1, nof_all_LP_vars)),
        known_vars=known_vars,
        equalities=inflation_matrix,
        default_non_negative=True,
        variables=variable_names,
        verbose=True,
        solverparameters=solverparameters,
    )
    print(solution["status"])


if __name__ == "__main__":
    n = 4
    main(n=n)
