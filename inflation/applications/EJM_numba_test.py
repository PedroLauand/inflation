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

# Ensure repo root is on sys.path so "import inflation" works when running directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inflation.distributions import EJMDistribution
from inflation.applications.Final_algo_numba import PrepLP
from inflation.lp.lp_utils import save_lp_solution


def main(*, n: int) -> None:
    distribution = EJMDistribution()
    prep = PrepLP(
        n,
        distribution,
        problem_name=f"EJM_n={n}",
        auto_discover_symmetries=True,
        compress_rows_under_discovered_group=True,
    )
    print(f"PrepLP initialized for n={n}; materializing LP inputs before Mosek.")
    _ = prep.variable_names
    _ = prep.known_vars
    print(
        f"LP inputs ready for n={n}: "
        f"rows={prep.nof_lp_constraints}, cols={prep.nof_lp_vars}. "
        "Starting Mosek setup."
    )

    solution = prep.solve(
        optimizer="interior_point",
        verbose=2,
    )
    print(f"Solution status for n={n}: {solution['status']}")
    if prep.output_path is not None:
        save_lp_solution(solution, prep.output_path)
        print(f"Saved LP solution archive to {prep.output_path}")


if __name__ == "__main__":
    n = 4
    main(n=n)
