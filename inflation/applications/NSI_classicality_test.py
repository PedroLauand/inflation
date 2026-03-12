"""
NSI_classicality_test.py
--------------------------------------------------------------------
Explore the NSI construction for rings up to size 4, and then up to size 6.
--------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repo root is on sys.path so "import inflation" works when running this file directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inflation.distributions import NSIPRDistribution
from inflation.applications.Final_algo_numba import PrepLP


def main(*, inflation_levels=(5, 6, 7, 8)) -> None:
    for n in inflation_levels:
        print(f"\n\n New problem: exploring NSI with inflation level {n}.")
        distribution = NSIPRDistribution()
        prep = PrepLP(
            n,
            distribution,
            problem_name=f"NSI_n={n}",
            auto_discover_symmetries=True,
            compress_rows_under_discovered_group=True,
        )
        print(f"PrepLP initialized for n={n}; materializing LP inputs before Mosek.")
        _ = prep.global_keys
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
        print(
            f"Exact feasibility for n={n}: {solution['success']}. "
            f"Incompatible fraction: {solution['incompatible_fraction']:.12g}"
        )
        if prep.output_path is not None:
            prep.save_solution(solution)
            print(f"Saved LP solution archive to {prep.output_path}")


if __name__ == "__main__":
    main()
