"""
EJM_4_test.py
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


def main(*, n: int = 4) -> None:
    distribution = EJMDistribution()
    prep = PrepLP(
        n,
        distribution,
        problem_name=f"EJM_n={n}",
        auto_discover_symmetries=True,
        compress_rows_under_discovered_group=True,
    )
    print(f"PrepLP initialized for n={n}; materializing LP inputs before Mosek.")
    _ = prep.global_keys
    print(
        f"LP inputs ready for n={n}: "
        f"rows={prep.nof_lp_constraints}, cols={prep.nof_lp_vars}. "
        "Starting relaxed incompatibility solve."
    )

    solution = prep.solve(
        optimizer="primal_simplex",
        verbose=2,
    )
    print(f"Relaxed LP status for EJM n={n}: {solution['status']}")
    print(
        f"Feasible within tolerance for EJM n={n}: {solution['success']}. "
        f"Incompatible fraction: {solution['incompatible_fraction']:.12g}"
    )
    print(
        f"Known mass: {solution['known_mass']:.12g}. "
        f"Optimized compatible mass: {solution['optimized_mass']:.12g}"
    )
    if solution["success"]:
        print("EJM is feasible within tolerance in the current ring LP.")
    else:
        print("EJM is detected as incompatible by the relaxed LP.")
        print("Dual certificate from the relaxed LP:")
        prep.print_certificate(solution)
    if prep.output_path is not None:
        prep.save_solution(solution)
        print(f"Saved relaxed LP solution archive to {prep.output_path}")


if __name__ == "__main__":
    main()
