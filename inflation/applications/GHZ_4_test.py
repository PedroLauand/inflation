"""
GHZ_4_test.py
--------------------------------------------------------------------
Test the specified binary GHZ ring distribution at n=4 and print the
direct row-basis certificate returned by the relaxed LP.
--------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repo root is on sys.path so "import inflation" works when running directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inflation.applications.Final_algo_numba import (
    PrepLP,
)
from inflation.distributions import GHZDistribution


def main(*, n: int = 4) -> None:
    distribution = GHZDistribution()
    prep = PrepLP(
        n,
        distribution,
        problem_name=None,
        auto_discover_symmetries=True,
        compress_rows_under_discovered_group=True,
    )
    print(f"PrepLP initialized for GHZ n={n}; materializing LP inputs before Mosek.")
    _ = prep.global_keys
    print(
        f"LP inputs ready for GHZ n={n}: "
        f"rows={prep.nof_lp_constraints}, cols={prep.nof_lp_vars}. "
        "Starting relaxed incompatibility solve."
    )

    relaxed_solution = prep.solve(
        optimizer="free_simplex",
        verbose=2,
    )
    print(f"Relaxed LP status for GHZ n={n}: {relaxed_solution['status']}")
    print(
        f"Exact feasibility for GHZ n={n}: {relaxed_solution['success']}. "
        f"Incompatible fraction: {relaxed_solution['incompatible_fraction']:.12g}"
    )
    print(
        f"Known mass: {relaxed_solution['known_mass']:.12g}. "
        f"Optimized compatible mass: {relaxed_solution['optimized_mass']:.12g}"
    )
    if relaxed_solution["success"]:
        print(
            "The specified all-equal GHZ distribution is feasible at n=4 in the current ring LP, "
            "so the relaxed LP does not provide an incompatibility certificate."
        )
    else:
        print("GHZ is detected as incompatible by the relaxed LP.")
        print("Dual certificate from the relaxed LP:")
        prep.print_certificate(relaxed_solution)

    if prep.output_path is not None:
        prep.save_solution(relaxed_solution)
        print(f"Saved relaxed LP solution archive to {prep.output_path}")


if __name__ == "__main__":
    main()
