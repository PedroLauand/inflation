"""
GHZ_4_debug.py
--------------------------------------------------------------------
Debug the GHZ ring pipeline at n=4 with the closest currently-available
filter to:
  "keep loops of length exactly 3, or lines of length 2 or 1".

In this branch, PrepLP only generates disjoint unions of cycles, so open
line marginals are not represented. As a result, the effective filter in
this script is "keep only single 3-cycles".
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
    _cycles_from_J,
    _perm_from_marginal,
)
from inflation.distributions import GHZDistribution


def keep_three_cycles_or_short_lines_if_available(marginal) -> bool:
    """
    Keep only marginals that are exactly one 3-cycle.

    Open line marginals of length 1 or 2 are not generated in the current
    ring pipeline, so they cannot be admitted here via marginal_filter_fn.
    """
    cycles = _cycles_from_J(_perm_from_marginal(marginal))
    return len(cycles) == 1 and len(cycles[0]) == 3


def main(*, n: int = 4) -> None:
    distribution = GHZDistribution()
    prep = PrepLP(
        n,
        distribution,
        problem_name=None,
        marginal_filter_fn=keep_three_cycles_or_short_lines_if_available,
        auto_discover_symmetries=True,
        compress_rows_under_discovered_group=True,
    )
    print(
        "Debug note: the current ring PrepLP only generates disjoint unions of cycles. "
        "Open line marginals of length 2 or 1 are not represented in this branch."
    )
    print(
        "A loop of length 3 does not automatically imply that lines of length 2 or 1 "
        "are generated or included. This debug run therefore keeps only single 3-cycles "
        "via marginal_filter_fn, with inflation level still fixed at 4."
    )
    print(f"PrepLP initialized for GHZ debug n={n}; materializing LP inputs before Mosek.")
    _ = prep.global_keys
    print(
        f"Filtered LP inputs ready for GHZ debug n={n}: "
        f"base_rows={prep.base_nof_marginals}, rows={prep.nof_lp_constraints}, cols={prep.nof_lp_vars}. "
        "Starting relaxed incompatibility solve."
    )

    relaxed_solution = prep.solve(
        optimizer="free_simplex",
        verbose=2,
    )
    print(f"Relaxed LP status for GHZ debug n={n}: {relaxed_solution['status']}")
    print(
        f"Exact feasibility for GHZ debug n={n}: {relaxed_solution['success']}. "
        f"Incompatible fraction: {relaxed_solution['incompatible_fraction']:.12g}"
    )
    print(
        f"Known mass: {relaxed_solution['known_mass']:.12g}. "
        f"Optimized compatible mass: {relaxed_solution['optimized_mass']:.12g}"
    )
    if relaxed_solution["success"]:
        print(
            "With only single 3-cycles retained, the filtered GHZ n=4 problem is compatible "
            "in the current ring LP."
        )
    else:
        print("The filtered GHZ n=4 problem is still detected as incompatible.")
        print("Dual certificate from the relaxed LP:")
        prep.print_certificate(relaxed_solution)


if __name__ == "__main__":
    main()
