"""
EJM_4_nofactorization.py
--------------------------------------------------------------------
Explore the EJM ring pipeline at n=4 while keeping only connected
marginals, i.e. those whose permutation decomposition has a single cycle.
This excludes disconnected product marginals.
--------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repo root is on sys.path so "import inflation" works when running directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inflation.applications.Final_algo_numba import PrepLP, keep_cycle_signatures
from inflation.distributions import EJMDistribution

def main(*, n: int = 4) -> None:
    distribution = EJMDistribution(coarsen=[[0,1],[2],[3]])
    allowed_signatures = [(length,) for length in range(2, n + 1)]
    prep = PrepLP(
        n,
        distribution,
        problem_name=f"EJM_n={n}_no_factorization_v2",
        marginal_filter_fn=keep_cycle_signatures(allowed_signatures),
        auto_discover_symmetries=True,
        compress_rows_under_discovered_group=True,
    )
    print(
        f"PrepLP initialized for EJM no-factorization n={n}; "
        "materializing LP inputs before Mosek."
    )
    _ = prep.global_keys
    print(
        f"Filtered LP inputs ready for EJM no-factorization n={n}: "
        f"base_rows={prep.base_nof_marginals}, rows={prep.nof_lp_constraints}, cols={prep.nof_lp_vars}. "
        "Starting relaxed incompatibility solve."
    )

    relaxed_solution = prep.solve(
        optimizer="free_simplex",
        verbose=2,
    )
    print(f"Relaxed LP status for EJM no-factorization n={n}: {relaxed_solution['status']}")
    print(
        f"Feasible within tolerance for EJM no-factorization n={n}: {relaxed_solution['success']}. "
        f"Incompatible fraction: {relaxed_solution['incompatible_fraction']:.12g}"
    )
    print(
        f"Known mass: {relaxed_solution['known_mass']:.12g}. "
        f"Optimized compatible mass: {relaxed_solution['optimized_mass']:.12g}"
    )
    if relaxed_solution["success"]:
        print("The EJM no-factorization n=4 problem is COMPATIBLE.")
    else:
        print("The EJM no-factorization n=4 problem is detected as INCOMPATIBLE.")
        print("Dual certificate from the relaxed LP:")
        prep.print_certificate(relaxed_solution)
    if prep.output_path is not None:
        prep.save_solution(relaxed_solution)
        print(f"Saved relaxed LP solution archive to {prep.output_path}")


if __name__ == "__main__":
    main()
