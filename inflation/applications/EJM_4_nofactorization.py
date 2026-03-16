"""
EJM_4_loops34_plus_2x2.py
--------------------------------------------------------------------
Explore the EJM ring pipeline at n=4 with a filter that keeps:
  - single loops of length 3,
  - single loops of length 4,
  - disjoint unions of exactly two loops of length 2,
while excluding marginals consisting of only one 2-cycle.
--------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repo root is on sys.path so "import inflation" works when running directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inflation.applications.Final_algo_numba import PrepLP, _cycles_from_J, _perm_from_marginal
from inflation.distributions import EJMDistribution


def keep_34_or_exactly_two_2cycles(marginal) -> bool:
    cycles = _cycles_from_J(_perm_from_marginal(marginal))
    lengths = sorted(len(cycle) for cycle in cycles)
    return lengths in ([3], [4], [2, 2])

def keep_234_no_factorization(marginal) -> bool:
    cycles = _cycles_from_J(_perm_from_marginal(marginal))
    return len(cycles)==1


def main(*, n: int = 4) -> None:
    distribution = EJMDistribution()
    prep = PrepLP(
        n,
        distribution,
        # problem_name=f"EJM_n={n}_loops34_plus_2x2",
        problem_name=f"EJM_n={n}_no_factorization",
        marginal_filter_fn=keep_234_no_factorization,
        auto_discover_symmetries=True,
        compress_rows_under_discovered_group=True,
    )
    # print(
    #     "This filtered run keeps single loops of length 3 or 4, plus marginals "
    #     "consisting of exactly two 2-cycles, while excluding isolated 2-cycles."
    # )
    print(
        f"PrepLP initialized for EJM loops34+2x2 n={n}; "
        "materializing LP inputs before Mosek."
    )
    _ = prep.global_keys
    print(
        f"Filtered LP inputs ready for EJM loops34+2x2 n={n}: "
        f"base_rows={prep.base_nof_marginals}, rows={prep.nof_lp_constraints}, cols={prep.nof_lp_vars}. "
        "Starting relaxed incompatibility solve."
    )

    relaxed_solution = prep.solve(
        optimizer="free_simplex",
        verbose=2,
    )
    print(f"Relaxed LP status for EJM loops34+2x2 n={n}: {relaxed_solution['status']}")
    print(
        f"Feasible within tolerance for EJM loops34+2x2 n={n}: {relaxed_solution['success']}. "
        f"Incompatible fraction: {relaxed_solution['incompatible_fraction']:.12g}"
    )
    print(
        f"Known mass: {relaxed_solution['known_mass']:.12g}. "
        f"Optimized compatible mass: {relaxed_solution['optimized_mass']:.12g}"
    )
    if relaxed_solution["success"]:
        print("The filtered EJM n=4 problem is COMPATIBLE.")
    else:
        print("The filtered EJM n=4 problem is detected as INCOMPATIBLE.")
        print("Dual certificate from the relaxed LP:")
        prep.print_certificate(relaxed_solution)
    if prep.output_path is not None:
        prep.save_solution(relaxed_solution)
        print(f"Saved relaxed LP solution archive to {prep.output_path}")


if __name__ == "__main__":
    main()
