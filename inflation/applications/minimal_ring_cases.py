"""
Compact runner for small ring-inflation cases.

The goal is to keep the EJM/NSI/GHZ examples in one place while we search for
small incompatibility witnesses.  Cases are deliberately named by the
distribution, inflation level, and marginal family.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable, Iterable

import sympy as sp

# Ensure repo root is on sys.path so "import inflation" works when running directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inflation.applications.Final_algo_numba import (  # noqa: E402
    PrepLP,
    keep_cycle_signatures,
    keep_loops_of_length,
)
from inflation.distributions import EJMDistribution, GHZDistribution, NSIPRDistribution  # noqa: E402


class ParityBinaryDistribution:
    """Small artificial binary distribution used as a minimal regression witness."""

    @property
    def nof_outcomes(self) -> int:
        return 2

    def prob_event_loop(self, outcomes: Iterable[int]) -> sp.Expr:
        return sp.Integer(1) if sum(tuple(outcomes)) % 2 == 0 else sp.Integer(0)

    def prob_event_line(self, outcomes: Iterable[int]) -> sp.Expr:
        return self.prob_event_loop(outcomes)


@dataclass(frozen=True)
class RingCase:
    name: str
    n: int
    distribution_factory: Callable[[], object]
    marginal_filter_factory: Callable[[], object | None]
    note: str


def _no_filter():
    return None


CASES = {
    "parity-n3": RingCase(
        name="parity-n3",
        n=3,
        distribution_factory=ParityBinaryDistribution,
        marginal_filter_factory=_no_filter,
        note="Artificial minimal incompatible regression witness.",
    ),
    "ghz-n3": RingCase(
        name="ghz-n3",
        n=3,
        distribution_factory=GHZDistribution,
        marginal_filter_factory=_no_filter,
        note="Feasible binary control case.",
    ),
    "nsi-n4": RingCase(
        name="nsi-n4",
        n=4,
        distribution_factory=NSIPRDistribution,
        marginal_filter_factory=_no_filter,
        note="NSI-PR n=4 control; currently feasible in this relaxation.",
    ),
    "nsi-n5": RingCase(
        name="nsi-n5",
        n=5,
        distribution_factory=NSIPRDistribution,
        marginal_filter_factory=_no_filter,
        note="Small NSI-PR incompatibility witness in the full n=5 relaxation.",
    ),
    "nsi-n6-triangles": RingCase(
        name="nsi-n6-triangles",
        n=6,
        distribution_factory=NSIPRDistribution,
        marginal_filter_factory=lambda: keep_cycle_signatures([(3, 3)]),
        note="NSI-PR n=6 with two-triangle marginals only.",
    ),
    "ejm-n4-single-loops": RingCase(
        name="ejm-n4-single-loops",
        n=4,
        distribution_factory=EJMDistribution,
        marginal_filter_factory=lambda: keep_cycle_signatures([(2,), (3,), (4,)]),
        note="Small EJM incompatibility witness found by subset sweep.",
    ),
    "ejm-n4-34-plus-22": RingCase(
        name="ejm-n4-34-plus-22",
        n=4,
        distribution_factory=EJMDistribution,
        marginal_filter_factory=lambda: keep_cycle_signatures([(3,), (4,), (2, 2)]),
        note="Alternative EJM incompatible family; larger than single-loops.",
    ),
    "ejm-n4-all-cycle-signatures": RingCase(
        name="ejm-n4-all-cycle-signatures",
        n=4,
        distribution_factory=EJMDistribution,
        marginal_filter_factory=lambda: keep_cycle_signatures([(2,), (3,), (4,), (2, 2)]),
        note="All n=4 cycle signatures used in the compact EJM sweep.",
    ),
    "ejm-n4-max3loop": RingCase(
        name="ejm-n4-max3loop",
        n=4,
        distribution_factory=EJMDistribution,
        marginal_filter_factory=lambda: keep_loops_of_length([1, 2, 3]),
        note="Matches the old EJM_4_max3loop.py filter.",
    ),
}


def build_prep(case: RingCase, *, show_progress: bool) -> PrepLP:
    return PrepLP(
        case.n,
        case.distribution_factory(),
        problem_name=None,
        marginal_filter_fn=case.marginal_filter_factory(),
        show_progress=show_progress,
        verbose_cache=False,
        auto_discover_symmetries=True,
        compress_rows_under_discovered_group=True,
    )


def run_case(case: RingCase, *, prepare_only: bool, show_progress: bool) -> None:
    print(f"case={case.name}")
    print(f"note={case.note}")
    prep = build_prep(case, show_progress=show_progress)
    print(
        "size="
        f"base_rows={prep.base_nof_marginals}, "
        f"rows={prep.nof_lp_constraints}, "
        f"cols={prep.nof_lp_vars}"
    )
    if prepare_only:
        return
    solution = prep.solve(mode="incompatible_fraction", optimizer="free_simplex", verbose=0)
    print(
        "solve="
        f"status={solution['status']}, "
        f"success={solution['success']}, "
        f"incompatible_fraction={float(solution['incompatible_fraction']):.12g}, "
        f"known_mass={float(solution['known_mass']):.12g}, "
        f"optimized_mass={float(solution['optimized_mass']):.12g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", choices=sorted(CASES), help="Case name to run.")
    parser.add_argument("--list", action="store_true", help="List available compact cases.")
    parser.add_argument("--prepare-only", action="store_true", help="Build LP inputs without solving.")
    parser.add_argument("--progress", action="store_true", help="Show detailed PrepLP progress.")
    args = parser.parse_args()

    if args.list:
        for case in CASES.values():
            print(f"{case.name}: n={case.n}; {case.note}")
        return
    if args.case is None:
        parser.error("provide a case name or use --list")
    run_case(CASES[args.case], prepare_only=args.prepare_only, show_progress=args.progress)


if __name__ == "__main__":
    main()
