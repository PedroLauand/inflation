"""
symmetric_inflation_test.py
--------------------------------------------------------------------
Expectation-driven wrapper of the canonical PrepLP pipeline.
--------------------------------------------------------------------
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import List, Tuple
import sys

import numpy as np
import sympy as sp

# Ensure repo root is on sys.path so "import inflation" works when running directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inflation.applications.Final_algo_numba import PrepLP
from inflation.distributions.protocols import RingDistributionProtocol


def loop_prob_from_correlators(
    outcomes: Tuple[int, ...] | List[int],
    E_line: dict[int, float],
    E_loop: dict[int, float],
) -> sp.Expr:
    """
    Probability for a loop of length m with binary outcomes {0,1},
    using line correlators E_k (k < m) and loop correlator E^o_m.
    """
    m = len(outcomes)
    if m == 0:
        raise ValueError("outcomes must be non-empty")
    if any(o not in (0, 1) for o in outcomes):
        raise ValueError("This demo assumes binary outcomes {0,1}.")

    x = [sp.Integer(1) if o == 0 else sp.Integer(-1) for o in outcomes]
    total = sp.Integer(1)

    for k in range(2, m):
        if k not in E_line:
            raise KeyError(f"missing E_line[{k}] for loop length {m}")
        seg_sum = sp.Integer(0)
        for start in range(m):
            prod = sp.Integer(1)
            for t in range(k):
                prod *= x[(start + t) % m]
            seg_sum += prod
        total += sp.sympify(E_line[k]) * seg_sum

    if m not in E_loop:
        raise KeyError(f"missing E_loop[{m}] for loop length {m}")
    prod_all = sp.Integer(1)
    for val in x:
        prod_all *= val
    total += sp.sympify(E_loop[m]) * prod_all

    return sp.simplify(total / (sp.Integer(2) ** m))


def sanity_check_loop_distributions(
    E_line: dict[int, float],
    E_loop: dict[int, float],
    *,
    tol: float = 1e-9,
) -> None:
    lengths = sorted(E_loop.keys())
    if not lengths:
        print("No loop lengths found in E_loop; skipping sanity checks.")
        return

    print("\nLoop distribution sanity checks:")
    for m in lengths:
        outcomes_list = list(product((0, 1), repeat=m))
        probs = np.array([float(sp.N(loop_prob_from_correlators(out, E_line, E_loop))) for out in outcomes_list])
        total = float(probs.sum())
        min_p = float(probs.min(initial=np.inf))
        max_p = float(probs.max(initial=-np.inf))
        neg_count = int((probs < -tol).sum())
        over_count = int((probs > 1.0 + tol).sum())

        print(f"  m={m}: sum={total:.12g} (|sum-1|={abs(total-1.0):.3g})")
        print(
            f"       min={min_p:.12g}, max={max_p:.12g}, "
            f"negatives={neg_count}, >1={over_count}"
        )


class ExpectationDistribution(RingDistributionProtocol):
    def __init__(self, E_line: dict[int, float], E_loop: dict[int, float]) -> None:
        self.E_line = E_line
        self.E_loop = E_loop

    @property
    def nof_outcomes(self) -> int:
        return 2

    def prob_event_loop(self, outcomes: Tuple[int, ...] | List[int]) -> sp.Expr:
        return loop_prob_from_correlators(outcomes, self.E_line, self.E_loop)

    def prob_event_line(self, outcomes: Tuple[int, ...] | List[int]) -> sp.Expr:
        # This test path only needs loop values, but protocol requires both methods.
        raise NotImplementedError("ExpectationDistribution is loop-only in this demo.")


class PrepLPExpectations(PrepLP):
    """Expectation-parameterized subclass of PrepLP."""

    def __init__(
        self,
        n: int,
        E_line: dict[int, float],
        E_loop: dict[int, float],
        *,
        add_normalization: bool = True,
        problem_name: str | None = None,
        show_progress: bool = True,
        marginal_filter_fn=None,
        auto_discover_symmetries: bool = True,
        compress_rows_under_discovered_group: bool = True,
        verbose_symmetry_discovery: bool = True,
    ) -> None:
        self.E_line = E_line
        self.E_loop = E_loop
        self.add_normalization = add_normalization
        super().__init__(
            n,
            ExpectationDistribution(E_line, E_loop),
            problem_name=problem_name,
            marginal_filter_fn=marginal_filter_fn,
            show_progress=show_progress,
            auto_discover_symmetries=auto_discover_symmetries,
            compress_rows_under_discovered_group=compress_rows_under_discovered_group,
            verbose_symmetry_discovery=verbose_symmetry_discovery,
        )


if __name__ == "__main__":
    n = 3
    sqrt2 = np.sqrt(2.0)
    E_line = {
        1: 0.0,
        2: sqrt2 - 1.0,
        3: 3.0 - 2.0 * sqrt2,
    }
    E_loop = {
        1: 0.0,
        2: 1.0,
        3: 2.0 - sqrt2,
    }

    sanity_check_loop_distributions(E_line, E_loop, tol=1e-9)
    prep = PrepLPExpectations(
        n,
        E_line,
        E_loop,
        add_normalization=True,
        problem_name=None,
        show_progress=True,
        auto_discover_symmetries=True,
        compress_rows_under_discovered_group=True,
    )
    _ = prep.global_keys

    solution = prep.solve(
        verbose=True,
    )
    print(solution["status"])
    print(
        f"Exact feasibility: {solution['success']}. "
        f"Incompatible fraction: {solution['incompatible_fraction']:.12g}"
    )
