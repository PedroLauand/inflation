"""
NSI_classicality_test.py
--------------------------------------------------------------------
Example test for classicality constraints in the off-diagonal ring
semantics, using all-zero loop probabilities and single-1 loop events set
to 0 in the binary-output case.

Setup:
  - outcomes = 2
  - n = 4
  - impose P(0...0) via the NSI-PR loop constructor
  - impose P(0...1...0)=0 for loop length >= 2 (all positions of the 1)
--------------------------------------------------------------------
"""

from __future__ import annotations
from typing import Iterable
from pathlib import Path
import sys

import sympy as sp

# Ensure repo root is on sys.path so "import inflation" works when running this file directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inflation.lp.lp_utils import solveLP_sparse
from scipy.sparse import coo_array
from inflation.distributions import NSIPRDistribution
from inflation.applications.Final_algo_numba import (
    PrepLP,
    _cycles_from_J,
    _perm_from_marginal,
    _outcomes_from_marginal,
)


_NSI = NSIPRDistribution()


def prob_zero_or_single_one_event(outcomes: Iterable[int]) -> sp.Expr:
    """
    Event probability for a cycle:
      - all zeros: NSI-PR prob_event_loop([0] * len(outcomes))
      - exactly one '1' and length >= 2: 0
    """
    out = tuple(int(x) for x in outcomes)
    if len(out) < 2:
        raise ValueError("Off-diagonal ring cycles must have length at least 2.")
    if all(x == 0 for x in out):
        return _NSI.prob_event_loop([0] * len(out))
    if sum(out) == 1 and all(x in (0, 1) for x in out):
        return sp.Integer(0)
    raise ValueError("Unsupported outcome pattern for this test.")


class ZeroSingleOneDistribution:
    @property
    def nof_outcomes(self) -> int:
        return 2

    def prob_event_loop(self, outcomes: Iterable[int]) -> sp.Expr:
        return prob_zero_or_single_one_event(outcomes)

    def prob_event_line(self, outcomes: Iterable[int]) -> sp.Expr:
        raise NotImplementedError("This test only uses loop-event probabilities.")


def _marginal_supported(marginal) -> bool:
    """
    Keep marginals whose cycle outcomes are either:
      - all zeros, or
      - exactly one 1 for cycles of length >= 2.
    """
    J = _perm_from_marginal(marginal)
    a = _outcomes_from_marginal(marginal)
    for cyc in _cycles_from_J(J):
        cyc_out = [a[i] for i in cyc]
        if all(x == 0 for x in cyc_out):
            continue
        if sum(cyc_out) == 1 and all(x in (0, 1) for x in cyc_out):
            continue
        return False
    return True


if __name__ == "__main__":
    n = 4
    distribution = ZeroSingleOneDistribution()

    prep = PrepLP(
        n,
        distribution,
        cache_name=None,
        marginal_filter_fn=_marginal_supported,
    )
    variable_names = prep.variable_names
    known_vars = prep.known_vars
    inflation_matrix = prep.inflation_matrix
    nof_all_LP_vars = inflation_matrix.shape[1]

    solution = solveLP_sparse(
        objective=coo_array(([], ([], [])), shape=(1, nof_all_LP_vars)),
        known_vars=known_vars,
        equalities=inflation_matrix,
        default_non_negative=True,
        variables=variable_names,
        verbose=True,
    )

    print(solution["status"])
