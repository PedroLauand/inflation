"""
NSI_classicality_test.py
--------------------------------------------------------------------
Example test for classicality constraints using all-zero loop
probabilities, random 1-body marginals, and single-1 loop events set to 0
in the binary-output case.

Setup:
  - outcomes = 2
  - n = 4
  - impose P(0...0) via the NSI-PR loop constructor
  - impose random 1-body marginal P(A=0)=p for single-site loops
  - impose P(0...1...0)=0 for loop length >= 3 (all positions of the 1)
--------------------------------------------------------------------
"""

from __future__ import annotations
from typing import Iterable
from pathlib import Path
import sys

# Ensure repo root is on sys.path so "import inflation" works when running this file directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inflation.lp.lp_utils import solveLP_sparse
from scipy.sparse import coo_array
from inflation.distributions.nsi_pr import prob_event_loop as nsi_pr_prob_event_loop
from inflation.applications.Final_algo_numba import (
    ring_problem,
    run_pipeline,
    _perm_from_marginal,
    _outcomes_from_marginal,
    _cycles_from_J,
)


# Uniform 1-body marginal: P(A=0)=P(A=1)=1/2 for any single-site loop.
_P1_ZERO = 0.5
_P1_ONE = 0.5


def prob_zero_or_single_one_event(outcomes: Iterable[int]) -> float:
    """
    Event probability for a cycle:
      - all zeros: NSI-PR prob_event_loop([0] * len(outcomes))
      - length 1: random 1-body marginal (P(0)=_P1_ZERO, P(1)=_P1_ONE)
      - exactly one '1' and length >= 3: 0
    """
    out = tuple(int(x) for x in outcomes)
    if len(out) == 1:
        if out[0] == 0:
            return _P1_ZERO
        if out[0] == 1:
            return _P1_ONE
        raise ValueError("Binary outcomes only for length-1 cycles.")
    if all(x == 0 for x in out):
        return nsi_pr_prob_event_loop([0] * len(out))
    if len(out) >= 3 and sum(out) == 1 and all(x in (0, 1) for x in out):
        return 0.0
    raise ValueError("Unsupported outcome pattern for this test.")


def _marginal_supported(marginal) -> bool:
    """
    Keep marginals whose cycle outcomes are either:
      - all zeros, or
      - any single-bit outcome for cycles of length 1, or
      - exactly one 1 for cycles of length >= 3.
    """
    J = _perm_from_marginal(marginal)
    a = _outcomes_from_marginal(marginal)
    for cyc in _cycles_from_J(J):
        cyc_out = [a[i - 1] for i in cyc]
        if len(cyc_out) == 1:
            if cyc_out[0] in (0, 1):
                continue
            return False
        if all(x == 0 for x in cyc_out):
            continue
        if len(cyc_out) >= 3 and sum(cyc_out) == 1 and all(x in (0, 1) for x in cyc_out):
            continue
        return False
    return True


if __name__ == "__main__":
    n, outcomes = 4, 2

    prob = ring_problem(n, outcomes)
    # Optional: add outcome relabeling symmetries
    # prob.add_symmetries(prob._setting_specific_outcome_relabelling_symmetries)
    print(f"uniform 1-body marginal P(A=0)={_P1_ZERO:.6f}")
    print("done with prob")

    variable_names, known_vars_coo_vec, inflation_matrix = run_pipeline(
        prob,
        event_prob_fn=prob_zero_or_single_one_event,
        marginal_filter_fn=_marginal_supported,
    )
    nof_all_LP_vars = inflation_matrix.shape[1]

    solution = solveLP_sparse(
        objective=coo_array(([], ([], [])), shape=(1, nof_all_LP_vars)),
        known_vars=known_vars_coo_vec,
        equalities=inflation_matrix,
        default_non_negative=True,
        variables=variable_names,
        verbose=True,
    )

    print(solution["status"])
