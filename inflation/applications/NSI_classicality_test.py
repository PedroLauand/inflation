"""
NSI_classicality_test.py
--------------------------------------------------------------------
Example test for classicality constraints using all-zero loop
probabilities, random 1-body marginals, and single-1 loop events set to 0
in the binary-output case.

Setup:
  - outcomes = 2
  - n = 4
  - impose P(0...0) via prob_loop(n)
  - impose random 1-body marginal P(A=0)=p for single-site loops
  - impose P(0...1...0)=0 for loop length >= 3 (all positions of the 1)
--------------------------------------------------------------------
"""

from __future__ import annotations
from typing import Iterable, Tuple
from pathlib import Path
import numpy as np
import sys

# Ensure repo root is on sys.path so "import inflation" works when running this file directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inflation import InflationProblem
from inflation.lp.lp_utils import solveLP_sparse
from tqdm.auto import tqdm
from scipy.sparse import coo_array
from numba import types
from numba.typed import Dict as NumbaDict

from postquantum_2outcomes import prob_loop
from Group_utils import build_sympy_group, prepare_group_chain
from symmetric_classical_test import (
    ring_problem,
    generate_minimal_marginal_events,
    factorized_marginal_value,
    representatives_of_global_extensions_uint64,
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
      - all zeros: prob_loop(len(outcomes))
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
        return prob_loop(len(out))
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


def run_pipeline_all_zero(
    prob: InflationProblem,
    *,
    show_progress: bool = True,
) -> Tuple[np.ndarray, coo_array, coo_array]:
    """
    Build the inflation matrix, known values vector, and variable names
    but only for all-zero outcome marginals.
    """
    n = prob.inflation_level_per_source[0]
    outcomes = prob.outcomes_per_party[0]
    raw_G = prob.symmetries
    if outcomes >= 255:
        raise ValueError("outcomes must be < 255 to fit in compact dtypes")
    assert n <= 5, "uint64 canonical events are only supported up to n=5"
    max_event_count = pow(outcomes, n * n)
    assert max_event_count <= np.iinfo(np.uint64).max, "events do not fit in uint64"

    # group acts on one-hot coordinates of size N = n^2 * outcomes
    N = (n * n) * outcomes
    if N > np.iinfo(np.uint16).max:
        raise ValueError("N exceeds uint16 range; use wider dtype for permutations")
    G = build_sympy_group(raw_G, N)
    level_invperms = prepare_group_chain(G, N)

    list_of_all_LP_variables = ["1"]
    all_marginals = generate_minimal_marginal_events(n, outcomes)
    marginals = [m for m in all_marginals if _marginal_supported(m)]
    nof_marginals = len(marginals)

    global_event_map = NumbaDict.empty(key_type=types.uint64, value_type=types.int32)
    next_event_idx = np.int32(1 + nof_marginals)
    global_extension_count = int(pow(outcomes, n * (n - 1)))
    total_entries = int(nof_marginals * (1 + global_extension_count))
    sparse_matrix_rows = np.empty(total_entries, dtype=np.int32)
    sparse_matrix_cols = np.empty(total_entries, dtype=np.int32)
    sparse_matrix_data = np.ones(total_entries, dtype=np.int8)
    sparse_matrix_rows[:nof_marginals] = np.arange(nof_marginals, dtype=np.int32)
    sparse_matrix_cols[:nof_marginals] = np.arange(1, nof_marginals + 1, dtype=np.int32)
    sparse_matrix_data[:nof_marginals] = -1
    row_grid = np.broadcast_to(
        np.arange(nof_marginals, dtype=np.int32)[:, None],
        (nof_marginals, global_extension_count),
    )
    sparse_matrix_rows[nof_marginals:] = row_grid.reshape(-1)

    known_values = np.empty(nof_marginals, dtype=float)
    for idx, marginal in enumerate(
        tqdm(marginals, desc="Computing all-zero marginal values...", disable=not show_progress)
    ):
        val = factorized_marginal_value(marginal, prob_zero_or_single_one_event)
        mkey = tuple(prob._lexrepr_to_names[prob.mon_to_lexrepr(marginal)])
        list_of_all_LP_variables.append("P_global(" + ",".join(mkey) + ")")
        known_values[idx] = val

    for row_num, marginal in enumerate(
        tqdm(marginals, desc="Finding global extensions...", disable=not show_progress)
    ):
        start = nof_marginals + row_num * global_extension_count
        next_event_idx = representatives_of_global_extensions_uint64(
            n=n,
            outcomes=outcomes,
            marginal=marginal,
            level_invperms=level_invperms,
            global_event_map=global_event_map,
            next_event_idx=next_event_idx,
            list_of_all_LP_variables=list_of_all_LP_variables,
            total=global_extension_count,
            sparse_matrix_cols=sparse_matrix_cols,
            start=start,
        )

    if int(next_event_idx) > np.iinfo(np.int32).max:
        raise ValueError("next_event_idx exceeds int32 range; use wider dtype")
    inflation_matrix = coo_array(
        (sparse_matrix_data, (sparse_matrix_rows, sparse_matrix_cols)),
        shape=(nof_marginals, int(next_event_idx)),
    )
    inflation_matrix.sum_duplicates()

    known_positions = np.arange(1, nof_marginals + 1, dtype=np.int32)
    known_rows = np.broadcast_to(np.int32(0), (nof_marginals,))
    known_vars_coo_vec = coo_array(
        (known_values, (known_rows, known_positions)),
        shape=(1, int(next_event_idx)),
    )
    variable_names = np.asarray(list_of_all_LP_variables, dtype=str)

    return variable_names, known_vars_coo_vec, inflation_matrix


if __name__ == "__main__":
    n, outcomes = 4, 2

    prob = ring_problem(n, outcomes)
    # Optional: add outcome relabeling symmetries
    # prob.add_symmetries(prob._setting_specific_outcome_relabelling_symmetries)
    print(f"uniform 1-body marginal P(A=0)={_P1_ZERO:.6f}")
    print("done with prob")

    variable_names, known_vars_coo_vec, inflation_matrix = run_pipeline_all_zero(prob)
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
