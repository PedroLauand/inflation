"""
symmetric_inflation_test.py
--------------------------------------------------------------------
Inflation-problem setup (2 sources, 1 party) and utilities to build
symmetrized marginalization matrices.

LP structure (as built here):
  - Variables: symmetrized global-event probabilities q'
  - Constraints: for each marginal operator list M,
      sum_{global extensions of M} q' = RHS(M)
    where RHS(M) is computed from loop correlators via factorization.

Nonnegativity/normalization will be added when the full feasibility
LP is constructed.
--------------------------------------------------------------------
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
from itertools import product
import sys
import numpy as np

# Ensure repo root is on sys.path so "import inflation" works when running directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inflation import InflationProblem
from scipy.sparse import coo_array
from tqdm.auto import tqdm
from numba import types
from numba.typed import Dict as NumbaDict

from Group_utils import build_sympy_group, prepare_group_chain
from symmetric_classical_test import (
    integer_partitions,
    representative_perm_for_partition,
    representatives_of_global_extensions_uint64,
    factorized_marginal_value,
)


def exists_shared_source_modified(
    inf_indices1: np.ndarray,
    inf_indices2: np.ndarray,
) -> bool:
    """Check if two inflation indices share a non-disjoint source assignment."""
    common_sources = np.logical_and(inf_indices1, inf_indices2)
    if not np.any(common_sources):
        return False
    return not set(inf_indices1[common_sources]).isdisjoint(set(inf_indices2[common_sources]))


def overlap_matrix(all_inflation_indxs: np.ndarray) -> np.ndarray:
    """Build overlap adjacency matrix for inflation indices."""
    n = len(all_inflation_indxs)
    adj_mat = np.eye(n, dtype=bool)
    for i in range(1, n):
        inf_indices_i = all_inflation_indxs[i]
        for j in range(i):
            inf_indices_j = all_inflation_indxs[j]
            if exists_shared_source_modified(inf_indices_i, inf_indices_j):
                adj_mat[i, j] = True
    adj_mat = np.logical_or(adj_mat, adj_mat.T)
    return adj_mat


def inf_problem(inflation_level: int, nof_outcomes: int = 2) -> InflationProblem:
    """
    Build the ring inflation problem with two sources feeding a single party.

    Parameters
    ----------
    inflation_level : int
        Number of inflation copies per source (n).
    nof_outcomes : int
        Number of outcomes per party.
    """
    inf_prob = InflationProblem(
        dag={"i1": ["A"], "i2": ["A"]},
        outcomes_per_party=[nof_outcomes],
        settings_per_party=[1],
        classical_sources=None,
        inflation_level_per_source=(inflation_level, inflation_level),
        order=("A",),
    )

    # Fix factorization (overlap between inflation indices)
    inf_prob._inflation_indices_overlap = overlap_matrix(
        inf_prob._all_unique_inflation_indices
    )

    # Stabilize symmetries: keep only those that preserve the diagonal structure
    to_stabilize = np.flatnonzero(inf_prob._lexorder[:, 1] == inf_prob._lexorder[:, 2])
    new_symmetries = np.array(
        [perm for perm in inf_prob.symmetries
         if np.array_equal(np.sort(perm[to_stabilize]), to_stabilize)],
        dtype=int,
    )
    inf_prob.symmetries = new_symmetries

    return inf_prob


def _build_group_chain(prob: InflationProblem) -> List[np.ndarray]:
    """
    Build the group chain (inverses) used to canonicalize global events.

    This delegates all group-specific logic to Group_utils.
    """
    n = prob.inflation_level_per_source[0]
    outcomes = prob.outcomes_per_party[0]
    # Group acts on one-hot coordinates of size N = n^2 * outcomes.
    N = (n * n) * outcomes
    if N > np.iinfo(np.uint16).max:
        raise ValueError("N exceeds uint16 range; use wider dtype for permutations")
    G = build_sympy_group(prob.symmetries, N)
    return prepare_group_chain(G, N)


def generate_all_marginal_events(n: int, outcomes: int) -> List[List[List[int]]]:
    """
    Generate all marginals [[1,i,J(i),0,a_i] for i=1..n] without outcome
    relabeling reduction. We still keep one representative per conjugacy
    class of permutations (cycle type), but include all outcome strings.
    """
    if n <= 0:
        return []
    all_marginals: List[List[List[int]]] = []
    all_outcomes = list(product(range(outcomes), repeat=n))
    for parts in integer_partitions(n):
        J = representative_perm_for_partition(np.array(parts, dtype=np.int64))  # 1-line
        for pat in all_outcomes:
            marginal = [[1, i, J[i - 1], 0, pat[i - 1]] for i in range(1, n + 1)]
            all_marginals.append(marginal)
    return all_marginals


def build_symmetrized_lhs(
    prob: InflationProblem,
    *,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, coo_array]:
    """
    Build the left-hand-side matrix N for constraints N q' = rhs,
    where q' are symmetrized global-event variables (columns).

    Returns
    -------
    row_labels : np.ndarray[object]
        Tuple-of-names for each marginal row.
    col_keys : np.ndarray[uint64]
        Canonical keys for symmetrized global events, ordered by discovery.
        Column index j corresponds to col_keys[j-1] for j >= 1
        (column 0 is unused in this raw form).
    lhs_matrix : coo_array
        Sparse matrix with ones at each compatible global extension.
    """
    n = prob.inflation_level_per_source[0]
    outcomes = prob.outcomes_per_party[0]

    if outcomes >= 255:
        raise ValueError("outcomes must be < 255 to fit in compact dtypes")
    assert n <= 5, "uint64 canonical events are only supported up to n=5"
    max_event_count = pow(outcomes, n * n)
    assert max_event_count <= np.iinfo(np.uint64).max, "events do not fit in uint64"

    # Build the symmetry group chain (via Group_utils) for canonicalization.
    level_invperms = _build_group_chain(prob)

    # 1) Enumerate marginals (operator lists) we want to constrain.
    marginals = generate_all_marginal_events(n, outcomes)
    nof_marginals = len(marginals)

    # 2) Prepare sparse matrix storage for all global extensions of each marginal.
    global_event_map = NumbaDict.empty(key_type=types.uint64, value_type=types.int32)
    list_of_all_global_keys: List[int] = []
    next_event_idx = np.int32(1)  # 0 is reserved as "not present" sentinel

    global_extension_count = int(pow(outcomes, n * (n - 1)))
    total_entries = int(nof_marginals * global_extension_count)
    sparse_matrix_rows = np.empty(total_entries, dtype=np.int32)
    sparse_matrix_cols = np.empty(total_entries, dtype=np.int32)
    sparse_matrix_data = np.ones(total_entries, dtype=np.int8)

    row_grid = np.broadcast_to(
        np.arange(nof_marginals, dtype=np.int32)[:, None],
        (nof_marginals, global_extension_count),
    )
    sparse_matrix_rows[:] = row_grid.reshape(-1)

    # 3) Build readable row labels (names of operators in the marginal).
    row_labels: List[Tuple[str, ...]] = []
    for marginal in marginals:
        mkey = tuple(prob._lexrepr_to_names[prob.mon_to_lexrepr(marginal)])
        row_labels.append(mkey)

    # 4) For each marginal, enumerate all compatible global extensions,
    #    canonicalize by symmetry, and write the column indices.
    for row_num, marginal in enumerate(
        tqdm(marginals, desc="Enumerating global extensions...", disable=not show_progress)
    ):
        start = row_num * global_extension_count
        next_event_idx = representatives_of_global_extensions_uint64(
            n=n,
            outcomes=outcomes,
            marginal=marginal,
            level_invperms=level_invperms,
            global_event_map=global_event_map,
            next_event_idx=next_event_idx,
            list_of_all_LP_variables=list_of_all_global_keys,
            total=global_extension_count,
            sparse_matrix_cols=sparse_matrix_cols,
            start=start,
        )

    if int(next_event_idx) > np.iinfo(np.int32).max:
        raise ValueError("next_event_idx exceeds int32 range; use wider dtype")

    lhs_matrix = coo_array(
        (sparse_matrix_data, (sparse_matrix_rows, sparse_matrix_cols)),
        shape=(nof_marginals, int(next_event_idx)),
    )
    lhs_matrix.sum_duplicates()

    return (
        np.asarray(row_labels, dtype=object),
        np.asarray(list_of_all_global_keys, dtype=np.uint64),
        lhs_matrix,
    )


def loop_prob_from_correlators(
    outcomes: Tuple[int, ...] | List[int],
    E_line: dict[int, float],
    E_loop: dict[int, float],
) -> float:
    """
    Probability for a loop of length m = len(outcomes) with binary outcomes {0,1},
    using line correlators E_k (k < m) and loop correlator E^o_m.

    Outcomes are mapped to x in {+1,-1} via x = (-1)^a (a in {0,1}).
    """
    m = len(outcomes)
    if m == 0:
        raise ValueError("outcomes must be non-empty")
    if any(o not in (0, 1) for o in outcomes):
        raise ValueError("this RHS demo assumes binary outcomes {0,1}")

    x = [1 if o == 0 else -1 for o in outcomes]
    total = 1.0

    # Connected k-body correlators around the loop (k=2..m-1).
    for k in range(2, m):
        if k not in E_line:
            raise KeyError(f"missing E_line[{k}] for loop length {m}")
        seg_sum = 0.0
        for start in range(m):
            prod = 1
            for t in range(k):
                prod *= x[(start + t) % m]
            seg_sum += prod
        total += E_line[k] * seg_sum

    # Full loop correlator.
    if m not in E_loop:
        raise KeyError(f"missing E_loop[{m}] for loop length {m}")
    prod_all = 1
    for val in x:
        prod_all *= val
    total += E_loop[m] * prod_all

    return total / (2**m)


def build_rhs_from_correlators(
    prob: InflationProblem,
    E_line: dict[int, float],
    E_loop: dict[int, float],
) -> np.ndarray:
    """
    Build RHS vector for the symmetrized marginals using loop correlators.
    """
    n = prob.inflation_level_per_source[0]
    outcomes = prob.outcomes_per_party[0]
    if outcomes != 2:
        raise NotImplementedError("RHS demo currently assumes binary outcomes")

    # 1) Enumerate the same marginals as in the LHS.
    marginals = generate_all_marginal_events(n, outcomes)

    def event_prob_fn(cycle_outcomes: Tuple[int, ...] | List[int]) -> float:
        return loop_prob_from_correlators(cycle_outcomes, E_line, E_loop)

    # 2) Evaluate each marginal using the factorization implied by its cycles.
    rhs = np.empty(len(marginals), dtype=float)
    for idx, marginal in enumerate(marginals):
        rhs[idx] = factorized_marginal_value(marginal, event_prob_fn)
    return rhs


def build_eq_system(
    prob: InflationProblem,
    E_line: dict[int, float],
    E_loop: dict[int, float],
    *,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, coo_array, np.ndarray]:
    """
    Build the equality system A_eq @ q = b_eq for the LP.

    Returns
    -------
    row_labels : np.ndarray[object]
        Labels for each marginal constraint row.
    col_keys : np.ndarray[uint64]
        Canonical keys for symmetrized global events (columns).
    A_eq : coo_array
        Sparse LHS matrix with columns aligned to col_keys (0-based).
    b_eq : np.ndarray
        RHS vector for the constraints.
    """
    row_labels, col_keys, lhs_raw = build_symmetrized_lhs(prob, show_progress=show_progress)
    b_eq = build_rhs_from_correlators(prob, E_line, E_loop)

    # Convert the raw LHS to 0-based column indexing by dropping the unused column 0.
    lhs_coo = lhs_raw.tocoo()
    if lhs_coo.nnz == 0:
        A_eq = coo_array(([], ([], [])), shape=(lhs_raw.shape[0], 0))
    else:
        A_eq = coo_array(
            (lhs_coo.data, (lhs_coo.row, lhs_coo.col - 1)),
            shape=(lhs_raw.shape[0], lhs_raw.shape[1] - 1),
        )
        A_eq.sum_duplicates()

    return row_labels, col_keys, A_eq, b_eq


def build_mosek_lp(
    prob: InflationProblem,
    E_line: dict[int, float],
    E_loop: dict[int, float],
    *,
    add_normalization: bool = True,
    show_progress: bool = True,
) -> Tuple[np.ndarray, coo_array, coo_array, np.ndarray]:
    """
    Assemble the LP in the format expected by solveLP_sparse (MOSEK backend).

    We convert A_eq q = b_eq into homogeneous equalities by introducing a
    constant variable c:

        A_eq q - b_eq * c = 0,  with  c = 1.

    This allows us to use the solver's "known_vars" mechanism to set c=1.
    Optionally, we add a normalization constraint:

        sum_i q_i - c = 0.
    """
    row_labels, col_keys, A_eq, b_eq = build_eq_system(
        prob, E_line, E_loop, show_progress=show_progress
    )

    # Number of symmetrized global-event variables (q).
    n_vars = A_eq.shape[1]
    const_col = n_vars  # last column is the constant variable c

    # Base equalities: A_eq q - b_eq * c = 0.
    A = A_eq.tocoo()
    base_rows = A.row.astype(np.int64, copy=False)
    base_cols = A.col.astype(np.int64, copy=False)
    base_data = A.data.astype(float, copy=False)

    rhs_rows = np.arange(A_eq.shape[0], dtype=np.int64)
    rhs_cols = np.full(A_eq.shape[0], const_col, dtype=np.int64)
    rhs_data = -b_eq.astype(float, copy=False)

    rows = [base_rows, rhs_rows]
    cols = [base_cols, rhs_cols]
    data = [base_data, rhs_data]

    # Optional normalization row: sum_i q_i - c = 0.
    if add_normalization:
        norm_row = A_eq.shape[0]
        norm_rows = np.full(n_vars, norm_row, dtype=np.int64)
        norm_cols = np.arange(n_vars, dtype=np.int64)
        norm_data = np.ones(n_vars, dtype=float)
        norm_c_row = np.array([norm_row], dtype=np.int64)
        norm_c_col = np.array([const_col], dtype=np.int64)
        norm_c_data = np.array([-1.0], dtype=float)
        rows.extend([norm_rows, norm_c_row])
        cols.extend([norm_cols, norm_c_col])
        data.extend([norm_data, norm_c_data])

    all_rows = np.concatenate(rows)
    all_cols = np.concatenate(cols)
    all_data = np.concatenate(data)
    n_rows = A_eq.shape[0] + (1 if add_normalization else 0)
    equalities = coo_array(
        (all_data, (all_rows, all_cols)),
        shape=(n_rows, n_vars + 1),
    )
    equalities.sum_duplicates()

    # "known_vars" encodes c = 1 (the constant variable).
    known_vars = coo_array(
        (np.array([1.0]), (np.array([0], dtype=np.int64), np.array([const_col], dtype=np.int64))),
        shape=(1, n_vars + 1),
    )

    # Variable names help debug solver output.
    variable_names = np.asarray([f"q[{k}]" for k in col_keys] + ["const_1"], dtype=str)

    return variable_names, known_vars, equalities, b_eq


def solve_lp_mosek(
    prob: InflationProblem,
    E_line: dict[int, float],
    E_loop: dict[int, float],
    *,
    add_normalization: bool = True,
    show_progress: bool = True,
    verbose: int = 1,
) -> dict:
    """
    Solve the feasibility LP using the MOSEK-backed solver.

    This uses:
      - equalities from build_mosek_lp()
      - default non-negativity (q >= 0)
      - zero objective (feasibility)
    """
    from inflation.lp.lp_utils import solveLP_sparse

    variable_names, known_vars, equalities, _b_eq = build_mosek_lp(
        prob,
        E_line,
        E_loop,
        add_normalization=add_normalization,
        show_progress=show_progress,
    )

    objective = coo_array(([], ([], [])), shape=(1, equalities.shape[1]))
    return solveLP_sparse(
        objective=objective,
        known_vars=known_vars,
        equalities=equalities,
        default_non_negative=True,
        variables=variable_names,
        verbose=verbose,
    )


if __name__ == "__main__":
    n, outcomes = 3, 2
    prob = inf_problem(n, outcomes)
    print(prob)

    # Correlators from Table I of arXiv:2102.03597 (Bancal & Gisin, 2021).
    # Values used here for the n=3 demo: E1, E2, E3 (line) and E^o_1, E^o_2, E^o_3 (loop).
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

    row_labels, col_keys, A_eq, b_eq = build_eq_system(
        prob, E_line, E_loop, show_progress=True
    )
    print("\nLHS matrix summary:")
    print(f"  rows (marginals): {A_eq.shape[0]}")
    print(f"  cols (symmetrized globals): {A_eq.shape[1]}")
    print(f"  nnz: {A_eq.nnz}")
    print("\nRHS vector summary:")
    print(f"  length: {b_eq.size}")
    print(f"  min/max: {b_eq.min():.6g} / {b_eq.max():.6g}")

    # Solve the feasibility LP via MOSEK (non-negativity + equalities).
    solution = solve_lp_mosek(
        prob,
        E_line,
        E_loop,
        add_normalization=True,
        show_progress=False,
        verbose=1,
    )
    print("\nMOSEK status:", solution["status"])
