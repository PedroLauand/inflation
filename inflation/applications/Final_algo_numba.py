# final_selfcontained_pipeline.py
# -------------------------------------------------------------------
# Canonical generic pipeline for the ring inflation workflow.
#
# Responsibilities:
#   1) generate symmetry-canonical marginal representatives from prob.symmetries,
#   2) evaluate factorized marginal values via injected loop-event callable,
#   3) enumerate symmetry-canonical global extensions and build LP matrices.
# -------------------------------------------------------------------

from __future__ import annotations

from itertools import permutations, product
from math import factorial
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple
import sys

import numpy as np
from numba import njit, types
from numba.typed import Dict as NumbaDict
from numba.typed import List as NumbaList
from scipy.sparse import coo_array
from tqdm.auto import tqdm

# Ensure repo root is on sys.path so "import inflation" works when running this file directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inflation import InflationProblem
from inflation.applications.Group_utils import (
    build_sympy_group,
    canonical_leximin_coset_chain_uint64,
    canonical_leximin_support_indices,
    prepare_group_chain,
)

ZERO_I32 = np.int32(0)


# ===================================
# Inflation problem
# ===================================
def exists_shared_source_modified(
    inf_indices1: np.ndarray,
    inf_indices2: np.ndarray,
) -> bool:
    common_sources = np.logical_and(inf_indices1, inf_indices2)
    if not np.any(common_sources):
        return False
    return not set(inf_indices1[common_sources]).isdisjoint(set(inf_indices2[common_sources]))


def overlap_matrix(all_inflation_indxs: np.ndarray) -> np.ndarray:
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


def ring_problem(inflation_level: int, nof_outcomes: int = 2) -> InflationProblem:
    inf_prob = InflationProblem(
        dag={"i1": ["A"], "i2": ["A"]},
        outcomes_per_party=[nof_outcomes],
        settings_per_party=[1],
        classical_sources=None,
        inflation_level_per_source=(inflation_level, inflation_level),
        order=("A",),
    )

    to_stabilize = np.flatnonzero(inf_prob._lexorder[:, 1] == inf_prob._lexorder[:, 2])

    # Fix factorization
    inf_prob._inflation_indices_overlap = overlap_matrix(inf_prob._all_unique_inflation_indices)

    # Fix symmetries
    new_symmetries = np.array(
        [perm for perm in inf_prob.symmetries if np.array_equal(np.sort(perm[to_stabilize]), to_stabilize)],
        dtype=int,
    )
    inf_prob.symmetries = new_symmetries

    return inf_prob


# =========================
# Integer partitions (kept public for scenario scripts)
# =========================
def integer_partitions(n: int) -> List[List[int]]:
    """All integer partitions of n in nonincreasing order."""
    out: List[List[int]] = []

    def rec(rem: int, mx: int, acc: List[int]) -> None:
        if rem == 0:
            out.append(acc[:])
            return
        for p in range(min(rem, mx), 0, -1):
            acc.append(p)
            rec(rem - p, p, acc)
            acc.pop()

    rec(n, n, [])
    return out


@njit(cache=True, fastmath=True)
def representative_perm_for_partition(parts: np.ndarray) -> np.ndarray:
    """
    Given partition parts of n (e.g., [3,1]), build a canonical 1-line
    permutation J of {1..n} with that cycle structure.
    """
    n = 0
    for i in range(parts.shape[0]):
        n += parts[i]
    J = np.arange(1, n + 1, dtype=np.int64)
    cur = 1
    for idx in range(parts.shape[0]):
        L = parts[idx]
        if L <= 1:
            cur += L
            continue
        for a in range(cur, cur + L - 1):
            J[a - 1] = a + 1
        J[cur + L - 2] = cur
        cur += L
    return J


def _prepare_group_chain(
    prob: InflationProblem,
) -> Tuple[int, int, int, NumbaList]:
    n = prob.inflation_level_per_source[0]
    outcomes = prob.outcomes_per_party[0]
    if outcomes >= 255:
        raise ValueError("outcomes must be < 255 to fit in compact dtypes")
    if n > 5:
        raise ValueError("uint64 canonical events are only supported up to n=5")
    max_event_count = pow(outcomes, n * n)
    if max_event_count > np.iinfo(np.uint64).max:
        raise ValueError("events do not fit in uint64")

    N = (n * n) * outcomes
    if N > np.iinfo(np.uint16).max:
        raise ValueError("N exceeds uint16 range; use wider dtype for permutations")
    G = build_sympy_group(prob.symmetries, N)
    level_invperms = prepare_group_chain(G, N)
    return n, outcomes, N, level_invperms


def _marginal_support_from_perm(
    perm: Sequence[int],
    outcome_pattern: Sequence[int],
    n: int,
    outcomes: int,
) -> np.ndarray:
    support = np.empty(n, dtype=np.int64)
    for i0 in range(n):
        j0 = int(perm[i0]) - 1
        support[i0] = (i0 * n + j0) * outcomes + int(outcome_pattern[i0])
    return support


def _marginal_from_support_key(
    support_key: Tuple[int, ...],
    n: int,
    outcomes: int,
) -> List[List[int]]:
    by_i: Dict[int, Tuple[int, int]] = {}
    for coord in support_key:
        slot, a = divmod(int(coord), outcomes)
        i = slot // n + 1
        j = slot % n + 1
        if i in by_i:
            raise ValueError("Invalid canonical marginal support: duplicate row assignment.")
        by_i[i] = (j, a)
    if len(by_i) != n:
        raise ValueError("Invalid canonical marginal support: missing row assignment.")
    return [[1, i, by_i[i][0], 0, by_i[i][1]] for i in range(1, n + 1)]


def _generate_canonical_marginals_with_chain(
    n: int,
    outcomes: int,
    N: int,
    level_invperms: NumbaList,
    *,
    marginal_filter_fn: Callable[[List[List[int]]], bool] | None,
    show_progress: bool,
) -> List[List[List[int]]]:
    seen_keys: set[Tuple[int, ...]] = set()
    marginals: List[List[List[int]]] = []
    base_outcomes = tuple(range(outcomes))
    perm_iter = permutations(range(1, n + 1))
    perm_iter = tqdm(
        perm_iter,
        total=factorial(n),
        desc="Canonicalizing marginals",
        disable=not show_progress,
    )
    for perm in perm_iter:
        for pat in product(base_outcomes, repeat=n):
            support = _marginal_support_from_perm(perm, pat, n, outcomes)
            canonical_support = canonical_leximin_support_indices(support, N, level_invperms)
            key = tuple(int(x) for x in canonical_support.tolist())
            if key in seen_keys:
                continue
            seen_keys.add(key)
            marginal = _marginal_from_support_key(key, n, outcomes)
            if marginal_filter_fn is not None and not marginal_filter_fn(marginal):
                continue
            marginals.append(marginal)
    return marginals


def generate_canonical_marginals(
    prob: InflationProblem,
    *,
    marginal_filter_fn: Callable[[List[List[int]]], bool] | None = None,
    show_progress: bool = False,
) -> List[List[List[int]]]:
    """
    Generate one canonical marginal representative per orbit under prob.symmetries.
    """
    n, outcomes, N, level_invperms = _prepare_group_chain(prob)
    return _generate_canonical_marginals_with_chain(
        n,
        outcomes,
        N,
        level_invperms,
        marginal_filter_fn=marginal_filter_fn,
        show_progress=show_progress,
    )


@njit(cache=True, fastmath=True)
def _fill_cols_uint64(
    evt: np.ndarray,
    remaining: np.ndarray,
    outcomes: int,
    level_invperms: NumbaList,
    global_event_map,
    next_event_idx: int,
    sparse_matrix_cols: np.ndarray,
    start: int,
    total: int,
    new_keys: NumbaList,
) -> int:
    """Writes into sparse_matrix_cols and updates global_event_map/new_keys."""
    next_event_idx = np.int32(next_event_idx)
    for pos in range(total):
        tmp = pos
        for r in range(remaining.size - 1, -1, -1):
            idx = remaining[r]
            evt[idx] = tmp % outcomes
            tmp //= outcomes
        key = canonical_leximin_coset_chain_uint64(evt, outcomes, level_invperms)
        event_idx = global_event_map.get(key, ZERO_I32)
        if event_idx == 0:
            event_idx = next_event_idx
            next_event_idx = np.int32(next_event_idx + 1)
            global_event_map[key] = event_idx
            new_keys.append(key)
        sparse_matrix_cols[start + pos] = event_idx
    return next_event_idx


# =========================
# Cycle extraction & factorized value for a marginal
# =========================
def _perm_from_marginal(marginal: List[List[int]]) -> List[int]:
    """Extract 1-line permutation J (1..n) from marginal [[1,i,j,0,a],...]."""
    n = len(marginal)
    J = [0] * n
    for (_, i, j, _, _) in marginal:
        J[i - 1] = j
    return J


def _outcomes_from_marginal(marginal: List[List[int]]) -> List[int]:
    """Extract outcome vector a_i from marginal [[1,i,j,0,a],...], in order i=1..n."""
    n = len(marginal)
    a = [0] * n
    for (_, i, _j, _, val) in marginal:
        a[i - 1] = val
    return a


def _cycles_from_J(J: List[int]) -> List[List[int]]:
    """Disjoint cycles of 1-line permutation J on {1..n}; each cycle as list (1-based)."""
    n = len(J)
    seen = [False] * (n + 1)
    cycles: List[List[int]] = []
    for start in range(1, n + 1):
        if seen[start]:
            continue
        cyc = []
        v = start
        while not seen[v]:
            seen[v] = True
            cyc.append(v)
            v = J[v - 1]
        cycles.append(cyc)
    return cycles


def factorized_marginal_value(
    marginal: List[List[int]],
    event_prob_fn: Callable[[Iterable[int]], float],
) -> float:
    """
    Multiply loop scalars over the disjoint cycles of J with outcomes in cycle order.
    """
    J = _perm_from_marginal(marginal)
    a = _outcomes_from_marginal(marginal)
    val = 1.0
    for cyc in _cycles_from_J(J):
        cyc_out = [a[i - 1] for i in cyc]
        val *= event_prob_fn(cyc_out)
    return val


def compute_known_values(
    prob: InflationProblem,
    marginals: List[List[List[int]]],
    event_prob_fn: Callable[[Iterable[int]], float],
    *,
    show_progress: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute known marginal values and their LP variable labels.
    """
    known_values = np.empty(len(marginals), dtype=float)
    known_labels: List[str] = []
    for idx, marginal in enumerate(
        tqdm(marginals, desc="Computing marginal values...", disable=not show_progress)
    ):
        known_values[idx] = factorized_marginal_value(marginal, event_prob_fn)
        mkey = tuple(prob._lexrepr_to_names[prob.mon_to_lexrepr(marginal)])
        known_labels.append("P_global(" + ",".join(mkey) + ")")
    return known_values, known_labels


def representatives_of_global_extensions_uint64(
    n: int,
    outcomes: int,
    marginal: List[List[int]],
    level_invperms: NumbaList,
    global_event_map,
    next_event_idx: int,
    list_of_all_LP_variables: List[int],
    total: int,
    sparse_matrix_cols: np.ndarray,
    start: int,
) -> int:
    """
    Uint64-keyed path using Numba typed dicts.
    Modifies global_event_map, sparse_matrix_cols, list_of_all_LP_variables.
    """
    fixed: Dict[int, int] = {}
    for (_one, i, j, _zero, a) in marginal:
        si = (i - 1) * n + (j - 1)
        if si in fixed and fixed[si] != a:
            return next_event_idx
        fixed[si] = a
    Nslots = n * n
    fixed_idx = np.fromiter(fixed.keys(), dtype=np.int64)
    fixed_val = np.fromiter(fixed.values(), dtype=np.uint8)
    mask = np.ones(Nslots, dtype=bool)
    mask[fixed_idx] = False
    remaining = np.nonzero(mask)[0]
    evt = np.zeros(Nslots, dtype=np.uint8)
    if fixed_idx.size:
        evt[fixed_idx] = fixed_val
    new_keys = NumbaList.empty_list(types.uint64)
    next_event_idx = _fill_cols_uint64(
        evt,
        remaining,
        outcomes,
        level_invperms,
        global_event_map,
        next_event_idx,
        sparse_matrix_cols,
        start,
        total,
        new_keys,
    )
    list_of_all_LP_variables.extend(new_keys)
    return next_event_idx


def _build_lhs_from_marginals_with_chain(
    prob: InflationProblem,
    marginals: List[List[List[int]]],
    *,
    outcomes: int,
    level_invperms: NumbaList,
    show_progress: bool,
) -> Tuple[np.ndarray, np.ndarray, coo_array]:
    nof_marginals = len(marginals)
    n = prob.inflation_level_per_source[0]

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

    row_labels: List[Tuple[str, ...]] = []
    for marginal in marginals:
        mkey = tuple(prob._lexrepr_to_names[prob.mon_to_lexrepr(marginal)])
        row_labels.append(mkey)

    for row_num, marginal in enumerate(
        tqdm(marginals, desc="Finding global extensions...", disable=not show_progress)
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


def build_lhs_from_marginals(
    prob: InflationProblem,
    marginals: List[List[List[int]]],
    *,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, coo_array]:
    """
    Build raw LHS matrix for provided marginals.

    Column 0 is reserved as sentinel and remains unused.
    """
    _n, outcomes, _N, level_invperms = _prepare_group_chain(prob)
    return _build_lhs_from_marginals_with_chain(
        prob,
        marginals,
        outcomes=outcomes,
        level_invperms=level_invperms,
        show_progress=show_progress,
    )


# =========================
# Top-level pipeline
# =========================
def run_pipeline(
    prob: InflationProblem,
    *,
    event_prob_fn: Callable[[Iterable[int]], float],
    marginal_filter_fn: Callable[[List[List[int]]], bool] | None = None,
    show_progress: bool = True,
) -> Tuple[np.ndarray, coo_array, coo_array]:
    """
    Build variable names, known-values vector, and inflation matrix.
    """
    n, outcomes, N, level_invperms = _prepare_group_chain(prob)

    marginals = _generate_canonical_marginals_with_chain(
        n,
        outcomes,
        N,
        level_invperms,
        marginal_filter_fn=marginal_filter_fn,
        show_progress=show_progress,
    )
    nof_marginals = len(marginals)

    known_values, known_labels = compute_known_values(
        prob,
        marginals,
        event_prob_fn,
        show_progress=show_progress,
    )
    _row_labels, global_keys, lhs_raw = _build_lhs_from_marginals_with_chain(
        prob,
        marginals,
        outcomes=outcomes,
        level_invperms=level_invperms,
        show_progress=show_progress,
    )

    lhs_coo = lhs_raw.tocoo()
    shifted_cols = lhs_coo.col.astype(np.int64, copy=True)
    mask = shifted_cols != 0
    shifted_cols[mask] += nof_marginals

    known_rows = np.arange(nof_marginals, dtype=np.int32)
    known_cols = np.arange(1, nof_marginals + 1, dtype=np.int32)
    known_data = -np.ones(nof_marginals, dtype=np.int8)

    all_rows = np.concatenate([known_rows, lhs_coo.row.astype(np.int32, copy=False)])
    all_cols = np.concatenate([known_cols, shifted_cols.astype(np.int32, copy=False)])
    all_data = np.concatenate([known_data, lhs_coo.data.astype(np.int8, copy=False)])

    total_cols = nof_marginals + lhs_raw.shape[1]
    inflation_matrix = coo_array(
        (all_data, (all_rows, all_cols)),
        shape=(nof_marginals, total_cols),
    )
    inflation_matrix.sum_duplicates()

    known_positions = np.arange(1, nof_marginals + 1, dtype=np.int32)
    known_rows0 = np.zeros(nof_marginals, dtype=np.int32)
    known_vars_coo_vec = coo_array(
        (known_values, (known_rows0, known_positions)),
        shape=(1, total_cols),
    )

    variable_names = np.asarray(
        ["1", *known_labels, *[str(k) for k in global_keys.tolist()]],
        dtype=str,
    )

    return variable_names, known_vars_coo_vec, inflation_matrix


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    from functools import partial

    from inflation.distributions.ejm import prob_event_loop as ejm_prob_event_loop
    from inflation.distributions.nsi_pr import prob_event_loop as nsi_pr_prob_event_loop
    from inflation.distributions.rgb import prob_event_loop as rgb_prob_event_loop
    from inflation.lp.lp_utils import solveLP_sparse

    # --- Pipeline configuration ---
    n, outcomes = 3, 2
    include_outcome_relabel_symmetries = True
    cache_name = "lp_cache_nsi_pi_n=3.npz"

    # Users should provide a fully configured callable, optionally via functools.partial.
    # EJM (raw):
    # event_prob_fn = ejm_prob_event_loop
    # RGB (coarse-grained):
    # event_prob_fn = partial(
    #     rgb_prob_event_loop,
    #     u=np.sqrt(0.9),
    #     lambda0=np.sqrt(0.5),
    #     coarsen=[[0, 3], [1], [2]],
    # )
    # NSI-PR:
    # event_prob_fn = nsi_pr_prob_event_loop
    # NSI-PI (same backend as NSI-PR module naming):
    nsi_pi_prob_event_loop = nsi_pr_prob_event_loop
    event_prob_fn = nsi_pi_prob_event_loop

    prob = ring_problem(n, outcomes)
    if include_outcome_relabel_symmetries:
        prob.add_symmetries(prob._setting_specific_outcome_relabelling_symmetries)
    print("done with prob")

    def _save_cache(
        path: Path,
        variable_names: np.ndarray,
        known_vars_coo_vec: coo_array,
        inflation_matrix: coo_array,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            inflation_matrix_columns_indices=inflation_matrix.col,
            inflation_matrix_row_indices=inflation_matrix.row,
            inflation_matrix_data_entries=inflation_matrix.data,
            variable_names=variable_names,
            known_values=known_vars_coo_vec.data,
            known_positions=known_vars_coo_vec.col,
        )

    def _load_cache(path: Path) -> Tuple[np.ndarray, coo_array, coo_array]:
        with np.load(path, allow_pickle=False) as z:
            row_idx = z["inflation_matrix_row_indices"]
            col_idx = z["inflation_matrix_columns_indices"]
            data = z["inflation_matrix_data_entries"]
            n_rows = int(np.max(row_idx)) + 1 if row_idx.size else 0
            n_cols = int(np.max(col_idx)) + 1 if col_idx.size else 0
            inflation_shape = (n_rows, n_cols)
            inflation_matrix = coo_array((data, (row_idx, col_idx)), shape=inflation_shape)
            known_positions = z["known_positions"]
            known_values = z["known_values"]
            if known_values.size == 0:
                known_vars_coo_vec = coo_array((1, inflation_shape[1]), dtype=float)
            else:
                known_rows = np.broadcast_to(
                    np.array(0, dtype=known_positions.dtype),
                    known_positions.shape,
                )
                known_vars_coo_vec = coo_array(
                    (known_values, (known_rows, known_positions)),
                    shape=(1, inflation_shape[1]),
                )
            variable_names = z["variable_names"]
            return variable_names, known_vars_coo_vec, inflation_matrix

    cache_dir = Path(__file__).resolve().parent / "cache"
    cache_path = cache_dir / cache_name

    if cache_path.exists():
        print(f"Loading cached LP constraints from {cache_path}")
        variable_names, known_vars_coo_vec, inflation_matrix = _load_cache(cache_path)
    else:
        variable_names, known_vars_coo_vec, inflation_matrix = run_pipeline(
            prob,
            event_prob_fn=event_prob_fn,
        )
        _save_cache(cache_path, variable_names, known_vars_coo_vec, inflation_matrix)

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
