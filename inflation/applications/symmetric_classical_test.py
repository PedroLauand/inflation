"""
symmetric_classical_test.py
--------------------------------------------------------------------
Pipeline for building an inflation LP from a chosen measurement
basis and a chosen state. The key idea is:
  1) compute event probabilities from (measurement, state),
  2) use those probabilities as constraints in a MOSEK LP via Inflation.

This file keeps the original logic, but reorganizes it so it is clear
where to swap the measurement and state.
--------------------------------------------------------------------
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Iterable, Union, Callable
from pathlib import Path
import numpy as np
import sys

# Ensure repo root is on sys.path so "import inflation" works when running this file directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from inflation import InflationProblem
from tqdm.auto import tqdm
from scipy.sparse import coo_array
from numba import njit, int64, uint8, types
from numba.typed import Dict as NumbaDict
from numba.typed import List as NumbaList
from GPT_utils import (
    default_ejm_measurement,
    default_state,
    build_loop_prob_fn,
    DEFAULT_EVENT_PROB,
    coarse_ejm,
    rgb4_measurement,
    rgb4_state,
    coarse_rgb,
)
from Group_utils import (
    to_lex_representation,
    from_lex_representation,
    build_sympy_group,
    prepare_group_chain,
    lexmin_with_invperms,
    canonical_leximin_coset_chain_uint64,
)

ZERO_I32 = np.int32(0)


#===================================
#Inflation Problem 
#===================================
def exists_shared_source_modified(inf_indices1: np.ndarray,
                            inf_indices2: np.ndarray) -> bool:
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
        dag={"i1": ["A"],
             "i2": ["A"], },
        outcomes_per_party=[nof_outcomes],
        settings_per_party=[1],
        classical_sources=None,
        inflation_level_per_source=(inflation_level,inflation_level),
        order=("A",))

    to_stabilize = np.flatnonzero(inf_prob._lexorder[:, 1] == inf_prob._lexorder[:, 2])


    #Fix factorization
    inf_prob._inflation_indices_overlap = overlap_matrix(inf_prob._all_unique_inflation_indices)

    # Fix symmetries
    new_symmetries = np.array([
        perm for perm in inf_prob.symmetries
        if np.array_equal(np.sort(perm[to_stabilize]), to_stabilize)
    ], dtype=int)
    inf_prob.symmetries = new_symmetries
    # inf_prob._interpretation_to_name = name_interpret_always_copy_indices

    return inf_prob

# =========================
# Integer partitions (conjugacy classes of S_n)
# =========================
def integer_partitions(n: int) -> List[List[int]]:
    """All integer partitions of n in nonincreasing order."""
    out: List[List[int]] = []
    def rec(rem: int, mx: int, acc: List[int]):
        if rem == 0:
            out.append(acc[:])
            return
        for p in range(min(rem, mx), 1 - 1, -1):
            acc.append(p)
            rec(rem - p, p, acc)
            acc.pop()
    rec(n, n, [])
    return out

@njit(cache=True, fastmath=True)
def representative_perm_for_partition(parts: np.ndarray) -> np.ndarray:
    """
    Given partition parts of n (e.g., [3,1]), build a canonical 1-line permutation J of {1..n}
    with that cycle structure: consecutive labels per cycle.
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

# =========================
# Outcome patterns up to relabeling (S_outcomes)
# =========================
def set_partitions_indices(n: int) -> List[List[List[int]]]:
    """All set partitions of {0,..,n-1} as list of blocks (lists)."""
    if n == 0:
        return [[[]]]
    parts = [[[0]]]
    for x in range(1, n):
        new_parts = []
        for part in parts:
            new_parts.append(part + [[x]])  # new block
            for i in range(len(part)):
                new_part = [blk[:] for blk in part]
                new_part[i].append(x)
                new_parts.append(new_part)
        parts = new_parts
    # normalize each partition blocks
    for p in parts:
        p.sort(key=lambda b: min(b))
        for b in p:
            b.sort()
    return parts

def canonical_outcome_patterns(n: int, outcomes: int) -> List[List[int]]:
    """
    One representative per equivalence class under outcome relabeling.
    Map blocks -> labels 0,1,2,... by block order (increasing min index).
    Only keep patterns with #blocks <= outcomes.
    """
    vecs: List[List[int]] = []
    for part in set_partitions_indices(n):
        if len(part) > outcomes:
            continue
        v = [0] * n
        for lbl, block in enumerate(part):
            for idx in block:
                v[idx] = lbl
        vecs.append(v)
    vecs.sort()
    return vecs

# =========================
# Generate symmetry-reduced marginals
# =========================
def generate_minimal_marginal_events(n: int, outcomes: int) -> List[List[List[int]]]:
    """
    Return a list of marginals in format [[1,i,J(i),0,a_i] for i=1..n],
    for each conjugacy-class representative J of S_n, and each canonical outcome pattern.
    """
    if n <= 0:
        return []
    all_marginals: List[List[List[int]]] = []
    outcome_reps = canonical_outcome_patterns(n, outcomes)
    for parts in integer_partitions(n):
        J = representative_perm_for_partition(np.array(parts, dtype=np.int64))  # 1-line
        for pat in outcome_reps:
            marginal = [[1, i, J[i - 1], 0, pat[i - 1]] for i in range(1, n + 1)]
            all_marginals.append(marginal)
    return all_marginals


"""
@njit(types.uint64(uint8[:], int64), cache=True, fastmath=True)
def event_to_uint64(evt: np.ndarray, outcomes: int) -> np.uint64:
    "Encode a compact event vector into a single uint64 in base `outcomes`."
    evt_u64 = evt.astype(np.uint64)
    outcomes_u64 = np.uint64(outcomes)
    acc = np.uint64(0)
    base = np.uint64(1)
    for i in range(evt_u64.size):
        acc += evt_u64[i] * base
        if i + 1 < evt_u64.size:
            base *= outcomes_u64
    return acc
"""

"""
def uint64_to_event_list(key: int, outcomes: int, n2: int) -> List[int]:
    "Decode a uint64 key into a compact event list in base `outcomes`."
    out = [0] * n2
    for i in range(n2):
        out[i] = key % outcomes
        key //= outcomes
    return out
"""

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
    """Disjoint cycles of 1-line permutation J on {1..n}; each cycle as a list in cycle order (1-based)."""
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
    Multiply EJM loop scalars over the disjoint cycles of J with outcomes taken in cycle order.
    """
    J = _perm_from_marginal(marginal)
    a = _outcomes_from_marginal(marginal)
    val = 1.0
    for cyc in _cycles_from_J(J):
        cyc_out = [a[i - 1] for i in cyc]
        val *= event_prob_fn(cyc_out)
    return val

def representatives_of_global_extensions_uint64(
    n: int,
    outcomes: int,
    marginal: List[List[int]],
    level_invperms: NumbaList,
    global_event_map,
    next_event_idx: int,
    list_of_all_LP_variables: List[str],
    total: int,
    sparse_matrix_cols: np.ndarray,
    start: int,
) -> int:
    """
    Uint64-keyed path using Numba typed dicts.
    Modifies global_event_map, sparse_matrix_cols, list_of_all_LP_variables.
    """
    fixed: Dict[int, int] = {}
    for (one, i, j, zero, a) in marginal:
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


# =========================
# Top-level pipeline
# =========================
def run_pipeline(
    prob: InflationProblem,
    *,
    event_prob_fn: Callable[[Iterable[int]], float] = DEFAULT_EVENT_PROB,
    show_progress: bool = True,
) -> Tuple[np.ndarray, coo_array, coo_array]:
    """
    Build the inflation matrix, known values vector, and variable names
    over all symmetry-reduced marginals for (n, outcomes).
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
    marginals = generate_minimal_marginal_events(n, outcomes)

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
    # Broadcast to build row indices for all global extensions without a per-row loop.
    row_grid = np.broadcast_to(
        np.arange(nof_marginals, dtype=np.int32)[:, None],
        (nof_marginals, global_extension_count),
    )
    sparse_matrix_rows[nof_marginals:] = row_grid.reshape(-1)

    # result = []
    # LOOP 0: Compute the marginal probabilities
    known_values = np.empty(nof_marginals, dtype=float)
    for idx, marginal in enumerate(
        tqdm(marginals, desc="Computing marginal values...", disable=not show_progress)
    ):
        # --- your existing body per marginal ---
        val = factorized_marginal_value(marginal, event_prob_fn)  ## This computes the numeric probabilities
        mkey = tuple(prob._lexrepr_to_names[prob.mon_to_lexrepr(marginal)])
        # mkey = tuple(tuple(x) for x in marginal)
        list_of_all_LP_variables.append("P_global("+",".join(mkey)+")")
        known_values[idx] = val

    # LOOP 1+2: Canonicalize globals and build sparse arrays on the fly
    for row_num, marginal in enumerate(tqdm(marginals, desc="Finding global extensions...",
                                           disable=not show_progress)):
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
    inflation_matrix = coo_array((sparse_matrix_data, (sparse_matrix_rows, sparse_matrix_cols)),
                          shape=(nof_marginals, int(next_event_idx)))
    inflation_matrix.sum_duplicates()

    known_positions = np.arange(1, nof_marginals + 1, dtype=np.int32)
    known_rows = np.broadcast_to(np.int32(0), (nof_marginals,))
    known_vars_coo_vec = coo_array((known_values, (known_rows, known_positions)),
                                   shape=(1, int(next_event_idx)))
    variable_names = np.asarray(list_of_all_LP_variables, dtype=str)

    return variable_names, known_vars_coo_vec, inflation_matrix

# =========================
# Example usage
# =========================
if __name__ == "__main__":
    from inflation.lp.lp_utils import solveLP_sparse
    # --- usage example ---
    # Choose inflation level (n), number of outcomes, measurement, and state.
    n, outcomes = 4, 3
    u = np.sqrt(0.9)
    lambda0 = np.sqrt(1.0 / 2.0)
    measurement = coarse_rgb(["03", "1", "2"], u=u)
    state = rgb4_state(lambda0)
    event_prob_fn = build_loop_prob_fn(measurement, state)

    # Build the inflation problem and symmetries.
    prob = ring_problem(n, outcomes)
    #prob.add_symmetries(prob._setting_specific_outcome_relabelling_symmetries)
    print("done with prob")

    
    def _save_cache(path: Path,
                    variable_names: np.ndarray,
                    known_vars_coo_vec: coo_array,
                    inflation_matrix: coo_array) -> None:
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
                known_vars_coo_vec = coo_array((known_values, (known_rows, known_positions)),
                                               shape=(1, inflation_shape[1]))
            variable_names = z["variable_names"]
            return variable_names, known_vars_coo_vec, inflation_matrix

    cache_dir = Path(__file__).resolve().parent / "cache"
    cache_path = cache_dir / f"lp_cache_n{n}_o{outcomes}.npz"

    # Cache loading is disabled.
    # if cache_path.exists():
    #     try:
    #         print(f"Loading cached LP constraints from {cache_path}")
    #         variable_names, known_vars_coo_vec, inflation_matrix = _load_cache(cache_path)
    #     except (EOFError, ValueError, OSError) as exc:
    #         print(f"Cache load failed ({exc}); rebuilding cache.")
    #         variable_names, known_vars_coo_vec, inflation_matrix = run_pipeline(
    #             prob,
    #             event_prob_fn=event_prob_fn,
    #         )
    #         # _save_cache(cache_path, variable_names, known_vars_coo_vec, inflation_matrix)
    # else:
    #     variable_names, known_vars_coo_vec, inflation_matrix = run_pipeline(
    #         prob,
    #         event_prob_fn=event_prob_fn,
    #     )
    #     # _save_cache(cache_path, variable_names, known_vars_coo_vec, inflation_matrix)
    variable_names, known_vars_coo_vec, inflation_matrix = run_pipeline(
        prob,
        event_prob_fn=event_prob_fn,
    )

    nof_all_LP_vars = inflation_matrix.shape[1]
    # # Print a small summary
    # for idx, (e1, e2) in enumerate(out):
    #     print(f"\nItem {idx}:")
    #     # marginal & value
    #     (marginal_key, val) = next(iter(e1.items()))
    #     print("  marginal:", list(list(t) for t in marginal_key))
    #     print("  value:   ", val)
    #     # reps & counts
    #     print("  reps (compact evt) -> count:")
    #     for rep, cnt in e2.items():
    #         print("   ", list(rep), "->", cnt)

    solution = solveLP_sparse(objective=coo_array(([], ([], [])), shape=(1, nof_all_LP_vars)),
                              known_vars=known_vars_coo_vec,
                              equalities=inflation_matrix,
                              default_non_negative=True,
                              variables=variable_names,
                              verbose=True)

    print(solution["status"])
