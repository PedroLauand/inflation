# final_selfcontained_pipeline.py
# -------------------------------------------------------------------
# Self-contained pipeline (no external imports from your project).
# It:
#   1) generates symmetry-reduced marginals (operator structure + outcome patterns),
#   2) evaluates each marginal's factorized EJM loop value,
#   3) enumerates global extensions and tallies canonical representatives under a group G.
#
# Output format: a list of items, each item is [e1, e2], where:
#   e1: { <marginal-as-tuple-of-tuples> : float_value }
#   e2: { <canonical-global-rep (tuple of length n^2)> : count }
#
# Example usage is at the bottom (n=2, outcomes=4).
# -------------------------------------------------------------------

from __future__ import annotations
from typing import List, Tuple, Dict, Iterable, Union, DefaultDict
from functools import lru_cache
from math import ldexp
from pathlib import Path
import numpy as np
import sys
from itertools import product

# Ensure repo root is on sys.path so "import inflation" works when running this file directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from inflation import InflationProblem
from collections import defaultdict
from tqdm.auto import tqdm
from scipy.sparse import coo_array
from numba import njit, int64, uint8, types
from numba.typed import Dict as NumbaDict
from numba.typed import List as NumbaList

ZERO_I32 = np.int32(0)


# =========================
# EJM distribution (4 outcomes)
# =========================
e0_un = np.array([[-1 - 1j, 0,     -2j,  -1 + 1j]], dtype=np.complex128).reshape(2, 2)
e1_un = np.array([[ 1 - 1j, 2j,     0,    1 + 1j]], dtype=np.complex128).reshape(2, 2)
e2_un = np.array([[-1 + 1j, 2j,     0,   -1 - 1j]], dtype=np.complex128).reshape(2, 2)
e3_un = np.array([[ 1 + 1j, 0,     -2j,   1 - 1j]], dtype=np.complex128).reshape(2, 2)
_ejm = np.stack([e0_un, e1_un, e2_un, e3_un])  # (4,2,2)

_psi = np.array([[0, 1, -1, 0]], dtype=np.complex128).reshape(2, 2)

@lru_cache(maxsize=None)
def _M() -> np.ndarray:
    """Precompute M[a] = ejm[a] @ psi, shape (4,2,2)."""
    return np.einsum('aij,jk->aik', _ejm, _psi, optimize=True)

def loop_prob_event(outcomes: Iterable[int]) -> float:
    """
    Probability P[a1,...,an] on an n-site RING (n=len(outcomes)), for outcomes in {0,1,2,3}.
    Uses unnormalized objects and applies the global factor 16^{-n}.
    """
    a = tuple(int(x) for x in outcomes)
    if not a:
        raise ValueError("Provide at least one outcome.")
    if any((x < 0 or x > 3) for x in a):
        raise ValueError("Outcomes must be in {0,1,2,3}.")
    M = _M()
    Pmat = np.eye(2, dtype=np.complex128)
    for x in a:
        Pmat = Pmat @ M[x]
    amp = np.trace(Pmat)
    prob = (amp.real * amp.real + amp.imag * amp.imag) * ldexp(1.0, -4 * len(a))  # 16^{-n}
    return float(prob)
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

# =========================
# One-hot lex helpers (for group action on events)
# =========================
@njit(types.boolean[:](uint8[:], int64), cache=True, fastmath=True)
def to_lex_representation(evt: np.ndarray, outcomes: int) -> np.ndarray:
    """Compact event -> one-hot lex vector (length n^2 * outcomes)."""
    n2 = evt.shape[0]
    lex = np.zeros(n2 * outcomes, dtype=np.bool_)
    base = np.uint16(outcomes)
    idx = evt + base * np.arange(n2, dtype=np.uint16)
    # Assumes 0 <= evt[s] < outcomes for all s.
    for s in range(n2):
        lex[idx[s]] = 1
    return lex

@njit(uint8[:](types.boolean[:], int64), cache=True, fastmath=True)
def from_lex_representation(lex_evt: np.ndarray, outcomes: int) -> np.ndarray:
    """One-hot lex vector -> compact event."""
    lex_evt = np.ascontiguousarray(lex_evt)
    n2 = lex_evt.shape[0] // outcomes
    evt = np.empty(n2, dtype=np.uint8)
    blocks = lex_evt.reshape(n2, outcomes)
    # Assumes length is divisible by outcomes and blocks are one-hot.
    for s in range(n2):
        idx = 0
        for j in range(outcomes):
            if blocks[s, j] != 0:
                idx = j
                break
        evt[s] = idx
    return evt


# =========================
# SymPy group utilities
# =========================
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.perm_groups import PermutationGroup

def _normalize_perm_list(g: List[int], N: int) -> List[int]:
    """Ensure 0-based bijection of range(N) from list (accepts 1- or 0-based)."""
    if len(g) != N:
        raise ValueError(f"Permutation length {len(g)} != {N}")
    if max(g) == N:  # likely 1-based
        g = [x - 1 for x in g]
    if sorted(g) != list(range(N)):
        raise ValueError("Invalid permutation (not a bijection).")
    return g

def build_sympy_group(raw_G: Union[List[List[int]], np.ndarray], N: int) -> PermutationGroup:
    """Build group from raw list permutations (0- or 1-based)."""
    gens = [Permutation(_normalize_perm_list(gl, N)) for gl in raw_G] or [Permutation(list(range(N)))]
    return PermutationGroup(gens)

def prepare_group_chain(G: PermutationGroup, N: int) -> NumbaList:
    """
    Run Schreier-Sims once. Returns inverse-transversal matrices per level
    for use in the canonicalizer (no recomputation).
    """
    G.schreier_sims()
    level_invperms = NumbaList()
    if N <= np.iinfo(np.uint8).max:
        perm_dtype = np.uint8
    else:
        perm_dtype = np.uint16
    for orbits, trans in zip(G.basic_orbits, G.basic_transversals):
        invperm_matrix = np.empty((len(orbits), N), dtype=perm_dtype)
        for i, u in enumerate(orbits):
            perm_arr = np.array(trans[u].array_form, dtype=perm_dtype)
            invperm = np.empty_like(perm_arr)
            invperm[perm_arr] = np.arange(perm_arr.size, dtype=perm_arr.dtype)
            invperm_matrix[i] = invperm
        level_invperms.append(invperm_matrix)
    return level_invperms

@njit(cache=True, fastmath=True)
def lexmin_with_invperms(current: np.ndarray, invperms: np.ndarray) -> Tuple[np.ndarray, int]:
    """Return lex-min vector and its index using inverse-permutation matrix."""
    m, n = invperms.shape
    best_idx = 0
    for i in range(1, m):
        for j in range(n):
            a = current[invperms[i, j]]
            b = current[invperms[best_idx, j]]
            if a < b:
                best_idx = i
                break
            if a > b:
                break
    best_vec = np.empty(n, dtype=current.dtype)
    for j in range(n):
        best_vec[j] = current[invperms[best_idx, j]]
    return best_vec, best_idx

@njit(cache=True, fastmath=True)
def canonical_leximin_coset_chain(
    evt: np.ndarray,
    outcomes: int,
    level_invperms: NumbaList,
) -> np.ndarray:
    """
    Canonical representative of 'evt' under G using the stabilizer chain:
    scans transversal reps at each level to minimize lex-image in one-hot space.
    """
    x = to_lex_representation(evt, outcomes)
    # witness permutation (inverse array) and current best image
    current_invperm = np.arange(len(x), dtype=level_invperms[0].dtype)
    best_vec = x

    # walk the chain
    levels = len(level_invperms)
    for k in range(levels):
        current = x[current_invperm]
        invperm_matrix = level_invperms[k]
        cand_vec, cand_idx = lexmin_with_invperms(current, invperm_matrix)
        current_invperm = current_invperm[invperm_matrix[cand_idx]]
        best_vec = cand_vec
    rep_evt = from_lex_representation(best_vec, outcomes)
    return rep_evt

@njit(cache=True, fastmath=True)
def canonical_leximin_coset_chain_uint64(
    evt: np.ndarray,
    outcomes: int,
    level_invperms: NumbaList,
) -> np.uint64:
    """
    Canonical representative of 'evt' under G using the stabilizer chain.
    Returns a uint64 key encoded directly from the lex-min one-hot vector.
    """
    x = to_lex_representation(evt, outcomes)
    current_invperm = np.arange(len(x), dtype=level_invperms[0].dtype)
    best_vec = x
    levels = len(level_invperms)
    for k in range(levels):
        current = x[current_invperm]
        invperm_matrix = level_invperms[k]
        cand_vec, cand_idx = lexmin_with_invperms(current, invperm_matrix)
        current_invperm = current_invperm[invperm_matrix[cand_idx]]
        best_vec = cand_vec

    outcomes_u64 = np.uint64(outcomes)
    acc = np.uint64(0)
    base = np.uint64(1)
    n2 = best_vec.size // outcomes
    offset = 0
    for s in range(n2):
        idx = 0
        for j in range(outcomes):
            if best_vec[offset + j] != 0:
                idx = j
                break
        acc += np.uint64(idx) * base  # Base-`outcomes` accumulation from one-hot blocks.
        if s + 1 < n2:
            base *= outcomes_u64
        offset += outcomes
    return acc

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

def factorized_marginal_value(marginal: List[List[int]]) -> float:
    """
    Multiply EJM loop scalars over the disjoint cycles of J with outcomes taken in cycle order.
    """
    J = _perm_from_marginal(marginal)
    a = _outcomes_from_marginal(marginal)
    val = 1.0
    for cyc in _cycles_from_J(J):
        cyc_out = [a[i - 1] for i in cyc]
        val *= loop_prob_event(cyc_out)
    return val

# =========================
# Canonicalize & count representatives of global extensions
# =========================
# Deprecated: kept for reference only (uint64 path used).
def representatives_of_global_extensions(
    n: int,
    outcomes: int,
    marginal: List[List[int]],
    level_invperms: NumbaList,
    global_event_map: DefaultDict[bytes, int],
    next_event_idx: int,
    list_of_all_LP_variables: List[str],
    total: int,
    sparse_matrix_cols: np.ndarray,
    start: int,
    show_progress: bool,
) -> int:
    """
    Iterate global extensions of 'marginal', canonicalize each under the group.
    Modifies global_event_map, sparse_matrix_cols, list_of_all_LP_variables.
    """
    # fixed slots
    fixed: Dict[int, int] = {}
    for (one, i, j, zero, a) in marginal:
        si = (i - 1) * n + (j - 1)
        if si in fixed and fixed[si] != a:
            return next_event_idx
        fixed[si] = a
    # build remaining indices
    Nslots = n * n
    fixed_idx = np.fromiter(fixed.keys(), dtype=np.int64)
    fixed_val = np.fromiter(fixed.values(), dtype=np.uint8)
    mask = np.ones(Nslots, dtype=bool)
    mask[fixed_idx] = False
    remaining = np.nonzero(mask)[0]
    # build base event (fixed slots set once)
    evt = np.zeros(Nslots, dtype=np.uint8)
    if fixed_idx.size:
        evt[fixed_idx] = fixed_val
    # iterate assignments without copying
    if remaining.size == 0:
        canon = canonical_leximin_coset_chain(evt, outcomes, level_invperms)
        event_key = canon.tobytes()
        event_idx = global_event_map[event_key]
        if event_idx == 0:
            event_idx = next_event_idx
            next_event_idx += 1
            global_event_map[event_key] = event_idx
            list_of_all_LP_variables.append("P_global("+",".join(map(str, canon))+")")
        sparse_matrix_cols[start] = event_idx
        return next_event_idx
    if show_progress:
        combos = tqdm(product(range(outcomes), repeat=remaining.size),
                      total=total,
                      desc="Canonicalizing globals...",
                      leave=True,
                      position=0)
    else:
        combos = product(range(outcomes), repeat=remaining.size)
    for pos, combo in enumerate(combos):
        evt[remaining] = combo
        canon = canonical_leximin_coset_chain(evt, outcomes, level_invperms)
        event_key = canon.tobytes()
        event_idx = global_event_map[event_key]
        if event_idx == 0:
            event_idx = next_event_idx
            next_event_idx += 1
            global_event_map[event_key] = event_idx
            list_of_all_LP_variables.append("P_global("+",".join(map(str, canon))+")")
        sparse_matrix_cols[start + pos] = event_idx
    return next_event_idx

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
        val = factorized_marginal_value(marginal)  ## This computes the numeric probabilities
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
# Example of usage
# =========================
if __name__ == "__main__":
    from inflation.lp.lp_utils import solveLP_sparse
    # Example: n=2, outcomes=4
    n, outcomes = 4, 4
    # One small example group on N = n^2 * outcomes = 4 * 4 = 16 coordinates:
    #   - identity
    #   - swap within each outcome block of the four operator slots (toy example)
    prob = ring_problem(n, outcomes)
    prob.add_symmetries(prob._setting_specific_outcome_relabelling_symmetries)
    
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

    if cache_path.exists():
        print(f"Loading cached LP constraints from {cache_path}")
        variable_names, known_vars_coo_vec, inflation_matrix = _load_cache(cache_path)
    else:
        variable_names, known_vars_coo_vec, inflation_matrix = run_pipeline(prob)
        _save_cache(cache_path, variable_names, known_vars_coo_vec, inflation_matrix)

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
