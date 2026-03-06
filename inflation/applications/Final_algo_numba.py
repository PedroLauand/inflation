# final_selfcontained_pipeline.py
# -------------------------------------------------------------------
# Canonical generic pipeline for the ring inflation workflow.
#
# Responsibilities:
#   1) generate symmetry-canonical marginal representatives from prob.symmetries,
#   2) evaluate factorized marginal values via distribution methods,
#   3) enumerate symmetry-canonical global extensions and build LP matrices.
# -------------------------------------------------------------------

from __future__ import annotations

from collections import Counter
from functools import cached_property
from itertools import permutations, product
from math import factorial
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import sys

import numpy as np
import sympy as sp
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
from inflation.distributions.protocols import RingDistributionProtocol
from inflation.symmetry_utils import discovery_symmetries_from_predicate

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


def ring_problem(inflation_level: int, distribution: RingDistributionProtocol) -> InflationProblem:
    nof_outcomes = int(distribution.nof_outcomes)
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


def _prepare_group_chain(
    prob: InflationProblem,
    symmetries: np.ndarray | None = None,
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
    if symmetries is None:
        symmetries = np.asarray(prob.symmetries, dtype=int)
    G = build_sympy_group(symmetries, N)
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


def _average_orbit_label(labels: Sequence[str]) -> str:
    if len(labels) == 1:
        return labels[0]
    counts = Counter(labels)
    weighted_terms = []
    for label in sorted(counts):
        mult = counts[label]
        if mult == 1:
            weighted_terms.append(label)
        else:
            weighted_terms.append(f"{mult}*{label}")
    return f"({' + '.join(weighted_terms)})/{len(labels)}"


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
    distribution: RingDistributionProtocol,
) -> sp.Expr:
    """
    Multiply loop scalars over the disjoint cycles of J with outcomes in cycle order.
    """
    J = _perm_from_marginal(marginal)
    a = _outcomes_from_marginal(marginal)
    val = sp.Integer(1)
    for cyc in _cycles_from_J(J):
        cyc_out = [a[i - 1] for i in cyc]
        val *= distribution.prob_event_loop(cyc_out)
    return val


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


# =========================
# OOP pipeline
# =========================
class PrepLP:
    """
    Prepare LP ingredients for the canonical ring pipeline.

    Main outputs:
      - variable_names
      - known_vars_symbolic
      - known_vars
      - inflation_matrix
    """

    def __init__(
        self,
        n: int,
        distribution: RingDistributionProtocol,
        *,
        cache_name: str | None = None,
        marginal_filter_fn=None,
        show_progress: bool = True,
        auto_discover_symmetries: bool = True,
        compress_rows_under_discovered_group: bool = True,
        verbose_symmetry_discovery: bool = True,
        verbose_cache: bool = True,
    ) -> None:
        self._requested_n = int(n)
        self.distribution = distribution
        self.cache_name = cache_name
        self.marginal_filter_fn = marginal_filter_fn
        self.show_progress = show_progress
        self.auto_discover_symmetries = auto_discover_symmetries
        self.compress_rows_under_discovered_group = compress_rows_under_discovered_group
        self.verbose_symmetry_discovery = verbose_symmetry_discovery
        self.verbose_cache = verbose_cache
        self.prob = ring_problem(self._requested_n, distribution)
        self._cached_variable_names: np.ndarray | None = None
        self._cached_known_vars: coo_array | None = None
        self._cached_inflation_matrix: coo_array | None = None
        self._cache_written = False
        self._load_cache_if_available()

    @cached_property
    def cache_path(self) -> Path | None:
        if self.cache_name is None:
            return None
        cache_dir = Path(__file__).resolve().parent / "cache"
        return cache_dir / self.cache_name

    def _load_cache_if_available(self) -> None:
        if self.cache_path is None or not self.cache_path.exists():
            return
        with np.load(self.cache_path, allow_pickle=False) as z:
            row_idx = z["inflation_matrix_row_indices"]
            col_idx = z["inflation_matrix_columns_indices"]
            data = z["inflation_matrix_data_entries"]
            n_rows = int(np.max(row_idx)) + 1 if row_idx.size else 0
            n_cols = int(np.max(col_idx)) + 1 if col_idx.size else 0
            inflation_shape = (n_rows, n_cols)
            self._cached_inflation_matrix = coo_array((data, (row_idx, col_idx)), shape=inflation_shape)

            known_positions = z["known_positions"]
            known_values = z["known_values"]
            if known_values.size == 0:
                self._cached_known_vars = coo_array((1, inflation_shape[1]), dtype=float)
            else:
                known_rows = np.broadcast_to(np.array(0, dtype=known_positions.dtype), known_positions.shape)
                self._cached_known_vars = coo_array(
                    (known_values, (known_rows, known_positions)),
                    shape=(1, inflation_shape[1]),
                )
            self._cached_variable_names = z["variable_names"]
        if self.verbose_cache:
            print(f"Loaded cached LP constraints from {self.cache_path}")

    def _save_cache(
        self,
        variable_names: np.ndarray,
        known_vars: coo_array,
        inflation_matrix: coo_array,
    ) -> None:
        if self.cache_path is None or self._cache_written:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.cache_path,
            inflation_matrix_columns_indices=inflation_matrix.col,
            inflation_matrix_row_indices=inflation_matrix.row,
            inflation_matrix_data_entries=inflation_matrix.data,
            variable_names=variable_names,
            known_values=known_vars.data,
            known_positions=known_vars.col,
        )
        self._cache_written = True
        if self.verbose_cache:
            print(f"Saved LP constraints cache to {self.cache_path}")

    @cached_property
    def core_symmetries(self) -> np.ndarray:
        """Initial ring-minimal symmetry elements used for base marginal discovery."""
        return np.asarray(self.prob.symmetries, dtype=int)

    @cached_property
    def candidate_symmetries(self) -> np.ndarray:
        """
        Candidate symmetry elements for automatic subgroup discovery.

        Uses the full closure from `all_possible_symmetries` so composition-only
        valid stabilizers are not missed.
        """
        candidates = np.asarray(self.prob.all_possible_symmetries, dtype=int)
        if candidates.ndim == 1:
            candidates = candidates[np.newaxis, :]
        return np.unique(np.vstack((self.core_symmetries, candidates)), axis=0)

    @cached_property
    def _core_group_chain_data(self) -> Tuple[int, int, int, NumbaList]:
        """Tuple `(n, outcomes, N, level_invperms)` prepared from core symmetries."""
        return _prepare_group_chain(self.prob, self.core_symmetries)

    @cached_property
    def _effective_group_chain_data(self) -> Tuple[int, int, int, NumbaList]:
        """Tuple `(n, outcomes, N, level_invperms)` prepared from discovered symmetries."""
        return _prepare_group_chain(self.prob, self.discovered_symmetries)

    @property
    def n(self) -> int:
        """Inflation level per source (number of copies)."""
        return self._core_group_chain_data[0]

    @property
    def outcomes(self) -> int:
        """Number of outcomes per party."""
        return self._core_group_chain_data[1]

    @property
    def N(self) -> int:
        """One-hot ambient dimension `n*n*outcomes` for group action."""
        return self._core_group_chain_data[2]

    @property
    def core_level_invperms(self) -> NumbaList:
        """Schreier-Sims inverse-transversal chain for the core symmetry group."""
        return self._core_group_chain_data[3]

    @property
    def level_invperms(self) -> NumbaList:
        """Schreier-Sims inverse-transversal chain for the discovered symmetry group."""
        return self._effective_group_chain_data[3]

    @cached_property
    def _base_marginal_payload(self) -> Tuple[List[List[List[int]]], List[Tuple[int, ...]]]:
        """
        Canonical marginal representatives under the initial core symmetries.
        """
        seen_keys: set[Tuple[int, ...]] = set()
        marginals: List[List[List[int]]] = []
        support_keys: List[Tuple[int, ...]] = []
        base_outcomes = tuple(range(self.outcomes))
        perm_iter = permutations(range(1, self.n + 1))
        perm_iter = tqdm(
            perm_iter,
            total=factorial(self.n),
            desc="Canonicalizing marginals",
            disable=not self.show_progress,
        )
        for perm in perm_iter:
            for pat in product(base_outcomes, repeat=self.n):
                # Candidate marginal -> sparse one-hot support in ambient coordinates.
                support = _marginal_support_from_perm(perm, pat, self.n, self.outcomes)
                # Canonicalize and dedupe by support key.
                canonical_support = canonical_leximin_support_indices(support, self.N, self.core_level_invperms)
                key = tuple(int(x) for x in canonical_support.tolist())
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                marginal = _marginal_from_support_key(key, self.n, self.outcomes)
                if self.marginal_filter_fn is not None and not self.marginal_filter_fn(marginal):
                    continue
                marginals.append(marginal)
                support_keys.append(key)
        return marginals, support_keys

    @property
    def base_marginals(self) -> List[List[List[int]]]:
        """Canonical marginals under core symmetries before automatic compression."""
        return self._base_marginal_payload[0]

    @property
    def base_support_keys(self) -> List[Tuple[int, ...]]:
        """Sorted support keys for base marginals."""
        return self._base_marginal_payload[1]

    @property
    def base_nof_marginals(self) -> int:
        """Number of base canonical marginals."""
        return len(self.base_marginals)

    @cached_property
    def base_row_labels(self) -> np.ndarray:
        """Operator-name tuple labels for each base marginal row."""
        return np.asarray(
            [tuple(self.prob._lexrepr_to_names[self.prob.mon_to_lexrepr(m)]) for m in self.base_marginals],
            dtype=object,
        )

    def _factorized_value_and_label(self, marginal: List[List[int]]) -> Tuple[sp.Expr, str]:
        """
        Compute value and copy-index-free cycle-factorized label for one marginal.

        Example:
          P_global(A^{1,1}=1,A^{2,2}=1,A^{3,3}=0)
          -> P_loop(A=1)^2*P_loop(A=0)
        """
        J = _perm_from_marginal(marginal)
        cycles = _cycles_from_J(J)
        by_i = {int(row[1]): row for row in marginal}
        loop_counts: Dict[str, int] = {}
        value = sp.Integer(1)

        for cyc in cycles:
            cycle_mon = np.asarray([by_i[i] for i in cyc], dtype=np.intc)
            lex = self.prob.mon_to_lexrepr(cycle_mon)
            copy_free_names = tuple(self.prob._lexrepr_to_copy_index_free_names[lex])
            label = "P_loop(" + ",".join(copy_free_names) + ")"
            loop_counts[label] = loop_counts.get(label, 0) + 1
            cyc_out = [int(by_i[i][4]) for i in cyc]
            value *= self.distribution.prob_event_loop(cyc_out)

        factors: List[str] = []
        for label, mult in loop_counts.items():
            if mult == 1:
                factors.append(label)
            else:
                factors.append(f"{label}^{mult}")
        return sp.simplify(value), "*".join(factors)

    @cached_property
    def _base_known_payload(self) -> Tuple[np.ndarray, List[str]]:
        """Known values and labels computed on base marginals."""
        known_values = np.empty(self.base_nof_marginals, dtype=object)
        known_labels: List[str] = []
        for idx, marginal in enumerate(
            tqdm(self.base_marginals, desc="Computing marginal values...", disable=not self.show_progress)
        ):
            value, label = self._factorized_value_and_label(marginal)
            known_values[idx] = value
            known_labels.append(label)
        return known_values, known_labels

    @property
    def base_known_labels(self) -> List[str]:
        """Base cycle-factorized known labels before orbit compression."""
        return self._base_known_payload[1]

    @property
    def base_known_values_symbolic(self) -> np.ndarray:
        """Base known marginal values (symbolic) before orbit compression."""
        return self._base_known_payload[0]

    @cached_property
    def discovered_symmetries(self) -> np.ndarray:
        """Largest discovered stabilizing subgroup used for final canonicalization."""
        if not self.auto_discover_symmetries:
            return self.core_symmetries
        support_to_idx = {key: idx for idx, key in enumerate(self.base_support_keys)}
        values = self.base_known_values_symbolic

        def _stabilizer_predicate(perm: np.ndarray) -> bool:
            for idx, key in enumerate(self.base_support_keys):
                mapped_support = np.asarray([int(perm[pos]) for pos in key], dtype=np.int64)
                mapped_canon = canonical_leximin_support_indices(
                    mapped_support,
                    self.N,
                    self.core_level_invperms,
                )
                mapped_key = tuple(int(x) for x in mapped_canon.tolist())
                mapped_idx = support_to_idx.get(mapped_key)
                if mapped_idx is None:
                    return False
                if sp.simplify(values[idx] - values[mapped_idx]) != 0:
                    return False
            return True

        discovered, _group = discovery_symmetries_from_predicate(
            stabilizer_predicate=_stabilizer_predicate,
            scenario=self.prob,
            initial_generators=self.core_symmetries,
            candidate_generators=self.candidate_symmetries,
            verbose=self.verbose_symmetry_discovery,
            return_group=True,
            progress_desc="Discovering ring stabilizing symmetries",
        )
        if discovered.size == 0:
            return self.core_symmetries
        return np.asarray(discovered, dtype=int)

    @cached_property
    def _row_compression_payload(
        self,
    ) -> Tuple[
        List[List[List[int]]],
        np.ndarray,
        np.ndarray,
        List[str],
        np.ndarray,
        List[Tuple[int, ...]],
        np.ndarray,
        List[str],
        List[Tuple[str, ...]],
    ]:
        """Compress base rows by discovered symmetry orbits and keep orbit metadata."""
        if not self.compress_rows_under_discovered_group:
            orbit_members = [(idx,) for idx in range(self.base_nof_marginals)]
            multiplicities = np.ones(self.base_nof_marginals, dtype=np.int64)
            member_labels = [(self.base_known_labels[idx],) for idx in range(self.base_nof_marginals)]
            row_labels = np.asarray([" ".join(label) for label in self.base_row_labels.tolist()], dtype=object)
            return (
                self.base_marginals,
                self.base_known_values_symbolic,
                np.asarray([float(sp.N(v)) for v in self.base_known_values_symbolic], dtype=float),
                self.base_known_labels,
                row_labels,
                orbit_members,
                multiplicities,
                self.base_known_labels,
                member_labels,
            )

        orbit_map: Dict[Tuple[int, ...], List[int]] = {}
        orbit_order: List[Tuple[int, ...]] = []
        for idx, key in enumerate(self.base_support_keys):
            support = np.asarray(key, dtype=np.int64)
            canon = canonical_leximin_support_indices(support, self.N, self.level_invperms)
            canon_key = tuple(int(x) for x in canon.tolist())
            if canon_key not in orbit_map:
                orbit_map[canon_key] = []
                orbit_order.append(canon_key)
            orbit_map[canon_key].append(idx)

        marginals: List[List[List[int]]] = []
        known_values_symbolic = np.empty(len(orbit_order), dtype=object)
        known_values_float = np.empty(len(orbit_order), dtype=float)
        known_labels: List[str] = []
        row_labels_list: List[str] = []
        orbit_members: List[Tuple[int, ...]] = []
        multiplicities = np.empty(len(orbit_order), dtype=np.int64)
        orbit_average_labels: List[str] = []
        orbit_member_labels: List[Tuple[str, ...]] = []

        for orbit_idx, canon_key in enumerate(orbit_order):
            members = tuple(orbit_map[canon_key])
            orbit_members.append(members)
            multiplicities[orbit_idx] = len(members)
            rep = members[0]
            marginals.append(self.base_marginals[rep])
            member_values = [self.base_known_values_symbolic[m] for m in members]
            representative_value = member_values[0]
            for other_value in member_values[1:]:
                if sp.simplify(other_value - representative_value) != 0:
                    raise ValueError("Orbit contains non-equal symbolic known values.")
            known_values_symbolic[orbit_idx] = representative_value
            known_values_float[orbit_idx] = float(sp.N(representative_value))

            member_known_labels = tuple(self.base_known_labels[m] for m in members)
            avg_known_label = _average_orbit_label(member_known_labels)
            known_labels.append(avg_known_label)
            orbit_average_labels.append(avg_known_label)
            orbit_member_labels.append(member_known_labels)

            member_row_labels = [" ".join(self.base_row_labels[m]) for m in members]
            row_labels_list.append(_average_orbit_label(member_row_labels))

        row_labels = np.asarray(row_labels_list, dtype=object)
        return (
            marginals,
            known_values_symbolic,
            known_values_float,
            known_labels,
            row_labels,
            orbit_members,
            multiplicities,
            orbit_average_labels,
            orbit_member_labels,
        )

    @property
    def marginals(self) -> List[List[List[int]]]:
        """Final marginals after optional discovered-group row compression."""
        return self._row_compression_payload[0]

    @property
    def known_values_symbolic(self) -> np.ndarray:
        """Known marginal values (symbolic) aligned with final marginals."""
        return self._row_compression_payload[1]

    @property
    def known_values(self) -> np.ndarray:
        """Known marginal values (float) aligned with final marginals."""
        return self._row_compression_payload[2]

    @property
    def known_labels(self) -> List[str]:
        """Known labels aligned with final marginals."""
        return self._row_compression_payload[3]

    @property
    def row_labels(self) -> np.ndarray:
        """Human-readable row labels aligned with final marginals."""
        return self._row_compression_payload[4]

    @property
    def row_orbit_members(self) -> List[Tuple[int, ...]]:
        """For each final row, indices of base rows in its discovered-group orbit."""
        return self._row_compression_payload[5]

    @property
    def row_orbit_multiplicities(self) -> np.ndarray:
        """Orbit multiplicities for each compressed row."""
        return self._row_compression_payload[6]

    @property
    def row_orbit_average_labels(self) -> List[str]:
        """Orbit-average known labels used as compressed row names."""
        return self._row_compression_payload[7]

    @property
    def row_orbit_member_labels(self) -> List[Tuple[str, ...]]:
        """Per-orbit list of known labels from base rows."""
        return self._row_compression_payload[8]

    @property
    def nof_marginals(self) -> int:
        """Number of final canonical marginals."""
        return len(self.marginals)

    @cached_property
    def _canonical_global_lhs_payload(self) -> Tuple[np.ndarray, coo_array]:
        """
        Pair `(global_keys, lhs_raw)` for canonical global-event extension constraints.
        """
        global_event_map = NumbaDict.empty(key_type=types.uint64, value_type=types.int32)
        list_of_all_global_keys: List[int] = []
        next_event_idx = np.int32(1)  # 0 is reserved as "not present" sentinel

        global_extension_count = int(pow(self.outcomes, self.n * (self.n - 1)))
        total_entries = int(self.nof_marginals * global_extension_count)
        sparse_matrix_rows = np.empty(total_entries, dtype=np.int32)
        sparse_matrix_cols = np.empty(total_entries, dtype=np.int32)
        sparse_matrix_data = np.ones(total_entries, dtype=np.int8)
        row_grid = np.broadcast_to(
            np.arange(self.nof_marginals, dtype=np.int32)[:, None],
            (self.nof_marginals, global_extension_count),
        )
        sparse_matrix_rows[:] = row_grid.reshape(-1)

        # Enumerate global extensions and canonicalize each extension to a column key.
        for row_num, marginal in enumerate(
            tqdm(self.marginals, desc="Finding global extensions...", disable=not self.show_progress)
        ):
            start = row_num * global_extension_count
            next_event_idx = representatives_of_global_extensions_uint64(
                n=self.n,
                outcomes=self.outcomes,
                marginal=marginal,
                level_invperms=self.level_invperms,
                global_event_map=global_event_map,
                next_event_idx=next_event_idx,
                list_of_all_LP_variables=list_of_all_global_keys,
                total=global_extension_count,
                sparse_matrix_cols=sparse_matrix_cols,
                start=start,
            )
        if int(next_event_idx) > np.iinfo(np.int32).max:
            raise ValueError("next_event_idx exceeds int32 range; use wider dtype")

        lhs_raw = coo_array(
            (sparse_matrix_data, (sparse_matrix_rows, sparse_matrix_cols)),
            shape=(self.nof_marginals, int(next_event_idx)),
        )
        lhs_raw.sum_duplicates()
        global_keys = np.asarray(list_of_all_global_keys, dtype=np.uint64)
        return global_keys, lhs_raw

    @property
    def global_keys(self) -> np.ndarray:
        """Canonical uint64 keys of global-event LP columns (excluding sentinel 0)."""
        return self._canonical_global_lhs_payload[0]

    @property
    def lhs_raw(self) -> coo_array:
        """Raw sparse LHS matrix with sentinel column 0 kept."""
        return self._canonical_global_lhs_payload[1]

    @cached_property
    def inflation_matrix(self) -> coo_array:
        """Final sparse equality matrix combining known-value and extension constraints."""
        if self._cached_inflation_matrix is not None:
            return self._cached_inflation_matrix
        lhs_coo = self.lhs_raw
        shifted_cols = lhs_coo.col.astype(np.int64, copy=True)
        mask = shifted_cols != 0
        shifted_cols[mask] += self.nof_marginals

        known_rows = np.arange(self.nof_marginals, dtype=np.int32)
        known_cols = np.arange(1, self.nof_marginals + 1, dtype=np.int32)
        known_data = -np.ones(self.nof_marginals, dtype=np.int8)

        all_rows = np.concatenate([known_rows, lhs_coo.row.astype(np.int32, copy=False)])
        all_cols = np.concatenate([known_cols, shifted_cols.astype(np.int32, copy=False)])
        all_data = np.concatenate([known_data, lhs_coo.data.astype(np.int8, copy=False)])

        total_cols = self.nof_marginals + self.lhs_raw.shape[1]
        inflation_matrix = coo_array(
            (all_data, (all_rows, all_cols)),
            shape=(self.nof_marginals, total_cols),
        )
        inflation_matrix.sum_duplicates()
        self._cached_inflation_matrix = inflation_matrix
        self._save_cache(self.variable_names, self.known_vars, inflation_matrix)
        return inflation_matrix

    @cached_property
    def known_vars_symbolic(self) -> coo_array:
        """Sparse symbolic known-variables row vector aligned with `inflation_matrix` columns."""
        total_cols = self.nof_marginals + self.lhs_raw.shape[1]
        known_positions = np.arange(1, self.nof_marginals + 1, dtype=np.int32)
        known_rows0 = np.zeros(self.nof_marginals, dtype=np.int32)
        return coo_array(
            (self.known_values_symbolic, (known_rows0, known_positions)),
            shape=(1, total_cols),
            dtype=object,
        )

    @cached_property
    def known_vars(self) -> coo_array:
        """Sparse float known-variables row vector aligned with `inflation_matrix` columns."""
        if self._cached_known_vars is not None:
            return self._cached_known_vars
        total_cols = self.nof_marginals + self.lhs_raw.shape[1]
        known_positions = np.arange(1, self.nof_marginals + 1, dtype=np.int32)
        known_rows0 = np.zeros(self.nof_marginals, dtype=np.int32)
        known_vars = coo_array(
            (self.known_values, (known_rows0, known_positions)),
            shape=(1, total_cols),
        )
        self._cached_known_vars = known_vars
        return known_vars

    @cached_property
    def variable_names(self) -> np.ndarray:
        """Variable names aligned with matrix columns: const, known marginals, global keys."""
        if self._cached_variable_names is not None:
            return self._cached_variable_names
        variable_names = np.asarray(
            ["1", *self.known_labels, *[str(k) for k in self.global_keys.tolist()]],
            dtype=str,
        )
        self._cached_variable_names = variable_names
        return variable_names


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    from inflation.distributions import EJMDistribution, NSIPRDistribution, RGBDistribution
    from inflation.lp.lp_utils import solveLP_sparse

    n = 4

    demos = [
        ("NSI-PR", NSIPRDistribution()),
        ("EJM", EJMDistribution()),
        ("EJM coarse [[0],[1],[2,3]]", EJMDistribution(coarsen=[[0], [1], [2, 3]])),
        ("RGB", RGBDistribution()),
    ]

    demo_preps: dict[str, PrepLP] = {}
    for label, distribution in demos:
        print(f"\n=== Symmetry demo: {label} (n={n}) ===")
        prep = PrepLP(
            n,
            distribution,
            cache_name=None,
            show_progress=True,
            auto_discover_symmetries=True,
            compress_rows_under_discovered_group=True,
            verbose_symmetry_discovery=True,
        )
        print(f"  base rows={prep.base_nof_marginals}, compressed rows={prep.nof_marginals}")
        demo_preps[label] = prep

    prep_nsi = demo_preps["NSI-PR"]
    variable_names = prep_nsi.variable_names
    known_vars = prep_nsi.known_vars
    inflation_matrix = prep_nsi.inflation_matrix

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

    def _evaluate_sparse_certificate_on_knowns(
        sparse_certificate: coo_array,
        known_vec: coo_array,
    ) -> float:
        """Evaluate sparse certificate on known assignments without densifying."""
        cert_coo = sparse_certificate
        known_cols = known_vec.col.astype(np.int64, copy=False)
        known_vals = known_vec.data.astype(float, copy=False)
        known_map = dict(zip(known_cols.tolist(), known_vals.tolist()))
        value = 0.0
        for col, coeff in zip(cert_coo.col.tolist(), cert_coo.data.tolist()):
            value += float(coeff) * float(known_map.get(int(col), 0.0))
        return value

    def _print_infeasibility_certificate_analysis(
        solution_dict: dict,
        known_vec: coo_array,
        *,
        chop_tol: float = 1e-10,
        top_k: int = 25,
    ) -> None:
        """Print a concise analysis of the dual infeasibility certificate."""
        cert_dict = solution_dict.get("dual_certificate", {})
        if not cert_dict:
            print("No dual certificate entries were returned.")
            return

        # Mimic InflationLP-style coefficient cleanup by chopping tiny entries.
        cleaned = {
            str(var): float(coeff)
            for var, coeff in cert_dict.items()
            if abs(float(coeff)) > chop_tol
        }
        if not cleaned:
            print(f"Dual certificate is numerically zero after chop_tol={chop_tol:g}.")
            return

        known_terms = {k: v for k, v in cleaned.items() if k.startswith("P_global(")}
        global_terms = {k: v for k, v in cleaned.items() if (k not in known_terms and k != "1")}
        const_coeff = cleaned.get("1", 0.0)

        cert_value = _evaluate_sparse_certificate_on_knowns(
            solution_dict["sparse_certificate"], known_vec
        )

        print("\nCertificate analysis:")
        print(f"  nonzero terms (after chop): {len(cleaned)}")
        print(f"  known-marginal terms: {len(known_terms)}")
        print(f"  global-event terms: {len(global_terms)}")
        print(f"  constant term coeff: {const_coeff:.12g}")
        print(f"  certificate value on knowns: {cert_value:.12g}")
        print("  incompatibility witness criterion: certificate < 0")

        top_terms = sorted(cleaned.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
        print(f"  top {len(top_terms)} terms by |coefficient|:")
        for var, coeff in top_terms:
            print(f"    {coeff:+.12g} * {var}")

    if not solution.get("success", False):
        _print_infeasibility_certificate_analysis(solution, known_vars)
