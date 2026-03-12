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
from itertools import combinations, permutations, product
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple
import sys

import numpy as np
import sympy as sp
from numba import njit, prange
from numba.typed import List as NumbaList
from scipy.sparse import coo_array, csr_array
from inflation.progress_utils import make_tqdm as tqdm, progress_stage

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
from inflation.applications.ring_utils import build_off_diagonal_ring_problem
from inflation.distributions.protocols import RingDistributionProtocol
from inflation.symmetry_utils import discovery_symmetries_from_predicate

CACHE_FORMAT_VERSION = np.int64(6)


def ring_problem(inflation_level: int, distribution: RingDistributionProtocol) -> InflationProblem:
    return build_off_diagonal_ring_problem(
        inflation_level,
        int(distribution.nof_outcomes),
        classical_sources="all",
    )


def _prepare_group_chain(
    prob: InflationProblem,
    symmetries: np.ndarray | None = None,
) -> Tuple[int, int, int, NumbaList]:
    n = prob.inflation_level_per_source[0]
    outcomes = prob.outcomes_per_party[0]
    if outcomes >= 255:
        raise ValueError("outcomes must be < 255 to fit in compact dtypes")
    if prob._nr_operators % outcomes != 0:
        raise ValueError("Ring lexorder width must be divisible by the number of outcomes")
    slot_count = prob._nr_operators // outcomes
    max_event_count = pow(outcomes, slot_count)
    if max_event_count > np.iinfo(np.uint64).max:
        raise ValueError("events do not fit in uint64")

    N = prob._nr_operators
    if N > np.iinfo(np.uint16).max:
        raise ValueError("N exceeds uint16 range; use wider dtype for permutations")
    if symmetries is None:
        symmetries = np.asarray(prob.symmetries, dtype=int)
    G = build_sympy_group(symmetries, N)
    level_invperms = prepare_group_chain(G, N)
    return n, outcomes, N, level_invperms


def _offdiag_slot_index(i: int, j: int, n: int) -> int:
    if i == j:
        raise ValueError("Off-diagonal ring slots do not support self-loops")
    i0 = i - 1
    j0 = j - 1
    return i0 * (n - 1) + j0 - int(j0 > i0)


def _slot_index_to_pair(slot: int, n: int) -> Tuple[int, int]:
    i0, pos = divmod(int(slot), n - 1)
    j0 = pos if pos < i0 else pos + 1
    return i0 + 1, j0 + 1


def _derangements(labels: Sequence[int]) -> Iterator[Tuple[int, ...]]:
    for perm in permutations(labels):
        if all(src != dst for src, dst in zip(labels, perm)):
            yield perm


def _marginal_support_from_cycle_cover(
    domain: Sequence[int],
    image: Sequence[int],
    outcome_pattern: Sequence[int],
    n: int,
    outcomes: int,
) -> np.ndarray:
    support = np.empty(len(domain), dtype=np.int64)
    for idx, (i, j, outcome) in enumerate(zip(domain, image, outcome_pattern)):
        slot = _offdiag_slot_index(int(i), int(j), n)
        support[idx] = slot * outcomes + int(outcome)
    return support


def _marginal_from_support_key(
    support_key: Tuple[int, ...],
    n: int,
    outcomes: int,
) -> List[List[int]]:
    by_i: Dict[int, Tuple[int, int]] = {}
    incoming: Dict[int, int] = {}
    for coord in support_key:
        slot, a = divmod(int(coord), outcomes)
        i, j = _slot_index_to_pair(slot, n)
        if i in by_i:
            raise ValueError("Invalid canonical marginal support: duplicate row assignment.")
        if j in incoming:
            raise ValueError("Invalid canonical marginal support: duplicate column assignment.")
        if i == j:
            raise ValueError("Invalid canonical marginal support: self-loops are forbidden.")
        by_i[i] = (j, a)
        incoming[j] = i
    if not by_i:
        raise ValueError("Invalid canonical marginal support: empty support.")
    if set(by_i) != set(incoming):
        raise ValueError("Invalid canonical marginal support: support is not a cycle cover.")
    return [[1, i, by_i[i][0], 0, by_i[i][1]] for i in sorted(by_i)]


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


@njit(cache=True, parallel=True, fastmath=True)
def _fill_global_extension_keys_parallel(
    row_entry_ptr: np.ndarray,
    row_fixed_ptr: np.ndarray,
    fixed_slots_flat: np.ndarray,
    fixed_vals_flat: np.ndarray,
    row_remaining_ptr: np.ndarray,
    remaining_slots_flat: np.ndarray,
    nof_off_diagonal_slots: int,
    outcomes: int,
    level_invperms: NumbaList,
) -> np.ndarray:
    """Enumerate canonical uint64 global-event keys for each row in parallel."""
    nof_rows = row_entry_ptr.size - 1
    total_entries = int(row_entry_ptr[-1])
    all_keys = np.empty(total_entries, dtype=np.uint64)
    for row_num in prange(nof_rows):
        evt = np.zeros(nof_off_diagonal_slots, dtype=np.uint8)
        fixed_start = int(row_fixed_ptr[row_num])
        fixed_end = int(row_fixed_ptr[row_num + 1])
        for pos in range(fixed_start, fixed_end):
            evt[fixed_slots_flat[pos]] = fixed_vals_flat[pos]
        remaining_start = int(row_remaining_ptr[row_num])
        remaining_end = int(row_remaining_ptr[row_num + 1])
        remaining_size = remaining_end - remaining_start
        entry_start = int(row_entry_ptr[row_num])
        entry_end = int(row_entry_ptr[row_num + 1])
        for offset in range(entry_end - entry_start):
            tmp = offset
            for rem_pos in range(remaining_size - 1, -1, -1):
                idx = remaining_slots_flat[remaining_start + rem_pos]
                evt[idx] = tmp % outcomes
                tmp //= outcomes
            all_keys[entry_start + offset] = canonical_leximin_coset_chain_uint64(
                evt,
                outcomes,
                level_invperms,
            )
    return all_keys


@njit(cache=True, parallel=True, fastmath=True)
def _sort_rows_and_count_unique(
    sparse_matrix_cols: np.ndarray,
    row_entry_ptr: np.ndarray,
) -> np.ndarray:
    """Sort each row slice in place and count unique columns per row."""
    nof_rows = row_entry_ptr.size - 1
    row_nnz = np.empty(nof_rows, dtype=np.int64)
    for row_num in prange(nof_rows):
        start = int(row_entry_ptr[row_num])
        end = int(row_entry_ptr[row_num + 1])
        if end > start:
            row_slice = sparse_matrix_cols[start:end]
            row_slice.sort()
            unique_cols = 1
            prev = row_slice[0]
            for pos in range(start + 1, end):
                col = sparse_matrix_cols[pos]
                if col != prev:
                    unique_cols += 1
                    prev = col
        else:
            unique_cols = 0
        row_nnz[row_num] = unique_cols
    return row_nnz


@njit(cache=True, parallel=True, fastmath=True)
def _fill_direct_csr_from_sorted_rows(
    sparse_matrix_cols: np.ndarray,
    row_entry_ptr: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
) -> None:
    """Fill the direct marginal-form CSR matrix from sorted per-row column ids."""
    nof_marginals = row_entry_ptr.size - 1
    for row_num in prange(nof_marginals):
        write_pos = int(indptr[row_num])
        start = int(row_entry_ptr[row_num])
        end = int(row_entry_ptr[row_num + 1])
        if end > start:
            current = sparse_matrix_cols[start]
            multiplicity = 1
            for pos in range(start + 1, end):
                col = sparse_matrix_cols[pos]
                if col == current:
                    multiplicity += 1
                else:
                    indices[write_pos] = current
                    data[write_pos] = float(multiplicity)
                    write_pos += 1
                    current = col
                    multiplicity = 1
            indices[write_pos] = current
            data[write_pos] = float(multiplicity)


@njit(cache=True, fastmath=True)
def _csr_to_column_payload(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    nof_cols: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build MOSEK-style column pointers and row indices from CSR arrays."""
    counts = np.zeros(nof_cols, dtype=np.int64)
    for pos in range(indices.size):
        counts[int(indices[pos])] += 1

    aptrb = np.empty(nof_cols, dtype=np.int64)
    aptre = np.empty(nof_cols, dtype=np.int64)
    running = np.int64(0)
    for col in range(nof_cols):
        aptrb[col] = running
        running += counts[col]
        aptre[col] = running

    asub = np.empty(indices.size, dtype=np.int32)
    aval = np.empty(data.size, dtype=np.float64)
    next_pos = aptrb.copy()
    nof_rows = indptr.size - 1
    for row_num in range(nof_rows):
        start = int(indptr[row_num])
        end = int(indptr[row_num + 1])
        for pos in range(start, end):
            col = int(indices[pos])
            write_pos = int(next_pos[col])
            asub[write_pos] = np.int32(row_num)
            aval[write_pos] = data[pos]
            next_pos[col] = write_pos + 1
    return aptrb, aptre, asub, aval


@njit(cache=True, fastmath=True)
def _csr_weighted_column_sums(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    row_weights: np.ndarray,
    nof_cols: int,
) -> np.ndarray:
    """Compute weighted column sums for a CSR matrix without format conversion."""
    result = np.zeros(nof_cols, dtype=np.float64)
    nof_rows = indptr.size - 1
    for row_num in range(nof_rows):
        weight = row_weights[row_num]
        start = int(indptr[row_num])
        end = int(indptr[row_num + 1])
        for pos in range(start, end):
            result[int(indices[pos])] += weight * data[pos]
    return result

def _stable_unique_inverse(keys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return first-appearance unique keys and inverse indices preserving row-major order."""
    if keys.size == 0:
        return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.int64)
    unique_keys, first_idx, inverse = np.unique(
        keys,
        return_index=True,
        return_inverse=True,
    )
    order = np.argsort(first_idx, kind="stable")
    remap = np.empty(order.size, dtype=np.int64)
    remap[order] = np.arange(order.size, dtype=np.int64)
    return unique_keys[order], remap[inverse]


def _boundkey_array(value, size: int) -> np.ndarray:
    """Allocate a MOSEK boundkey array with the current binding's accepted dtype."""
    return np.full(size, value, dtype=np.int32)


def _resolve_mosek_optimizer(name: str):
    """Map a user-facing optimizer choice to the MOSEK optimizer enum."""
    import mosek

    normalized = str(name).strip().lower()
    optimizer_map = {
        "simplex": mosek.optimizertype.free_simplex,
        "free_simplex": mosek.optimizertype.free_simplex,
        "primal_simplex": mosek.optimizertype.primal_simplex,
        "dual_simplex": mosek.optimizertype.dual_simplex,
        "interior_point": mosek.optimizertype.intpnt,
        "intpnt": mosek.optimizertype.intpnt,
    }
    try:
        return optimizer_map[normalized]
    except KeyError as exc:
        raise ValueError(
            "Unknown optimizer choice. Expected one of: "
            "simplex, free_simplex, primal_simplex, dual_simplex, interior_point, intpnt."
        ) from exc


def _resolve_ring_solve_mode(mode: str) -> str:
    """Normalize the supported direct ring LP solve modes."""
    normalized = str(mode).strip().lower()
    if normalized not in {"feasibility", "incompatible_fraction", "generalized_robustness"}:
        raise ValueError(
            "Unknown solve mode. Expected one of: "
            "feasibility, incompatible_fraction, generalized_robustness."
        )
    return normalized


def _normalize_prep_solution_path(path: str | Path) -> Path:
    """Normalize a PrepLP solution archive path to `.npz`."""
    archive_path = Path(path)
    if archive_path.suffix == "":
        return archive_path.with_suffix(".npz")
    if archive_path.suffix.lower() == ".npz":
        return archive_path
    raise ValueError("Archive path must omit the extension or end with '.npz'.")


def save_prep_lp_solution(
    solution: Dict,
    path: str | Path,
    *,
    compression: bool = True,
) -> Path:
    """Save the direct-basis PrepLP solution dictionary to an NPZ archive."""
    archive_path = _normalize_prep_solution_path(path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    global_keys = np.asarray(list(solution["x"].keys()), dtype=np.uint64)
    x_values = np.asarray([float(solution["x"][key]) for key in global_keys.tolist()], dtype=np.float64)
    constraint_names = np.asarray(solution["constraint_names"], dtype=str)
    sparse_certificate = solution["sparse_certificate"].tocoo(copy=False)
    if sparse_certificate.nnz:
        order = np.argsort(sparse_certificate.col, kind="stable")
        certificate_col = sparse_certificate.col[order].astype(np.int64, copy=False)
        certificate_data = sparse_certificate.data[order].astype(np.float64, copy=False)
    else:
        certificate_col = np.empty(0, dtype=np.int64)
        certificate_data = np.empty(0, dtype=np.float64)
    term_code, term_desc = solution["term_code"]

    save_fn = np.savez_compressed if compression else np.savez
    save_fn(
        archive_path,
        status=np.asarray(str(solution["status"])),
        success=np.asarray(bool(solution["success"])),
        solver_success=np.asarray(bool(solution["solver_success"])),
        primal_value=np.asarray(float(solution["primal_value"])),
        dual_value=np.asarray(float(solution["dual_value"])),
        term_code=np.asarray(str(term_code)),
        term_desc=np.asarray(str(term_desc)),
        mode=np.asarray(str(solution["mode"])),
        known_mass=np.asarray(float(solution["known_mass"])),
        optimized_mass=np.asarray(float(solution["optimized_mass"])),
        incompatible_fraction=np.asarray(float(solution["incompatible_fraction"])),
        generalized_robustness=np.asarray(float(solution["generalized_robustness"])),
        global_keys=global_keys,
        x_values=x_values,
        constraint_names=constraint_names,
        certificate_col=certificate_col,
        certificate_data=certificate_data,
    )
    return archive_path


def read_prep_lp_solution(path: str | Path, *, allow_pickle: bool = True) -> Dict:
    """Read a PrepLP solution archive and reconstruct the direct-basis solution dictionary."""
    archive_path = _normalize_prep_solution_path(path)
    with np.load(archive_path, allow_pickle=allow_pickle) as z:
        global_keys = np.asarray(z["global_keys"], dtype=np.uint64)
        x_values = np.asarray(z["x_values"], dtype=np.float64)
        constraint_names = np.asarray(z["constraint_names"], dtype=str)
        certificate_col = np.asarray(z["certificate_col"], dtype=np.int64)
        certificate_data = np.asarray(z["certificate_data"], dtype=np.float64)
        cert_row = np.zeros(certificate_col.shape[0], dtype=np.int32)
        sparse_certificate = coo_array(
            (certificate_data, (cert_row, certificate_col)),
            shape=(1, constraint_names.size),
        )
        dual_certificate = dict(
            zip(constraint_names[certificate_col].tolist(), certificate_data.tolist())
        )
        return {
            "primal_value": float(np.asarray(z["primal_value"]).item()),
            "dual_value": float(np.asarray(z["dual_value"]).item()),
            "status": str(np.asarray(z["status"]).item()),
            "success": bool(np.asarray(z["success"]).item()),
            "solver_success": bool(np.asarray(z["solver_success"]).item()),
            "mode": str(np.asarray(z["mode"]).item()),
            "known_mass": float(np.asarray(z["known_mass"]).item()),
            "optimized_mass": float(np.asarray(z["optimized_mass"]).item()),
            "incompatible_fraction": float(np.asarray(z["incompatible_fraction"]).item()),
            "generalized_robustness": float(np.asarray(z["generalized_robustness"]).item()),
            "dual_certificate": dual_certificate,
            "sparse_certificate": sparse_certificate,
            "constraint_names": constraint_names,
            "x": dict(zip(global_keys.tolist(), x_values.tolist())),
            "term_code": (
                str(np.asarray(z["term_code"]).item()),
                str(np.asarray(z["term_desc"]).item()),
            ),
        }


def load_prep_lp_solution(path: str | Path, *, allow_pickle: bool = True) -> Dict:
    """Compatibility alias for `read_prep_lp_solution()`."""
    return read_prep_lp_solution(path, allow_pickle=allow_pickle)


# =========================
# Cycle extraction & factorized value for a marginal
# =========================
def _perm_from_marginal(marginal: List[List[int]]) -> Dict[int, int]:
    """Extract the participating partial permutation `i -> j` from a marginal."""
    J: Dict[int, int] = {}
    incoming: set[int] = set()
    for (_one, i, j, _zero, _a) in marginal:
        i_int = int(i)
        j_int = int(j)
        if i_int == j_int:
            raise ValueError("Ring marginals cannot contain self-loops.")
        if i_int in J:
            raise ValueError("Marginal has duplicate outgoing assignments.")
        if j_int in incoming:
            raise ValueError("Marginal has duplicate incoming assignments.")
        J[i_int] = j_int
        incoming.add(j_int)
    if set(J) != incoming:
        raise ValueError("Marginal is not a disjoint union of cycles.")
    return J


def _outcomes_from_marginal(marginal: List[List[int]]) -> Dict[int, int]:
    """Extract the outcome assignment for the participating copy labels."""
    a: Dict[int, int] = {}
    for (_one, i, _j, _zero, val) in marginal:
        i_int = int(i)
        if i_int in a:
            raise ValueError("Marginal has duplicate outcome assignments.")
        a[i_int] = int(val)
    return a


def _cycles_from_J(J: Dict[int, int]) -> List[List[int]]:
    """Disjoint cycles of a partial permutation on its participating subset."""
    if set(J) != set(J.values()):
        raise ValueError("Partial permutation is not a cycle cover.")
    seen: set[int] = set()
    cycles: List[List[int]] = []
    for start in sorted(J):
        if start in seen:
            continue
        cyc = []
        v = start
        while v not in seen:
            seen.add(v)
            cyc.append(v)
            v = J[v]
        if v != start:
            raise ValueError("Partial permutation orbit does not close to a cycle.")
        if len(cyc) == 1:
            raise ValueError("Ring marginals cannot contain 1-cycles.")
        cycles.append(cyc)
    return cycles


def factorized_marginal_value(
    marginal: List[List[int]],
    distribution: RingDistributionProtocol,
) -> sp.Expr:
    """
    Multiply loop scalars over the disjoint cycles on the participating subset.
    """
    J = _perm_from_marginal(marginal)
    a = _outcomes_from_marginal(marginal)
    val = sp.Integer(1)
    for cyc in _cycles_from_J(J):
        cyc_out = [a[i] for i in cyc]
        val *= distribution.prob_event_loop(cyc_out)
    return val


def _build_row_extension_descriptors(
    marginals: List[List[List[int]]],
    *,
    n: int,
    outcomes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten per-row fixed/remaining slot metadata for Numba parallel kernels."""
    nof_rows = len(marginals)
    nof_slots = n * (n - 1)

    row_entry_ptr = np.empty(nof_rows + 1, dtype=np.int64)
    row_fixed_ptr = np.empty(nof_rows + 1, dtype=np.int64)
    row_remaining_ptr = np.empty(nof_rows + 1, dtype=np.int64)
    row_entry_ptr[0] = 0
    row_fixed_ptr[0] = 0
    row_remaining_ptr[0] = 0

    fixed_slots_chunks: List[np.ndarray] = []
    fixed_vals_chunks: List[np.ndarray] = []
    remaining_slots_chunks: List[np.ndarray] = []

    for row_num, marginal in enumerate(marginals):
        fixed_slots = np.empty(len(marginal), dtype=np.int64)
        fixed_vals = np.empty(len(marginal), dtype=np.uint8)
        seen_slots: set[int] = set()
        for idx, (_one, i, j, _zero, a) in enumerate(marginal):
            slot = _offdiag_slot_index(int(i), int(j), n)
            if slot in seen_slots:
                raise ValueError("Invalid marginal: duplicate fixed off-diagonal slot.")
            seen_slots.add(slot)
            fixed_slots[idx] = slot
            fixed_vals[idx] = np.uint8(a)
        mask = np.ones(nof_slots, dtype=bool)
        if fixed_slots.size:
            mask[fixed_slots] = False
        remaining_slots = np.nonzero(mask)[0].astype(np.int64, copy=False)
        total = pow(outcomes, int(remaining_slots.size))
        row_entry_ptr[row_num + 1] = row_entry_ptr[row_num] + np.int64(total)
        row_fixed_ptr[row_num + 1] = row_fixed_ptr[row_num] + np.int64(fixed_slots.size)
        row_remaining_ptr[row_num + 1] = row_remaining_ptr[row_num] + np.int64(remaining_slots.size)
        fixed_slots_chunks.append(fixed_slots)
        fixed_vals_chunks.append(fixed_vals)
        remaining_slots_chunks.append(remaining_slots)

    fixed_slots_flat = (
        np.concatenate(fixed_slots_chunks).astype(np.int64, copy=False)
        if fixed_slots_chunks
        else np.empty(0, dtype=np.int64)
    )
    fixed_vals_flat = (
        np.concatenate(fixed_vals_chunks).astype(np.uint8, copy=False)
        if fixed_vals_chunks
        else np.empty(0, dtype=np.uint8)
    )
    remaining_slots_flat = (
        np.concatenate(remaining_slots_chunks).astype(np.int64, copy=False)
        if remaining_slots_chunks
        else np.empty(0, dtype=np.int64)
    )
    return (
        row_entry_ptr,
        row_fixed_ptr,
        fixed_slots_flat,
        fixed_vals_flat,
        row_remaining_ptr,
        remaining_slots_flat,
    )


# =========================
# OOP pipeline
# =========================
class PrepLP:
    """
    Prepare LP ingredients for the canonical ring pipeline.

    Main outputs:
      - global_keys
      - known_values_symbolic
      - known_values
      - inflation_matrix
    """

    def __init__(
        self,
        n: int,
        distribution: RingDistributionProtocol,
        *,
        problem_name: str | None = None,
        marginal_filter_fn=None,
        show_progress: bool = True,
        auto_discover_symmetries: bool = True,
        compress_rows_under_discovered_group: bool = True,
        verbose_symmetry_discovery: bool = True,
        verbose_cache: bool = True,
    ) -> None:
        self._requested_n = int(n)
        self.distribution = distribution
        self._problem_name = self._normalize_problem_name(problem_name)
        self.marginal_filter_fn = marginal_filter_fn
        self.show_progress = show_progress
        self.auto_discover_symmetries = auto_discover_symmetries
        self.compress_rows_under_discovered_group = compress_rows_under_discovered_group
        self.verbose_symmetry_discovery = verbose_symmetry_discovery
        self.verbose_cache = verbose_cache
        self.prob = ring_problem(self._requested_n, distribution)
        self._cached_inflation_matrix: csr_array | None = None
        self._cached_global_keys: np.ndarray | None = None
        self._cached_nof_caonical_global_events: int | None = None
        self._cached_solver_column_payload: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        self._cache_written = False
        self._initialize_cache()

    @staticmethod
    def _normalize_problem_name(problem_name: str | None) -> str | None:
        if problem_name is None:
            return None
        normalized = str(problem_name).strip()
        if not normalized:
            return None
        if normalized.lower().endswith(".npz"):
            return normalized[:-4]
        return normalized

    @property
    def problem_name(self) -> str | None:
        return self._problem_name

    @property
    def cache_name(self) -> str | None:
        if self.problem_name is None:
            return None
        return f"{self.problem_name}_lp_input_cache.npz"

    @property
    def output_name(self) -> str | None:
        if self.problem_name is None:
            return None
        return f"{self.problem_name}_lp_solution_output.npz"

    @cached_property
    def cache_path(self) -> Path | None:
        if self.cache_name is None:
            return None
        cache_dir = Path(__file__).resolve().parent / "cache"
        return cache_dir / self.cache_name

    @cached_property
    def output_path(self) -> Path | None:
        if self.output_name is None:
            return None
        output_dir = Path(__file__).resolve().parent / "outputs"
        return output_dir / self.output_name

    @staticmethod
    def _incompatible_cache_error() -> ValueError:
        return ValueError("Incompatible cache already exists with that name")

    def _initialize_cache(self) -> None:
        if self.cache_path is None:
            return
        if self.cache_path.exists():
            with progress_stage(
                f"Checking LP input cache at {self.cache_path}",
                enabled=self.verbose_cache,
                end_message=lambda elapsed: (
                    "Loaded cached LP constraints from "
                    f"{self.cache_path} in {elapsed:.2f}s "
                    f"(rows={self._cached_inflation_matrix.shape[0]}, "
                    f"cols={self._cached_inflation_matrix.shape[1]})"
                ),
            ):
                self._load_cache_if_available()
        return

    def _load_cache_if_available(self) -> None:
        if self.cache_path is None or not self.cache_path.exists():
            return
        try:
            with np.load(self.cache_path, allow_pickle=True) as z:
                required = {
                    "cache_format_version",
                    "requested_n",
                    "outcomes",
                    "ambient_dimension",
                    "original_symmetry_generators",
                    "discovered_symmetry_generators",
                    "inflation_matrix_shape",
                    "inflation_matrix_indptr",
                    "inflation_matrix_indices",
                    "inflation_matrix_data_entries",
                    "global_keys",
                    "solver_aptrb",
                    "solver_aptre",
                    "solver_asub",
                    "solver_aval",
                    "row_names",
                }
                if any(key not in z.files for key in required):
                    raise self._incompatible_cache_error()
                if int(z["cache_format_version"]) != int(CACHE_FORMAT_VERSION):
                    raise self._incompatible_cache_error()
                if int(z["requested_n"]) != self._requested_n:
                    raise self._incompatible_cache_error()
                if int(z["outcomes"]) != self.outcomes:
                    raise self._incompatible_cache_error()
                if int(z["ambient_dimension"]) != self.N:
                    raise self._incompatible_cache_error()

                cached_core_symmetries = np.asarray(z["original_symmetry_generators"], dtype=int)
                if not np.array_equal(cached_core_symmetries, self.core_symmetries):
                    raise self._incompatible_cache_error()

                cached_discovered_symmetries = np.asarray(z["discovered_symmetry_generators"], dtype=int)
                if not np.array_equal(cached_discovered_symmetries, self.discovered_symmetries):
                    raise self._incompatible_cache_error()

                row_names = np.asarray(z["row_names"], dtype=str)
                inflation_shape_raw = np.asarray(z["inflation_matrix_shape"], dtype=np.int64)
                if inflation_shape_raw.shape != (2,):
                    raise self._incompatible_cache_error()
                inflation_shape = tuple(int(x) for x in inflation_shape_raw.tolist())
                if inflation_shape[0] != self.nof_marginals:
                    raise self._incompatible_cache_error()
                if inflation_shape[1] != int(np.asarray(z["global_keys"], dtype=np.uint64).size):
                    raise self._incompatible_cache_error()
                if not np.array_equal(row_names, np.asarray(self.row_labels, dtype=str)):
                    raise self._incompatible_cache_error()

                global_keys = np.asarray(z["global_keys"], dtype=np.uint64)
                nof_caonical_global_events = int(global_keys.size)
                if nof_caonical_global_events != inflation_shape[1]:
                    raise self._incompatible_cache_error()

                try:
                    indptr = np.asarray(z["inflation_matrix_indptr"], dtype=np.int64)
                    indices = np.asarray(z["inflation_matrix_indices"], dtype=np.int64)
                    data = np.asarray(z["inflation_matrix_data_entries"], dtype=np.float64)
                    solver_aptrb = np.asarray(z["solver_aptrb"], dtype=np.int64)
                    solver_aptre = np.asarray(z["solver_aptre"], dtype=np.int64)
                    solver_asub = np.asarray(z["solver_asub"], dtype=np.int32)
                    solver_aval = np.asarray(z["solver_aval"], dtype=np.float64)
                except (TypeError, ValueError) as exc:
                    raise self._incompatible_cache_error() from exc

                self._cached_inflation_matrix = csr_array(
                    (data, indices, indptr),
                    shape=inflation_shape,
                )
                self._cached_global_keys = global_keys
                self._cached_nof_caonical_global_events = nof_caonical_global_events
                self._cached_solver_column_payload = (
                    np.ascontiguousarray(solver_aptrb),
                    np.ascontiguousarray(solver_aptre),
                    np.ascontiguousarray(solver_asub),
                    np.ascontiguousarray(solver_aval),
                )
        except ValueError as exc:
            if str(exc) == str(self._incompatible_cache_error()):
                raise
            raise self._incompatible_cache_error() from exc
        except (OSError, TypeError, KeyError) as exc:
            raise self._incompatible_cache_error() from exc

    def _save_cache(
        self,
        inflation_matrix: csr_array,
        solver_payload: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        if self.cache_path is None or self._cache_written:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        solver_aptrb, solver_aptre, solver_asub, solver_aval = solver_payload
        with progress_stage(
            f"Saving LP input cache to {self.cache_path}",
            enabled=self.verbose_cache,
            end_message=lambda elapsed: (
                f"Saved LP constraints cache to {self.cache_path} in {elapsed:.2f}s"
            ),
        ):
            np.savez_compressed(
                self.cache_path,
                cache_format_version=CACHE_FORMAT_VERSION,
                requested_n=np.int64(self._requested_n),
                outcomes=np.int64(self.outcomes),
                ambient_dimension=np.int64(self.N),
                original_symmetry_generators=np.asarray(self.core_symmetries, dtype=int),
                discovered_symmetry_generators=np.asarray(self.discovered_symmetries, dtype=int),
                inflation_matrix_shape=np.asarray(inflation_matrix.shape, dtype=np.int64),
                inflation_matrix_indptr=np.asarray(inflation_matrix.indptr, dtype=np.int64),
                inflation_matrix_indices=np.asarray(inflation_matrix.indices, dtype=np.int64),
                inflation_matrix_data_entries=np.asarray(inflation_matrix.data, dtype=np.float64),
                global_keys=np.asarray(self.global_keys, dtype=np.uint64),
                solver_aptrb=np.asarray(solver_aptrb, dtype=np.int64),
                solver_aptre=np.asarray(solver_aptre, dtype=np.int64),
                solver_asub=np.asarray(solver_asub, dtype=np.int32),
                solver_aval=np.asarray(solver_aval, dtype=np.float64),
                row_names=np.asarray(self.row_labels, dtype=str),
            )
        self._cache_written = True

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
        """One-hot ambient dimension `n*(n-1)*outcomes` for group action."""
        return self._core_group_chain_data[2]

    @property
    def nof_off_diagonal_slots(self) -> int:
        """Number of off-diagonal copy-index slots in the ring global event."""
        return self.n * (self.n - 1)

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
        copy_labels = tuple(range(1, self.n + 1))
        total_candidates = sum(
            sum(1 for _ in _derangements(subset)) * (self.outcomes ** len(subset))
            for subset_size in range(2, self.n + 1)
            for subset in combinations(copy_labels, subset_size)
        )
        progress = tqdm(
            total=total_candidates,
            desc="Canonicalizing marginals",
            disable=not self.show_progress,
        )
        for subset_size in range(2, self.n + 1):
            for subset in combinations(copy_labels, subset_size):
                for image in _derangements(subset):
                    for pat in product(base_outcomes, repeat=subset_size):
                        support = _marginal_support_from_cycle_cover(subset, image, pat, self.n, self.outcomes)
                        canonical_support = canonical_leximin_support_indices(support, self.N, self.core_level_invperms)
                        key = tuple(int(x) for x in canonical_support.tolist())
                        if key not in seen_keys:
                            marginal = _marginal_from_support_key(key, self.n, self.outcomes)
                            if self.marginal_filter_fn is None or self.marginal_filter_fn(marginal):
                                seen_keys.add(key)
                                marginals.append(marginal)
                                support_keys.append(key)
                        progress.update(1)
        progress.close()
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
          P_global(A^{1,2}=1,A^{2,3}=0,A^{3,1}=1)
          -> P_loop(A=1,A=0,A=1)
        """
        J = _perm_from_marginal(marginal)
        cycles = _cycles_from_J(J)
        by_i = {int(row[1]): row for row in marginal}
        loop_counts: Dict[str, int] = {}
        value = sp.Integer(1)

        for cyc in cycles:
            if len(cyc) < 2:
                raise ValueError("Ring marginals cannot contain 1-cycles.")
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
            verbose=False,
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
    def row_extension_counts(self) -> np.ndarray:
        """Number of compatible global extensions for each final marginal row."""
        counts = np.empty(self.nof_marginals, dtype=np.int64)
        for row_num, marginal in enumerate(self.marginals):
            free_slots = self.nof_off_diagonal_slots - len(marginal)
            if free_slots < 0:
                raise ValueError("Marginal fixes more slots than the off-diagonal ring supports.")
            counts[row_num] = pow(self.outcomes, free_slots)
        return counts

    @cached_property
    def _row_extension_descriptor_payload(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Flat per-row metadata for the parallel global-extension kernel."""
        return _build_row_extension_descriptors(
            self.marginals,
            n=self.n,
            outcomes=self.outcomes,
        )

    @cached_property
    def _canonical_global_lhs_payload(
        self,
    ) -> Tuple[
        csr_array,
        int,
        np.ndarray,
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]:
        """
        Tuple `(inflation_matrix, nof_caonical_global_events, global_keys,
        solver_column_payload)` for the direct ring LP system.
        """
        if (
            self._cached_inflation_matrix is not None
            and self._cached_global_keys is not None
            and self._cached_nof_caonical_global_events is not None
            and self._cached_solver_column_payload is not None
        ):
            return (
                self._cached_inflation_matrix,
                self._cached_nof_caonical_global_events,
                self._cached_global_keys,
                self._cached_solver_column_payload,
            )

        (
            row_entry_ptr,
            row_fixed_ptr,
            fixed_slots_flat,
            fixed_vals_flat,
            row_remaining_ptr,
            remaining_slots_flat,
        ) = self._row_extension_descriptor_payload
        total_entries = int(row_entry_ptr[-1]) if row_entry_ptr.size else 0

        with progress_stage(
            "Finding global extensions...",
            enabled=self.show_progress,
            end_message=lambda elapsed: (
                f"Enumerated {total_entries} canonicalized extensions in {elapsed:.2f}s"
            ),
        ):
            all_keys = _fill_global_extension_keys_parallel(
                row_entry_ptr,
                row_fixed_ptr,
                fixed_slots_flat,
                fixed_vals_flat,
                row_remaining_ptr,
                remaining_slots_flat,
                self.nof_off_diagonal_slots,
                self.outcomes,
                self.level_invperms,
            )

        with progress_stage(
            "Finalizing sparse extension matrix...",
            enabled=self.show_progress,
            end_message=lambda elapsed: (
                "Constraint matrix finalized: "
                f"rows={inflation_matrix.shape[0]}, "
                f"cols={inflation_matrix.shape[1]}, "
                f"nnz={inflation_matrix.nnz} "
                f"in {elapsed:.2f}s"
            ),
        ):
            global_keys, inverse = _stable_unique_inverse(all_keys)
            nof_caonical_global_events = int(global_keys.size)
            if nof_caonical_global_events > np.iinfo(np.int32).max:
                raise ValueError("Ring LP exceeds the current MOSEK Python binding variable limit.")

            sparse_matrix_cols = inverse.astype(np.int64, copy=False)
            top_row_nnz = _sort_rows_and_count_unique(sparse_matrix_cols, row_entry_ptr)
            indptr = np.empty(self.nof_marginals + 1, dtype=np.int64)
            indptr[0] = 0
            if top_row_nnz.size > 0:
                indptr[1:] = np.cumsum(top_row_nnz, dtype=np.int64)
            indices = np.empty(int(indptr[-1]), dtype=np.int64)
            data = np.empty(int(indptr[-1]), dtype=np.float64)
            _fill_direct_csr_from_sorted_rows(
                sparse_matrix_cols,
                row_entry_ptr,
                indptr,
                indices,
                data,
            )
            inflation_matrix = csr_array(
                (data, indices, indptr),
                shape=(self.nof_marginals, nof_caonical_global_events),
            )
            solver_payload = _csr_to_column_payload(
                indptr,
                indices,
                data,
                nof_caonical_global_events,
            )
            solver_payload = tuple(np.ascontiguousarray(arr) for arr in solver_payload)

        self._cached_inflation_matrix = inflation_matrix
        self._cached_global_keys = global_keys
        self._cached_nof_caonical_global_events = nof_caonical_global_events
        self._cached_solver_column_payload = solver_payload
        self._save_cache(inflation_matrix, solver_payload)
        return (
            inflation_matrix,
            nof_caonical_global_events,
            global_keys,
            solver_payload,
        )

    @property
    def global_keys(self) -> np.ndarray:
        """Canonical uint64 keys of global-event LP columns."""
        if self._cached_global_keys is not None:
            return self._cached_global_keys
        return self._canonical_global_lhs_payload[2]

    @property
    def nof_caonical_global_events(self) -> int:
        """Number of canonical global-event LP columns."""
        if self._cached_nof_caonical_global_events is not None:
            return self._cached_nof_caonical_global_events
        return self._canonical_global_lhs_payload[1]

    @property
    def inflation_matrix(self) -> csr_array:
        """Direct marginal-form sparse matrix `A_direct`."""
        if self._cached_inflation_matrix is not None:
            return self._cached_inflation_matrix
        return self._canonical_global_lhs_payload[0]

    @property
    def solver_column_payload(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Cached column-major equality payload for direct MOSEK task assembly."""
        if self._cached_solver_column_payload is not None:
            return self._cached_solver_column_payload
        return self._canonical_global_lhs_payload[3]

    @property
    def nof_lp_vars(self):
        """Number of direct global-event variables in the final LP."""
        if self._cached_global_keys is not None:
            return int(self._cached_global_keys.size)
        if self._cached_nof_caonical_global_events is not None:
            return int(self._cached_nof_caonical_global_events)
        return self.nof_caonical_global_events

    @property
    def nof_lp_constraints(self) -> int:
        """Number of LP constraints in the direct marginal formulation."""
        if self._cached_inflation_matrix is not None:
            return int(self._cached_inflation_matrix.shape[0])
        return self.nof_marginals

    def variable_names(self) -> np.ndarray:
        """Compatibility alias for the direct global-event uint64 keys."""
        return self.global_keys

    @cached_property
    def _mass_weights(self) -> np.ndarray:
        """Row multiplicities as float weights for direct mass objectives."""
        return self.row_orbit_multiplicities.astype(np.float64, copy=False)

    @cached_property
    def _mass_objective(self) -> np.ndarray:
        """Column objective giving the multiplicity-corrected global mass."""
        matrix = self.inflation_matrix
        return _csr_weighted_column_sums(
            matrix.indptr.astype(np.int64, copy=False),
            matrix.indices.astype(np.int64, copy=False),
            matrix.data.astype(np.float64, copy=False),
            self._mass_weights,
            self.nof_lp_vars,
        )

    @cached_property
    def _known_mass(self) -> float:
        """Multiplicity-corrected total known mass for direct relaxations."""
        return float(np.dot(self._mass_weights, self.known_values.astype(np.float64, copy=False)))

    def solve(
        self,
        *,
        mode: str = "incompatible_fraction",
        optimizer: str = "free_simplex",
        verbose: int = 0,
    ) -> Dict:
        """Solve the direct ring LP in feasibility or relaxed incompatibility modes."""
        import mosek
        from inflation.lp.lp_utils import streamprinter

        if verbose > 1:
            t0 = perf_counter()
            t_total = perf_counter()
            print("Starting pre-processing for the LP solver...")

        solve_mode = _resolve_ring_solve_mode(mode)
        aptrb, aptre, asub, aval = self.solver_column_payload
        nof_lp_vars = self.nof_lp_vars
        nof_constraints = self.nof_lp_constraints
        known_values = self.known_values.astype(np.float64, copy=False)
        global_keys = self.global_keys.astype(np.uint64, copy=False)
        constraint_names = np.asarray(self.row_labels, dtype=object)
        known_mass = self._known_mass
        mass_objective = self._mass_objective.astype(np.float64, copy=False)

        if verbose > 1:
            print("Proceeding with ring LP initialization...")

        numcon = nof_constraints
        numvar = nof_lp_vars
        if numcon > np.iinfo(np.int32).max or numvar > np.iinfo(np.int32).max:
            raise ValueError("Ring LP exceeds the current MOSEK Python binding dimension limit.")

        optimizer_choice = _resolve_mosek_optimizer(optimizer)
        tolerance = 1e-9
        rhs = np.ascontiguousarray(known_values, dtype=np.float64)
        if solve_mode == "feasibility":
            objective_vector = np.zeros(numvar, dtype=np.float64)
            bkc = _boundkey_array(mosek.boundkey.fx, numcon)
            blc = rhs.copy()
            buc = rhs.copy()
            objective_sense = mosek.objsense.maximize
        elif solve_mode == "incompatible_fraction":
            if known_mass <= 0.0:
                raise ValueError("known_mass must be positive to compute incompatible_fraction.")
            objective_vector = mass_objective.copy()
            bkc = _boundkey_array(mosek.boundkey.up, numcon)
            blc = np.zeros(numcon, dtype=np.float64)
            buc = rhs.copy()
            objective_sense = mosek.objsense.maximize
        else:
            if known_mass <= 0.0:
                raise ValueError("known_mass must be positive to compute generalized_robustness.")
            objective_vector = mass_objective.copy()
            bkc = _boundkey_array(mosek.boundkey.lo, numcon)
            blc = rhs.copy()
            buc = np.zeros(numcon, dtype=np.float64)
            objective_sense = mosek.objsense.minimize

        bkx = _boundkey_array(mosek.boundkey.lo, numvar)
        blx = np.zeros(numvar, dtype=np.float64)
        bux = np.zeros(numvar, dtype=np.float64)

        aptrb = np.ascontiguousarray(aptrb, dtype=np.int64)
        aptre = np.ascontiguousarray(aptre, dtype=np.int64)
        asub = np.ascontiguousarray(asub, dtype=np.int32)
        aval = np.ascontiguousarray(aval, dtype=np.float64)
        objective_vector = np.ascontiguousarray(objective_vector, dtype=np.float64)
        bkc = np.ascontiguousarray(bkc, dtype=np.int32)
        blc = np.ascontiguousarray(blc, dtype=np.float64)
        buc = np.ascontiguousarray(buc, dtype=np.float64)
        bkx = np.ascontiguousarray(bkx, dtype=np.int32)
        blx = np.ascontiguousarray(blx, dtype=np.float64)
        bux = np.ascontiguousarray(bux, dtype=np.float64)

        with mosek.Env() as env:
            with mosek.Task(env) as task:
                task.putintparam(mosek.iparam.sim_reformulation, mosek.simreform.aggressive)
                task.putintparam(mosek.iparam.sim_switch_optimizer, mosek.onoffkey.on)
                task.putintparam(mosek.iparam.optimizer, optimizer_choice)
                task.putintparam(mosek.iparam.sim_solve_form, mosek.solveform.primal)
                if verbose > 0:
                    task.set_Stream(mosek.streamtype.log, streamprinter)
                    task.putintparam(mosek.iparam.log_include_summary, mosek.onoffkey.on)
                    task.putintparam(mosek.iparam.log_storage, 1)
                if verbose < 2:
                    task.putintparam(mosek.iparam.log_sim, 0)
                    task.putintparam(mosek.iparam.log_intpnt, 0)

                task.putobjsense(objective_sense)
                if verbose > 0:
                    print(f"Size of constraint matrix: ({numcon}, {numvar})")

                with progress_stage(
                    "Starting task.inputdata in Mosek...",
                    enabled=verbose > 1,
                    end_message=lambda elapsed: f"Mosek input data loaded in {elapsed:.2f}s",
                ):
                    task.inputdata(
                        numcon,
                        numvar,
                        objective_vector,
                        0.0,
                        aptrb,
                        aptre,
                        asub,
                        aval,
                        bkc,
                        blc,
                        buc,
                        bkx,
                        blx,
                        bux,
                    )

                if verbose > 1:
                    print("Pre-processing took", format(perf_counter() - t0, ".4f"), "seconds.\n")
                    t0 = perf_counter()
                if verbose > 2:
                    with progress_stage(
                        "Writing problem to debug_lp.ptf...",
                        enabled=True,
                        end_message=lambda elapsed: f"Wrote debug_lp.ptf in {elapsed:.2f}s",
                    ):
                        task.writedata("debug_lp.ptf")

                if verbose > 0:
                    print("\nSolving the problem...\n")
                trmcode = task.optimize()
                if verbose > 1:
                    print("Solving took", format(perf_counter() - t0, ".4f"), "seconds.")

                basic = mosek.soltype.bas
                (
                    problemsta,
                    solutionsta,
                    skc,
                    skx,
                    skn,
                    xc,
                    xx,
                    yy,
                    slc,
                    suc,
                    slx,
                    sux,
                    snx,
                ) = task.getsolution(basic)

                xx = np.asarray(xx, dtype=np.float64)
                yy = np.asarray(yy, dtype=np.float64)
                primal = task.getprimalobj(basic)
                dual = task.getdualobj(basic)

                status_str = solutionsta.__repr__()
                solver_success = solutionsta != mosek.solsta.unknown
                has_optimal_primal = solutionsta == mosek.solsta.optimal
                term_tuple = mosek.Env.getcodedesc(trmcode)
                if solutionsta == mosek.solsta.unknown and verbose > 0:
                    print("The solution status is unknown.")
                    print(f"   Termination code: {term_tuple}")

                optimized_mass = float(np.dot(mass_objective, xx)) if has_optimal_primal else np.nan
                if solve_mode == "feasibility":
                    success = bool(has_optimal_primal)
                    incompatible_fraction = np.nan
                    generalized_robustness = np.nan
                elif solve_mode == "incompatible_fraction":
                    success = bool(has_optimal_primal and optimized_mass >= known_mass - tolerance)
                    incompatible_fraction = (
                        max(0.0, 1.0 - (optimized_mass / known_mass))
                        if has_optimal_primal
                        else np.nan
                    )
                    generalized_robustness = np.nan
                else:
                    success = bool(has_optimal_primal and optimized_mass <= known_mass + tolerance)
                    incompatible_fraction = np.nan
                    generalized_robustness = (
                        max(0.0, (optimized_mass / known_mass) - 1.0)
                        if has_optimal_primal
                        else np.nan
                    )

                cert_data = yy.astype(np.float64, copy=False)
                cert_col = np.nonzero(~np.isclose(cert_data, 0.0))[0].astype(np.int64, copy=False)
                cert_row = np.zeros(cert_col.size, dtype=np.int32)
                cert_vals = cert_data[cert_col]
                sparse_certificate = coo_array(
                    (cert_vals, (cert_row, cert_col)),
                    shape=(1, numcon),
                )

                x_values = dict(zip(global_keys.tolist(), xx.tolist()))
                certificate = dict(zip(constraint_names.tolist(), cert_data.tolist()))
                for var in list(certificate):
                    if np.isclose(certificate[var], 0.0):
                        del certificate[var]

                if verbose > 1:
                    print("\nTotal execution time:", format(perf_counter() - t_total, ".4f"), "seconds.")

                return {
                    "primal_value": primal,
                    "dual_value": dual,
                    "status": status_str,
                    "success": bool(success),
                    "solver_success": bool(solver_success),
                    "mode": solve_mode,
                    "known_mass": float(known_mass),
                    "optimized_mass": float(optimized_mass),
                    "incompatible_fraction": incompatible_fraction,
                    "generalized_robustness": generalized_robustness,
                    "dual_certificate": certificate,
                    "sparse_certificate": sparse_certificate,
                    "constraint_names": np.asarray(constraint_names, dtype=str),
                    "x": x_values,
                    "term_code": term_tuple,
                }

    def save_solution(
        self,
        solution: Dict,
        path: str | Path | None = None,
        *,
        compression: bool = True,
    ) -> Path:
        """Save a direct-basis PrepLP solution archive."""
        target_path = self.output_path if path is None else Path(path)
        if target_path is None:
            raise ValueError("No output path configured for this PrepLP instance.")
        return save_prep_lp_solution(solution, target_path, compression=compression)

    @staticmethod
    def read_solution(path: str | Path, *, allow_pickle: bool = True) -> Dict:
        """Read a direct-basis PrepLP solution archive."""
        return read_prep_lp_solution(path, allow_pickle=allow_pickle)

    @staticmethod
    def load_solution(path: str | Path, *, allow_pickle: bool = True) -> Dict:
        """Compatibility alias for `read_solution()`."""
        return read_prep_lp_solution(path, allow_pickle=allow_pickle)


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    from inflation.distributions import EJMDistribution, NSIPRDistribution, RGBDistribution

    n = 4

    demos = [
        ("EJM", EJMDistribution()),
        ("EJM coarse [[0],[1],[2,3]]", EJMDistribution(coarsen=[[0], [1], [2, 3]])),
        ("RGB", RGBDistribution()),
        ("NSI-PR", NSIPRDistribution()),
    ]

    demo_preps: dict[str, PrepLP] = {}
    for label, distribution in demos:
        print(f"\n=== Symmetry demo: {label} (n={n}) ===")
        prep = PrepLP(
            n,
            distribution,
            problem_name=f"NSI-PR_n={n}" if label == "NSI-PR" else None,
            show_progress=True,
            auto_discover_symmetries=True,
            compress_rows_under_discovered_group=True,
            verbose_symmetry_discovery=True,
        )
        print(f"  base rows={prep.base_nof_marginals}, compressed rows={prep.nof_marginals}")
        demo_preps[label] = prep

    prep_nsi = demo_preps["NSI-PR"]
    print("PrepLP initialized for NSI-PR demo; materializing LP inputs before Mosek.")
    _ = prep_nsi.global_keys
    print(
        "LP inputs ready for NSI-PR demo: "
        f"rows={prep_nsi.nof_lp_constraints}, cols={prep_nsi.nof_lp_vars}. "
        "Starting Mosek setup."
    )

    solution = prep_nsi.solve(
        verbose=2,
    )

    print(solution["status"])
    print(
        f"Exact feasibility: {solution['success']}. "
        f"Incompatible fraction: {solution['incompatible_fraction']:.12g}"
    )
    if prep_nsi.output_path is not None:
        prep_nsi.save_solution(solution)
        print(f"Saved LP solution archive to {prep_nsi.output_path}")

    def _evaluate_sparse_certificate_on_knowns(
        sparse_certificate: coo_array,
        known_values: np.ndarray,
    ) -> float:
        """Evaluate a direct row-basis certificate on the known marginal values."""
        cert_coo = sparse_certificate
        value = 0.0
        for col, coeff in zip(cert_coo.col.tolist(), cert_coo.data.tolist()):
            value += float(coeff) * float(known_values[int(col)])
        return value

    def _print_infeasibility_certificate_analysis(
        solution_dict: dict,
        known_values: np.ndarray,
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

        cert_value = _evaluate_sparse_certificate_on_knowns(
            solution_dict["sparse_certificate"], known_values
        )

        print("\nCertificate analysis:")
        print(f"  nonzero terms (after chop): {len(cleaned)}")
        print(f"  constraint-row terms: {len(cleaned)}")
        print(f"  certificate value on knowns: {cert_value:.12g}")
        print("  incompatibility witness criterion: certificate < 0")

        top_terms = sorted(cleaned.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
        print(f"  top {len(top_terms)} terms by |coefficient|:")
        for var, coeff in top_terms:
            print(f"    {coeff:+.12g} * {var}")

    if not solution.get("success", False):
        _print_infeasibility_certificate_analysis(
            solution,
            prep_nsi.known_values.astype(np.float64, copy=False),
        )
