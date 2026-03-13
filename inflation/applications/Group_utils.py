from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
from numba import int64, njit, uint8
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.permutations import Permutation


@njit(uint8[:](uint8[:], int64), cache=True, fastmath=True)
def to_lex_representation(evt: np.ndarray, outcomes: int) -> np.ndarray:
    """Compact event -> one-hot lex vector (length n^2 * outcomes)."""
    n2 = evt.shape[0]
    lex = np.zeros(n2 * outcomes, dtype=np.uint8)
    base = np.uint16(outcomes)
    idx = evt + base * np.arange(n2, dtype=np.uint16)
    for s in range(n2):
        lex[idx[s]] = 1
    return lex


@njit(uint8[:](uint8[:], int64), cache=True, fastmath=True)
def from_lex_representation(lex_evt: np.ndarray, outcomes: int) -> np.ndarray:
    """One-hot lex vector -> compact event."""
    lex_evt = np.ascontiguousarray(lex_evt)
    n2 = lex_evt.shape[0] // outcomes
    evt = np.empty(n2, dtype=np.uint8)
    blocks = lex_evt.reshape(n2, outcomes)
    for s in range(n2):
        idx = 0
        for j in range(outcomes):
            if blocks[s, j] != 0:
                idx = j
                break
        evt[s] = idx
    return evt


def _normalize_perm_list(g: List[int], N: int) -> List[int]:
    """Ensure 0-based bijection of range(N) from list (accepts 1- or 0-based)."""
    if len(g) != N:
        raise ValueError(f"Permutation length {len(g)} != {N}")
    if max(g) == N:
        g = [x - 1 for x in g]
    if sorted(g) != list(range(N)):
        raise ValueError("Invalid permutation (not a bijection).")
    return g


def build_sympy_group(raw_G: Union[List[List[int]], np.ndarray], N: int) -> PermutationGroup:
    """Build group from raw list permutations (0- or 1-based)."""
    gens = [Permutation(_normalize_perm_list(gl, N)) for gl in raw_G] or [Permutation(list(range(N)))]
    return PermutationGroup(gens)


def _sorted_group_elements_matrix(group: PermutationGroup) -> np.ndarray:
    group.schreier_sims()
    elements = np.array(list(group.generate_schreier_sims(af=True)), dtype=int)
    if elements.ndim == 1:
        elements = elements[np.newaxis, :]
    return elements[np.lexsort(np.rot90(elements))]


def _perm_dtype_for_width(width: int):
    if width <= np.iinfo(np.uint8).max:
        return np.uint8
    if width <= np.iinfo(np.uint16).max:
        return np.uint16
    if width <= np.iinfo(np.uint32).max:
        return np.uint32
    return np.uint64


def prepare_group_chain(
    G: PermutationGroup,
    N: int,
    outcomes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Precompute exact group actions for support and event canonicalization.

    Returns
    -------
    tuple
        `(group_perms, slot_sources, outcome_maps)` where:
        - `group_perms[g, coord]` is the forward image of coordinate `coord`,
        - `slot_sources[g, out_slot]` is the input slot contributing to output slot `out_slot`,
        - `outcome_maps[g, out_slot, in_outcome]` is the output outcome obtained when the
          input slot has outcome `in_outcome`.
    """
    if N % outcomes != 0:
        raise ValueError("The ambient lex width must be divisible by the number of outcomes.")

    slot_count = N // outcomes
    perm_dtype = _perm_dtype_for_width(N)
    slot_dtype = _perm_dtype_for_width(slot_count)
    outcome_dtype = _perm_dtype_for_width(outcomes)

    group_perms = _sorted_group_elements_matrix(G).astype(perm_dtype, copy=False)
    inv_perms = np.empty_like(group_perms)
    identity = np.arange(N, dtype=perm_dtype)
    for row in range(group_perms.shape[0]):
        inv_perms[row, group_perms[row]] = identity

    slot_sources = np.empty((group_perms.shape[0], slot_count), dtype=slot_dtype)
    outcome_maps = np.empty((group_perms.shape[0], slot_count, outcomes), dtype=outcome_dtype)

    for row in range(inv_perms.shape[0]):
        inv_perm = inv_perms[row]
        for out_slot in range(slot_count):
            src_slot = None
            seen_outcomes: set[int] = set()
            for out_outcome in range(outcomes):
                out_coord = out_slot * outcomes + out_outcome
                pre_coord = int(inv_perm[out_coord])
                src_slot_candidate, src_outcome = divmod(pre_coord, outcomes)
                if src_slot is None:
                    src_slot = src_slot_candidate
                elif src_slot_candidate != src_slot:
                    raise ValueError("Symmetry does not preserve outcome blocks by slot.")
                outcome_maps[row, out_slot, src_outcome] = out_outcome
                seen_outcomes.add(src_outcome)
            if src_slot is None or len(seen_outcomes) != outcomes:
                raise ValueError("Symmetry does not induce a per-slot outcome permutation.")
            slot_sources[row, out_slot] = src_slot

    return group_perms, slot_sources, outcome_maps


@njit(cache=True, fastmath=True)
def _lexicographically_less_int64(left: np.ndarray, right: np.ndarray) -> bool:
    for idx in range(left.size):
        if left[idx] < right[idx]:
            return True
        if left[idx] > right[idx]:
            return False
    return False


@njit(cache=True, fastmath=True)
def canonical_leximin_coset_chain_uint64(
    evt: np.ndarray,
    outcomes: int,
    slot_sources: np.ndarray,
    outcome_maps: np.ndarray,
) -> np.uint64:
    """
    Exact canonical representative of `evt` under the full symmetry group.

    The comparison order is the lexicographically minimal compact event sequence,
    which corresponds to the lexicographically maximal one-hot representation.
    """
    group_order = slot_sources.shape[0]
    slot_count = slot_sources.shape[1]
    best_group = 0

    for group_idx in range(1, group_order):
        cand_better = False
        best_better = False
        for out_slot in range(slot_count):
            cand_digit = int(outcome_maps[group_idx, out_slot, evt[int(slot_sources[group_idx, out_slot])]])
            best_digit = int(outcome_maps[best_group, out_slot, evt[int(slot_sources[best_group, out_slot])]])
            if cand_digit < best_digit:
                cand_better = True
                break
            if cand_digit > best_digit:
                best_better = True
                break
        if cand_better and not best_better:
            best_group = group_idx

    outcomes_u64 = np.uint64(outcomes)
    acc = np.uint64(0)
    base = np.uint64(1)
    for out_slot in range(slot_count):
        digit = np.uint64(
            outcome_maps[best_group, out_slot, evt[int(slot_sources[best_group, out_slot])]]
        )
        acc += digit * base
        if out_slot + 1 < slot_count:
            base *= outcomes_u64
    return acc


@njit(cache=True, fastmath=True)
def canonical_leximin_support_indices(
    support_indices: np.ndarray,
    N: int,
    group_perms: np.ndarray,
) -> np.ndarray:
    """
    Canonicalize a sparse one-hot support under the exact full group action.
    """
    if support_indices.size == 0:
        return np.empty(0, dtype=np.int64)

    best = np.empty(support_indices.size, dtype=np.int64)
    candidate = np.empty(support_indices.size, dtype=np.int64)

    for idx in range(support_indices.size):
        best[idx] = np.int64(support_indices[idx])
    best.sort()

    for group_idx in range(group_perms.shape[0]):
        perm = group_perms[group_idx]
        for idx in range(support_indices.size):
            candidate[idx] = np.int64(perm[int(support_indices[idx])])
        candidate.sort()
        if _lexicographically_less_int64(candidate, best):
            best[:] = candidate

    return best
